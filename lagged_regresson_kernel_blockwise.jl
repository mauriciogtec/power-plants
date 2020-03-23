using Pkg
Pkg.activate(".")

using Flux
using CSV
using FileIO
using DataFrames
using SparseArrays
using LinearAlgebra
using Statistics
using Printf
using Random


"""
# Model
    X: power-plant pollution (nobs x nlags x nplants)
    Y: zip-code/grid pollution (nobs x nrows x ncols)
    Y_miss: boolean of y-missing data (same size as Y)
    eta: spatial smoothness hyper-parameter
    nu: effect shrinking regularization
"""
struct LagModel
    μ::Array{Float32,3}
    γ::Array{Float32,3}
    λ::Array{Float32,3}
    Y::Array{Float32,3}
    X::Array{Float32,3}
    Y_miss::Array{Float32,3}
    R::Int  # nrows
    C::Int  # ncols
    T::Int  # nlags
    N::Int  # nobs
    P::Int  # nplants
    function LagModel(
            X::AbstractArray{Float32,3},
            Y::AbstractArray{Float32,3},
            Y_miss::AbstractArray{Float32,3},
            η::Float32,
            ν::Float32)
        @assert length(size(X)) == 3
        @assert length(size(Y)) == 3
        @assert size(X, 1) == size(Y, 1)
        @assert size(Y) == size(Y_miss)
        nobs, nrows, ncols = size(Y)
        _, nlags, nplants = size(X)
        μ = zeros(nplants, nrows, ncols)
        λ = zeros(nplants, nrows, ncols)
        γ = zeros(nplants, nrows, ncols)
        new(μ, γ, λ, X, Y, Y_miss, nrows,
            ncols, nlags, nplants, η)
    end
end


function spatial_neighbors(
        mod::LagModel,
        p::Int, 
        r::Int,
        c::Int,
        linear::Bool=true)
    nbrs = []
    if (r > 1) push!(nbrs, (r - 1, c)) end
    if (c > 1) push!(nbrs, (r, c - 1)) end
    if (r < mod.R) push!(nbrs, (r + 1, c)) end
    if (c < mod.C) push!(nbrs, (r, c + 1)) end
    if linear
        ix = LinearIndices(mod.P, mod.R, mod.C)
        nbrs = [ix[p, r, c] for (r, c) in nbrs]
    end
    return nbrs
end

function kernel(
        t::AbstractVector{Float32},
        μ::Float32,
        λ::Float32;
        derivatives::Bool=true)
    # function
    δ = μ .- t
    Δ = 0.5 .* δ^2
    ψ = exp.(- λ .* Δ)
    β = λ .* ψ ./ √(2π)
    if !derivatives
        return β
    end
    # derivatives
    ∂μ = - (λ .* δ .* β)
    ∂λ = (ψ ./ √(2π)) .- (Δ .* β)
    ∇β = [∂λ, ∂μ]
    # second derivatives
    ∂²μ = - (λ .* β) .+ (λ .* δ .* ∂μ)
    ∂²λ = - (Δ .* ψ ./ √(2π)) .- (Δ .* ∂λ)
    ∇²β = [∂²λ, ∂²μ]
    # ∂μ∂λ = - δ .* (β + λ .* ∂λ)
    return β, ∇β, ∇²β
end


"""
Performs updates cycling through all variables.
Each step is linear time with small constant.
"""
function learn!(mod::LagModel)
    tvec = collect(Float32, range(0, mod.T - 1))
    for c in 1:mod.C, r in 1:mod.R
        # params for row, col
        γ = view(mod.γ, :, r, c)
        μ = view(mod.μ, :, r, c)
        λ = view(mod.λ, :, r, c)
        y = view(mod.Y, :, r, c)
        X = [view(mod.X, :, :, p) for p in 1:mod.P]
        y_miss = view(mod.Y_miss, :, r, c)
        # lagged effect
        kout = kernel.(tvec, μ, λ)
        β = [kout[p][1] for p in 1:mod.p]
        ϕ = [y_miss[p] * X[p] * β[p] for p in 1:mod.P]
        # errors
        y_hat = sum(γ .* ϕ) 
        ε = (1.0 - y_miss) .* (y_hat .- y)
        # can use multithread here with small overhead
        for p in 1:mod.C
            nbrs = spatial_neighbors(mod, p, r, c)
            # gamma updates (exact)
            γ_nbrs_sum = sum(mod.γ[nbrs])
            γ_lp =  ϕ[p] * γ[p] .- ε
            b =  dot(ϕ[p], γ_lp) + mod.η * γ_nbrs_sum
            H = dot(ϕ[p], ϕ[p]) + mod.η * length(nbrs) + mod.ν
            mod.γ[r, c] =  b / H

            # mu update
            μ_nbrs_sum = sum(mod.μ[nbrs])
            ∂β∂μ = kout[p][2][1]  # size t
            ∂²β∂²μ = kout[p][3][1]  # size t
            ∇ = dot(X[p] * ∂β∂μ, ε) + mod.η * (μ[p] - μ_nbrs_sum)
            ∇² = dot(X[p] * ∂²β∂²μ, ε) + mod.η
            mod.μ[p, r, c] -= ∇ / ∇²

            # lam update
            λ_nbrs_sum = sum(mod.λ[nbrs])
            ∂β∂λ = kout[p][2][2]  # size t
            ∂²β∂²λ = kout[p][3][2]  # size t
            ∇ = dot(X[p] * ∂β∂λ, ε) + mod.η * (λ[p] - λ_nbrs_sum)
            ∇² = dot(X[p] * ∂²β∂²λ, ε) + mod.η  # scalar
            mod.λ[p, r, c] -= ∇ / ∇²
        end
    end
end


function load_obs_unit_data()
    #
    N_OBS = 5
    N_LAGS = 12
    TARGET_YEAR = 2015
    TARGET_MONTH = 12
    #
    file = CSV.file("./model_dev_data/grid_pm25_subset.csv")
    grid = DataFrame(file)
    sort!(grid, (:month, :year, :lon, :lat))
    δ = 0.01  # grid width in
    d = Int(log10(1.0 / δ))
    max_lon = maximum(grid.lon)
    min_lon = minimum(grid.lon)
    max_lat = maximum(grid.lat)
    min_lat = minimum(grid.lat)
    nr = Int(round(max_lat - min_lat, digits=d) ÷ δ) + 1
    nc = Int(round(max_lon - min_lon, digits=d) ÷ δ) + 1
    num_nodes = nr * nc
    grid.row = Int.(round.(grid.lat .- min_lat, digits=d) .÷ δ) .+ 1
    grid.col = Int.(round.(grid.lon .- min_lon, digits=d) .÷ δ) .+ 1
    #
    # file = CSV.file("./model_dev_data/so2_data_subset.csv")
    # plants = DataFrame(file)
    # plants.lag = (12TARGET_YEAR + TARGET_MONTH) .- 12plants.year .- plants.month
    # filter!(r -> r.so2_tons > 0.0 && r.lag < N_LAGS, plants)
    # plants.row = Int.(round.(plants.lat .- min_lat, digits=d) .÷ δ) .+ 1
    # plants.col = Int.(round.(plants.lon .- min_lon, digits=d) .÷ δ) .+ 1
    # unique_ids = unique(plants.id)
    # num_plants = length(unique_ids)
    # nid_dict = Dict(zip(unique_ids, 1:num_plants))
    # plants.nid = [nid_dict[x] for x in plants.id]
    # return grid, plants
end

function load_power_plant_data()

end


function main()
    load_data()
    # -- 1.b extract y and X
    # y = Union{Missing, Float32}[missing for _ in 1:num_nodes]
    y = zeros(Float32, num_nodes)
    missmask = zeros(Float32, num_nodes)
    イ = LinearIndices((nr, nc))
    for (datum, i, j) in zip(grid.pm25, grid.row, grid.col)
        y[イ[i, j]] = datum
        missmask[イ[i, j]] = 1.0
    end
    y_mean = mean(y[y .> 0])
    y_std = std(y[y .> 0])
    @views y[y .> 0] .-= y_mean 
    @views y[y .> 0] ./ y_std

    X = zeros(Float32, (num_plants, nt))
    target_t = target_year * 12 + target_month
    for (datum, lag, nid) in zip(plants.so2_tons, plants.lag, plants.nid)
        if lag < nt
            X[nid, lag + 1] = datum
        end
    end

    #  standardize
    X_means = Float32[]
    X_stds = Float32[]
    for n in 1:num_plants
        x = @view X[n, :]
        μ = mean(x)
        σ = std(x)
        ix = x .> 0.0
        @views x[ix] .-= μ
        @views x[ix] ./= σ
        push!(X_means, μ)
        push!(X_stds, σ)
    end
    X = reshape(X, (1, :))

    # # -- 1.c. 
    num_params = nt * num_plants
    β = zeros(Float32, (num_params, num_nodes))
    α = zeros(Float32, num_nodes)
    θ = Flux.Params([α, β])


    function reg_loss(edges)
        e0, e1, ω = edges
        β_reg = sum(ω .* (β[e0] .- β[e1]) .^ 2)
        α_reg = sum(α .^ 2)
        1f-5 * α_reg + 1f-3 * β_reg
    end
    
    
    function logll_loss(vertices, X, y, missmask)
        u, ω = vertices
        @views yhat = α[u] .+ X * β[:, u]
        sum(ω .* missmask[u] .* (y[u] .- yhat) .^ 2)
    end
    
    
    function loss(vertices, edges, X, y, missmask)
        reg_loss(edges) + logll_loss(vertices, X, y, missmask)
    end

    # -- 2 Optimization
    opt = Flux.ADAGrad(1e-4)


    training_data = vec(Tuple.(CartesianIndices((nr, nc, nt))));
    N = length(training_data)
    shuffle!(training_data);


    λ = 0.01
    l_ = 0.0
    r_ = 0.0
    print_every = 10
    train_every = 1

    iter = 1
    num_epochs = 10
    for e in 1:num_epochs
        batch = []
        for (b, (r, c, t)) in enumerate(training_data)
            vertices = batch_vertex_builder(r, c, nr, nc)
            edges = batch_edge_builder(r, c, t, nr, nc, nt)
            datum = (vertices, edges, X, y, missmask)
            push!(batch, datum)

            if b % train_every == 0
                Flux.Optimise.train!(loss, θ, batch, opt)
                batch = []
            end
            
            # print stats
            if iter == 1
                l_ = 100.0 * logll_loss(vertices, X, y, missmask)
                r_ = 10000.0 * reg_loss(edges)
            else            
                l_ += λ * (100.0 * logll_loss(vertices, X, y, missmask) - l_)
                r_ += λ * (10000.0 * reg_loss(edges) - r_)
            end

            if iter % print_every == 0
                mβ = 100.0 * minimum(β)
                μβ = 100.0 * mean(β)
                Mβ = 100.0 * maximum(β)
                mα = 100.0 * minimum(α)
                μα = 100.0 * mean(α)
                Mα = 100.0 * maximum(α)
                c = 100.0 * iter / (num_epochs * N)
                @printf "%.3f%%, ハ: %d, え: %d, ロス: %.2f, レグ: %.2f, β: [%.3f, %.3f, %.4f], α: [%.3f, %.3f, %.4f]\n" c b e l_ r_ mβ Mβ μβ mα Mα μα
            end

            iter += 1
        end
    end

    writedlm("output/betas.csv", β, ',')
end

main()

