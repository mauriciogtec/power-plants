using Pkg; Pkg.activate(".")
using CSV
using DataFrames
using LinearAlgebra
using Statistics
using Printf
using Dates
using Base.Threads: @threads
using ProgressBars


"""
# Model
    X: power-plant pollution (nobs x nlags x nplants)
    Y: zip-code/grid pollution (nobs x nrows x ncols)
    Y_miss: boolean of y-missing data (same size as Y)
    eta: spatial smoothness hyper-parameter
    nu: effect shrinking regularization
"""
mutable struct LagModel
    μ::Array{Float32,3}  # kernel center
    γ::Array{Float32,3}  # kernel intensity
    λ::Array{Float32,3}  # kernel decay
    α::Float32  # overall intercept
    η::Float32  # spatial regularization
    ν::Float32  # power plant regularization
    X::Array{Float32,3}
    Y::Array{Float32,3}
    R::Int  # nrows
    C::Int  # ncols
    T::Int  # nlags
    N::Int  # nobs
    P::Int  # nplants
    function LagModel(
            X::AbstractArray{Float32,3},
            Y::AbstractArray{Float32,3},
            η::Float32,
            ν::Float32)
        @assert length(size(X)) == 3
        @assert length(size(Y)) == 3
        @assert size(X, 1) == size(Y, 1)
        nobs, nrows, ncols = size(Y)
        _, nlags, nplants = size(X)
        μ = zeros(Float32, (nplants, nrows, ncols))
        λ = ones(Float32, (nplants, nrows, ncols))
        γ = ones(Float32, (nplants, nrows, ncols))
        α = 0.0f0
        new(μ, γ, λ, α,
            η, ν,
            X, Y, 
            nrows, ncols, nlags, nobs, nplants)
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
        ix = LinearIndices((mod.P, mod.R, mod.C))
        nbrs = [ix[p, r, c] for (r, c) in nbrs]
    end
    return nbrs
end

function kernel(
        t::AbstractVector{Float32},
        μ::Float32,
        λ::Float32;
        derivatives::Bool=true)
    # eval kernel
    C = √(2π)
    δ = @. μ - t
    Δ = @. 0.5 * δ ^ 2
    ψ = @. exp(- λ * Δ)
    β = @. λ * ψ / C

    if !derivatives 
        return β
    end

    # derivatives
    ∂μ = @. - λ * δ * β
    ∂²μ = @. - λ * (β + δ * ∂μ)
    ∂λ = @. ψ / C - Δ * β
    ∂²λ = @. - Δ * (ψ / C - ∂λ)

    ∇β = [∂λ, ∂μ]
    ∇²β = [∂²λ, ∂²μ]

    return β, ∇β, ∇²β
end


"""
Performs updates cycling through all variables.
Each step is linear time with small constant.
"""
function learn!(mod::LagModel)
    tvec = Float32.(0:(mod.T - 1))
    prev_loss = 0.0f0
    grid = [(r, c) for c in 1:mod.C for r in 1:mod.R]
    cum_error = 0.0f0   # we'll accumulate error to set new intercept
    for (r, c) in ProgressBar(grid)
        # params for row, col
        γ = view(mod.γ, :, r, c)
        μ = view(mod.μ, :, r, c)
        λ = view(mod.λ, :, r, c)
        y = view(mod.Y, :, r, c)
        X = [view(mod.X, :, :, p) for p in 1:mod.P]
        XtX = [X'[p] * X[p] for p in 1:mod.P]
        # lagged effect
        kernels = [kernel(tvec, μ[p], λ[p]) for p in 1:mod.P]
        β = [kernels[p][1] for p in 1:mod.P]
        ϕ = [X[p] * β[p] for p in 1:mod.P]
        # errors
        Φ = sum(γ .* ϕ)
        cum_error += mean(Φ .- y)
        ε = mod.α .+ Φ .- y
        prev_loss += 0.5 * sum(ε .^ 2)
        # can use multithread here with small overhead
        @threads for p in 1:mod.P
            nbrs = spatial_neighbors(mod, p, r, c)
            ∂β∂μ, ∂β∂λ = kernels[p][2]  # size t each
            ∂²β∂²μ, ∂²β∂²λ = kernels[p][3]  # size t each

            # mu update
            μ_nbrs_sum = sum(mod.μ[nbrs])
            prev_loss += 0.25 * mod.η * (μ[p] - μ_nbrs_sum)^2
            ∇ = γ[p] * ε' * X[p] * ∂β∂μ +
                mod.η * (μ[p] - μ_nbrs_sum)
            ∇² = γ[p]^2 * ∂β∂μ' * XtX[p] * ∂β∂μ +
                γ[p] * ε' * X[p] * ∂²β∂²μ +
                mod.η * length(nbrs)
            mod.μ[p, r, c] -= ∇ / ∇²

            # lam update
            λ_nbrs_sum = sum(mod.λ[nbrs])
            prev_loss += 0.25 * mod.η * (λ[p] - λ_nbrs_sum)^2
            ∇ = γ[p] * ε' * X[p] * ∂β∂λ +
                mod.η * (λ[p] - λ_nbrs_sum)
            ∇² = γ[p]^2 * ∂β∂λ' * XtX[p] * ∂β∂λ +
                γ[p] * ε' * X[p] * ∂²β∂²λ +
                mod.η * length(nbrs)
            mod.λ[p, r, c] -= ∇ / ∇²

            # gamma updates (exact)
            γ_nbrs_sum = sum(mod.γ[nbrs])
            prev_loss += 0.25 * mod.η * (γ[p] - γ_nbrs_sum)^2
            prev_loss += mod.ν * γ[p]^2
            b =  γ[p] * ϕ[p] .- ε
            H = ϕ[p]' * ϕ[p] + mod.η * length(nbrs) + mod.ν
            mod.γ[p, r, c] = b / H
        end
    end
    # update alpha (this is a approx. delayed update for \alpha)
    mod.α = cum_error / (mod.R * mod.C + mod.ν)
    return prev_loss
end


function load_data()
    # conf
    T = 6  # number of lags
    # read observation units file
    file = CSV.file("./model_dev_data/grid_pm25_subset.csv")
    grid = DataFrame(file)
    # add row index
    unique_lat = sort(unique(grid.lat))
    R = length(unique_lat)
    row_dict = Dict(x => i for (i, x) in enumerate(unique_lat))
    grid.row = [row_dict[x] for x in grid.lat]
    # add col index
    unique_lon = sort(unique(grid.lon))
    C = length(unique_lon)
    col_dict = Dict(x => i for (i, x) in enumerate(unique_lon))
    grid.col = [col_dict[x] for x in grid.lon]
    # add obs (date) index
    grid.date = Date.(grid.year, grid.month)
    obs_unique_dates = sort(unique(grid.date))
    N = length(obs_unique_dates)
    obs_dict = Dict(x => i for (i, x) in enumerate(obs_unique_dates))
    grid.obs = [obs_dict[x] for x in grid.date]
    # fill data matrix
    Y = zeros(Float32, (N, R, C))
    for d in eachrow(grid)
        Y[d.obs, d.row, d.col] = d.pm25
    end

    # read power plant file
    file = CSV.file("./model_dev_data/so2_data.csv")
    plants = DataFrame(file)
    unique_plants = sort(unique(plants.fid))
    plants_dict = Dict(x => i for (i, x) in enumerate(unique_plants))
    # enumerate dates as in observation data
    plants.date = Date.(plants.year, plants.month)
    unique_lags = sort(unique(plants.date), rev=true)
    lags_dict = Dict(x => i for (i, x) in enumerate(unique_lags))
    # first make matrix with per date entry
    # this code assums that the most recent date in X and Y
    # are the same, otherwise it will not be correct
    X_ = zeros(Float32, (N + T, length(unique_plants)))
    for d in eachrow(plants)
        i = lags_dict[d.date]
        if i <= N + T
            p = plants_dict[d.fid]
            X_[i, p] = d.so2_tons
        end
    end
    # keep only plants with no missing information
    full_columns = findall(vec(sum(X_, dims=1)) .> 0)
    X_ = X_[:,full_columns]
    P = length(full_columns)
    # now transform in (nobs, nlags, nplants) format copying slices
    X = zeros(Float32, N, T, P)
    for i in 1:N
        X[i,:,:] = X_[i:(i + T - 1),:]
    end

    return X, Y
end


function main()
    # load data
    X, Y = load_data()
    # build model
    η = 0.1f0  # spatio-temporal smoothing
    ν = 0.1f0  # power plant effect overall shrinkage
    mod = LagModel(X, Y, η, ν)
    # step
    for i in 1:5
        println("Iteration $i")
        @time loss = learn!(mod);
        println("Starting loss: $loss\n")
        
    end
end

main()

