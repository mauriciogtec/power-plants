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
struct LagModel
    μ::Array{Float32,3}  # kernel center
    γ::Array{Float32,3}  # kernel intensity
    λ::Array{Float32,3}  # kernel decay
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
        μ = zeros(nplants, nrows, ncols)
        λ = zeros(nplants, nrows, ncols)
        γ = zeros(nplants, nrows, ncols)
        new(μ, γ, λ,
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
    # function
    δ = @. μ - t
    Δ = @. 0.5 * δ ^ 2
    ψ = @. exp(- λ * Δ)
    C = √(2π)
    β = @. λ * ψ / C
    if !derivatives
        return β
    end
    # derivatives
    ∂μ = @. - λ * δ * β
    ∂λ = @. ψ / C - Δ * β
    ∇β = [∂λ, ∂μ]
    # second derivatives
    ∂²μ = @. - λ * β - λ * δ * ∂μ
    ∂²λ = @. - Δ * ψ / C - Δ * ∂λ
    ∇²β = [∂²λ, ∂²μ]
    # ∂μ∂λ = - width .* (β + λ .* ∂λ)
    return β, ∇β, ∇²β
end


"""
Performs updates cycling through all variables.
Each step is linear time with small constant.
"""
function learn!(mod::LagModel)
    tvec = Float32.(0:(mod.T - 1))
    starting_loss = 0.0
    grid = [(r, c) for c in 1:mod.C for r in 1:mod.R]
    for (r, c) in ProgressBar(grid)
        # params for row, col
        γ = view(mod.γ, :, r, c)
        μ = view(mod.μ, :, r, c)
        λ = view(mod.λ, :, r, c)
        y = view(mod.Y, :, r, c)
        X = [view(mod.X, :, :, p) for p in 1:mod.P]
        # lagged effect
        kout = [kernel(tvec, μ[p], λ[p]) for p in 1:mod.P]
        β = [kout[p][1] for p in 1:mod.P]
        ϕ = [X[p] * β[p] for p in 1:mod.P]
        # errors
        y_hat = sum(γ .* ϕ) 
        ε = y_hat .- y
        starting_loss += 0.5 * sum(ε .^ 2)
        # can use multithread here with small overhead
        @threads for p in 1:mod.P
            nbrs = spatial_neighbors(mod, p, r, c)
        
            # mu update
            μ_nbrs_sum = sum(mod.μ[nbrs])
            starting_loss += mod.η * (μ[p] - μ_nbrs_sum)^2
            ∂β∂μ = kout[p][2][1]  # size t
            ∂²β∂²μ = kout[p][3][1]  # size t
            ∇ = dot(X[p] * ∂β∂μ, ε) + mod.η * (μ[p] - μ_nbrs_sum)
            ∇² = dot(X[p] * ∂²β∂²μ, ε) + mod.η
            mod.μ[p, r, c] -= ∇ / ∇²

            # lam update
            λ_nbrs_sum = sum(mod.λ[nbrs])
            starting_loss += mod.η * (λ[p] - λ_nbrs_sum)^2
            ∂β∂λ = kout[p][2][2]  # size t
            ∂²β∂²λ = kout[p][3][2]  # size t
            ∇ = dot(X[p] * ∂β∂λ, ε) + mod.η * (λ[p] - λ_nbrs_sum)
            ∇² = dot(X[p] * ∂²β∂²λ, ε) + mod.η  # scalar
            mod.λ[p, r, c] -= ∇ / ∇²

            # gamma updates (exact)
            γ_nbrs_sum = sum(mod.γ[nbrs])
            starting_loss += mod.η * (γ[p] - γ_nbrs_sum)^2
            starting_loss += mod.ν * γ[p]^2
            γ_lp =  γ[p] * ϕ[p] .- ε
            b =  dot(ϕ[p], γ_lp) + mod.η * γ_nbrs_sum
            H = dot(ϕ[p], ϕ[p]) + mod.η * length(nbrs) + mod.ν
            mod.γ[p, r, c] =  b / H
        end
    end
    return starting_loss
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
        println("Starting loss: $i\n")
        
    end
end

main()

