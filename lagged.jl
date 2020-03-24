using Pkg; Pkg.activate(".")
using CSV
using DataFrames
using LinearAlgebra
using Statistics
using Dates
using Printf: @printf
using Base.Threads: @threads
using ProgressBars: ProgressBar
using BSON: @save, @load


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
    use_intercept::Bool
    function LagModel(
            X::AbstractArray{Float32,3},
            Y::AbstractArray{Float32,3},
            η::Float32,
            ν::Float32;
            use_intercept::Bool=false)
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
            nrows, ncols, nlags, nobs, nplants,
            use_intercept)
    end
end


function spatial_neighbors(
        m::LagModel,
        p::Int, 
        r::Int,
        c::Int,
        linear::Bool=true)
    nbrs = []
    if (r > 1) push!(nbrs, (r - 1, c)) end
    if (c > 1) push!(nbrs, (r, c - 1)) end
    if (r < m.R) push!(nbrs, (r + 1, c)) end
    if (c < m.C) push!(nbrs, (r, c + 1)) end
    if linear
        ix = LinearIndices((m.P, m.R, m.C))
        nbrs = [ix[p, r, c] for (r, c) in nbrs]
    end
    return nbrs
end

function kernel(
        τ::AbstractVector{Float32},
        μ::Float32,
        λ::Float32;
        derivatives::Bool=true)
    # eval kernel
    δ = @. μ - τ
    Δ = @. 0.5f0 * δ ^ 2
    β = @. exp(- λ * Δ)

    if !derivatives 
        return β
    end

    # derivatives
    ∂μ = @. - λ * δ * β
    ∂²μ = @. - λ * (β + δ * ∂μ)
    ∂λ = @. - Δ * β
    ∂²λ = @. Δ^2 * β

    ∇β = [∂μ, ∂λ]
    ∇²β = [∂²μ, ∂²λ]

    return β, ∇β, ∇²β
end


"""
Performs updates cycling through all variables.
Each step is linear time with small constant.
"""
function learn!(m::LagModel)
    τ = Float32.(0:(m.T - 1))
    prev_loss = 0.0f0
    grid = [(r, c) for c in 1:m.C for r in 1:m.R]
    cum_error = 0.0f0   # we'll accumulate error to set new intercept

    X = [view(m.X, :, :, p) for p in 1:m.P]
    XtX = [X'[p] * X[p] for p in 1:m.P]  # can precm
    
    @threads for (r, c) in ProgressBar(grid)
        # params for row, col
        γ = view(m.γ, :, r, c)
        μ = view(m.μ, :, r, c)
        λ = view(m.λ, :, r, c)
        Y = view(m.Y, :, r, c)
        
        # lagged effect
        kernels = [kernel(τ, μ[p], λ[p]) for p in 1:m.P]
        β = [kernels[p][1] for p in 1:m.P]
        ϕ = [X[p] * β[p] for p in 1:m.P]

        # error
        ε = m.α .+ sum(γ .* ϕ) .- Y

        # add to loss and cum_error for intercept
        prev_loss += 0.5f0 * sum(ε .^ 2)
        cum_error += mean(ε) - m.α
        
        # can use multithread here with small overhead
        for p in 1:m.P
            nbrs = spatial_neighbors(m, p, r, c)
            N_nbrs = Float32(length(nbrs))

            ∂β∂μ, ∂β∂λ = kernels[p][2]  # size t each
            ∂²β∂²μ, ∂²β∂²λ = kernels[p][3]  # size t each

            # mu update
            μ_nbrs_sum = sum(m.μ[nbrs])
            prev_loss += 0.25f0 * m.η * (μ[p] - μ_nbrs_sum)^2
            ∇ = γ[p] * ε' * X[p] * ∂β∂μ +
                m.η * (μ[p] - μ_nbrs_sum)
            ∇² = γ[p]^2 * ∂β∂μ' * XtX[p] * ∂β∂μ +
                γ[p] * ε' * X[p] * ∂²β∂²μ +
                m.η * N_nbrs
            m.μ[p, r, c] = max(m.μ[p, r, c] - ∇ / max(∇², 2.0f0), 0f0)

            # lam update
            λ_nbrs_sum = sum(m.λ[nbrs])
            prev_loss += 0.25f0 * m.η * (λ[p] - λ_nbrs_sum)^2
            ∇ = γ[p] * ε' * X[p] * ∂β∂λ +
                m.η * (λ[p] - λ_nbrs_sum)
            ∇² = γ[p]^2 * ∂β∂λ' * XtX[p] * ∂β∂λ +
                γ[p] * ε' * X[p] * ∂²β∂²λ +
                m.η * N_nbrs
            m.λ[p, r, c] = max(m.λ[p, r, c] - ∇ / max(∇², 2.0f0), 1f-3)

            # gamma updates (exact)
            γ_nbrs_sum = sum(m.γ[nbrs])
            prev_loss += 0.25 * m.η * (γ[p] - γ_nbrs_sum)^2
            prev_loss += m.ν * γ[p]^2
            b =  γ[p] * ϕ[p] .- ε
            H = ϕ[p]' * ϕ[p] + m.η * N_nbrs + m.ν
            m.γ[p, r, c] = max(ϕ[p]' * b / H, 1f-6)
        end
    end
    # update alpha (this is a approx. delayed update for \alpha)
    if m.use_intercept
        m.α = cum_error / (m.R * m.C + m.ν)
    end
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
    # loop conf
    save_every = 10
    print_every = 1
    niter = 150
    
    # load data
    X, Y = load_data()
    
    # build model
    η = 0.1f0  # spatio-temporal smoothing
    ν = 0.1f0  # power plant effect overall shrinkage
    m = LagModel(X, Y, η, ν)

    # step
    for iter in 1:niter
        println("Iteration $iter")
        
        @time loss = learn!(m);
        println("Starting loss: $loss\n") 

        if iter ==1 || iter % print_every == 0
            @printf "γ: (%.2f, %.2f, %.2f) " minimum(m.γ) mean(m.γ) maximum(m.γ)
            @printf "μ: (%.2f, %.2f, %.2f) " minimum(m.μ) mean(m.μ) maximum(m.μ)
            @printf "λ: (%.2f, %.2f, %.2f)\n" minimum(m.λ) mean(m.λ) maximum(m.λ)
        end
        
        if iter == 1 || iter % save_every == 0
            @save "results/model.bson" m
        end
    end

end

main()

