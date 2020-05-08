using Pkg; Pkg.activate("."); Pkg.instantiate()
using CSV
using DataFrames
using LinearAlgebra
using Statistics
using Dates
using Printf: @printf
using Base.Threads: @threads
using ProgressBars: ProgressBar
using BSON: @save, @load
using DelimitedFiles: writedlm
using Random


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
    α::Matrix{Float32}  # intercept
    η::Float32  # spatial regularization
    ν::Float32  # power plant regularization
    X::Array{Float32,3}
    y::Array{Float32,3}
    shape::NamedTuple
    function LagModel(
            X::AbstractArray{Float32,3},
            y::AbstractArray{Float32,3},
            η::Float32,
            ν::Float32;)
        @assert length(size(X)) == 3
        @assert length(size(y)) == 3
        @assert size(X, 1) == size(y, 1)
        nobs, nrows, ncols = size(y)
        _, nlags, nplants = size(X)
        μ = zeros(Float32, (nplants, nrows, ncols))
        λ = ones(Float32, (nplants, nrows, ncols))
        γ = ones(Float32, (nplants, nrows, ncols))
        α = zeros(Float32, (nrows, ncols))
        shape = (R=nrows, C=ncols, T=nlags, N=nobs, P=nplants)
        new(μ, γ, λ, α,  # parameters
            η, ν,  # hyperperameters
            X, y, shape) # data
    end
end


function export_params_to_csv(m::LagModel; suffix::String="")
    dir = "exports"
    shape = Tuple(m.shape)
    writedlm(open("$dir/shape$suffix.csv", "w"), shape, ',')
    writedlm(open("$dir/mu$suffix.csv", "w"), reshape(m.μ, :), ',')
    writedlm(open("$dir/lambda$suffix.csv", "w"), reshape(m.λ, :), ',')
    writedlm(open("$dir/gamma$suffix.csv", "w"), reshape(m.γ, :), ',')
    writedlm(open("$dir/alpha$suffix.csv", "w"), reshape(m.α, :), ',')
end


function spatial_neighbors(
        m::LagModel,
        p::Int,
        r::Int,
        c::Int,
        linear_indices::Bool=true)
    nbrs = []
    R, C, T, N, P = m.shape

    if (r > 1) push!(nbrs, (p, r - 1, c)) end
    if (r < R) push!(nbrs, (p, r + 1, c)) end
    if (c > 1) push!(nbrs, (p, r, c - 1)) end  
    if (c < C) push!(nbrs, (p, r, c + 1)) end

    if linear_indices
        ix = LinearIndices((P, R, C))
        nbrs = [ix[nbr...] for nbr in nbrs]
    end

    return nbrs
end


function kernel_gaussian(
        τ::AbstractVector{Float32},
        μ::Float32,
        λ::Float32,
        γ::Float32)

    # eval kernel
    δ = @. μ - τ
    Δ = @. 0.5f0 * δ ^ 2
    ψ = @. exp(- λ * Δ)
    β = @. γ * ψ

    # derivatives
    ∂μ = @. - λ * δ * β
    ∂λ = @. - Δ * β
    ∂γ = @. ψ

    ∇β = [∂μ, ∂λ, ∂γ]

    return β, ∇β
end


function kernel_laplace(
        τ::AbstractVector{Float32},
        μ::Float32,
        λ::Float32,
        γ::Float32)

    # eval kernel
    δ = @. μ - τ
    sgn = @. sign(δ)
    δabs = @. sgn * δ
    ψ = @. exp(- λ * δabs)
    β = @. γ * ψ

    # derivatives
    ∂μ = @. - λ * sgn * β
    ∂λ = @. - δabs * β
    ∂γ = @. ψ


    ∇β = [∂μ, ∂λ, ∂γ]

    return β, ∇β
end


"""
Performs updates cycling through all variables.
Each step is linear time with small constant.
"""
function learn!(m::LagModel)
    lr = 1.0f-4  # GRADIENT LEARNING RATE
    prev_loss = 0.0f0
    prev_tv = 0.0f0
    prev_ridge = 0.0f0

    kernel = kernel_laplace
    update_center = false
    use_intercept = true
    
    R, C, T, N, P = m.shape

    τ = Float32.(0:(T - 1))
    X = [view(m.X, :, :, p) for p in 1:P]
    XtX = [X'[p] * X[p] for p in 1:P]  # can precm
    
    grid = [(r, c) for r in 1:R for c in 1:C]
    shuffle!(grid)

    @threads for (r, c) in ProgressBar(grid)
        # offload notation
        @views begin # avoids copying memorys
            μ = m.μ[:, r, c]
            λ = m.λ[:, r, c]
            γ = m.γ[:, r, c]
            y = m.y[:, r, c]
        end 
        α = m.α[r, c]

        # lagged effect
        kernels = [kernel(τ, μ[p], λ[p], γ[p]) for p in 1:P]
        β = [kernels[p][1] for p in 1:P]
        Φ = [X[p] * β[p] for p in 1:P]

        # error
        ε = α .+ sum(Φ) .- y
        prev_loss += 0.5f0 * (ε' * ε)

        # update intercept
        if use_intercept
            m.α[r, c] -= lr * sum(ε)
        end

        # can use multithread here with small overhead
        for p in 1:P
            # kernel derivatives
            ∂μ_β, ∂λ_β, ∂γ_β = kernels[p][2]

            # neighbors of point [r, c, p]
            nbrs = spatial_neighbors(m, p, r, c)
            N_nbrs = Float32(length(nbrs))

            # pre-compute for efficiency
            εXp = ε' * X[p] 

            # mu update
            if update_center
                μ_nbrs_sum = sum(m.μ[nbrs])
                prev_tv += 0.25f0 * m.η * (N_nbrs * μ[p]  - μ_nbrs_sum)^2     
                ∂μ_L = εXp * ∂μ_β + m.η * (N_nbrs * μ[p] - μ_nbrs_sum)
                μ_new = μ[p] - lr * ∂μ_L
                m.μ[p, r, c] = clamp(μ_new, 0f0, Float32(T))
            end
                                 
            # lam update
            λ_nbrs_sum = sum(m.λ[nbrs])
            prev_tv += 0.25f0 * m.η * (N_nbrs * λ[p]  - λ_nbrs_sum)^2      
            ∂λ_L = εXp * ∂λ_β + m.η * (N_nbrs * λ[p] - λ_nbrs_sum)
            λ_new = λ[p] - lr * ∂λ_L
            m.λ[p, r, c] = clamp(λ_new, Float32(1.0 / T), Float32(T))

            # γ update
            γ_nbrs_sum = sum(m.γ[nbrs])
            prev_tv += 0.25f0 * m.η * (N_nbrs * γ[p]  - γ_nbrs_sum)^2
            prev_ridge += m.ν * γ[p]^2
            ∂γ_L = εXp * ∂γ_β + m.η * (N_nbrs * γ[p] - γ_nbrs_sum) + m.ν * γ[p]
            γ_new = γ[p] - lr  * ∂γ_L
            m.γ[p, r, c] = clamp(γ_new, 1f-3, 1f3)
        end
    end

    # normalize by grid size for interpretability
    prev_loss /= R * C
    prev_tv /= R * C
    prev_ridge /= R * C

    return prev_loss, prev_tv, prev_ridge  
end


function load_data()
    # params
    normalize_independently = false

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
    full_columns = findall(vec(minimum(X_, dims=1)) .> 0)
    unique_plants = unique_plants[full_columns]
    X_ = X_[:, full_columns]
    P = length(full_columns)
    # now transform in (nobs, nlags, nplants) format copying slices
    X = zeros(Float32, N, T, P)
    for i in 1:N
        X[i, :, :] = X_[i:(i + T - 1), :]
    end

    # standardize
    if normalize_independently
        X ./= std(X, dims=(2,3))
        Y ./= std(Y, dims=(2,3))
    else
        X ./= std(X)
        Y ./= std(Y)
    end

    # export necessary information
    dir = "exports"
    writedlm(open("$dir/lat.csv", "w"), unique_lat, ',')
    writedlm(open("$dir/lon.csv", "w"), unique_lon, ',')
    writedlm(open("$dir/power-plant-id.csv", "w"), unique_plants, ',')
    writedlm(open("$dir/dates.csv", "w"), Dates.format.(obs_unique_dates, "yyyy-mm-dd"), ',')

    return X, Y
end


function main()
    # loop conf
    save_every = 5 # saves model as bson
    print_every = 1
    export_every = 5 # exports kernel params to csv
    niter = 500
    load = false
    suffix = "_unnorm"
    
    # load data
    X, y = load_data()
    
    # build model
    η = 10f-1  # spatio-temporal smoothing
    ν = 1f-1  # power plant effect overall shrinkage
    if !load
        m = LagModel(X, y, η, ν)
    else
        @load "results/model_laplace$suffix.bson" m
        m.ν = ν
        m.η = η
    end

    # step
    for iter in 1:niter
        println("Iteration $iter")
        
        @time loss, tv, ridge = learn!(m);
        println("Starting loss: $loss\t|\ttv: $tv\t|\tridge: $ridge\n") 

        if iter ==1 || iter % print_every == 0
            @printf "γ: (%.4f, %.4f, %.4f) " minimum(m.γ) mean(m.γ) maximum(m.γ)
            @printf "μ: (%.3f, %.3f, %.3f) " minimum(m.μ) mean(m.μ) maximum(m.μ)
            @printf "λ: (%.3f, %.3f, %.3f)\n" minimum(m.λ) mean(m.λ) maximum(m.λ)
        end
        
        if iter % save_every == 0
            @save "results/model_laplace$suffix.bson" m
        end

        if iter % export_every == 0
            export_params_to_csv(m, suffix=suffix)
        end
    end

end

main()

