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
function learn!(m::LagModel; lr=1e-4, progress=false, update_center=false)
    lr = Float32(lr)  # GRADIENT LEARNING RATE
    prev_loss = 0.0f0
    prev_tv = 0.0f0
    prev_ridge = 0.0f0

    kernel = kernel_laplace
    use_intercept = true
    
    R, C, T, N, P = m.shape

    τ = Float32.(0:(T - 1))
    X = [view(m.X, :, :, p) for p in 1:P]
    XtX = [X'[p] * X[p] for p in 1:P]  # can precm
    
    grid = [(r, c) for r in 1:R for c in 1:C]
    shuffle!(grid)

    if progress
        grid = ProgressBar(grid)
    end

    @threads for (r, c) in grid
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
                μ_nbrs_sum = N_nbrs > 0 ? sum(m.μ[nbrs]) : 0.0f0
                prev_tv += 0.25f0 * m.η * (N_nbrs * μ[p]  - μ_nbrs_sum)^2     
                ∂μ_L = εXp * ∂μ_β + m.η * (N_nbrs * μ[p] - μ_nbrs_sum)
                μ_new = μ[p] - lr * ∂μ_L
                m.μ[p, r, c] = clamp(μ_new, 0f0, Float32(T))
            end
                                 
            # lam update
            λ_nbrs_sum = N_nbrs > 0 ?  sum(m.λ[nbrs]) : 0.0f0
            prev_tv += 0.25f0 * m.η * (N_nbrs * λ[p]  - λ_nbrs_sum)^2      
            ∂λ_L = εXp * ∂λ_β + m.η * (N_nbrs * λ[p] - λ_nbrs_sum)
            λ_new = λ[p] - lr * ∂λ_L
            m.λ[p, r, c] = clamp(λ_new, Float32(1.0 / T), Float32(T))

            # γ update
            γ_nbrs_sum = N_nbrs > 0 ? sum(m.γ[nbrs]) : 0.0f0
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
