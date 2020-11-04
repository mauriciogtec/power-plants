using Flux
using JLD2
using Random
using NPZ


function load_data(seasonal::Bool; lags::Int = 30)
    fn = seasonal ? "seasonal" : "no_seasonal"
    data = load("data/simulation/$(fn).jld2")
    X_ = data["power_plants"];
    Y_ = data["states"];
    sources = data["sources"]

    # change dimensions to match expectation by Lagged Model
    # make X lagged
    nplants, nobs = size(X_)
    X = zeros(Float32, nobs - lags + 1, lags, nplants);
    for t in 1:(nobs - lags + 1)
        X[t, :, :] = X_[:, t:(t + lags - 1)]';
    end

    R, C, nobs = size(Y_)
    Y_ = permutedims(Y_, (3, 1, 2))
    y = Float32.(Y_[lags:end, :, :])

    return X, y, sources
end


struct GaussFilter
    μ
    logσ
    logλ
end

function GaussFilter(dim::Integer, kernel_size::Integer)
    μ = randn(dim, kernel_size)
    logσ = randn(dim, kernel_size)
    logλ = randn(dim, kernel_size) 
    GaussFilter(μ, logσ, logλ)
end

(m::GaussFilter)(x) = depthwise()


function main()
    seasonal = false
    X, y = load_data(seasonal);

    η = 0f0 # spatial regularization
    ν = 0f0  # power plant regularization
end