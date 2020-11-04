using JLD2

include("lagged.jl")

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


function main()
    seasonal = false
    η = 0f0 # spatial regularization
    ν = 0f0  # power plant regularization
    X, y = load_data(seasonal);

    # for speed reduce y to a mini 2x2 grid
    R, C = size(y)[2:3]
    y = y[:, (R ÷ 2):(R ÷ 2 + 1), (C ÷ 2):(C ÷ 2 + 1)]

    m = LagModel(X, y, η, ν);

    # step
    niter = 100000
    print_every = 10000
    for iter in 1:niter
        
        loss, tv, ridge = learn!(m, lr=5e-5, update_center=true);
        
        if iter ==1 || iter % print_every == 0
            println("Iteration $iter")
            println("Starting loss: $loss\t|\ttv: $tv\t|\tridge: $ridge\n") 
            @printf "γ: (%.4f, %.4f, %.4f) " minimum(m.γ) mean(m.γ) maximum(m.γ)
            @printf "μ: (%.3f, %.3f, %.3f) " minimum(m.μ) mean(m.μ) maximum(m.μ)
            @printf "λ: (%.3f, %.3f, %.3f)\n" minimum(m.λ) mean(m.λ) maximum(m.λ)
        end
    end

    m.γ
    spurious = map(sources) do src
        src.pos[1] < mid_point[1] || src.pos[2] < mid_point[2]
    end
    effects_spurious = m.γ[spurious, :, :]
    effects_not_spurious = m.γ[.!spurious, :, :]

    mean(spurious)
    mean(.!spurious)
end