using Pkg; Pkg.activate(".")
using Distributions
using Plots
using Random
using DelimitedFiles
using MultivariateStats
using JLD2
using FileIO
using NPZ


mutable struct PowerPlant
    σ::Float64
    pos::Tuple{Int,Int}
    state::Float64
    φ::Float64
    ξ::Float64

    function PowerPlant(σ, pos; φ=0.99, ξ=0.1)
        state = 0.0
        new(σ, pos, state, φ, ξ)
    end
end


function emit!(src::PowerPlant)::Float64
    src.state = src.φ * src.state + src.ξ * randn()
    # return rand(Exponential(src.σ * exp(src.state)))
    return src.σ * exp(src.state)
end


mutable struct PollutionGrid
    sources::Vector{PowerPlant}
    wind_vecs::Vector{Vector{Int}}
    wind_wts::Vector{Float64}
    state::Matrix{Float64}
    effects::Array{Float64, 3}
    decay::Float64
    speed::Int
    eps::Float64
    seasonal::Vector{Float64}
    state_season::Int
    season_eps::Float64
    local_noise::Matrix{Float64}
    local_autcorr::Float64
    double_seasonal::Bool

    function PollutionGrid(
        shape,
        sources,
        wind_vecs,
        wind_wts,
        decay;
        speed=1,
        eps=1e-1,
        seasonal=[0.0],
        season_eps=0.001,
        local_autcorr=0.75,
        double_seasonal=false
    )
        @assert decay > season_eps

        state_season = 1
        local_noise = zeros(shape)
        state = zeros(shape)
        effects = zeros(shape..., length(sources))
        new(
            sources, wind_vecs, wind_wts, state, effects, decay,
            speed, eps, seasonal, state_season, season_eps,
            local_noise, local_autcorr, double_seasonal
        )
    end
end


function plot_grid(g::PollutionGrid; clims=(1e-3, 0.1))
    R, C = size(g.state)
    z = log.(1 .+ g.state)
    plt = heatmap(z, c=:terrain, clims=clims, leg=false)
    x = [s.pos[2] for s in g.sources]
    y = [s.pos[1] for s in g.sources]

    scale = minimum([s.σ for s in g.sources])
    sizes = [s.σ * exp(s.state) / scale for s in g.sources]
    if !ismissing(g.seasonal)
        L = length(g.seasonal)
        season_comp = exp(g.seasonal[1 + (g.state_season - 1) % L])
        sizes .*= season_comp
    end

    plot!(
        plt, x, y,
        ms=sizes,
        st=:scatter,
        marker=:dtriangle,
        c=:red
    )
    plt
end



function plot_effects(g::PollutionGrid, p::Int; clims=(1e-3, 0.1))
    R, C = size(g.state)
    z = log.(1 .+ g.effects[:, :, p])
    plt = heatmap(z, c=:terrain, clims=clims, leg=false)
    x = [s.pos[2] for s in g.sources]
    y = [s.pos[1] for s in g.sources]

    scale = minimum([s.σ for s in g.sources])
    sizes = [3s.σ * exp(s.state) / scale for s in g.sources]
    if !ismissing(g.seasonal)
        L = length(g.seasonal)
        season_comp = g.seasonal[1 + g.state_season % L]
        sizes .*= exp(season_comp)
    end

    plot!(
        plt, x, y,
        ms=sizes,
        st=:scatter,
        marker=:dtriangle,
        c=:red
    )
    plt
end




function propagate!(g::PollutionGrid)
    # propagate with wind vector with decay
    R, C = size(g.state)
    local emissions
    # add new sources
    L = length(g.seasonal)
    season_comp = g.seasonal[1 + g.state_season % L]
        
    for _ in 1:g.speed
        ψ = g.local_autcorr 
        g.local_noise = ψ .* g.local_noise .+ g.eps * randn((R, C))
        g.state .*= (1. - g.decay)
        prop = sum(g.wind_wts)
        δ = exp.(g.local_noise) * exp(g.double_seasonal * season_comp)
        new_state = g.state * (1. - prop) .+ g.eps .* δ
        new_effects = g.effects * (1. - prop)
        for c in 1:C
            for r in 1:R
                for (v, wt) in zip(g.wind_vecs, g.wind_wts)
                    r1, c1 = [r, c] + v
                    if 1 <= r1 <= R && 1 <= c1 <= C
                        new_state[r1, c1] += g.state[r, c] * wt
                        new_effects[r1, c1, :] .+= g.effects[r, c, :] .* wt
                    end
                end
            end
        end
        g.state = new_state
        g.effects = new_effects

        emissions = map(enumerate(g.sources)) do (p, src)
            emission = emit!(src) * exp(season_comp)
            g.state[src.pos...] += emission
            g.effects[src.pos..., p] += emission
            emission
        end
    end

    g.state_season += 1

    return emissions
end


function main()
    Random.seed!(1983)

    nrow = 40
    ncol = 100
    shape = (nrow, ncol)
    n_srcs = 15
    # sources = map(1:n_srcs) do _
    #     x = rand(DiscreteUniform(1, nrow))
    #     y = rand(DiscreteUniform(1, ncol))
    #     σ = rand(Uniform(0, 10))
    #     PowerPlant(σ, (x, y))
    # end
    n_srcs = 3
    y = [30, 50, 70]
    x = [25, 25, 25]
    σ = [0.3, 0.3, 0.3]
    sources = PowerPlant.(σ, zip(x, y))
    sigs = [p.σ for p in sources]
    locs = vcat([[p.pos[1], p.pos[2]]' for p in sources]...)
    wind_vecs = [[0, -1], [-1, 0], [-1, -1], [-2, 0], [0, -2]]
    wind_wts = [0.3, 0.3, 0.1, 0.1, 0.1]
    decay = 0.15
    speed = 32
    eps = 0.001

    # no seasonal effect
    seasonal = [0.]
    g = PollutionGrid(
        shape, sources, wind_vecs, wind_wts, decay;
        speed, eps, seasonal, local_autcorr=0.5
    );
    
    max_t = 60
    X = zeros((n_srcs, max_t));
    Y = zeros((nrow, ncol, max_t));

    # high seasonal effect
    warmup = 60
    anim = @animate for t in 1:(max_t + warmup)
        pollution = propagate!(g)
        println(round(maximum(g.state), digits=4))
        if t > warmup
            X[:, t - warmup] = pollution
            Y[:, :, t - warmup] = g.state
        end
        plot_grid(g, clims=(0, 0.1))
    end
    gif(anim, "outputs/pollution.gif", fps=1)
    savedict = Dict("power_plants" => X, "states" => Y, "sources" => sources)
    save("data/simulation/no_seasonal.jld2", savedict)
    savedict = Dict("power_plants" => X, "states" => Y, "sigs" => sigs, "locs" => locs)
    npzwrite("data/simulation/no_seasonal.npz", savedict)

    # seasonal effect
    max_τ = 12
    τ = float.(0:(max_τ - 1))
    seasonal = 0.33 * (sin.(τ * 2π / max_τ) + 0.5sin.(τ * π / max_τ))
    seasonal = 4seasonal  # multplier
    plt = plot(exp.(seasonal), title="seasonal effect", leg=false);
    
    hline!(plt, [1.0], c="red");
    savefig(plt, "outputs/seasonal_effect.png")
    
    # decay = 0.1
    g.seasonal = seasonal

    X = zeros((n_srcs, max_t));
    Y = zeros((nrow, ncol, max_t));
    anim = @animate for t in 1:(max_t + warmup)
        pollution = propagate!(g)
        println(maximum(g.state))
        if t > warmup
            X[:, t - warmup] = pollution
            Y[:, :, t - warmup] = g.state
        end
        plot_grid(g, clims=(0, 0.1))
    end
    print(maximum(g.state))
    gif(anim, "outputs/pollution_seasonal.gif", fps=1)

    savedict = Dict("power_plants" => X, "states" => Y, "sources" => sources)
    save("data/simulation/seasonal.jld2", savedict)
    savedict = Dict("power_plants" => X, "states" => Y, "sigs" => sigs, "locs" => locs)
    npzwrite("data/simulation/seasonal.npz", savedict)


    # decay = 0.1
    g.double_seasonal = true

    X = zeros((n_srcs, max_t));
    Y = zeros((nrow, ncol, max_t));
    anim = @animate for t in 1:(max_t + warmup)
        pollution = propagate!(g)
        println(maximum(g.state))
        if t > warmup
            X[:, t - warmup] = pollution
            Y[:, :, t - warmup] = g.state
        end
        plot_grid(g, clims=(0, 0.1))
    end
    print(maximum(g.state))
    gif(anim, "outputs/pollution_double_seasonal.gif", fps=1)

    savedict = Dict("power_plants" => X, "states" => Y, "sources" => sources)
    save("data/simulation/double_seasonal.jld2", savedict)
    savedict = Dict("power_plants" => X, "states" => Y, "sigs" => sigs, "locs" => locs)
    npzwrite("data/simulation/double_seasonal.npz", savedict)

    # make a plot of spurious correlations
    # mid_point = (nrow ÷ 2, ncol ÷ 2)
    # λ_ridge = 1e-6
    # betas = zeros(n_srcs, 10)
    # for k in 1:10
    #     g.seasonal = seasonal / k
    #     X = zeros((n_srcs, max_t));
    #     y = zeros(max_t)
    #     for t in 1:max_t
    #         X[:, t] = propagate!(g)
    #         y[t] = g.state[mid_point...]
    #         plot_grid(g)
    #     end
    #     betas[:, k] = ridge(log.(X'), log.(y), λ_ridge)[2:end]
    # end
    # spurious = map(g.sources) do src
    #     src.pos[1] > mid_point[1] || src.pos[2] > mid_point[2]
    # end
    # beta_true
end

main()