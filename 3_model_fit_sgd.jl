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
using DelimitedFiles
using Random




function get_nbrs(r::Int, c::Int, nr::Int, nc::Int)
    nbrs = Int[]
    if (r - 1 > 1) push!(nbrs, r - 1) end
    if (c - 1 > 1) push!(nbrs, c - 1) end
    if (r + 1 < nr) push!(nbrs, r + 1) end
    if (r - 1 < nc) push!(nbrs, c + 1) end
    return nbrs
end


function main()
    # 1.-- read and process grid location
    grid = DataFrame(CSV.file("./model_dev_data/grid_pm25_subset.csv"))
    sort!(grid, (:lon, :lat))
    plants = DataFrame(CSV.file("./model_dev_data/so2_data_subset.csv"))
    target_year = 2015
    target_month = 12
    nt = 12
    plants.lag = (12target_year + target_month) .- 12plants.year .- plants.month
    filter!(r -> r.so2_tons > 0.0 && r.lag < nt, plants)
    unique_ids = unique(plants.id)
    num_plants = length(unique_ids)
    nid_dict = Dict(zip(unique_ids, 1:num_plants))
    plants.nid = [nid_dict[x] for x in plants.id]

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
    plants.row = Int.(round.(plants.lat .- min_lat, digits=d) .÷ δ) .+ 1
    plants.col = Int.(round.(plants.lon .- min_lon, digits=d) .÷ δ) .+ 1


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

