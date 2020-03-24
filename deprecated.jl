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



# 1.-- read and process grid location
grid = DataFrame(CSV.file("./model_dev_data/grid_pm25_subset.csv"))
sort!(grid, (:lon, :lat))
plants = DataFrame(CSV.file("./model_dev_data/so2_data_subset.csv"))
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
nr = Int(round(max_lat - min_lat, digits=3) ÷ δ) + 1
nc = Int(round(max_lon - min_lon, digits=3) ÷ δ) + 1
num_nodes = nr * nc
num_lags = 12
grid.row = Int.(round.(grid.lat .- min_lat, digits=d) .÷ δ) .+ 1
grid.col = Int.(round.(grid.lon .- min_lon, digits=d) .÷ δ) .+ 1
plants.row = Int.(round.(plants.lat .- min_lat, digits=d) .÷ δ) .+ 1
plants.col = Int.(round.(plants.lon .- min_lon, digits=d) .÷ δ) .+ 1
target_year = 2015
target_month = 12
plants.lag = (12target_year + target_month) .- 12plants.year .- plants.month


# -- 1.a. make edges
ix = LinearIndices((nr, nc))
se0 = Int[]
se1 = Int[]
for i in 1:nr
    for j in 1:nc
        if j < nc
            push!(se0, ix[i, j])
            push!(se1, ix[i, j + 1])
        end
        if i < nr
            push!(se0, ix[i, j])
            push!(se1, ix[i + 1, j])     
        end
    end
end
sedges = (se0, se1)


# -- 1.b extract y and X
# y = Union{Missing, Float32}[missing for _ in 1:num_nodes]
y = zeros(Float32, num_nodes)
missing_mask = zeros(Float32, num_nodes)
for (datum, i, j) in zip(grid.pm25, grid.row, grid.col)
    y[ix[i, j]] = datum
    missing_mask[ix[i, j]] = 1.0
end
y_mean = mean(y[y .> 0])
y_std = std(y[y .> 0])
y = (y .- y_mean) ./ y_std

# X = Union{Missing, Float32}[missing for _ in 1:num_plants, _ in 1:num_lags]
X = zeros(Float32, num_plants * num_lags)
ixp = LinearIndices((num_plants, num_lags))
target_time = target_year * 12 + target_month
for (datum, lag, i) in zip(plants.so2_tons, plants.lag, plants.nid)
    if lag < num_lags
        X[ixp[i, lag + 1]] = datum
    end
end
X_mean = mean(X[X .> 0])
X_std = std(X[X .> 0])
X = (X .- X_mean) ./ X_std


# -- 1.c. vector of coefficients
β = zeros(Float32, (num_nodes, num_plants * num_lags))
α = zeros(Float32, num_nodes)

te0 = Int[]
te1 = Int[]
for v in 1:num_plants
    for t in 1:num_lags - 1
        push!(te0, ixp[v, t])
        push!(te1, ixp[v, t + 1])
    end
end
tedges = (te0, te1)


## get final edges
edges = (Int[] , Int[])
# e0 = Int[]
# e1 = Int[]
# ixb = LinearIndices(size(β))
# for (v, w) in zip(te0, te1)
#     for z in 1:num_nodes
#         push!(e0, ixb[z, v])
#         push!(e1, ixb[z, w])
#     end
# end
# for (v, w) in zip(se0, se1)
#     for z in 1:(num_plants * num_lags)
#         push!(e0, ixb[v, z])
#         push!(e1, ixb[w, z])
#     end
# end
# edges = (e0, e1)
# num_edges = length(edges)


# -- 1.d. loss function
function reg(β, edges)
    e0, e1 = edges
    @views β_reg = 1e-2 * sum((β[e0] .- β[e1]) .^ 2)
    α_reg = 1e-4 * sum(α .^ 2)
    return α_reg + β_reg
end


function predicted(β, α, X, y)
    return α .+ β * X 
end


function loss(X, y, edges, missing_mask)
    yhat = predicted(β, α, X, y)
    Δ = missing_mask .* (y - yhat)
    negll = mean(Δ .^ 2)
    return negll # + reg(β, edges)
end


# -- 2 Optimization
opt = Flux.Nesterov(0.03, 0.95)
parlist = [α, β]
θ = Flux.Params(parlist)
grads = Flux.gradient(() -> loss(X, y, edges, missing_mask), θ)


num_epochs = 100000
for e in 1:num_epochs
    for par in parlist
        Flux.Optimise.update!(opt, par, grads[par])
    end
    l_ = 100000.0 * loss(X, y, edges, missing_mask)
    r_ = 0.0
    # r_ = 100000.0 * reg(β, edges)
    mβ = 100000.0 * minimum(β)
    μβ = 100000.0 * mean(β)
    Mβ = 100000.0 * maximum(β)
    mα = 100000.0 * minimum(α)
    μα = 100000.0 * mean(α)
    Mα = 100000.0 * maximum(α)
    c = 100.0 * e / num_epochs
    @printf "い: %.2f%%, ろ: %.2f, れ: %.2f, β: %.2f, %.2f, %.2f, α: %.2f, %.2f, %.2f\n" c l_ r_ mβ μβ Mβ mα μα Mα
end


# η = β * reshape()
# writedlm("output/betas", β, ',')
