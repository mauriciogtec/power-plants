# this code runs a simple sampler to generate uniform points
# inside a polygon. It samples from a uniform box enclosing
# the polygon and tester whether the point is inside or not the
# polygon.

using Pkg; Pkg.activate(".")
using Shapefile
import Shapefile: Point
using ProgressMeter
using DataFrames
using Base.Threads
using Distributions
using DelimitedFiles
using CSV
using ArchGDAL
const AG = ArchGDAL


function inside_polygon(p::Point, polygon::Vector{Point})
    N = length(polygon)
    angle = 0.0
    for i in 1:N - 1
        x = polygon[i].x - p.x
        y = polygon[i].y - p.y
        x1 = polygon[i + 1].x - p.x
        y1 = polygon[i + 1].y - p.y
        angle += angle2d(x, y, x1, y1)
    end
    abs(angle) < π
end


function angle2d(x1::T, y1::T, x2::T, y2::T) where T <: Float64
    θ₁ = atan(y1, x1)
    θ₂ = atan(y2, x2)
    δθ = θ₂ - θ₁
    while δθ > π
        δθ -= 2π
    end
    while δθ < -π
        δθ += 2π
    end
    δθ
end

function get_bounds(polygon::Vector{Point}; ϵ::Float64=1e-6)
    xmin, xmax, ymin, ymax = Inf, -Inf, Inf, -Inf
    for p in polygon
        xmin = min(p.x, xmin)
        xmax = max(p.x, xmax)
        ymin = min(p.y, ymin)
        ymax = max(p.y, ymax)
    end
    return xmin - ϵ, xmax + ϵ, ymin - ϵ, ymax + ϵ
end

function main(n_samples, shapefile; max_attempts=100_000)
    table = Shapefile.Table(shapefile)
    polygons = [x.points for x in Shapefile.shapes(table)]
    n_poly = length(polygons)

    lons = fill(-999999., n_samples, n_poly)
    lats = fill(-999999., n_samples, n_poly)

    pbar = Progress(n_poly)
    println("Generating random points...")
    @threads for i=1:n_poly
        xmin, xmax, ymin, ymax = get_bounds(polygons[i])

        sampled = 0
        for attempts in 1:max_attempts
            x = rand(Uniform(xmin, xmax))
            y = rand(Uniform(ymin, ymax))
            if inside_polygon(Point(x, y), polygons[i])
                sampled += 1
                lons[sampled, i] = x
                lats[sampled, i] = y
                sampled >= n_samples && break
        end end
        next!(pbar)
    end

    # df = DataFrame(
    #     :lon=>lons, :lat=>lats, :idx=>indexes, :lab=>labs
    # )
    # CSV.write("data/zip_samples.csv", df)

    # raster_dir = "data/SO4/"
    # raster_files = []
    # for (root, dirs, files) in walkdir(raster_dir)
    #     for file in files
    #         if occursin("NoNegs.tif" , file)
    #             path = 
    #             push!(raster_files, joinpath(root, file))
    # end end end
    # n_layers = length(raster_files)
    
    # # pts = CSV.read("data/zip_samples.csv", types=Dict(:lab=>String))
    # # n_pts = nrow(pts)
    
    # W = zeros(Int, n_samples, n_poly)
    # H = zeros(Int, n_samples, n_poly)
    # oob = zeros(Bool, n_samples, n_poly)

    # rast1 = AG.readraster(raster_files[1])
    # nc, nr, _ = size(rast1)
    # gt = AG.getgeotransform(rast1)
    # lon_min, δlon, _, lat_max, _, δlat = gt

    # for j in 1:n_poly 
    #     for i in 1:n_samples
    #         h = round((lats[i, j] - lat_max) / δlat)
    #         w = round((lons[i, j] - lon_min) / δlon)
    #         W[i, j] = w
    #         H[i, j] = h
    #         oob[i, j] = (h < 1 || h > nr || w < 1 || w > nc)
    # end end

    # vals = zeros(n_samples, n_poly, n_layers)
    # miss = zeros(n_samples, n_poly, n_layers)
    # vnames = fill("", n_layers)
    
    # println("Averaging raster layers...")
    # pbar = Progress(n_layers)
    # @threads for k in 1:n_layers
    #     rast = AG.readraster(raster_files[k])
    #     for j in 1:n_poly
    #         for i in 1:n_samples
    #             w, h = W[i, j], H[i, j]
    #             vals[i, j, k] = oob[i, j] ? -999999. : rast[w, h, 1]
    #             miss[i, j, k] = oob[i, j] || rast[w, h, 1] ≤ 0
    #     end end
    #     vnames[k] = "so4_" * raster_files[k][end-23:end-11]
    #     next!(pbar)
    # end

    # df = DataFrame(:lab=>table.GEOID10)
    # counts = dropdims(sum(miss .== 0, dims=1), dims=1)

    # meanvals = dropdims(sum(vals .* (1.0 .- miss), dims=1), dims=1)
    # meanvals ./= (counts .+ 1e-12)

    # for k in 1:n_layers
    #     df[Symbol(vnames[k])] = meanvals[:, k]
    # end

    # CSV.write("model_dev_data/zip_averages.csv", df)


    # now repeat for covariates
    raster_dir = "model_dev_data/covariates/"
    raster_files = []
    for (root, dirs, files) in walkdir(raster_dir)
        for file in files
            if occursin(".tif" , file)
                path = 
                push!(raster_files, joinpath(root, file))
    end end end
    n_rasts = length(raster_files)
    
    W = zeros(Int, n_samples, n_poly)
    H = zeros(Int, n_samples, n_poly)
    oob = zeros(Bool, n_samples, n_poly)

    rast1 = AG.readraster(raster_files[1])
    nc, nr, nl = size(rast1)
    gt = AG.getgeotransform(rast1)
    lon_min, δlon, _, lat_max, _, δlat = gt

    # override number of ratser layhers
    # cnames = ["temp", "apcp", "rhum", "vwnd", "uwnd", "wspd", "phi"]
    nl = 4
    cnames = ["temp", "apcp", "rhum", "vwnd"]

    for j in 1:n_poly 
        for i in 1:n_samples
            h = round((lats[i, j] - lat_max) / δlat)
            w = round((lons[i, j] - lon_min) / δlon)
            W[i, j] = w
            H[i, j] = h
            oob[i, j] = (h < 1 || h > nr || w < 1 || w > nc)
    end end

    vals = zeros(n_samples, n_poly, nl, n_rasts)
    miss = zeros(n_samples, n_poly, nl, n_rasts)
    vnames = fill("", nl, n_rasts)
    
    println("Averaging raster layers...")
    
    pbar = Progress(n_rasts)
    @threads for k in 1:n_rasts
        rast = AG.readraster(raster_files[k])
        for l in 1:nl
            nodataval = AG.getnodatavalue(AG.getband(rast, l))
            for j in 1:n_poly
                for i in 1:n_samples
                    w, h = W[i, j], H[i, j]
                    vals[i, j, l, k] = oob[i, j] ? 0.0 : rast[w, h, l]
                    miss[i, j, l, k] = oob[i, j] || rast[w, h, l] == nodataval
            end end
            vnames[l, k] = "so4_" * raster_files[k][end-9:end-4] * "_" * cnames[l]
        end
        next!(pbar)
    end

    # Averaging
    println("Averaging...")

    df = DataFrame(:lab=>table.GEOID10)
    counts = dropdims(sum(miss .== 0, dims=1), dims=1)

    meanvals = dropdims(sum(vals .* (1.0 .- miss), dims=1), dims=1)
    meanvals ./= (counts .+ 1e-12)

    for k in 1:n_rasts
        for l in 1:nl
            df[Symbol(vnames[l, k])] = meanvals[:, l, k]
    end end

    CSV.write("model_dev_data/covars_avs.csv", df)

end

n_samples = 20  # 100 for pollution 20 for covariates
shapefile = "data/tl_2016_us_zcta510/tl_2016_us_zcta510.shp"
max_attempts = 100_000

main(n_samples, shapefile, max_attempts=max_attempts)
