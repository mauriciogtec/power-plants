using Pkg; Pkg.activate(".")
using Shapefile
using ProgressMeter
using DataFrames
using Base.Threads
using CSV
using FileIO

in01(t::T, eps::T) where T <: Float64 = (-eps ≤ t) && (t ≤ 1.0 + eps)

slope(a1::T, b1::T, a2::T, b2::T) where T <: Float64 = (b2 - b1) / (a2 - a1)

aeq(a::T, b::T, eps::T) where T <: Float64 = abs(a - b) < eps

function segment_intersect(
    a1::T, b1::T, a2::T, b2::T, c1::T, d1::T, c2::T, d2::T, eps::T
)::Bool where T <: Float64
    # case 1: two vertical lines
    if a1 == a2 && c1 == c2
        if !aeq(c1, a1, eps)
            return false
        end
        t = (b1 - c1) / (c2 - c1)
        s = (b2 - c1) / (c2 - c1)
        return in01(t, eps) || in01(s, eps)
    end

    # case 2: two non-vertical lines
    if a1 != a2 && c1 != c2
        m1 = slope(a1, b1, a2, b2)
        m2 = slope(c1, d1, c2, d2)

        # parallel lines
        if m1 == m2
            intercept1 = b1 - m1 * a1
            intercept2 = d1 - m2 * c1
            if !aeq(intercept1, intercept2, eps)
                return false
            end
            t = (b1 - c1) / (c2 - c1)
            s = (b2 - c1) / (c2 - c1)
            return in01(t, eps) || in01(s, eps)
        end
    end
        
    # case 3: two non-parallel lines
    t2 = (
        ((c1 - a1) * (b2 - b1) - (d1 - b1) * (a2 - a1))
        / ((d2 - d1) * (a2 - a1) - (c2 - c1) * (b2 - b1))
    )
    t1 = (d1 - b1 + t2 * (d2 - d1)) / (b2 - b1)
    return in01(t1, eps) && in01(t2, eps)
end

function polygon_intersect(
    coords1::Vector{Vector{T}},
    coords2::Vector{Vector{T}},
    eps::T
)::Bool where T <: Float64
    n = length(coords1)
    m = length(coords2)
    out = false
    @threads for i in 1:n - 1
        for j in 1:m - 1
            intersect = segment_intersect(
                coords1[i]...,
                coords1[i + 1]...,
                coords2[j]...,
                coords2[j + 1]...,
                eps
            )
            intersect && (out = true)
            out && break
        end
    end
    return false
end


function polygon_intersect(
    coords1::Vector{Shapefile.Point},
    coords2::Vector{Shapefile.Point},
    eps::T
)::Bool where T <: Float64
    n = length(coords1)
    m = length(coords2)
    out = false
    @threads for i in 1:n - 1
        for j in 1:m - 1
            intersect = segment_intersect(
                coords1[i].x,
                coords1[i].y,
                coords1[i + 1].x,
                coords1[i + 1].y,
                coords2[j].x,
                coords2[j].y,
                coords2[j + 1].x,
                coords2[j + 1].y,
                eps
            )
            intersect && (out = true)
            out && break
        end
    end
    return out
end


function harvesine(λ₁::T, φ₁::T, λ₂::T, φ₂::T)::T where T <: Float64
    Δλ = λ₂ - λ₁  # longitudes
    Δφ = φ₂ - φ₁  # latitudes
    # haversine formula
    a = sind(Δφ/2)^2 + cosd(φ₁)*cosd(φ₂)*sind(Δλ/2)^2
    # distance on the sphere
    # take care of floating point errors
    radius = 6371  # kms at poles
    2radius * asin( min(√a, one(a)) )
end


function find_polygon_adjacency(
    polygons::Vector{Vector{T}},
    lon::Vector{S},
    lat::Vector{S};
    min_kms::S = 50.0,  # mean distance to test
    eps::S = 0.001,  # error for the intersection
    max_matches::Int = 10
  ) where {T <: Union{Vector{Float64}, Shapefile.Point}, S <: Float64}
    N = length(polygons)
    src = Int[]
    tgt = Int[]
    dist = Float64[]
    @showprogress for i in 1:(N - 1)
        curr_matches = 0
        for j in (i + 1):N
            # curr_matches == 10 && return DataFrame(src=src, tgt=tgt, dist=dist)
            # do not test pts that are very far to save computation
            center_dist = harvesine(lon[i], lat[i], lon[j], lat[j])
            if center_dist < min_kms
                intersected = polygon_intersect(polygons[i], polygons[j], eps)
                if intersected
                    push!(src, i)
                    push!(tgt, j)
                    push!(dist, center_dist)
                    curr_matches += 1
                    curr_matches == max_matches && break
                end
            end
        end
    end
    return DataFrame(src=src, tgt=tgt, dist=dist)
end

function main()
    path = "data/tl_2016_us_zcta510/tl_2016_us_zcta510.shp"
    table = Shapefile.Table(path)
    polygons = [x.points for x in Shapefile.shapes(table)]
    lon = parse.(Float64, table.INTPTLON10)
    lat = parse.(Float64, table.INTPTLAT10)
    edges = find_polygon_adjacency(
        polygons, lon, lat, min_kms=120.0, max_matches=20, eps=0.01
    )
    edges.dist = round.(edges.dist, digits=3)
    edges[:src_lab] = [table.GEOID10[i] for i in edges.src]
    edges[:tgt_lab] = [table.GEOID10[i] for i in edges.tgt]
    CSV.write("model_dev_data/zip_adjacency.csv", edges)
end

main()