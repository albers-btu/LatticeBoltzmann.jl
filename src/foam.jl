"""
    set_foam!(model; D=0, k_H=0, k_Π=0, q=0, V_m=0, γ_b=1, c0=0, ρ_liquid=1)

Dissolved-gas parameters. Henry uses the paper, `c_H = k_H * ρ_b / 3`.
LBfoam's C++ uses `c = kh_LB * ρ_b / 4`, so `k_H = kh_LB` is 4/3 of that
interface concentration and `k_H = (3/4) * kh_LB` matches an XML run.
Their `pi_LB` is `k_Π * d_max` (`d_max = 4`), so `k_Π = pi_LB / 4`.
"""
function set_foam!(model; D=0, k_H=0, k_Π=0, q=0, V_m=0, γ_b=1, c0=0, ρ_liquid=1)
    @static if FOAM
        for domain in model.domains
            CT = typeof(domain.D)
            domain.D = CT(D)
            domain.k_H = CT(k_H)
            domain.k_Π = CT(k_Π)
            domain.q = CT(q)
            domain.V_m = CT(V_m)
            domain.γ_b = CT(γ_b)
            domain.c0 = CT(c0)
            domain.ρ_liquid = CT(ρ_liquid)
        end
        return model
    else
        throw(ArgumentError("FOAM is false"))
    end
end

"""
One bubble row. `frozen` is stored for a later PR and is never read.
`ratio` is not multiplied by `V_ref / V`; that factor is the imposed
density, and this PR still writes `ρb = 1`.
"""
struct Bubble
    V::Float64
    V_ref::Float64
    ratio::Float64
    frozen::Bool
end

const TAG_NONE = Int32(0)
const TAG_ATM = Int32(-1)

# Face neighbors only. Corner contact must not merge two bubbles.
const _FACE6 = (
    (1, 0, 0),
    (-1, 0, 0),
    (0, 1, 0),
    (0, -1, 0),
    (0, 0, 1),
    (0, 0, -1),
)

"""
Host bubble table plus the grid buffers reused every substep.
`Array(device)` each step would copy the whole lattice.
"""
mutable struct FoamHost{CType<:AbstractFloat}
    bubbles::Vector{Union{Nothing,Bubble}}
    free_ids::Vector{Int32}
    flags::Vector{UInt8}
    ϕ::Vector{CType}
    tag::Vector{Int32}
    tag_prev::Vector{Int32}
    ρb::Vector{CType}
    component::Vector{Int32}
    queue::Vector{Int}
end

function FoamHost{CType}(N::Int) where {CType<:AbstractFloat}
    return FoamHost{CType}(
        Vector{Union{Nothing,Bubble}}(),
        Int32[],
        Vector{UInt8}(undef, N),
        Vector{CType}(undef, N),
        zeros(Int32, N),
        zeros(Int32, N),
        fill(one(CType), N),
        zeros(Int32, N),
        Vector{Int}(undef, N),
    )
end

function _fit_host!(foam::FoamHost{CType}, N::Int) where {CType}
    length(foam.flags) == N && return foam
    resize!(foam.flags, N)
    resize!(foam.ϕ, N)
    resize!(foam.tag, N)
    resize!(foam.tag_prev, N)
    resize!(foam.ρb, N)
    fill!(foam.ρb, one(CType))
    resize!(foam.component, N)
    resize!(foam.queue, N)
    return foam
end

@inline function _cell_xyz(n::Int, Nx::Int, Ny::Int)
    n0 = n - 1
    x = n0 % Nx
    y = (n0 ÷ Nx) % Ny
    z = n0 ÷ (Nx * Ny)
    return x, y, z
end

@inline function _xyz_n(x::Int, y::Int, z::Int, Nx::Int, Ny::Int)
    return x + y * Nx + z * Nx * Ny + 1
end

function _live_bubble(foam::FoamHost, id::Integer)
    i = Int(id)
    1 <= i <= length(foam.bubbles) || return nothing
    return foam.bubbles[i]
end

function _alloc_bubble_id!(foam::FoamHost)
    if !isempty(foam.free_ids)
        return pop!(foam.free_ids)
    end
    id = length(foam.bubbles) + 1
    if id > MAX_BUBBLES
        error("MAX_BUBBLES exceeded (got $id); cap is $(MAX_BUBBLES)")
    end
    push!(foam.bubbles, nothing)
    return Int32(id)
end

function _release_bubble_id!(foam::FoamHost, id::Integer)
    i = Int(id)
    if 1 <= i <= length(foam.bubbles)
        foam.bubbles[i] = nothing
    end
    fid = Int32(i)
    for k in eachindex(foam.free_ids)
        foam.free_ids[k] == fid && return nothing
    end
    push!(foam.free_ids, fid)
    return nothing
end

function _require_foam(model)
    @static if FOAM
        return model
    else
        throw(ArgumentError("FOAM is false"))
    end
end

"""
    bubble_count(model) -> Int

Number of live rows. Tag `-1` (atmosphere) is not a row.
"""
function bubble_count(model)
    _require_foam(model)
    @static if FOAM
        n = 0
        for row in model.foam.bubbles
            row === nothing || (n += 1)
        end
        return n
    end
end

"""
    bubble_ids(model) -> Vector{Int32}

Live ids, ascending. Ids start at 1. Vanished ids stay on the free list
and are not returned.
"""
function bubble_ids(model)
    _require_foam(model)
    @static if FOAM
        ids = Int32[]
        for (i, row) in enumerate(model.foam.bubbles)
            row === nothing && continue
            push!(ids, Int32(i))
        end
        return ids
    end
end

"""
    bubble_volume(model, id) -> Float64

Gas volume from the last fill (or the punch, if no fill has run):
`Σ (1 − clamp(ϕ, 0, 1))` over cells tagged `id`.
"""
function bubble_volume(model, id)
    return _require_bubble(model, id).V
end

"""
    bubble_ratio(model, id) -> Float64
"""
function bubble_ratio(model, id)
    return _require_bubble(model, id).ratio
end

function _require_bubble(model, id)
    _require_foam(model)
    @static if FOAM
        row = _live_bubble(model.foam, id)
        row === nothing && throw(ArgumentError("no bubble with id $id"))
        return row
    end
end

"""
    bubble_stats(model) -> NamedTuple

`(n, ΣV, mean_ratio, max_abs_ρb, max_Π)`. No clamp-hit count: κ is not
in a copy of `ρb` and `Pi`.
"""
function bubble_stats(model)
    @static if FOAM
        n = 0
        ΣV = 0.0
        Σratio = 0.0
        for row in model.foam.bubbles
            row === nothing && continue
            n += 1
            ΣV += row.V
            Σratio += row.ratio
        end
        mean_ratio = n == 0 ? 0.0 : Σratio / n
        max_abs_ρb = 0.0
        max_Π = -Inf
        for domain in model.domains
            ρb = Array(domain.ρb.data)
            Pi = Array(domain.Pi.data)
            isempty(ρb) || (max_abs_ρb = max(max_abs_ρb, Float64(maximum(abs, ρb))))
            isempty(Pi) || (max_Π = max(max_Π, Float64(maximum(Pi))))
        end
        max_Π == -Inf && (max_Π = 0.0)
        return (; n, ΣV, mean_ratio, max_abs_ρb, max_Π)
    else
        throw(ArgumentError("FOAM is false"))
    end
end

# --- Poisson disk (centers only; does not punch) -------------------------

# SplitMix64. Seeded and local so a stdlib import is not required.
mutable struct _MixRNG
    state::UInt64
end

function _MixRNG(seed::Integer)
    s = UInt64(Int64(seed))
    s == zero(UInt64) && (s = 0x9e3779b97f4a7c15)
    return _MixRNG(s)
end

function _mix!(rng::_MixRNG)
    rng.state += 0x9e3779b97f4a7c15
    z = rng.state
    z = (z ⊻ (z >> 30)) * 0xbf58476d1ce4e5b9
    z = (z ⊻ (z >> 27)) * 0x94d049bb133111eb
    return z ⊻ (z >> 31)
end

_unit(rng::_MixRNG) = Float64(_mix!(rng) >> 11) * (1.0 / Float64(UInt64(1) << 53))

function _pick(rng::_MixRNG, n::Int)
    return Int(_mix!(rng) % UInt64(n)) + 1
end

"""
    poisson_disk_centers(Nx, Ny, Nz, rmin, n; seed=1) -> Vector{NTuple{3,Float64}}

Bridson sample in `[0, Nx) × [0, Ny) × [0, Nz)`. Centers are at least
`rmin` apart. For nuclei of radius `R`, pass `rmin = R + 1` so the shells
do not touch. Does not write flags or the bubble table.
"""
function poisson_disk_centers(Nx, Ny, Nz, rmin, n; seed=1)
    Nx = Int(Nx); Ny = Int(Ny); Nz = Int(Nz)
    n = Int(n)
    n >= 0 || throw(ArgumentError("poisson_disk_centers n must be non-negative (got $n)"))
    n == 0 && return NTuple{3,Float64}[]
    rmin = Float64(rmin)
    rmin > 0 || throw(ArgumentError("poisson_disk_centers rmin must be positive (got $rmin)"))
    rng = _MixRNG(seed)
    cell = rmin / sqrt(3.0)
    gx = max(1, ceil(Int, Nx / cell))
    gy = max(1, ceil(Int, Ny / cell))
    gz = max(1, ceil(Int, Nz / cell))
    grid = zeros(Int, gx, gy, gz)
    pts = NTuple{3,Float64}[]
    active = Int[]
    rmin2 = rmin * rmin

    function grid_index(p)
        ix = clamp(floor(Int, p[1] / cell) + 1, 1, gx)
        iy = clamp(floor(Int, p[2] / cell) + 1, 1, gy)
        iz = clamp(floor(Int, p[3] / cell) + 1, 1, gz)
        return ix, iy, iz
    end
    function far_enough(p)
        ix, iy, iz = grid_index(p)
        for dz in -2:2, dy in -2:2, dx in -2:2
            jx = ix + dx; jy = iy + dy; jz = iz + dz
            (1 <= jx <= gx && 1 <= jy <= gy && 1 <= jz <= gz) || continue
            k = grid[jx, jy, jz]
            k == 0 && continue
            q = pts[k]
            dxp = p[1] - q[1]; dyp = p[2] - q[2]; dzp = p[3] - q[3]
            if dxp * dxp + dyp * dyp + dzp * dzp < rmin2
                return false
            end
        end
        return true
    end
    function accept!(p)
        push!(pts, p)
        k = length(pts)
        push!(active, k)
        ix, iy, iz = grid_index(p)
        grid[ix, iy, iz] = k
        return k
    end

    accept!((_unit(rng) * Nx, _unit(rng) * Ny, _unit(rng) * Nz))
    ktry = 30
    while !isempty(active) && length(pts) < n
        ai = _pick(rng, length(active))
        base = pts[active[ai]]
        found = false
        for _try in 1:ktry
            # Uniform direction, radius in [rmin, 2 rmin].
            z = 2 * _unit(rng) - 1
            φ = 2π * _unit(rng)
            s = sqrt(max(0.0, 1 - z * z))
            rad = rmin * (1 + _unit(rng))
            cand = (
                base[1] + rad * s * cos(φ),
                base[2] + rad * s * sin(φ),
                base[3] + rad * z,
            )
            if cand[1] < 0 || cand[1] >= Nx || cand[2] < 0 || cand[2] >= Ny || cand[3] < 0 || cand[3] >= Nz
                continue
            end
            if far_enough(cand)
                accept!(cand)
                found = true
                break
            end
        end
        found || deleteat!(active, ai)
    end
    if length(pts) < n
        error("poisson_disk_centers placed $(length(pts))/$n centers at rmin=$rmin")
    end
    return pts[1:n]
end

# --- Punch ----------------------------------------------------------------

function _cube_dist2_bounds(cx, cy, cz, x, y, z)
    closest(c, a) = c <= a ? a : (c >= a + 1 ? a + 1 : c)
    farthest(c, a) = abs(c - a) >= abs(c - (a + 1)) ? a : a + 1
    qx = closest(cx, x); qy = closest(cy, y); qz = closest(cz, z)
    fx = farthest(cx, x); fy = farthest(cy, y); fz = farthest(cz, z)
    dmin2 = (qx - cx)^2 + (qy - cy)^2 + (qz - cz)^2
    dmax2 = (fx - cx)^2 + (fy - cy)^2 + (fz - cz)^2
    return dmin2, dmax2
end

# 0 outside, 1 fully inside, 2 cut. Touching from the outside is outside.
function _cube_class(cx, cy, cz, x, y, z, R)
    dmin2, dmax2 = _cube_dist2_bounds(cx, cy, cz, x, y, z)
    R2 = R * R
    dmin2 >= R2 && return 0
    dmax2 <= R2 && return 1
    return 2
end

# plic_cube maps liquid volume → plane offset and increases with ϕ.
# Twenty bisections are enough at init; this is not a kernel.
function _invert_plic(α::Float64, nrm::SVector{3,Float64})
    lo = 0.0
    hi = 1.0
    for _ in 1:20
        mid = (lo + hi) * 0.5
        if plic_cube(mid, nrm) < α
            lo = mid
        else
            hi = mid
        end
    end
    return (lo + hi) * 0.5
end

# Liquid is outside the sphere. The normal points toward the gas
# (toward the center), and α = d − R is the plane offset along that normal.
function _cap_phi(cx, cy, cz, x, y, z, R)
    dx = (x + 0.5) - cx
    dy = (y + 0.5) - cy
    dz = (z + 0.5) - cz
    d = sqrt(dx * dx + dy * dy + dz * dz)
    d < 1e-12 && return 0.0
    nrm = SVector{3,Float64}(-dx / d, -dy / d, -dz / d)
    return clamp(_invert_plic(d - R, nrm), 0.0, 1.0)
end

function _cell_of_point(c, N)
    c <= 0 && return 0
    c >= N && return N - 1
    return floor(Int, c)
end

function _reject_punch_cell!(foam, n, cx, cy, cz)
    fl = foam.flags[n]
    if (fl & TYPE_S) != 0
        throw(ArgumentError("nucleate_bubbles! center ($cx, $cy, $cz) overlaps TYPE_S"))
    end
    if foam.tag[n] > 0 || (fl & TYPE_SU) == TYPE_G || (fl & TYPE_SU) == TYPE_I
        throw(ArgumentError("nucleate_bubbles! center ($cx, $cy, $cz) overlaps an existing nucleus"))
    end
    return nothing
end

function _punch_sphere!(foam::FoamHost, center, R::Float64, Nx::Int, Ny::Int, Nz::Int)
    R > 0 || throw(ArgumentError("nucleate_bubbles! radius must be positive (got $R)"))
    cx = Float64(center[1]); cy = Float64(center[2]); cz = Float64(center[3])
    ix = _cell_of_point(cx, Nx)
    iy = _cell_of_point(cy, Ny)
    iz = _cell_of_point(cz, Nz)
    nctr = _xyz_n(ix, iy, iz, Nx, Ny)
    if (foam.flags[nctr] & TYPE_S) != 0
        throw(ArgumentError("nucleate_bubbles! center ($cx, $cy, $cz) lands in TYPE_S"))
    end

    pad = R + 1.5
    x0 = max(0, floor(Int, cx - pad))
    x1 = min(Nx - 1, floor(Int, cx + pad))
    y0 = max(0, floor(Int, cy - pad))
    y1 = min(Ny - 1, floor(Int, cy + pad))
    z0 = max(0, floor(Int, cz - pad))
    z1 = min(Nz - 1, floor(Int, cz + pad))

    cells = Int[]
    phis = Float64[]
    gas = Bool[]
    for z in z0:z1, y in y0:y1, x in x0:x1
        cls = _cube_class(cx, cy, cz, x, y, z, R)
        cls == 0 && continue
        n = _xyz_n(x, y, z, Nx, Ny)
        _reject_punch_cell!(foam, n, cx, cy, cz)
        push!(cells, n)
        if cls == 1
            push!(phis, 0.0)
            push!(gas, true)
        else
            push!(phis, _cap_phi(cx, cy, cz, x, y, z, R))
            push!(gas, false)
        end
    end
    isempty(cells) && throw(ArgumentError("nucleate_bubbles! sphere at ($cx, $cy, $cz) covers no cell"))

    CT = eltype(foam.ϕ)
    for (k, n) in enumerate(cells)
        if gas[k]
            foam.flags[n] = (foam.flags[n] & ~TYPE_SU) | TYPE_G
            foam.ϕ[n] = zero(CT)
        else
            foam.flags[n] = (foam.flags[n] & ~TYPE_SU) | TYPE_I
            foam.ϕ[n] = CT(phis[k])
        end
    end

    # initialize_body! turns a TYPE_G cell that sees TYPE_F on the D3Q19
    # stencil into TYPE_I with ϕ = 0.5. Paint those neighbors first so the
    # punched cap is what initialize keeps.
    vel = velocities(:D3Q19)
    extra = Int[]
    extraϕ = Float64[]
    seen = Set(cells)
    for (k, n) in enumerate(cells)
        gas[k] || continue
        x, y, z = _cell_xyz(n, Nx, Ny)
        for i in 2:length(vel)
            j = src_index(x, y, z, vel[i][1], vel[i][2], vel[i][3], Nx, Ny, Nz)
            (foam.flags[j] & TYPE_SU) == TYPE_F || continue
            j in seen && continue
            _reject_punch_cell!(foam, j, cx, cy, cz)
            push!(seen, j)
            xj, yj, zj = _cell_xyz(j, Nx, Ny)
            push!(extra, j)
            # Fully outside the sphere: ϕ = 1 so the blocker adds no gas
            # volume. A cut neighbor still gets the cap.
            cls = _cube_class(cx, cy, cz, xj, yj, zj, R)
            push!(extraϕ, cls == 0 ? 1.0 : _cap_phi(cx, cy, cz, xj, yj, zj, R))
        end
    end
    for (k, n) in enumerate(extra)
        foam.flags[n] = (foam.flags[n] & ~TYPE_SU) | TYPE_I
        foam.ϕ[n] = CT(extraϕ[k])
    end

    id = _alloc_bubble_id!(foam)
    V = 0.0
    for (k, n) in enumerate(cells)
        foam.tag[n] = id
        V += 1 - phis[k]
    end
    for (k, n) in enumerate(extra)
        ϕk = extraϕ[k]
        foam.tag[n] = id
        foam.ϕ[n] = CT(ϕk)
        V += 1 - ϕk
    end
    foam.bubbles[Int(id)] = Bubble(V, V, 1.0, false)
    return id
end

"""
    nucleate_bubbles!(model, centers, radii)

Punch one sphere per center **before** `initialize!`. Interior cells are
`TYPE_G` (`ϕ = 0`); cells cut by the sphere are `TYPE_I` with `ϕ` the
liquid volume outside the plane at distance `R` from the center. One table
row per ball, `ratio = 1`, `V_ref` = punched gas volume. Populations are
left for `initialize!`. A center in `TYPE_S`, or a ball that overlaps
another nucleus or a solid cell, throws.
"""
function nucleate_bubbles!(model, centers::AbstractVector, radii::AbstractVector)
    @static if !FOAM
        throw(ArgumentError("FOAM is false"))
    else
        if length(centers) != length(radii)
            throw(ArgumentError("nucleate_bubbles! got $(length(centers)) centers and $(length(radii)) radii"))
        end
        length(model.domains) == 1 ||
            throw(ArgumentError("nucleate_bubbles! v1 supports a single domain"))
        domain = model.domains[1]
        foam = model.foam
        _fit_host!(foam, length(domain.flags))
        KernelAbstractions.synchronize(model.backend)
        copyto!(foam.flags, domain.flags.data)
        copyto!(foam.ϕ, domain.ϕ.data)
        copyto!(foam.tag, domain.tag.data)
        Nx = Int(domain.Nx); Ny = Int(domain.Ny); Nz = Int(domain.Nz)
        for (center, radius) in zip(centers, radii)
            _punch_sphere!(foam, center, Float64(radius), Nx, Ny, Nz)
        end
        copyto!(domain.flags.data, foam.flags)
        copyto!(domain.ϕ.data, foam.ϕ)
        copyto!(domain.tag.data, foam.tag)
        return model
    end
end

# --- Flood fill ------------------------------------------------------------

function _flood_gas!(foam::FoamHost, Nx::Int, Ny::Int, Nz::Int)
    N = Nx * Ny * Nz
    flags = foam.flags
    comp = foam.component
    q = foam.queue
    fill!(comp, Int32(0))
    ncomp = 0
    for seed in 1:N
        comp[seed] != 0 && continue
        (flags[seed] & TYPE_SU) != TYPE_G && continue
        ncomp += 1
        cid = Int32(ncomp)
        comp[seed] = cid
        qh = 1
        qt = 1
        q[1] = seed
        while qh <= qt
            n = q[qh]
            qh += 1
            x, y, z = _cell_xyz(n, Nx, Ny)
            for (cx, cy, cz) in _FACE6
                j = src_index(x, y, z, cx, cy, cz, Nx, Ny, Nz)
                comp[j] != 0 && continue
                (flags[j] & TYPE_SU) != TYPE_G && continue
                comp[j] = cid
                qt += 1
                q[qt] = j
            end
        end
    end
    return ncomp
end

function _collect_parents(foam::FoamHost, ncomp::Int, N::Int)
    parents = [Int32[] for _ in 1:ncomp]
    atm = falses(ncomp)
    comp = foam.component
    prev = foam.tag_prev
    for n in 1:N
        c = Int(comp[n])
        c == 0 && continue
        t = prev[n]
        if t == TAG_ATM
            atm[c] = true
        elseif t > 0
            _live_bubble(foam, t) === nothing && continue
            ids = parents[c]
            seen = false
            for k in eachindex(ids)
                if ids[k] == t
                    seen = true
                    break
                end
            end
            seen || push!(ids, t)
        end
    end
    return parents, atm
end

# PR3 merge/split does not conserve Σ ratio·V_ref. Each child keeps the
# minimum old id's ratio and sets V_ref to its own current V. A 1–1
# transition keeps the old id and the old V_ref. Gas that overlaps neither
# a live id nor tag -1 is atmosphere, not a new bubble: nucleation is the
# only source of positive ids (split children are the one exception, so
# two components do not share a row).
function _assign_components!(foam::FoamHost, parents, atm, ncomp::Int)
    comp_tag = fill(TAG_ATM, ncomp)
    merged = falses(ncomp)
    # A component that also overlaps tag -1 has burst: those rows are dropped.
    doomed = Set{Int32}()
    for c in 1:ncomp
        atm[c] || continue
        for id in parents[c]
            push!(doomed, id)
        end
    end

    groups = Dict{Int32,Vector{Int}}()
    for c in 1:ncomp
        (atm[c] || isempty(parents[c])) && continue
        live = Int32[]
        for id in parents[c]
            id in doomed || push!(live, id)
        end
        if isempty(live)
            comp_tag[c] = TAG_ATM
            continue
        end
        keep = minimum(live)
        if length(live) > 1
            merged[c] = true
        end
        comp_tag[c] = keep
        push!(get!(groups, keep, Int[]), c)
    end

    # Non-min parents of a merge are not kept. A component that still
    # claimed one of them gets a fresh id (split away from the merge).
    absorbed = Set{Int32}()
    for c in 1:ncomp
        merged[c] || continue
        keep = comp_tag[c]
        for id in parents[c]
            (id == keep || id in doomed) && continue
            push!(absorbed, id)
        end
    end

    meta = Dict{Int32,Tuple{Int8,Float64,Float64}}()
    for id in absorbed
        cs = pop!(groups, id, nothing)
        cs === nothing && continue
        row = _live_bubble(foam, id)
        r = row === nothing ? 1.0 : row.ratio
        for c in cs
            nid = _alloc_bubble_id!(foam)
            comp_tag[c] = nid
            meta[nid] = (Int8(2), r, 0.0)
        end
    end

    for (id, cs) in groups
        row = _live_bubble(foam, id)
        r = row === nothing ? 1.0 : row.ratio
        vr = row === nothing ? 0.0 : row.V_ref
        css = sort(cs)
        if length(css) == 1
            c = css[1]
            comp_tag[c] = id
            meta[id] = (merged[c] ? Int8(2) : Int8(1), r, vr)
        else
            for (k, c) in enumerate(css)
                cid = k == 1 ? id : _alloc_bubble_id!(foam)
                comp_tag[c] = cid
                meta[cid] = (Int8(2), r, vr)
            end
        end
    end
    return comp_tag, meta
end

function _tag_interface!(flags, tag, Nx, Ny, Nz)
    N = Nx * Ny * Nz
    for n in 1:N
        (flags[n] & TYPE_SU) == TYPE_I || continue
        x, y, z = _cell_xyz(n, Nx, Ny)
        best = TAG_NONE
        npos = 0
        atm = false
        for (cx, cy, cz) in _FACE6
            j = src_index(x, y, z, cx, cy, cz, Nx, Ny, Nz)
            (flags[j] & TYPE_SU) == TYPE_G || continue
            tj = tag[j]
            if tj > 0
                npos += 1
                best = npos == 1 ? tj : min(best, tj)
            elseif tj < 0
                atm = true
            end
        end
        if npos > 0
            tag[n] = best
        elseif atm
            tag[n] = TAG_ATM
        else
            tag[n] = TAG_NONE
        end
    end
    return nothing
end

function _commit_rows!(foam::FoamHost, meta, N::Int)
    nb = length(foam.bubbles)
    vol = zeros(Float64, nb)
    tag = foam.tag
    ϕ = foam.ϕ
    for n in 1:N
        t = Int(tag[n])
        t > 0 || continue
        ϕn = Float64(ϕ[n])
        ϕn = ϕn < 0 ? 0.0 : (ϕn > 1 ? 1.0 : ϕn)
        vol[t] += 1 - ϕn
    end
    used = falses(nb)
    for (id, (kind, r, vr)) in meta
        i = Int(id)
        V = vol[i]
        if !(V > 0)
            continue
        end
        used[i] = true
        # kind 2: merge or split. V_ref = this child's V, not the parent's.
        foam.bubbles[i] = kind == Int8(1) ? Bubble(V, vr, r, false) : Bubble(V, V, r, false)
    end
    for i in 1:nb
        used[i] && continue
        foam.bubbles[i] === nothing && continue
        _release_bubble_id!(foam, i)
    end
    return nothing
end

function _retag!(foam::FoamHost, Nx::Int, Ny::Int, Nz::Int)
    N = Nx * Ny * Nz
    ncomp = _flood_gas!(foam, Nx, Ny, Nz)
    fill!(foam.tag, TAG_NONE)
    if ncomp == 0
        _tag_interface!(foam.flags, foam.tag, Nx, Ny, Nz)
        for i in 1:length(foam.bubbles)
            foam.bubbles[i] === nothing && continue
            _release_bubble_id!(foam, i)
        end
        return nothing
    end
    parents, atm = _collect_parents(foam, ncomp, N)
    comp_tag, meta = _assign_components!(foam, parents, atm, ncomp)
    comp = foam.component
    tag = foam.tag
    for n in 1:N
        c = Int(comp[n])
        c == 0 && continue
        tag[n] = comp_tag[c]
    end
    _tag_interface!(foam.flags, tag, Nx, Ny, Nz)
    _commit_rows!(foam, meta, N)
    return nothing
end

"""
Host section at the start of the substep. Reads post-`surface_3` flags
from the previous substep. Writes a complete tag array and `ρb = 1`
(ideal-gas `ρb` is a later PR) before `surface_0`.
"""
function foam_host!(model, domain)
    @static if FOAM
        foam = model.foam
        _fit_host!(foam, length(domain.flags))
        # The trailing sync covers the previous step!. This one covers surface_3
        # when the substep is entered again in the same step!.
        KernelAbstractions.synchronize(model.backend)
        copyto!(foam.flags, domain.flags.data)
        copyto!(foam.ϕ, domain.ϕ.data)
        copyto!(foam.tag_prev, domain.tag.data)
        _retag!(foam, Int(domain.Nx), Int(domain.Ny), Int(domain.Nz))
        fill!(foam.ρb, one(eltype(foam.ρb)))
        copyto!(domain.tag.data, foam.tag)
        copyto!(domain.ρb.data, foam.ρb)
        # Δm → ratio is a later PR. Drop the bin so it cannot accumulate.
        fill!(domain.flux.data, zero(eltype(domain.flux.data)))
    end
    return nothing
end
