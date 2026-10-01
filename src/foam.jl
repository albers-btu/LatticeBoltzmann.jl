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
            # A zero coefficient must not leave surface_0 subtracting a stale 3Π.
            domain.k_Π == zero(CT) && fill!(domain.Pi.data, zero(CT))
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
One bubble row. `frozen` is set when every interface cell of this id is
solid (`is_solid_fraction`: liquid fraction below 10⁻³) and is never cleared.
A frozen row keeps its ratio and its id: dissolved flux is ignored, it does
not merge, and gas that touches atmosphere does not delete it.
The imposed density is `ratio * (V_ref / V)^γ_b`.
While the row is liquid, dissolved mass adds `Δm * V_m * ρ_liquid / V_ref` to `ratio`.
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
    blockers::Vector{Int}               # Class-0 cells painted only for initialize!
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
        Int[],
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

"""
    bubble_frozen(model, id) -> Bool

True after every interface cell of `id` has solidified. Stays true.
"""
function bubble_frozen(model, id)
    return _require_bubble(model, id).frozen
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

# Minimum image on a periodic axis. The fill wraps through src_index.
function _wrap_delta(d::Float64, L::Float64)
    return d - L * round(d / L)
end

function _wrap_pos(x::Float64, L::Int)
    y = mod(x, Float64(L))
    return y == Float64(L) ? 0.0 : y
end

function _wrap_idx(i::Int, n::Int)
    return mod(i - 1, n) + 1
end

"""
    poisson_disk_centers(Nx, Ny, Nz, rmin, n; seed=1) -> Vector{NTuple{3,Float64}}
    poisson_disk_centers(Nx, Ny, Nz, rmin; seed=1, margin=0) -> Vector{NTuple{3,Float64}}

Bridson sample. Centers are at least `rmin` apart. The five-argument
method fills the periodic box `[0, Nx) × [0, Ny) × [0, Nz)` and errors
if it cannot place `n` points. The four-argument method places as many
as fit. `margin > 0` insets that box to `[margin, N − margin)` and drops
the periodic images, so a closed solid wall can sit in the margin.
For nuclei of radius `R`, pass `rmin >= 2R + 1` so the shells do not
intersect, and `margin >= R + 2.5` so the punch stays off the wall cells.
Does not write flags or the bubble table.
"""
function poisson_disk_centers(Nx, Ny, Nz, rmin, n::Integer; seed=1)
    pts = _poisson_disk_centers(Nx, Ny, Nz, rmin; seed=seed, margin=0.0, nmax=Int(n))
    if length(pts) < Int(n)
        error("poisson_disk_centers placed $(length(pts))/$n centers at rmin=$rmin")
    end
    return pts
end

function poisson_disk_centers(Nx, Ny, Nz, rmin; seed=1, margin::Real=0)
    return _poisson_disk_centers(Nx, Ny, Nz, rmin; seed=seed, margin=Float64(margin), nmax=typemax(Int))
end

function _poisson_disk_centers(Nx, Ny, Nz, rmin; seed=1, margin::Float64, nmax::Int)
    Nx = Int(Nx); Ny = Int(Ny); Nz = Int(Nz)
    nmax >= 0 || throw(ArgumentError("poisson_disk_centers n must be non-negative (got $nmax)"))
    nmax == 0 && return NTuple{3,Float64}[]
    rmin = Float64(rmin)
    rmin > 0 || throw(ArgumentError("poisson_disk_centers rmin must be positive (got $rmin)"))
    margin >= 0 || throw(ArgumentError("poisson_disk_centers margin must be non-negative (got $margin)"))
    periodic = margin == 0
    x0, x1 = margin, Float64(Nx) - margin
    y0, y1 = margin, Float64(Ny) - margin
    z0, z1 = margin, Float64(Nz) - margin
    (x0 < x1 && y0 < y1 && z0 < z1) ||
        throw(ArgumentError("poisson_disk_centers margin=$margin leaves no interior in $(Nx)×$(Ny)×$(Nz)"))
    rng = _MixRNG(seed)
    cell = rmin / sqrt(3.0)
    gx = max(1, ceil(Int, (x1 - x0) / cell))
    gy = max(1, ceil(Int, (y1 - y0) / cell))
    gz = max(1, ceil(Int, (z1 - z0) / cell))
    grid = zeros(Int, gx, gy, gz)
    pts = NTuple{3,Float64}[]
    active = Int[]
    rmin2 = rmin * rmin
    Lx = Float64(Nx); Ly = Float64(Ny); Lz = Float64(Nz)

    function grid_index(p)
        ix = floor(Int, (p[1] - x0) / cell) + 1
        iy = floor(Int, (p[2] - y0) / cell) + 1
        iz = floor(Int, (p[3] - z0) / cell) + 1
        if periodic
            return _wrap_idx(ix, gx), _wrap_idx(iy, gy), _wrap_idx(iz, gz)
        end
        return clamp(ix, 1, gx), clamp(iy, 1, gy), clamp(iz, 1, gz)
    end
    function far_enough(p)
        ix, iy, iz = grid_index(p)
        for dz in -2:2, dy in -2:2, dx in -2:2
            jx, jy, jz = ix + dx, iy + dy, iz + dz
            if periodic
                jx, jy, jz = _wrap_idx(jx, gx), _wrap_idx(jy, gy), _wrap_idx(jz, gz)
            elseif !(1 <= jx <= gx && 1 <= jy <= gy && 1 <= jz <= gz)
                continue
            end
            k = grid[jx, jy, jz]
            k == 0 && continue
            q = pts[k]
            dxp = periodic ? _wrap_delta(p[1] - q[1], Lx) : p[1] - q[1]
            dyp = periodic ? _wrap_delta(p[2] - q[2], Ly) : p[2] - q[2]
            dzp = periodic ? _wrap_delta(p[3] - q[3], Lz) : p[3] - q[3]
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
    function inside(p)
        return x0 <= p[1] < x1 && y0 <= p[2] < y1 && z0 <= p[3] < z1
    end

    accept!((x0 + _unit(rng) * (x1 - x0), y0 + _unit(rng) * (y1 - y0), z0 + _unit(rng) * (z1 - z0)))
    ktry = 30
    while !isempty(active) && length(pts) < nmax
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
            if periodic
                cand = (_wrap_pos(cand[1], Nx), _wrap_pos(cand[2], Ny), _wrap_pos(cand[3], Nz))
            elseif !inside(cand)
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
    return pts
end

"""
    seed_poisson_bubbles!(model, R; rmin=2R+1, margin=R+2.5, seed=1)

Punch a Poisson-disk packing of radius-`R` nuclei into the domain.
Flags must already be painted (liquid in the interior, solid on the wall).
`rmin` is the center spacing. `margin` keeps each punch off the wall.
"""
function seed_poisson_bubbles!(model, R; rmin=nothing, margin=nothing, seed=1)
    R = Float64(R)
    R > 0 || throw(ArgumentError("seed_poisson_bubbles! radius must be positive (got $R)"))
    rmin === nothing && (rmin = 2R + 1)
    margin === nothing && (margin = R + 2.5)
    length(model.domains) == 1 ||
        throw(ArgumentError("seed_poisson_bubbles! v1 supports a single domain"))
    domain = model.domains[1]
    pts = poisson_disk_centers(Int(domain.Nx), Int(domain.Ny), Int(domain.Nz), rmin;
                               seed=seed, margin=Float64(margin))
    isempty(pts) && throw(ArgumentError("seed_poisson_bubbles! placed no centers at rmin=$rmin"))
    nucleate_bubbles!(model, pts, fill(R, length(pts)))
    return pts
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
# A center that sits on the cell center has no radial direction; bisect
# along −z anyway so the cell is not forced to ϕ = 0.
function _cap_phi(cx, cy, cz, x, y, z, R)
    dx = (x + 0.5) - cx
    dy = (y + 0.5) - cy
    dz = (z + 0.5) - cz
    d = sqrt(dx * dx + dy * dy + dz * dz)
    if d < 1e-12
        nrm = SVector{3,Float64}(0.0, 0.0, -1.0)
        return clamp(_invert_plic(-R, nrm), 0.0, 1.0)
    end
    nrm = SVector{3,Float64}(-dx / d, -dy / d, -dz / d)
    return clamp(_invert_plic(d - R, nrm), 0.0, 1.0)
end

function _cell_of_point(c, N)
    c <= 0 && return 0
    c >= N && return N - 1
    return floor(Int, c)
end

function _drop_blocker!(foam, n)
    bs = foam.blockers
    for k in eachindex(bs)
        bs[k] == n || continue
        bs[k] = bs[end]
        pop!(bs)
        return nothing
    end
    return nothing
end

function _reject_punch_cell!(foam, n, cx, cy, cz)
    fl = foam.flags[n]
    if (fl & TYPE_S) != 0
        throw(ArgumentError("nucleate_bubbles! center ($cx, $cy, $cz) overlaps TYPE_S"))
    end
    # Tag 0 and TYPE_I is a class-0 blocker, not a nucleus. A later sphere may claim it.
    if foam.tag[n] > 0 || (fl & TYPE_SU) == TYPE_G
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
        _drop_blocker!(foam, n)
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
    extra_out = Bool[]
    seen = Set(cells)
    for (k, n) in enumerate(cells)
        gas[k] || continue
        x, y, z = _cell_xyz(n, Nx, Ny)
        for i in 2:length(vel)
            j = src_index(x, y, z, vel[i][1], vel[i][2], vel[i][3], Nx, Ny, Nz)
            (foam.flags[j] & TYPE_SU) == TYPE_F || continue
            j in seen && continue
            _reject_punch_cell!(foam, j, cx, cy, cz)
            _drop_blocker!(foam, j)
            push!(seen, j)
            xj, yj, zj = _cell_xyz(j, Nx, Ny)
            push!(extra, j)
            cls = _cube_class(cx, cy, cz, xj, yj, zj, R)
            # Class 0 is outside the sphere. ϕ = 1 and it is not part of the
            # bubble; initialize! would otherwise rewrite the interior.
            push!(extra_out, cls == 0)
            push!(extraϕ, cls == 0 ? 1.0 : _cap_phi(cx, cy, cz, xj, yj, zj, R))
        end
    end
    shell = Int[]
    shellϕ = Float64[]
    for (k, n) in enumerate(extra)
        if extra_out[k]
            foam.flags[n] = (foam.flags[n] & ~TYPE_SU) | TYPE_I
            foam.ϕ[n] = one(CT)
            push!(foam.blockers, n)
        else
            foam.flags[n] = (foam.flags[n] & ~TYPE_SU) | TYPE_I
            foam.ϕ[n] = CT(extraϕ[k])
            push!(shell, n)
            push!(shellϕ, extraϕ[k])
        end
    end

    id = _alloc_bubble_id!(foam)
    V = 0.0
    for (k, n) in enumerate(cells)
        foam.tag[n] = id
        V += 1 - phis[k]
    end
    for (k, n) in enumerate(shell)
        ϕk = shellϕ[k]
        foam.tag[n] = id
        V += 1 - ϕk
    end
    foam.bubbles[Int(id)] = Bubble(V, V, 1.0, false)
    return id
end

function _table_snapshot(foam::FoamHost)
    return (copy(foam.bubbles), copy(foam.free_ids), copy(foam.blockers))
end

function _restore_table!(foam::FoamHost, snap)
    rows, free, blockers = snap
    empty!(foam.bubbles)
    append!(foam.bubbles, rows)
    empty!(foam.free_ids)
    append!(foam.free_ids, free)
    empty!(foam.blockers)
    append!(foam.blockers, blockers)
    return nothing
end

"""
    nucleate_bubbles!(model, centers, radii)

Punch one sphere per center **before** `initialize!`. Interior cells are
`TYPE_G` (`ϕ = 0`); cells cut by the sphere are `TYPE_I` with `ϕ` the
liquid volume outside the plane at distance `R` from the center. One table
row per ball, `ratio = 1`, `V_ref` = punched gas volume. Populations are
left for `initialize!`. A center in `TYPE_S`, or a ball that overlaps
another nucleus or a solid cell, throws.

After `initialize!`, use `spawn_bubbles!`. This punch leaves the class-0
neighbors as temporary interface cells so `initialize_body!` does not
rewrite the interior, and it does not rebuild populations.
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
        # A later center can throw. The device grid is still the pre-batch
        # state until every sphere has been accepted.
        snap = _table_snapshot(foam)
        try
            for (center, radius) in zip(centers, radii)
                _punch_sphere!(foam, center, Float64(radius), Nx, Ny, Nz)
            end
        catch
            _restore_table!(foam, snap)
            copyto!(foam.flags, domain.flags.data)
            copyto!(foam.ϕ, domain.ϕ.data)
            copyto!(foam.tag, domain.tag.data)
            rethrow()
        end
        copyto!(domain.flags.data, foam.flags)
        copyto!(domain.ϕ.data, foam.ϕ)
        copyto!(domain.tag.data, foam.tag)
        return model
    end
end

# Class-0 D3Q19 neighbors are TYPE_I only so initialize_body! does not
# rewrite interior gas. Once that has run, they are liquid again.
function _restore_punch_blockers!(model)
    @static if FOAM
        foam = model.foam
        isempty(foam.blockers) && return nothing
        domain = model.domains[1]
        copyto!(foam.flags, domain.flags.data)
        copyto!(foam.ϕ, domain.ϕ.data)
        copyto!(foam.tag, domain.tag.data)
        CT = eltype(foam.ϕ)
        for n in foam.blockers
            foam.flags[n] = (foam.flags[n] & ~TYPE_SU) | TYPE_F
            foam.ϕ[n] = one(CT)
            foam.tag[n] = TAG_NONE
        end
        copyto!(domain.flags.data, foam.flags)
        copyto!(domain.ϕ.data, foam.ϕ)
        copyto!(domain.tag.data, foam.tag)
        empty!(foam.blockers)
    end
    return nothing
end

# A spawned pore was liquid. Its populations are a liquid equilibrium, and
# the new interface reads the gas cell on the next surface pass. Write a
# rest equilibrium into the new gas cells only. Class-0 blockers are already
# liquid again, so they are left alone. Neighbor AA slots that happen to
# sit on the new gas cell are replaced; the compact this is aimed at is
# still near ρ = 1, u = 0, so that replacement is the same equilibrium.
function _equilibrate_spawned!(model, domain, flags_before::Vector{UInt8})
    flags = Array(domain.flags.data)
    ϕ = Array(domain.ϕ.data)
    ρ = domain.ρ.data
    u = domain.u.data
    mass = domain.mass.data
    T = domain.T.data
    fi = domain.fi.data
    gi = domain.gi.data
    N = length(flags)
    CT = eltype(ρ)
    oneρ = one(CT)
    zeroρ = zero(CT)
    w = model.weights
    vel = model.velocities
    @inbounds for n in eachindex(flags)
        su0 = flags_before[n] & TYPE_SU
        su1 = flags[n] & TYPE_SU
        su0 == su1 && continue
        if su1 == TYPE_G
            ρ[n] = oneρ
            u[n, 1] = zeroρ
            u[n, 2] = zeroρ
            u[n, 3] = zeroρ
            mass[n] = zeroρ
            store_feq_local!(fi, n, oneρ, zeroρ, zeroρ, zeroρ, w, vel, N)
            @static if TEMPERATURE
                store_geq_local!(gi, n, CT(T[n]), N)
            end
        elseif su1 == TYPE_I
            mass[n] = CT(ϕ[n]) * CT(ρ[n])
        end
    end
    # The punch is not a fill change. ϕ_correction would otherwise credit
    # −c Δϕ onto the new row.
    copyto!(domain.ϕ_old.data, domain.ϕ.data)
    return nothing
end

"""
    spawn_bubbles!(model, centers, radii) -> Vector{Int32}

Punch spheres **after** `initialize!` and return the new ids. Same geometry
as `nucleate_bubbles!`, then the class-0 blockers are turned back into
liquid and each new gas cell is written as a rest equilibrium at ρ = 1.
The shell keeps its temperature and solid fraction; its mass is set to
`ϕ ρ`. A center in `TYPE_S`, or a ball that overlaps another pore, throws
and leaves the grid as it was.
"""
function spawn_bubbles!(model, centers::AbstractVector, radii::AbstractVector)
    @static if !FOAM
        throw(ArgumentError("FOAM is false"))
    else
        model.initialized ||
            throw(ArgumentError("spawn_bubbles! runs after initialize!; use nucleate_bubbles! before it"))
        length(model.domains) == 1 ||
            throw(ArgumentError("spawn_bubbles! v1 supports a single domain"))
        domain = model.domains[1]
        before = Set(bubble_ids(model))
        flags_before = Array(domain.flags.data)
        nucleate_bubbles!(model, centers, radii)
        _restore_punch_blockers!(model)
        _equilibrate_spawned!(model, domain, flags_before)
        spawned = Int32[]
        for id in bubble_ids(model)
            id in before || push!(spawned, id)
        end
        return spawned
    end
end

"""
    add_dissolved!(model, δ)

Add a dissolved-concentration source. `δ[n]` is added to the rest
population of cell `n` and to `c`. The rest slot does not stream, so the
next concentration collide carries the mass. Gas and solid cells are
skipped. `δ` follows the same linear index as `flags`.
"""
function add_dissolved!(model, δ::AbstractVector)
    @static if !FOAM
        throw(ArgumentError("FOAM is false"))
    else
        model.initialized ||
            throw(ArgumentError("add_dissolved! runs after initialize!"))
        length(model.domains) == 1 ||
            throw(ArgumentError("add_dissolved! v1 supports a single domain"))
        domain = model.domains[1]
        N = length(domain.flags)
        length(δ) == N ||
            throw(ArgumentError("add_dissolved! got $(length(δ)) values for $N cells"))
        any(!iszero, δ) || return nothing
        flags = Array(domain.flags.data)
        c = Array(domain.c.data)
        ci = Array(domain.ci.data)
        CT = eltype(c)
        @inbounds for n in eachindex(δ)
            d = δ[n]
            iszero(d) && continue
            su = flags[n] & TYPE_SU
            (su == TYPE_F || su == TYPE_I) || continue
            dd = CT(d)
            c[n] += dd
            ci[f_index(n, 1, N)] += dd
        end
        copyto!(domain.c.data, c)
        copyto!(domain.ci.data, ci)
        return nothing
    end
end

# --- Flood fill ------------------------------------------------------------

function _face_touches_gas(flags, n::Int, Nx::Int, Ny::Int, Nz::Int)
    x, y, z = _cell_xyz(n, Nx, Ny)
    for (cx, cy, cz) in _FACE6
        j = src_index(x, y, z, cx, cy, cz, Nx, Ny, Nz)
        (flags[j] & TYPE_SU) == TYPE_G && return true
    end
    return false
end

# TYPE_G, or a TYPE_I shell that does not face gas and whose previous tag
# is not already on a gas cell. The second case is a punched sphere with
# no full cell; its punch id has to survive the fill. Shells that still
# touch gas are labeled later by _tag_interface!.
function _joins_component(flags, prev, gas_tags, n, Nx, Ny, Nz)
    su = flags[n] & TYPE_SU
    su == TYPE_G && return true
    su == TYPE_I || return false
    _face_touches_gas(flags, n, Nx, Ny, Nz) && return false
    t = prev[n]
    return t > 0 && t ∉ gas_tags
end

function _frozen_mask(foam::FoamHost)
    nb = length(foam.bubbles)
    mask = falses(nb)
    @inbounds for i in 1:nb
        row = foam.bubbles[i]
        row !== nothing && row.frozen && (mask[i] = true)
    end
    return mask
end

# `lock > 0` is a solidified bubble: the walk stays on that id.
# An unlocked walk (liquid gas, atmosphere) does not enter one.
function _frozen_link_ok(prev, frozen::BitVector, lock::Int32, j::Int)
    pj = prev[j]
    if lock > Int32(0)
        return pj == lock
    end
    return !(pj > 0 && pj <= length(frozen) && frozen[pj])
end

function _flood_gas!(foam::FoamHost, Nx::Int, Ny::Int, Nz::Int, frozen::BitVector)
    N = Nx * Ny * Nz
    flags = foam.flags
    prev = foam.tag_prev
    comp = foam.component
    q = foam.queue
    gas_tags = Set{Int32}()
    for n in 1:N
        (flags[n] & TYPE_SU) == TYPE_G || continue
        t = prev[n]
        t > 0 && push!(gas_tags, t)
    end
    fill!(comp, Int32(0))
    ncomp = 0
    for seed in 1:N
        comp[seed] != 0 && continue
        _joins_component(flags, prev, gas_tags, seed, Nx, Ny, Nz) || continue
        pt = prev[seed]
        lock = (pt > 0 && pt <= length(frozen) && frozen[pt]) ? pt : Int32(0)
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
                _joins_component(flags, prev, gas_tags, j, Nx, Ny, Nz) || continue
                _frozen_link_ok(prev, frozen, lock, j) || continue
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

# Connected overlaps are one transition. 1–1 keeps the id, ratio, and V_ref.
# A merge keeps the minimum old id; ratio is the V_ref-weighted average and
# V_ref is the sum, so Σ(ratio * V_ref) is unchanged. A split copies ratio
# and sets V_ref,child = V_ref * V_child / Σ V; the lowest component keeps
# the parent id. Overlap with tag -1 drops those rows. Orphan gas stays -1.
function _assign_components!(foam::FoamHost, parents, atm, ncomp::Int)
    comp_tag = fill(TAG_ATM, ncomp)
    doomed = Set{Int32}()
    for c in 1:ncomp
        atm[c] || continue
        for id in parents[c]
            row = _live_bubble(foam, id)
            # A solidified bubble that numerically touches tag −1 stays.
            row !== nothing && row.frozen && continue
            push!(doomed, id)
        end
    end

    comp_parents = [Int32[] for _ in 1:ncomp]
    id_comps = Dict{Int32,Vector{Int}}()
    for c in 1:ncomp
        atm[c] && continue
        for id in parents[c]
            id in doomed && continue
            row = _live_bubble(foam, id)
            # Frozen ids are not merged. Their cells are written back from tag_prev.
            row !== nothing && row.frozen && continue
            push!(comp_parents[c], id)
            push!(get!(id_comps, id, Int[]), c)
        end
    end

    seen = falses(ncomp)
    groups = Tuple{Vector{Int32},Float64,Float64}[]
    for c0 in 1:ncomp
        seen[c0] && continue
        isempty(comp_parents[c0]) && continue
        comps = Int[]
        olds = Int32[]
        old_seen = Set{Int32}()
        stack = [c0]
        seen[c0] = true
        while !isempty(stack)
            c = pop!(stack)
            push!(comps, c)
            for id in comp_parents[c]
                id in old_seen && continue
                push!(old_seen, id)
                push!(olds, id)
                for c2 in id_comps[id]
                    seen[c2] && continue
                    seen[c2] = true
                    push!(stack, c2)
                end
            end
        end
        sort!(comps)
        sort!(olds)
        if length(olds) == 1
            row = _live_bubble(foam, olds[1])
            ratio = row === nothing ? 0.0 : row.ratio
            sum_vr = row === nothing ? 0.0 : row.V_ref
        else
            sum_vr = 0.0
            sum_rvr = 0.0
            for id in olds
                row = _live_bubble(foam, id)
                row === nothing && continue
                sum_vr += row.V_ref
                sum_rvr += row.ratio * row.V_ref
            end
            if sum_vr > 0
                ratio = sum_rvr / sum_vr
            else
                row = _live_bubble(foam, olds[1])
                ratio = row === nothing ? 0.0 : row.ratio
            end
        end
        keep = olds[1]
        new_ids = Int32[]
        for (k, c) in enumerate(comps)
            cid = k == 1 ? keep : _alloc_bubble_id!(foam)
            comp_tag[c] = cid
            push!(new_ids, cid)
        end
        push!(groups, (new_ids, ratio, sum_vr))
    end
    return comp_tag, groups
end

function _tag_interface!(flags, tag, comp, prev, frozen, Nx, Ny, Nz)
    N = Nx * Ny * Nz
    for n in 1:N
        (flags[n] & TYPE_SU) == TYPE_I || continue
        # Orphan-shell cells already hold the component id.
        comp[n] != 0 && continue
        # A solidified shell stays on the bubble it already belonged to.
        pt = prev[n]
        if pt > 0 && pt <= length(frozen) && frozen[pt]
            tag[n] = pt
            continue
        end
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

function _commit_rows!(foam::FoamHost, groups, N::Int)
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
    for (new_ids, ratio, sum_vr) in groups
        live = Int32[]
        for id in new_ids
            i = Int(id)
            1 <= i <= nb && vol[i] > 0 && push!(live, id)
        end
        isempty(live) && continue
        sumV = 0.0
        for id in live
            sumV += vol[Int(id)]
        end
        nchild = length(live)
        for id in live
            i = Int(id)
            V = vol[i]
            vr = nchild == 1 ? sum_vr : sum_vr * V / sumV
            used[i] = true
            foam.bubbles[i] = Bubble(V, vr, ratio, false)
        end
    end
    for i in 1:nb
        used[i] && continue
        row = foam.bubbles[i]
        row === nothing && continue
        # Solidified gas still carries this id. Keep ratio and V_ref; V is the fill.
        if row.frozen && i <= length(vol) && vol[i] > 0
            foam.bubbles[i] = Bubble(vol[i], row.V_ref, row.ratio, true)
            continue
        end
        _release_bubble_id!(foam, i)
    end
    return nothing
end

function _retag!(foam::FoamHost, Nx::Int, Ny::Int, Nz::Int)
    N = Nx * Ny * Nz
    frozen = _frozen_mask(foam)
    prev = foam.tag_prev
    ncomp = _flood_gas!(foam, Nx, Ny, Nz, frozen)
    fill!(foam.tag, TAG_NONE)
    if ncomp == 0
        _tag_interface!(foam.flags, foam.tag, foam.component, prev, frozen, Nx, Ny, Nz)
        for i in 1:length(foam.bubbles)
            foam.bubbles[i] === nothing && continue
            _release_bubble_id!(foam, i)
        end
        return nothing
    end
    parents, atm = _collect_parents(foam, ncomp, N)
    comp_tag, groups = _assign_components!(foam, parents, atm, ncomp)
    comp = foam.component
    tag = foam.tag
    for n in 1:N
        c = Int(comp[n])
        c != 0 && (tag[n] = comp_tag[c])
        pt = prev[n]
        if pt > 0 && pt <= length(frozen) && frozen[pt]
            tag[n] = pt
        end
    end
    _tag_interface!(foam.flags, tag, comp, prev, frozen, Nx, Ny, Nz)
    _commit_rows!(foam, groups, N)
    return nothing
end

# surface_0 staged the Henry slot drop on flux. Spread it across the liquid
# rests and clear it, so the drop is not also given to the pore. Eq. 27
# then books the pore and is still in flux at the next host. A domain with
# no TYPE_F cell is a bare film: the staged drop stays on the pore.
function _park_henry_slot!(model, domain)
    flux = Array(domain.flux.data)
    total = sum(Float64, flux)
    total == 0.0 && return nothing
    flags = model.foam.flags
    N = length(flags)
    nF = 0
    @inbounds for f in flags
        (f & TYPE_SU) == TYPE_F && (nF += 1)
    end
    if nF == 0
        return nothing
    end
    share = total / nF
    ci = Array(domain.ci.data)
    CT = eltype(ci)
    @inbounds for n in 1:N
        (flags[n] & TYPE_SU) == TYPE_F || continue
        r = f_index(n, 1, N)
        ci[r] = CT(Float64(ci[r]) + share)
    end
    copyto!(domain.ci.data, ci)
    fill!(domain.flux.data, zero(eltype(domain.flux.data)))
    return nothing
end

# Eq. 27 has been added to flux and the Henry drop has already been parked.
# Spread that pore credit back off the liquid rests so the bath falls as
# the pores grow. The interface rest is left alone.
function _debit_dissolved!(model, domain)
    flux = Array(domain.flux.data)
    total = sum(Float64, flux)
    total == 0.0 && return nothing
    flags = model.foam.flags
    N = length(flags)
    nF = 0
    @inbounds for f in flags
        (f & TYPE_SU) == TYPE_F && (nF += 1)
    end
    nF == 0 && return nothing
    share = total / nF
    ci = Array(domain.ci.data)
    CT = eltype(ci)
    @inbounds for n in 1:N
        (flags[n] & TYPE_SU) == TYPE_F || continue
        r = f_index(n, 1, N)
        ci[r] = CT(Float64(ci[r]) - share)
    end
    copyto!(domain.ci.data, ci)
    return nothing
end

# Population flux is already in concentration units. Tag -1 has no row.
function _add_dissolved_inventory!(foam::FoamHost, flux, V_m, ρ_liquid)
    scale = Float64(V_m) * Float64(ρ_liquid)
    scale == 0.0 && return nothing
    nbuf = length(flux)
    @inbounds for i in eachindex(foam.bubbles)
        row = foam.bubbles[i]
        row === nothing && continue
        row.frozen && continue
        (1 <= i <= nbuf) || continue
        Δm = Float64(flux[i])
        Δm == 0.0 && continue
        Vref = row.V_ref
        Vref > 0 || continue
        foam.bubbles[i] = Bubble(row.V, row.V_ref, row.ratio + Δm * scale / Vref, row.frozen)
    end
    return nothing
end

# γ_b = 1 is a division. pow(x, 1) is not what this writes.
function _impose_bubble_density!(foam::FoamHost, γ_b)
    γ = Float64(γ_b)
    CType = eltype(foam.ρb)
    fill!(foam.ρb, one(CType))
    @inbounds for n in eachindex(foam.tag)
        t = Int(foam.tag[n])
        t > 0 || continue
        row = _live_bubble(foam, t)
        row === nothing && continue
        V = row.V
        V > 0 || continue
        vr_over_v = row.V_ref / V
        factor = γ == 1.0 ? vr_over_v : vr_over_v^γ
        foam.ρb[n] = CType(row.ratio * factor)
    end
    return nothing
end

# One-way. A bubble freezes when it has a shell and every TYPE_I cell of its
# tag is solid. fs = 0 (the default fill) does not freeze. No remelt.
function _freeze_shells!(foam::FoamHost, fs)
    nb = length(foam.bubbles)
    nb == 0 && return nothing
    n_iface = zeros(Int, nb)
    n_liquid = zeros(Int, nb)
    flags = foam.flags
    tag = foam.tag_prev
    for n in eachindex(flags)
        (flags[n] & TYPE_SU) == TYPE_I || continue
        t = Int(tag[n])
        1 <= t <= nb || continue
        foam.bubbles[t] === nothing && continue
        n_iface[t] += 1
        is_solid_fraction(fs[n]) || (n_liquid[t] += 1)
    end
    for i in 1:nb
        row = foam.bubbles[i]
        row === nothing && continue
        row.frozen && continue
        n_iface[i] == 0 && continue
        n_liquid[i] == 0 || continue
        foam.bubbles[i] = Bubble(row.V, row.V_ref, row.ratio, true)
    end
    return nothing
end

# A pore that shares a face with the headspace is one component and is dropped.
# The link is often not a direct face: one untagged gas cell sits between the
# pore and the atmosphere, the flood follows it, and the whole pore is lost.
# Mark atmosphere plus untagged gas that touches it, and turn each live pore
# cell that faces that set back into liquid. The finger is plugged and the
# pore keeps its id. A frozen pore is left alone.
function _seal_atmosphere!(foam::FoamHost, Nx::Int, Ny::Int, Nz::Int)
    N = Nx * Ny * Nz
    flags = foam.flags
    prev = foam.tag_prev
    ϕ = foam.ϕ
    frozen = _frozen_mask(foam)
    connected = falses(N)
    queue = Int[]
    for n in 1:N
        (flags[n] & TYPE_SU) == TYPE_G || continue
        prev[n] == TAG_ATM || continue
        connected[n] = true
        push!(queue, n)
    end
    qh = 1
    while qh <= length(queue)
        n = queue[qh]
        qh += 1
        x, y, z = _cell_xyz(n, Nx, Ny)
        for (cx, cy, cz) in _FACE6
            j = src_index(x, y, z, cx, cy, cz, Nx, Ny, Nz)
            connected[j] && continue
            (flags[j] & TYPE_SU) == TYPE_G || continue
            prev[j] == TAG_NONE || continue
            connected[j] = true
            push!(queue, j)
        end
    end
    contacts = Int[]
    for n in 1:N
        (flags[n] & TYPE_SU) == TYPE_G || continue
        t = prev[n]
        t > 0 || continue
        t <= length(frozen) && frozen[t] && continue
        x, y, z = _cell_xyz(n, Nx, Ny)
        touch = false
        for (cx, cy, cz) in _FACE6
            j = src_index(x, y, z, cx, cy, cz, Nx, Ny, Nz)
            if connected[j]
                touch = true
                break
            end
        end
        touch && push!(contacts, n)
    end
    CT = eltype(ϕ)
    for n in contacts
        flags[n] = (flags[n] & ~TYPE_SU) | TYPE_F
        ϕ[n] = one(CT)
        prev[n] = TAG_NONE
    end
    return contacts
end

# A plugged cell was gas. Its populations are not a liquid equilibrium, and the
# interface reads them on the next surface pass. Replace them with liquid at
# rest so the plug does not kick the film.
function _quiet_plugs!(model, domain, cells)
    isempty(cells) && return nothing
    N = length(domain.flags)
    fi = domain.fi.data
    ci = domain.ci.data
    ρ = domain.ρ.data
    u = domain.u.data
    mass = domain.mass.data
    CT = eltype(ρ)
    oneρ = one(CT)
    zeroρ = zero(CT)
    zc = zero(eltype(ci))
    for n in cells
        ρ[n] = oneρ
        u[n, 1] = zeroρ
        u[n, 2] = zeroρ
        u[n, 3] = zeroρ
        mass[n] = oneρ
        store_feq_local!(fi, n, oneρ, zeroρ, zeroρ, zeroρ, model.weights, model.velocities, N)
        ci[f_index(n, 1, N)] = zc
        for i in 2:7
            ci[f_index(n, i, N)] = zc
        end
    end
    return nothing
end

"""
Host section at the start of the substep. Reads post-`surface_3` flags
from the previous substep. Freezes any bubble whose shell is solid, adds
the binned dissolved flux to the liquid rows, then writes tags and the
imposed `ρb` before `surface_0`.
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
        @static if TEMPERATURE
            _freeze_shells!(foam, Array(domain.fs.data))
        end
        flux = Array(domain.flux.data)
        # Old rows, so a merge weights every parent's Δm and a split copies one ratio.
        # Frozen rows are skipped: the shell no longer exchanges dissolved gas.
        _add_dissolved_inventory!(foam, flux, domain.V_m, domain.ρ_liquid)
        plugs = _seal_atmosphere!(foam, Int(domain.Nx), Int(domain.Ny), Int(domain.Nz))
        _retag!(foam, Int(domain.Nx), Int(domain.Ny), Int(domain.Nz))
        fill!(domain.flux.data, zero(eltype(domain.flux.data)))
        _impose_bubble_density!(foam, domain.γ_b)
        copyto!(domain.tag.data, foam.tag)
        copyto!(domain.ρb.data, foam.ρb)
        if !isempty(plugs)
            copyto!(domain.flags.data, foam.flags)
            copyto!(domain.ϕ.data, foam.ϕ)
            _quiet_plugs!(model, domain, plugs)
        end
    end
    return nothing
end
