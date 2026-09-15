# Enclosed TYPE_G bubbles: p V = n T_gas (lattice). Atmosphere is TYPE_G
# connected to the top lid; those cells keep p_atm (ρ_gas = 1 at σ = 0).
# Reconstruction uses ρ_gas = 3 p − 6 σ κ with p = n T / V on enclosed
# components. Coalescence/split conserves n by carrying n/V on gas cells.
#
# Off by default (p_gas stays p_atm). Körner / LBfoam closed-bubble model.

const P_ATM_LAT = Float32(1) / Float32(3)

# Homogeneous nucleation in supersaturated liquid (LBfoam Poisson-disk).
# Off unless `enabled`. A nucleus is a (2R+1)³ TYPE_G cube; the outer shell
# becomes TYPE_I, leaving a G core. New bubbles get n = n_over × p_eq V.
mutable struct Nucleation{T<:AbstractFloat}
    enabled::Bool
    every::Int
    d_min::T             # min centre–centre spacing (cells)
    R::Int               # G-cube half-width; 1 → 3³
    c_star::T            # nucleate if c > c_star; 0 → no concentration gate
    p_cell::T            # keep candidate with this probability
    n_max::Int           # max nuclei per attempt
    n_over::T            # n / (p_atm V); 1 = atmospheric, >1 grows
    n_total_max::Int
    n_planted::Int
end

# LBfoam Π = k_Π (d_max − d) between two different bubble interfaces.
# Subtracted from p before ρ_gas = 3p − 6σκ. k_Π = 0 → off.
# Paper: d_max = 4, k_Π ≈ 0.08 holds a lamella; 0 coalesces on contact.

function Nucleation{T}(;
    enabled = true,
    every = 20,
    d_min = 8,
    R = 1,
    c_star = zero(T),
    p_cell = one(T),
    n_max = 8,
    n_over = T(1.2),
    n_total_max = 256,
) where {T<:AbstractFloat}
    R < 1 && throw(ArgumentError("R must be ≥ 1"))
    every < 1 && throw(ArgumentError("every must be ≥ 1"))
    return Nucleation{T}(
        enabled, Int(every), T(d_min), Int(R), T(c_star), T(p_cell),
        Int(n_max), T(n_over), Int(n_total_max), 0,
    )
end
Nucleation(; kwargs...) = Nucleation{Float32}(; kwargs...)

mutable struct BubbleTracker{T<:AbstractFloat}
    enabled::Bool
    every::Int
    T_gas::T
    p_atm::T
    n_mol::Vector{T}
    vol::Vector{T}       # G-cell count (n transport and EOS volume)
    p::Vector{T}
    label::Vector{Int32} # per-cell enclosed id; 0 = not a bubble
    nb::Int
    nucleation::Any
    k_Π::T
    d_max::T
end

function BubbleTracker{T}(;
    enabled = true,
    every = 1,
    T_gas = one(T),
    p_atm = T(1) / T(3),
    nucleation = nothing,
    k_Π = zero(T),
    d_max = T(4),
) where {T<:AbstractFloat}
    every < 1 && throw(ArgumentError("every must be ≥ 1"))
    return BubbleTracker{T}(
        enabled, Int(every), T(T_gas), T(p_atm),
        T[], T[], T[], Int32[], 0, nucleation, T(k_Π), T(d_max),
    )
end
BubbleTracker(; kwargs...) = BubbleTracker{Float32}(; kwargs...)

young_laplace_p(σ, R, p_atm=P_ATM_LAT) = p_atm + 2 * σ / max(R, eps(typeof(σ)))
bubble_radius(V) = (3 * max(V, 0) / (4 * π))^(1 / 3)

# Quasi-static EP with constant p (σ=0 or R ≫ 2σ/p). c is T-like, c_s = k_H p.
# R²(t) = R0² + 2 D (c∞/k_H − p) / p · t
function epstein_plesset_R2(R0, t, D, c∞, k_H, p=P_ATM_LAT)
    cs = k_H * p
    return R0 * R0 + 2 * D * (c∞ - cs) / p * t
end

@inline function _uf_find!(parent, i::Int)
    r = i
    @inbounds while parent[r] != r
        r = parent[r]
    end
    @inbounds while parent[i] != r
        j = parent[i]
        parent[i] = r
        i = j
    end
    return r
end

@inline function _uf_union!(parent, a::Int, b::Int)
    ra, rb = _uf_find!(parent, a), _uf_find!(parent, b)
    ra != rb && (parent[rb] = ra)
    return ra
end

# Face-connected TYPE_G components. Returns (root_of_cell, ncomp, which, is_atm).
# `which[k]` is the 1-based component id of union-find root k (0 if unused).
function label_gas_components(flags, Nx::Int, Ny::Int, Nz::Int)
    N = Nx * Ny * Nz
    parent = zeros(Int, N)
    @inbounds for n in 1:N
        (flags[n] & TYPE_SU) == TYPE_G || continue
        parent[n] = n
    end
    @inbounds for n in 1:N
        parent[n] == 0 && continue
        n0 = n - 1
        x = n0 % Nx
        y = (n0 ÷ Nx) % Ny
        z = n0 ÷ (Nx * Ny)
        # Face-connected (LBfoam flood-fill). 26-connect merges through corners
        # and eats one-cell films before disjoining pressure can act.
        for (cx, cy, cz) in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))
            j = src_index(x, y, z, cx, cy, cz, Nx, Ny, Nz)
            parent[j] == 0 && continue
            _uf_union!(parent, n, j)
        end
    end
    which = zeros(Int, N)
    ncomp = 0
    @inbounds for n in 1:N
        parent[n] == 0 && continue
        r = _uf_find!(parent, n)
        if which[r] == 0
            ncomp += 1
            which[r] = ncomp
        end
    end
    is_atm = fill(false, ncomp)
    @inbounds for n in 1:N
        parent[n] == 0 && continue
        z = (n - 1) ÷ (Nx * Ny) + 1
        z < Nz - 1 && continue
        is_atm[which[_uf_find!(parent, n)]] = true
    end
    return parent, ncomp, which, is_atm
end

function _component_cells(parent, which, ncomp::Int)
    cells = [Int[] for _ in 1:ncomp]
    @inbounds for n in eachindex(parent)
        parent[n] == 0 && continue
        push!(cells[which[_uf_find!(parent, n)]], n)
    end
    return cells
end

function update_bubbles!(B::BubbleTracker{T}, flags, ϕ, p_gas, bid, Nx::Int, Ny::Int, Nz::Int) where {T}
    B.enabled || return B
    N = Nx * Ny * Nz
    length(B.label) == N || (B.label = zeros(Int32, N))
    old_label = copy(B.label)
    old_n = copy(B.n_mol)
    fill!(B.label, zero(Int32))

    parent, ncomp, which, is_atm = label_gas_components(flags, Nx, Ny, Nz)
    if ncomp == 0
        B.nb = 0
        empty!(B.n_mol); empty!(B.vol); empty!(B.p)
        fill!(p_gas, B.p_atm)
        fill!(bid, zero(eltype(bid)))
        return B
    end
    cells = _component_cells(parent, which, ncomp)

    bub_ids = Int[]
    @inbounds for c in 1:ncomp
        is_atm[c] || push!(bub_ids, c)
    end
    nb = length(bub_ids)
    n_mol = zeros(T, nb)
    volg = zeros(T, nb)
    old_in_new = [Int[] for _ in 1:nb]
    @inbounds for (k, c) in enumerate(bub_ids)
        gs = cells[c]
        volg[k] = T(length(gs))
        seen = Set{Int}()
        for n in gs
            B.label[n] = Int32(k)
            ol = Int(old_label[n])
            if 1 <= ol <= length(old_n) && ol ∉ seen
                push!(old_in_new[k], ol)
                push!(seen, ol)
            end
        end
    end
    nold = length(old_n)
    new_of_old = [Int[] for _ in 1:nold]
    @inbounds for k in 1:nb
        for ol in old_in_new[k]
            push!(new_of_old[ol], k)
        end
    end
    @inbounds for ol in 1:nold
        dest = new_of_old[ol]
        isempty(dest) && continue
        Vsum = zero(T)
        for k in dest
            Vsum += volg[k]
        end
        Vsum <= zero(T) && continue
        for k in dest
            n_mol[k] += old_n[ol] * volg[k] / Vsum
        end
    end

    p_atm = B.p_atm
    Tg = B.T_gas
    p = zeros(T, nb)
    @inbounds for k in 1:nb
        Vp = max(volg[k], T(1e-6))
        if n_mol[k] <= zero(T)
            n_mol[k] = p_atm * Vp / Tg
        end
        pk = n_mol[k] * Tg / Vp
        p[k] = clamp(pk, p_atm / T(20), T(20) * p_atm)
    end

    B.n_mol = n_mol
    B.vol = volg
    B.p = p
    B.nb = nb
    _write_p_gas!(B, flags, ϕ, p_gas, bid, Nx, Ny, Nz)
    return B
end

function _write_p_gas!(B::BubbleTracker{T}, flags, ϕ, p_gas, bid, Nx::Int, Ny::Int, Nz::Int) where {T}
    N = Nx * Ny * Nz
    pg = fill(B.p_atm, N)
    bd = zeros(T, N)
    @inbounds for n in 1:N
        k = Int(B.label[n])
        if 1 <= k <= B.nb
            pg[n] = B.p[k]
            bd[n] = T(k)
        end
    end
    @inbounds for n in 1:N
        (flags[n] & TYPE_SU) == TYPE_I || continue
        n0 = n - 1
        x = n0 % Nx
        y = (n0 ÷ Nx) % Ny
        z = n0 ÷ (Nx * Ny)
        pmax = zero(T)
        id = 0
        hit = false
        many = false
        for cz in -1:1, cy in -1:1, cx in -1:1
            (cx == 0 && cy == 0 && cz == 0) && continue
            j = src_index(x, y, z, cx, cy, cz, Nx, Ny, Nz)
            k = Int(B.label[j])
            if 1 <= k <= B.nb
                if !hit
                    id = k
                    pmax = B.p[k]
                    hit = true
                elseif k != id
                    many = true
                    pmax = max(pmax, B.p[k])
                end
            end
        end
        if hit
            pg[n] = pmax
            bd[n] = T(id)
        end
    end
    if B.k_Π > zero(T)
        _apply_disjoining!(B, flags, ϕ, pg, bd, Nx, Ny, Nz)
    end
    copyto!(p_gas, pg)
    copyto!(bid, bd)
    return nothing
end

# Distance to another bubble's I/G in the liquid half-space (along ∇ϕ).
# LBfoam uses a normal ray + PLIC; for d_max ≲ 4 a forward stencil is equivalent
# and does not miss thin films.
function _film_distance(flags, bid, ϕ, Nx, Ny, Nz, n, my_id, d_max::T) where {T}
    n0 = n - 1
    x = n0 % Nx
    y = (n0 ÷ Nx) % Ny
    z = n0 ÷ (Nx * Ny)
    r = max(1, ceil(Int, d_max))
    dmin = T(-1)
    @inbounds for dz in (-r):r, dy in (-r):r, dx in (-r):r
        (dx == 0 && dy == 0 && dz == 0) && continue
        dist = sqrt(T(dx * dx + dy * dy + dz * dz))
        dist > d_max && continue
        ix, iy, iz = x + 1 + dx, y + 1 + dy, z + 1 + dz
        (1 <= ix <= Nx && 1 <= iy <= Ny && 1 <= iz <= Nz) || continue
        j = _cell_n(ix, iy, iz, Nx, Ny)
        fl = flags[j]
        (fl & TYPE_S) != 0x00 && continue
        su = fl & TYPE_SU
        idj = Int(round(bid[j]))
        ((su == TYPE_I || su == TYPE_G) && idj > 0 && idj != my_id) || continue
        dmin = dmin < zero(T) ? dist : min(dmin, dist)
    end
    return dmin
end

function _apply_disjoining!(B::BubbleTracker{T}, flags, ϕ, pg, bd, Nx, Ny, Nz) where {T}
    kΠ = B.k_Π
    dmax = B.d_max
    kΠ <= zero(T) && return nothing
    N = Nx * Ny * Nz
    @inbounds for n in 1:N
        (flags[n] & TYPE_SU) == TYPE_I || continue
        my_id = Int(round(bd[n]))
        if my_id < 1 || my_id > B.nb
            n0 = n - 1
            x = n0 % Nx
            y = (n0 ÷ Nx) % Ny
            z = n0 ÷ (Nx * Ny)
            for cz in -1:1, cy in -1:1, cx in -1:1
                (cx == 0 && cy == 0 && cz == 0) && continue
                j = src_index(x, y, z, cx, cy, cz, Nx, Ny, Nz)
                k = Int(B.label[j])
                k < 1 && (k = Int(round(bd[j])))
                if 1 <= k <= B.nb
                    my_id = k
                    break
                end
            end
        end
        (1 <= my_id <= B.nb) || continue
        d = _film_distance(flags, bd, ϕ, Nx, Ny, Nz, n, my_id, dmax)
        d < zero(T) && continue
        Π = kΠ * max(dmax - d, zero(T))
        pg[n] = max(pg[n] - Π, B.p_atm / T(20))
    end
    return nothing
end

function bubble_records(B::BubbleTracker{T}) where {T}
    recs = NamedTuple{(:id, :n, :V, :p, :R), Tuple{Int, T, T, T, T}}[]
    @inbounds for k in 1:B.nb
        V = B.vol[k]
        push!(recs, (id=k, n=B.n_mol[k], V=V, p=B.p[k], R=T(bubble_radius(Float64(V)))))
    end
    return recs
end

function set_bubble_n!(B::BubbleTracker{T}, id::Integer, n, flags, ϕ, p_gas, bid, Nx, Ny, Nz) where {T}
    (1 <= id <= B.nb) || throw(ArgumentError("bubble id $id is not in 1:$(B.nb)"))
    B.n_mol[id] = T(n)
    Vp = max(B.vol[id], T(1e-6))
    B.p[id] = clamp(B.n_mol[id] * B.T_gas / Vp, B.p_atm / T(20), T(20) * B.p_atm)
    _write_p_gas!(B, flags, ϕ, p_gas, bid, Nx, Ny, Nz)
    return B
end

function apply_dissolved_flux!(B::BubbleTracker{T}, nflux, bid, flags, ϕ, p_gas, Nx, Ny, Nz) where {T}
    B.nb == 0 && return B
    nf = Array(nflux)
    bd = Array(bid)
    acc = zeros(T, B.nb)
    @inbounds for i in eachindex(nf)
        k = round(Int, bd[i])
        1 <= k <= B.nb || continue
        acc[k] += T(nf[i])
    end
    @inbounds for k in 1:B.nb
        B.n_mol[k] = max(B.n_mol[k] + acc[k], zero(T))
        Vp = max(B.vol[k], T(1e-6))
        B.p[k] = clamp(B.n_mol[k] * B.T_gas / Vp, B.p_atm / T(20), T(20) * B.p_atm)
    end
    fill!(nflux, zero(eltype(nflux)))
    _write_p_gas!(B, flags, ϕ, p_gas, bid, Nx, Ny, Nz)
    return B
end

function update_bubbles!(model::Model, domain::Domain)
    B = model.bubbles
    B isa BubbleTracker || return nothing
    B.enabled || return nothing
    every = B.every
    t = Int(domain.t)
    (every > 1 && t % every != 0 && t != 0) && return nothing
    flags = Array(domain.flags.data)
    ϕA = Array(domain.ϕ.data)
    Nx, Ny, Nz = Int(domain.Nx), Int(domain.Ny), Int(domain.Nz)
    update_bubbles!(B, flags, ϕA, domain.p_gas.data, domain.bid.data, Nx, Ny, Nz)
    return B
end

function update_bubbles!(model::Model)
    for domain in model.domains
        update_bubbles!(model, domain)
    end
    return model.bubbles
end

function bubble_records(model::Model)
    model.bubbles isa BubbleTracker || return NamedTuple[]
    return bubble_records(model.bubbles)
end

# LBfoam-bucket diagnostics: enclosed porosity, free-surface height, freeze.
function foam_metrics(model::Model)
    domain = model.domains[1]
    flags = Array(domain.flags.data)
    ϕA = Array(domain.ϕ.data)
    fsA = Array(domain.fs.data)
    Nx, Ny, Nz = Int(domain.Nx), Int(domain.Ny), Int(domain.Nz)
    nF = 0
    nI = 0
    nG = 0
    n_frozen = 0
    Vmet = 0.0
    hsum = 0.0
    hcnt = 0
    @inbounds for z in 1:Nz, y in 1:Ny, x in 1:Nx
        n = x + (y - 1) * Nx + (z - 1) * Nx * Ny
        su = flags[n] & TYPE_SU
        if su == TYPE_F
            nF += 1
            Vmet += Float64(ϕA[n])
            is_solid_fraction(fsA[n]) && (n_frozen += 1)
        elseif su == TYPE_I
            nI += 1
            Vmet += Float64(ϕA[n])
            is_solid_fraction(fsA[n]) && (n_frozen += 1)
        elseif su == TYPE_G
            nG += 1
        end
    end
    @inbounds for y in 2:(Ny - 1), x in 2:(Nx - 1)
        ztop = 0.0
        for z in (Nz - 1):-1:2
            n = x + (y - 1) * Nx + (z - 1) * Nx * Ny
            su = flags[n] & TYPE_SU
            if su == TYPE_I || (su == TYPE_F && ϕA[n] > 0.05f0)
                ztop = Float64(z - 1) + Float64(ϕA[n])
                break
            end
        end
        if ztop > 0
            hsum += ztop
            hcnt += 1
        end
    end
    B = model.bubbles
    recs = B isa BubbleTracker ? bubble_records(B) : NamedTuple[]
    Vbub = B isa BubbleTracker ? sum(Float64, B.vol; init=0.0) : 0.0
    n_gas = B isa BubbleTracker ? sum(Float64, B.n_mol; init=0.0) : 0.0
    por = (Vmet + Vbub) > 0 ? Vbub / (Vmet + Vbub) : 0.0
    inv = agent_inventory(domain)
    return (;
        nF, nI, nG, n_frozen,
        nb = length(recs),
        V_metal = Vmet,
        V_bubble = Vbub,
        porosity = por,
        fill_z = hcnt > 0 ? hsum / hcnt : 0.0,
        n_gas,
        n_planted = (B isa BubbleTracker && B.nucleation isa Nucleation) ?
            B.nucleation.n_planted : 0,
        a = inv.a, a_res = inv.res, dissolved = inv.dissolved,
        agent_total = inv.total,
    )
end

function set_bubble_n!(model::Model, id::Integer, n)
    B = model.bubbles
    B isa BubbleTracker || throw(ArgumentError("model.bubbles is not a BubbleTracker"))
    domain = model.domains[1]
    flags = Array(domain.flags.data)
    ϕA = Array(domain.ϕ.data)
    Nx, Ny, Nz = Int(domain.Nx), Int(domain.Ny), Int(domain.Nz)
    set_bubble_n!(B, id, n, flags, ϕA, domain.p_gas.data, domain.bid.data, Nx, Ny, Nz)
    return B
end

function initialize_dissolved!(model::Model, domain::Domain)
    return nothing
end

function advance_blowing_agent!(model::Model, domain::Domain)
    domain.k_a > 0 || return nothing
    N = get_N(domain)
    blowing_agent_kernel!(model.backend, model.workgroup)(
        domain.a.data, domain.a_res.data, domain.c.data,
        domain.flags.data, domain.ϕ.data, domain.fs.data, domain.T.data,
        domain.k_a, domain.E_a, domain.Y_a, domain.k_H, domain.a_fs_max;
        ndrange = N)
    KernelAbstractions.synchronize(model.backend)
    return nothing
end

function advance_dissolved_gas!(model::Model, domain::Domain, _t_odd::Bool)
    domain.ω_c > 0 || return nothing
    N = get_N(domain)
    Nx, Ny, Nz = Int(domain.Nx), Int(domain.Ny), Int(domain.Nz)
    D = dissolved_D(domain)
    dissolved_diffuse_kernel!(model.backend, model.workgroup)(
        domain.ci.data, domain.c.data, domain.flags.data, domain.u.data,
        D, Nx, Ny, Nz; ndrange = N)
    dissolved_henry_kernel!(model.backend, model.workgroup)(
        domain.c.data, domain.nflux.data, domain.ci.data, domain.flags.data,
        domain.ϕ.data, domain.p_gas.data, domain.k_H; ndrange = N)
    KernelAbstractions.synchronize(model.backend)
    B = model.bubbles
    if B isa BubbleTracker && B.enabled && domain.k_H > 0
        flags = Array(domain.flags.data)
        ϕA = Array(domain.ϕ.data)
        apply_dissolved_flux!(B, domain.nflux.data, domain.bid.data, flags, ϕA,
                              domain.p_gas.data, Nx, Ny, Nz)
    end
    return nothing
end

@inline function _cell_n(x::Int, y::Int, z::Int, Nx::Int, Ny::Int)
    return x + (y - 1) * Nx + (z - 1) * Nx * Ny
end

function _cube_is_fluid(flags, fs, Nx, Ny, Nz, cx, cy, cz, R)
    @inbounds for dz in (-R):R, dy in (-R):R, dx in (-R):R
        x, y, z = cx + dx, cy + dy, cz + dz
        (1 < x < Nx && 1 < y < Ny && 1 < z < Nz) || return false
        n = _cell_n(x, y, z, Nx, Ny)
        (flags[n] & TYPE_SU) == TYPE_F || return false
        is_solid_fraction(fs[n]) && return false
    end
    return true
end

# Only enclosed bubbles (and nuclei planted this pass) occupy space.
# Atmosphere TYPE_G and the free-surface TYPE_I must not push nuclei to the
# bottom of the pad. Walls do occupy, so a liquid pad cannot nucleate on the floor.
function _too_close(flags, parent, which, is_atm, bid, Nx, Ny, Nz, cx, cy, cz, d_min)
    r = max(1, ceil(Int, d_min))
    d2 = d_min * d_min
    @inbounds for dz in (-r):r, dy in (-r):r, dx in (-r):r
        dx * dx + dy * dy + dz * dz > d2 && continue
        x, y, z = cx + dx, cy + dy, cz + dz
        (1 <= x <= Nx && 1 <= y <= Ny && 1 <= z <= Nz) || continue
        n = _cell_n(x, y, z, Nx, Ny)
        fl = flags[n]
        (fl & TYPE_BO) == TYPE_S && return true
        su = fl & TYPE_SU
        if su == TYPE_G
            if parent[n] != 0
                r0 = _uf_find!(parent, n)
                is_atm[which[r0]] && continue
            end
            return true
        elseif su == TYPE_I
            id = bid === nothing ? 0 : Int(round(bid[n]))
            id > 0 && return true
        end
    end
    return false
end

function _plant_nucleus!(
    flags, ϕ, mass, ρ, u, fs, fi, gi, Tfield,
    w, vel, N, Nx, Ny, Nz, cx, cy, cz, R, t_odd, ::Type{CType},
) where {CType}
    cells = Int[]
    @inbounds for dz in (-R):R, dy in (-R):R, dx in (-R):R
        x, y, z = cx + dx, cy + dy, cz + dz
        n = _cell_n(x, y, z, Nx, Ny)
        flags[n] = TYPE_G
        ϕ[n] = zero(CType)
        mass[n] = zero(CType)
        u[n, 1] = zero(CType); u[n, 2] = zero(CType); u[n, 3] = zero(CType)
        fs[n] = zero(CType)
        push!(cells, n)
    end
    @inbounds for n in cells
        n0 = n - 1
        x = n0 % Nx
        y = (n0 ÷ Nx) % Ny
        z = n0 ÷ (Nx * Ny)
        toI = false
        for i in 2:length(vel)
            j = src_index(x, y, z, vel[i][1], vel[i][2], vel[i][3], Nx, Ny, Nz)
            (flags[j] & TYPE_SU) == TYPE_F && (toI = true)
        end
        if toI
            flags[n] = TYPE_I
            ϕ[n] = CType(0.5)
            mass[n] = CType(0.5) * ρ[n]
        end
        store_feq!(fi, n, x, y, z, ρ[n], zero(CType), zero(CType), zero(CType),
                   w, vel, N, Nx, Ny, Nz, t_odd)
        store_geq!(gi, n, x, y, z, Tfield[n], zero(CType), zero(CType), zero(CType),
                   N, Nx, Ny, Nz, t_odd, CType)
    end
    return _cell_n(cx, cy, cz, Nx, Ny)
end

# Returns linear indices of planted G cores.
function nucleate_bubbles!(
    Nuc::Nucleation{T}, flags, c, ϕ, mass, ρ, u, fs, fi, gi, Tfield,
    w, vel, N, Nx, Ny, Nz, t_odd, nb_now::Int, ::Type{CType};
    bid = nothing,
) where {T, CType}
    Nuc.enabled || return Int[]
    nb_now >= Nuc.n_total_max && return Int[]
    R = Nuc.R
    # One liquid cell between the G-cube and the wall so outer G see TYPE_F
    # (not TYPE_S) and convert to TYPE_I.
    lo, hi_x, hi_y, hi_z = 3 + R, Nx - 2 - R, Ny - 2 - R, Nz - 2 - R
    hi_x < lo && return Int[]
    parent, ncomp, which, is_atm = label_gas_components(flags, Nx, Ny, Nz)
    cands = Tuple{Int,Int,Int}[]
    @inbounds for z in lo:hi_z, y in lo:hi_y, x in lo:hi_x
        n = _cell_n(x, y, z, Nx, Ny)
        (flags[n] & TYPE_SU) == TYPE_F || continue
        is_solid_fraction(fs[n]) && continue
        Nuc.c_star > 0 && c[n] <= Nuc.c_star && continue
        Nuc.p_cell < 1 && rand(T) > Nuc.p_cell && continue
        # R+1 liquid shell: a 3³ nucleus in a 3-cell film opens the free
        # surface and the laser keyhole blows. Homogeneous nucleation needs
        # bulk liquid around the embryo.
        _cube_is_fluid(flags, fs, Nx, Ny, Nz, x, y, z, R + 1) || continue
        _too_close(flags, parent, which, is_atm, bid, Nx, Ny, Nz, x, y, z, Nuc.d_min) && continue
        push!(cands, (x, y, z))
    end
    isempty(cands) && return Int[]
    for i in length(cands):-1:2
        j = rand(1:i)
        cands[i], cands[j] = cands[j], cands[i]
    end
    planted = Int[]
    ncap = min(Nuc.n_max, Nuc.n_total_max - nb_now)
    @inbounds for (x, y, z) in cands
        length(planted) >= ncap && break
        _too_close(flags, parent, which, is_atm, bid, Nx, Ny, Nz, x, y, z, Nuc.d_min) && continue
        _cube_is_fluid(flags, fs, Nx, Ny, Nz, x, y, z, R + 1) || continue
        core = _plant_nucleus!(
            flags, ϕ, mass, ρ, u, fs, fi, gi, Tfield,
            w, vel, N, Nx, Ny, Nz, x, y, z, R, t_odd, CType)
        push!(planted, core)
        Nuc.n_planted += 1
    end
    return planted
end

function nucleate_bubbles!(model::Model, domain::Domain; force::Bool=false)
    B = model.bubbles
    B isa BubbleTracker || return Int[]
    Nuc = B.nucleation
    Nuc isa Nucleation || return Int[]
    Nuc.enabled || return Int[]
    t = Int(domain.t)
    (!force && Nuc.every > 1 && t % Nuc.every != 0 && t != 0) && return Int[]
    Nx, Ny, Nz = Int(domain.Nx), Int(domain.Ny), Int(domain.Nz)
    Nd = Int(domain.N)
    flags = Array(domain.flags.data)
    cA = Array(domain.c.data)
    ϕA = Array(domain.ϕ.data)
    massA = Array(domain.mass.data)
    ρA = Array(domain.ρ.data)
    uA = Array(domain.u.data)
    fsA = Array(domain.fs.data)
    fiA = Array(domain.fi.data)
    giA = Array(domain.gi.data)
    TA = Array(domain.T.data)
    t_odd = Val(isodd(t))
    CType = eltype(ρA)
    cores = nucleate_bubbles!(
        Nuc, flags, cA, ϕA, massA, ρA, uA, fsA, fiA, giA, TA,
        model.weights, model.velocities, Nd, Nx, Ny, Nz, t_odd, B.nb, CType;
        bid = Array(domain.bid.data))
    isempty(cores) && return cores
    copyto!(domain.flags.data, flags)
    copyto!(domain.ϕ.data, ϕA)
    copyto!(domain.mass.data, massA)
    copyto!(domain.u.data, uA)
    copyto!(domain.fs.data, fsA)
    copyto!(domain.fi.data, fiA)
    copyto!(domain.gi.data, giA)
    return cores
end

function _seed_new_nuclei!(B::BubbleTracker{T}, cores, flags, ϕ, p_gas, bid, σ, Nx, Ny, Nz) where {T}
    (B.nucleation isa Nucleation && !isempty(cores)) || return B
    n_over = B.nucleation.n_over
    @inbounds for n in cores
        k = Int(B.label[n])
        1 <= k <= B.nb || continue
        Rnuc = T(bubble_radius(Float64(B.vol[k])))
        peq = young_laplace_p(T(σ), max(Rnuc, T(0.5)), B.p_atm)
        set_bubble_n!(B, k, n_over * peq * B.vol[k], flags, ϕ, p_gas, bid, Nx, Ny, Nz)
    end
    return B
end

