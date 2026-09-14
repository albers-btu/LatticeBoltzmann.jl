# Ballistic powder parcels. Sampled from a Gaussian at a nozzle plane,
# streamed along the jet, deposited into mp (or mass if τ_p=0) on TYPE_I/F.

mutable struct PowderJet{T<:AbstractFloat}
    enabled::Bool
    mdot::T              # captured mass rate [kg/s]
    w::T                 # 1/e² radius in the nozzle plane [cells]
    v::T                 # parcel speed [cells / step]
    x::T
    y::T
    z::T
    dx::T
    dy::T
    dz::T
    nparcels::Int        # spawned per LBM step
    nmax::Int
    px::Vector{T}
    py::Vector{T}
    pz::Vector{T}
    pvx::Vector{T}
    pvy::Vector{T}
    pvz::Vector{T}
    pm::Vector{T}        # lattice mass (units of ρ×cell)
    alive::Vector{Bool}
end

@inline function _jet_basis(dx::T, dy::T, dz::T) where {T}
    if abs(dz) < T(0.9)
        e1x, e1y, e1z = dy, -dx, zero(T)
    else
        e1x, e1y, e1z = zero(T), dz, -dy
    end
    n1 = sqrt(e1x*e1x + e1y*e1y + e1z*e1z)
    n1 = n1 > zero(T) ? n1 : one(T)
    e1x /= n1; e1y /= n1; e1z /= n1
    e2x = dy*e1z - dz*e1y
    e2y = dz*e1x - dx*e1z
    e2z = dx*e1y - dy*e1x
    n2 = sqrt(e2x*e2x + e2y*e2y + e2z*e2z)
    n2 = n2 > zero(T) ? n2 : one(T)
    return e1x, e1y, e1z, e2x/n2, e2y/n2, e2z/n2
end

function PowderJet{T}(;
    mdot = zero(T),
    w = one(T),
    v = one(T),
    x = one(T),
    y = one(T),
    z = one(T),
    dir = (zero(T), zero(T), -one(T)),
    nparcels = 16,
    nmax = 2048,
    enabled = true,
) where {T<:AbstractFloat}
    d = SVector{3,T}(T(dir[1]), T(dir[2]), T(dir[3]))
    nrm = sqrt(d[1]*d[1] + d[2]*d[2] + d[3]*d[3])
    nrm <= 0 && (d = SVector{3,T}(zero(T), zero(T), -one(T)); nrm = one(T))
    d = d / nrm
    nmax = max(Int(nparcels), Int(nmax))
    return PowderJet{T}(
        enabled, T(mdot), T(w), T(v), T(x), T(y), T(z), d[1], d[2], d[3],
        Int(nparcels), nmax,
        zeros(T, nmax), zeros(T, nmax), zeros(T, nmax),
        zeros(T, nmax), zeros(T, nmax), zeros(T, nmax),
        zeros(T, nmax), fill(false, nmax),
    )
end

function PowderJet(U::Units{T};
    mdot, w, v,
    x = one(T), y = one(T), z = one(T),
    dir = (0, 0, -1),
    nparcels = 16, nmax = 2048,
    enabled = true,
) where {T}
    md = mdot isa Quantity ? T(ustrip(u"kg/s", uconvert(u"kg/s", mdot))) : T(mdot)
    wl = w isa Quantity ? T(ustrip(u"m", w) / U.m) : T(w)
    vl = v isa Quantity ? T(ustrip(u"m/s", v) * U.s / U.m) : T(v)
    return PowderJet{T}(; mdot=md, w=wl, v=vl, x=T(x), y=T(y), z=T(z), dir=dir,
                        nparcels=nparcels, nmax=nmax, enabled=enabled)
end

function set_powder_jet_position!(J::PowderJet{T}, x, y, z=J.z) where {T}
    J.x = T(x); J.y = T(y); J.z = T(z)
    return J
end

function aim_powder_jet!(J::PowderJet{T}, tx, ty, tz) where {T}
    dx = T(tx) - J.x
    dy = T(ty) - J.y
    dz = T(tz) - J.z
    nrm = sqrt(dx*dx + dy*dy + dz*dz)
    nrm <= 0 && return J
    J.dx = dx/nrm; J.dy = dy/nrm; J.dz = dz/nrm
    return J
end

@inline function _parcel_slot(J::PowderJet)
    @inbounds for i in 1:J.nmax
        J.alive[i] || return i
    end
    return 0
end

function _spawn_parcels!(J::PowderJet{T}, m_each::T) where {T}
    m_each <= 0 && return nothing
    e1x, e1y, e1z, e2x, e2y, e2z = _jet_basis(J.dx, J.dy, J.dz)
    σ = T(0.5) * J.w
    @inbounds for _ in 1:J.nparcels
        i = _parcel_slot(J)
        i == 0 && break
        u1 = rand(T)
        u2 = rand(T)
        r = sqrt(-T(2) * log(max(u1, eps(T))))
        a = T(2π) * u2
        ox = σ * r * cos(a)
        oy = σ * r * sin(a)
        J.px[i] = J.x + ox*e1x + oy*e2x
        J.py[i] = J.y + ox*e1y + oy*e2y
        J.pz[i] = J.z + ox*e1z + oy*e2z
        J.pvx[i] = J.v * J.dx
        J.pvy[i] = J.v * J.dy
        J.pvz[i] = J.v * J.dz
        J.pm[i] = m_each
        J.alive[i] = true
    end
    return nothing
end

@inline function _deposit_parcel!(mp, mass, flags, n::Int, pmass, τ_p)
    if τ_p > 0
        @inbounds mp[n] += pmass
        return pmass
    else
        @inbounds begin
            su = flags[n] & TYPE_SU
            if su == TYPE_I || su == TYPE_F
                mass[n] += pmass
                return pmass
            end
        end
        return zero(pmass)
    end
end

# DDA along dir for at most dist_max cells. Hits I/F → deposit; S/OOB → die.
@inline function _walk_parcel!(
    mp, mass, flags, ϕ, τ_p,
    o0x::T, o0y::T, o0z::T, dirx::T, diry::T, dirz::T,
    dist_max::T, pmass::T, Nx::Int, Ny::Int, Nz::Int,
) where {T}
    ox, oy, oz = o0x, o0y, o0z
    traveled = zero(T)
    max_step = Nx + Ny + Nz + 8
    @inbounds for _ in 1:max_step
        remaining = dist_max - traveled
        remaining <= T(1e-6) && return ox, oy, oz, true, zero(T)
        ix = floor(Int, ox + T(0.5))
        iy = floor(Int, oy + T(0.5))
        iz = floor(Int, oz + T(0.5))
        if ix < 1 || ix > Nx || iy < 1 || iy > Ny || iz < 1 || iz > Nz
            return ox, oy, oz, false, zero(T)
        end
        n = ix + (iy - 1) * Nx + (iz - 1) * Nx * Ny
        fl = flags[n]
        su = fl & TYPE_SU
        if (fl & TYPE_BO) == TYPE_S
            return ox, oy, oz, false, zero(T)
        elseif su == TYPE_I
            ϕ0 = T(ϕ[n])
            phij = gather_phi_d3q27(ϕ, ϕ0, ix - 1, iy - 1, iz - 1, Nx, Ny, Nz)
            hit, t, _nϕ = plic_hit(ϕ0, phij, ox, oy, oz, dirx, diry, dirz,
                                   T(ix), T(iy), T(iz))
            if hit && t > zero(T) && t <= remaining + T(0.5)
                dm = _deposit_parcel!(mp, mass, flags, n, pmass, τ_p)
                return ox, oy, oz, false, dm
            end
        elseif su == TYPE_F
            dm = _deposit_parcel!(mp, mass, flags, n, pmass, τ_p)
            return ox, oy, oz, false, dm
        end
        tMaxX = dirx > 0 ? (T(ix) + T(0.5) - ox) / dirx :
                dirx < 0 ? (ox - (T(ix) - T(0.5))) / (-dirx) : T(Inf)
        tMaxY = diry > 0 ? (T(iy) + T(0.5) - oy) / diry :
                diry < 0 ? (oy - (T(iy) - T(0.5))) / (-diry) : T(Inf)
        tMaxZ = dirz > 0 ? (T(iz) + T(0.5) - oz) / dirz :
                dirz < 0 ? (oz - (T(iz) - T(0.5))) / (-dirz) : T(Inf)
        tstep = min(max(tMaxX, zero(T)), max(tMaxY, zero(T)), max(tMaxZ, zero(T)))
        tstep = min(tstep, remaining) + T(1e-5)
        ox += tstep * dirx
        oy += tstep * diry
        oz += tstep * dirz
        traveled += tstep
    end
    return ox, oy, oz, false, zero(T)
end

function advance_powder_jet!(model, domain)
    J = model.powder_jet
    (J === nothing || (!J.enabled && !any(J.alive))) && return nothing
    T = eltype(J.px)
    U = model.units
    τ_p = domain.τ_p
    Nx, Ny, Nz = Int(domain.Nx), Int(domain.Ny), Int(domain.Nz)
    flags = Array(domain.flags.data)
    ϕ = Array(domain.ϕ.data)
    mp = Array(domain.mp.data)
    mass = Array(domain.mass.data)
    if J.enabled && J.mdot > 0 && J.nparcels > 0
        m_step = T(J.mdot * U.s / U.kg)
        _spawn_parcels!(J, m_step / T(J.nparcels))
    end
    invv = J.v > 0 ? one(T) / J.v : one(T)
    captured = zero(T)
    @inbounds for i in 1:J.nmax
        J.alive[i] || continue
        dirx, diry, dirz = J.pvx[i]*invv, J.pvy[i]*invv, J.pvz[i]*invv
        nd = sqrt(dirx*dirx + diry*diry + dirz*dirz)
        if nd <= 0
            J.alive[i] = false
            continue
        end
        dirx /= nd; diry /= nd; dirz /= nd
        ox, oy, oz, live, dm = _walk_parcel!(
            mp, mass, flags, ϕ, τ_p,
            J.px[i], J.py[i], J.pz[i], dirx, diry, dirz,
            J.v, J.pm[i], Nx, Ny, Nz,
        )
        J.px[i] = ox; J.py[i] = oy; J.pz[i] = oz
        J.alive[i] = live
        captured += dm
    end
    @static if TEMPERATURE
        domain.E_powder += captured * domain.T_p
    end
    copyto!(domain.mp.data, mp)
    τ_p > 0 || copyto!(domain.mass.data, mass)
    return nothing
end
