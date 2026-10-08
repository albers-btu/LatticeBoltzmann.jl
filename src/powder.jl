# Powder parcels, sampled from a Gaussian and streamed along the jet.
# Heated by the beam in flight. Cold solid landings go into mp; molten
# parcels and landings in liquid join mass.
# model.powder_jet is one PowderJet or a vector of them. One feed is split
# evenly across the vector. The beam shadow cap is the sum of every jet.
# Sphere diameters are one size (`d`) or a lognormal drawn from d10/d50/d90
# and rejected outside [dmin, dmax]. The sampled diameter is the optical
# diameter: parcel mass stays mdot/nparcels, and N = m_parcel/m_sphere(d).
mutable struct PowderJet{T<:AbstractFloat}
    enabled::Bool        # Boolean to enable powder source
    mdot::T              # Source mass rate in kg/s
    w::T                 # 1/e² radius in the nozzle plane in cells (unit)
    v::T                 # Parcel velocity in cells/step
    x::T                 # Nozzle origin X in lattice cells
    y::T                 # Nozzle origin Y in lattice cells
    z::T                 # Nozzle origin Z in lattice cells
    dx::T                # Nozzle axis in X in lattice cells 
    dy::T                # Nozzle axis in Y in lattice cells
    dz::T                # Nozzle axis in Z in lattice cells
    nparcels::Int        # Number of parcels spawned per time step
    nmax::Int            # Maximum number of parcels
    px::Vector{T}        # Per-parcel X position in lattice cells
    py::Vector{T}        # Per-parcel Y position in lattice cells
    pz::Vector{T}        # Per-parcel Z position in lattice cells
    pvx::Vector{T}       # Per-parcel X velocity in lattice cells
    pvy::Vector{T}       # Per-parcel Y velocity in lattice cells
    pvz::Vector{T}       # Per-parcel Z velocity in lattice cells
    pm::Vector{T}        # Per-parcel mass in ρ×cell
    alive::Vector{Bool}  # Per-parcel boolean state if alive
    pT::Vector{T}        # Per-parcel lattice temperature
    d::T                 # Median / fallback sphere diameter in cells. 0 disables absorption when pd is 0.
    Tfeed::T             # Spawn temperature. 0 uses domain.T_p.
    absorbed::T          # Geometric shadow, watts removed from the wall beam
    qhold::Vector{T}     # Enthalpy debit currently subtracted from Q
    pd::Vector{T}        # Per-parcel sphere diameter in cells. 0 uses d.
    d10::T               # Lognormal D10 in cells. 0, with d50 and d90, disables the distribution.
    d50::T               # Median diameter in cells.
    d90::T               # Lognormal D90 in cells.
    dmin::T              # Smallest accepted diameter in cells.
    dmax::T              # Largest accepted diameter in cells.
end

# Standard-normal 10% and 90% quantiles. Same width rule as a lognormal
# powder specification: σ is the average of the widths implied by D10 and D90.
const _PSD_Z10 = -1.2815515655446004
const _PSD_Z90 = 1.2815515655446004

function _check_powder_psd(d10, d50, d90, dmin, dmax)
    (d10 > 0 || d50 > 0 || d90 > 0) || return nothing
    0 < d10 < d50 < d90 || throw(ArgumentError("powder size distribution needs 0 < d10 < d50 < d90"))
    dmin > 0 || throw(ArgumentError("powder dmin must be positive"))
    dmin < dmax || throw(ArgumentError("powder dmin must be below dmax"))
    μ = log(d50)
    σ10 = (log(d10) - μ) / _PSD_Z10
    σ90 = (log(d90) - μ) / _PSD_Z90
    σ = (σ10 + σ90) / 2
    if !(σ > 0)
        throw(ArgumentError("powder d10 and d90 must lie on opposite sides of d50"))
    end
    if abs(σ10 - σ90) > 0.25 * σ
        @warn "D10 and D90 imply different lognormal widths; using the average" σ10 σ90 σ
    end
    return nothing
end

# One diameter in cells. Without a distribution, every parcel uses J.d.
function _sample_diameter(J::PowderJet{T}) where {T}
    (J.d10 > 0 && J.d50 > 0 && J.d90 > 0) || return J.d
    μ = log(J.d50)
    σ10 = (log(J.d10) - μ) / T(_PSD_Z10)
    σ90 = (log(J.d90) - μ) / T(_PSD_Z90)
    σ = T(0.5) * (σ10 + σ90)
    σ > 0 || return J.d50
    lo = J.dmin
    hi = J.dmax
    for _ in 1:64
        u1 = rand(T)
        u2 = rand(T)
        z = sqrt(-T(2) * log(max(u1, eps(T)))) * cos(T(2π) * u2)
        d = exp(μ + σ * z)
        if d >= lo && d <= hi
            return d
        end
    end
    d = J.d50
    d = d < lo ? lo : d
    return d > hi ? hi : d
end

# Returns two unit vectors in the nozzle plane, these are perpendicular
# to the jet axis.
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
    mdot = zero(T),                     # Source mass rate in kg/s
    w = one(T),                         # 1/e² radius in the nozzle plane in cells (unit)
    v = one(T),                         # Parcel velocity in cells/step
    x = one(T),                         # Nozzle origin X in lattice cells
    y = one(T),                         # Nozzle origin Y in lattice cells
    z = one(T),                         # Nozzle origin Z in lattice cells
    dir = (zero(T), zero(T), -one(T)),  # Jet direction, default is (0, 0, -1)
    d = zero(T),                        # Sphere diameter in cells. 0 = no absorption.
    d10 = zero(T),                      # Lognormal D10 in cells. 0 disables the distribution.
    d50 = zero(T),                      # Median diameter in cells.
    d90 = zero(T),                      # Lognormal D90 in cells.
    dmin = zero(T),                     # Smallest accepted diameter in cells.
    dmax = zero(T),                     # Largest accepted diameter in cells.
    Tfeed = zero(T),                    # Spawn temperature. 0 → domain.T_p at spawn.
    nparcels = 16,                      # Number of powder parcels
    nmax = 2048,                        # Maximum allowed parcels
    enabled = true,                     # Enable powder source
) where {T<:AbstractFloat}
    d10 = T(d10); d50 = T(d50); d90 = T(d90); dmin = T(dmin); dmax = T(dmax)
    _check_powder_psd(d10, d50, d90, dmin, dmax)
    d = T(d)
    if !(d > 0) && d50 > 0
        d = d50
    end
    ax = SVector{3,T}(T(dir[1]), T(dir[2]), T(dir[3]))
    nrm = sqrt(ax[1]*ax[1] + ax[2]*ax[2] + ax[3]*ax[3])
    nrm <= 0 && (ax = SVector{3,T}(zero(T), zero(T), -one(T)); nrm = one(T))
    ax = ax / nrm
    nmax = max(Int(nparcels), Int(nmax))
    return PowderJet{T}(
        enabled, T(mdot), T(w), T(v), T(x), T(y), T(z), ax[1], ax[2], ax[3],
        Int(nparcels), nmax,
        zeros(T, nmax), zeros(T, nmax), zeros(T, nmax),
        zeros(T, nmax), zeros(T, nmax), zeros(T, nmax),
        zeros(T, nmax), fill(false, nmax), fill(T(Tfeed), nmax),
        T(d), T(Tfeed), zero(T), T[],
        zeros(T, nmax), d10, d50, d90, dmin, dmax,
    )
end

function PowderJet(U::Units{T};
    mdot, w, v,
    d = zero(T),
    d10 = zero(T), d50 = zero(T), d90 = zero(T),
    dmin = zero(T), dmax = zero(T),
    Tfeed = nothing,
    x = one(T), y = one(T), z = one(T),
    dir = (0, 0, -1),
    nparcels = 16, nmax = 2048,
    enabled = true,
) where {T}
    md = mdot isa Quantity ? T(ustrip(u"kg/s", uconvert(u"kg/s", mdot))) : T(mdot)
    wl = w isa Quantity ? T(ustrip(u"m", w) / U.m) : T(w)
    vl = v isa Quantity ? T(ustrip(u"m/s", v) * U.s / U.m) : T(v)
    cells = x -> x isa Quantity ? T(ustrip(u"m", x) / U.m) : T(x)
    dl = cells(d)
    Tf = Tfeed === nothing ? zero(T) :
         Tfeed isa Quantity ? T(lbm_T(U, Tfeed)) : T(Tfeed)
    return PowderJet{T}(; mdot=md, w=wl, v=vl, d=dl, Tfeed=Tf, x=T(x), y=T(y), z=T(z),
                        d10=cells(d10), d50=cells(d50), d90=cells(d90),
                        dmin=cells(dmin), dmax=cells(dmax),
                        dir=dir, nparcels=nparcels, nmax=nmax, enabled=enabled)
end

# Set the position of the powder jet origin in lattice units.
function set_powder_jet_position!(J::PowderJet{T}, x, y, z=J.z) where {T}
    J.x = T(x); J.y = T(y); J.z = T(z)
    return J
end

# Set the nozzle axis of the powder jet in lattice units.
function aim_powder_jet!(J::PowderJet{T}, tx, ty, tz) where {T}
    dx = T(tx) - J.x
    dy = T(ty) - J.y
    dz = T(tz) - J.z
    nrm = sqrt(dx*dx + dy*dy + dz*dz)
    nrm <= 0 && return J
    J.dx = dx/nrm; J.dy = dy/nrm; J.dz = dz/nrm
    return J
end

# One jet, or the list stored on the model. An empty list does no work.
_powder_jet_list(::Nothing) = ()
_powder_jet_list(J::PowderJet) = (J,)
_powder_jet_list(J::AbstractVector{<:PowderJet}) = J

# Even share of one feed. The shares sum to mdot.
powder_feed_share(mdot, n::Integer) = n == 1 ? mdot : mdot / n

# n jets, each carrying an even share of mdot. Other keywords match PowderJet.
function make_powder_jets(U::Units; mdot, n::Integer, kwargs...)
    n >= 1 || throw(ArgumentError("powder feed needs at least one jet"))
    share = powder_feed_share(mdot, n)
    return [PowderJet(U; mdot = share, kwargs...) for _ in 1:n]
end

function set_powder_enabled!(J::PowderJet, on::Bool)
    J.enabled = on
    return J
end
function set_powder_enabled!(jets::AbstractVector{<:PowderJet}, on::Bool)
    for J in jets
        J.enabled = on
    end
    return jets
end

powder_enabled(::Nothing) = false
powder_enabled(J::PowderJet) = J.enabled
powder_enabled(jets::AbstractVector{<:PowderJet}) = any(J -> J.enabled, jets)

# Place every jet on a circle about the focus and aim it there.
# tilt_rad is the angle from the vertical. azimuth_rad is around the beam,
# 0 along the scan (sgn), positive toward +y when sgn is +1.
# The circle radius is (z_noz − z_aim) tan(tilt). A jet that would leave the
# box stops at the wall; the aim stays on the focus.
function place_powder_jets!(jets, x_las, y_las, z_noz, z_aim, Nx, Ny, sgn, tilt_rad, azimuth_rad)
    n = length(jets)
    n == length(azimuth_rad) || throw(ArgumentError("one azimuth per powder jet"))
    n == 0 && return jets
    T = eltype(first(jets).x)
    drop = max(float(z_noz) - float(z_aim), 1.0)
    R = drop * tan(float(tilt_rad))
    sg = float(sgn)
    for (J, ψ) in zip(jets, azimuth_rad)
        c = cos(float(ψ))
        s = sin(float(ψ))
        x_noz = clamp(float(x_las) + sg * R * c, 2.5, float(Nx) - 1.5)
        y_noz = clamp(float(y_las) + sg * R * s, 2.5, float(Ny) - 1.5)
        set_powder_jet_position!(J, T(x_noz), T(y_noz), T(z_noz))
        aim_powder_jet!(J, x_las, y_las, z_aim)
    end
    return jets
end

# Returns the first free parcel slot, or 0 if all slots are alive.
@inline function _parcel_slot(J::PowderJet)
    @inbounds for i in 1:J.nmax
        J.alive[i] || return i
    end
    return 0
end

# Populate the parcels, sampled from 2D Gaussian. Tspawn is the feed temperature.
function _spawn_parcels!(J::PowderJet{T}, m_each::T, Tspawn::T) where {T}
    m_each <= 0 && return nothing
    e1x, e1y, e1z, e2x, e2y, e2z = _jet_basis(J.dx, J.dy, J.dz)
    σ = T(0.5) * J.w
    @inbounds for _ in 1:J.nparcels
        i = _parcel_slot(J)
        i == 0 && break

        # Box-Muller in the plane
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
        J.pT[i] = Tspawn
        J.pd[i] = _sample_diameter(J)
        J.alive[i] = true
    end
    return nothing
end

# Places the parcel into cell n.
# τ_p > 0: a molten parcel, or a landing in liquid, joins the metal.
# Specific enthalpy becomes the mass-weighted mixture of the metal already
# in the cell and the parcel. Earlier parcels this step are in qhold, so
# h_now = h_base − qhold. Q changes only by the new mixture; qhold stores
# the whole debit against h_base, and the next step adds it back so a held
# laser source is not applied twice. An empty mass field uses ρ as the
# heat capacity and writes that metal into mass.
# A cold parcel on solid metal stays in mp and sheds on τ_p.
# τ_p == 0: mass on TYPE_I / TYPE_F, no temperature.
# Returns (lattice mass captured, lattice enthalpy brought in).
@inline function _deposit_parcel!(
    mp, mass, Q, qhold, Tfield, fs, ρ, flags, n::Int,
    pmass::T, pT::T, τ_p, Ts::T, Λ::T, γs::T, γl::T,
) where {T}
    pmass <= zero(T) && return zero(T), zero(T)
    if τ_p > 0
        @inbounds begin
            Tc = T(Tfield[n])
            fsn = T(fs[n])
            liquid = (Tc >= Ts) || !is_solid_fraction(fsn)
            molten = pT >= Ts
            if liquid || molten
                ρn = T(ρ[n])
                ρn = ρn > zero(T) ? ρn : one(T)
                m_store = T(mass[n])
                m_now = m_store > zero(T) ? m_store : ρn
                γc = blend_phase(fsn, γs, γl)
                h_base = cell_enthalpy(Tc, fsn, Λ, γc)
                held = T(qhold[n])
                h_now = h_base - held
                γp = molten ? γl : γs
                h_p = sensible_H(pT, γp) + (molten ? Λ : zero(T))
                m_new = m_now + pmass
                h_mix = (m_now * h_now + pmass * h_p) / m_new
                mass[n] = m_new
                debit = h_base - h_mix
                Q[n] -= debit - held
                qhold[n] = debit
                return pmass, pmass * h_p
            else
                mp[n] += pmass
                return pmass, pmass * sensible_H(pT, γs)
            end
        end
    else
        @inbounds begin
            su = flags[n] & TYPE_SU
            if su == TYPE_I || su == TYPE_F
                mass[n] += pmass
                return pmass, pmass * sensible_H(pT, γs)
            end
        end
        return zero(T), zero(T)
    end
end

# Digital Differential Analyzer (DDA) along direction (dir) for at most 
# dist_max cells. Hits I/F leads to deposit; S leads to death.
# Returns position, alive, captured mass, and captured enthalpy.
@inline function _walk_parcel!(
    mp, mass, Q, qhold, Tfield, fs, ρ, flags, ϕ, τ_p,
    o0x::T, o0y::T, o0z::T, dirx::T, diry::T, dirz::T,
    dist_max::T, pmass::T, pT::T, Ts::T, Λ::T, γs::T, γl::T,
    Nx::Int, Ny::Int, Nz::Int,
) where {T}
    @static if DIM == 3
        ox, oy, oz = o0x, o0y, o0z
        traveled = zero(T)
        max_step = Nx + Ny + Nz + 8
        @inbounds for _ in 1:max_step
            remaining = dist_max - traveled

            # Stopped short
            remaining <= T(1e-6) && return ox, oy, oz, true, zero(T), zero(T)

            ix = floor(Int, ox + T(0.5))
            iy = floor(Int, oy + T(0.5))
            iz = floor(Int, oz + T(0.5))

            # Out of bounds
            if ix < 1 || ix > Nx || iy < 1 || iy > Ny || iz < 1 || iz > Nz
                return ox, oy, oz, false, zero(T), zero(T)
            end

            n = ix + (iy - 1) * Nx + (iz - 1) * Nx * Ny
            fl = flags[n]
            su = fl & TYPE_SU

            # Cancel on solid boundary
            if (fl & TYPE_BO) == TYPE_S
                return ox, oy, oz, false, zero(T), zero(T)
            # Deposit into Interface
            elseif su == TYPE_I
                ϕ0 = T(ϕ[n])
                phij = gather_phi_d3q27(ϕ, ϕ0, ix - 1, iy - 1, iz - 1, Nx, Ny, Nz)
                hit, t, _nϕ = plic_hit(ϕ0, phij, ox, oy, oz, dirx, diry, dirz,
                                       T(ix), T(iy), T(iz))
                # t = 0 is the face the step just crossed on a full cell.
                if hit && t >= zero(T) && t <= remaining + T(0.5)
                    dm, dE = _deposit_parcel!(
                        mp, mass, Q, qhold, Tfield, fs, ρ, flags, n,
                        pmass, pT, τ_p, Ts, Λ, γs, γl)
                    return ox, oy, oz, false, dm, dE
                end
            # Deposit into Fluid
            elseif su == TYPE_F
                dm, dE = _deposit_parcel!(
                        mp, mass, Q, qhold, Tfield, fs, ρ, flags, n,
                        pmass, pT, τ_p, Ts, Λ, γs, γl)
                return ox, oy, oz, false, dm, dE
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
        return ox, oy, oz, false, zero(T), zero(T)
    elseif DIM == 2
        # Vertical parcel cannot cross an x or y face, so it dies. Slab is z = 1.
        ox, oy = o0x, o0y
        oz = one(T)
        dirz = zero(T)
        nd = sqrt(dirx * dirx + diry * diry)
        if nd <= zero(T)
            return ox, oy, oz, false, zero(T), zero(T)
        end
        dirx /= nd
        diry /= nd
        traveled = zero(T)
        max_step = Nx + Ny + 16
        @inbounds for _ in 1:max_step
            remaining = dist_max - traveled

            # Stopped short
            remaining <= T(1e-6) && return ox, oy, oz, true, zero(T), zero(T)

            ix = floor(Int, ox + T(0.5))
            iy = floor(Int, oy + T(0.5))
            iz = floor(Int, oz + T(0.5))

            # Out of bounds
            if ix < 1 || ix > Nx || iy < 1 || iy > Ny || iz < 1 || iz > Nz
                return ox, oy, oz, false, zero(T), zero(T)
            end

            n = ix + (iy - 1) * Nx + (iz - 1) * Nx * Ny
            fl = flags[n]
            su = fl & TYPE_SU

            # Cancel on solid boundary
            if (fl & TYPE_BO) == TYPE_S
                return ox, oy, oz, false, zero(T), zero(T)
            # Deposit into Interface
            elseif su == TYPE_I
                ϕ0 = T(ϕ[n])
                phij = gather_phi_d2q9(ϕ, ϕ0, ix - 1, iy - 1, iz - 1, Nx, Ny, Nz)
                hit, t, _nϕ = plic_hit(ϕ0, phij, ox, oy, oz, dirx, diry, dirz,
                                       T(ix), T(iy), T(iz))
                if hit && t >= zero(T) && t <= remaining + T(0.5)
                    dm, dE = _deposit_parcel!(
                        mp, mass, Q, qhold, Tfield, fs, ρ, flags, n,
                        pmass, pT, τ_p, Ts, Λ, γs, γl)
                    return ox, oy, oz, false, dm, dE
                end
            # Deposit into Fluid
            elseif su == TYPE_F
                dm, dE = _deposit_parcel!(
                        mp, mass, Q, qhold, Tfield, fs, ρ, flags, n,
                        pmass, pT, τ_p, Ts, Λ, γs, γl)
                return ox, oy, oz, false, dm, dE
            end
            tMaxX = dirx > 0 ? (T(ix) + T(0.5) - ox) / dirx :
                    dirx < 0 ? (ox - (T(ix) - T(0.5))) / (-dirx) : T(Inf)
            tMaxY = diry > 0 ? (T(iy) + T(0.5) - oy) / diry :
                    diry < 0 ? (oy - (T(iy) - T(0.5))) / (-diry) : T(Inf)
            tstep = min(max(tMaxX, zero(T)), max(tMaxY, zero(T)))
            tstep = min(tstep, remaining) + T(1e-5)
            ox += tstep * dirx
            oy += tstep * diry
            traveled += tstep
        end
        return ox, oy, oz, false, zero(T), zero(T)
    end
end

# Raw geometric shadow of each parcel, watts, before the cap at beam power.
# nothing when this jet shades nothing. Downstream of the source only.
function _powder_raw_shadow(J::PowderJet{FT}, L, U) where {FT}
    any(J.alive) || return nothing
    has_d = J.d > 0
    if !has_d
        @inbounds for i in eachindex(J.alive)
            if J.alive[i] && J.pd[i] > 0
                has_d = true
                break
            end
        end
    end
    has_d || return nothing
    P = FT(L.P)
    w = FT(L.w)
    wm = w * FT(U.m)
    I0 = FT(2) * P / (FT(π) * wm * wm)
    ρsi = FT(U.kg) / (FT(U.m)^3)
    w2 = w * w
    raw = zeros(FT, J.nmax)
    @inbounds for i in 1:J.nmax
        J.alive[i] || continue
        di = J.pd[i] > 0 ? J.pd[i] : J.d
        di > 0 || continue
        dm = di * FT(U.m)
        area1 = FT(π) * dm * dm / FT(4)
        m_one = ρsi * FT(π) * dm * dm * dm / FT(6)
        m_kg = J.pm[i] * FT(U.kg)
        (m_kg > 0 && m_one > 0) || continue
        rx = J.px[i] - FT(L.x)
        ry = J.py[i] - FT(L.y)
        rz = J.pz[i] - FT(L.z)
        axial = rx * FT(L.dx) + ry * FT(L.dy) + rz * FT(L.dz)
        axial > 0 || continue
        r2 = rx * rx + ry * ry + rz * rz - axial * axial
        r2 = r2 > 0 ? r2 : zero(FT)
        I = I0 * exp(-FT(2) * r2 / w2)
        # N = m/m_one spheres. Shadow per mass scales as area/volume ∝ 1/d.
        raw[i] = I * (m_kg / m_one) * area1
    end
    return raw
end

# Fraction of the beam that still reaches the plate. The shadow is the
# geometric cross section of every jet, so a particle already at T_v still
# blocks the wall.
function powder_beam_transmit(model)
    L = model.laser
    L === nothing && return 1.0f0
    FT = eltype(L.P)
    !(L.P > 0) && return one(FT)
    jets = _powder_jet_list(model.powder_jet)
    isempty(jets) && return one(FT)
    shadowed = zero(FT)
    for J in jets
        shadowed += FT(J.absorbed)
    end
    return clamp(one(FT) - shadowed / FT(L.P), zero(FT), one(FT))
end

# Optically thin cloud. Parcel i, standing for N = m/m_one spheres of diameter
# d, shadows S_i = I(r) N π d²/4 and absorbs A_i = η S_i. r is the perpendicular
# distance to the beam axis, and only downstream of the source. η is
# normal-incidence Fresnel. I(r) = 2P/(π w²) exp(-2 r²/w²), w the 1/e² radius.
# The laser rewrites Q every `every` steps and that Q is applied on each of the
# held steps, so this refresh deposits `every` steps of beam energy at once.
# The sum of S_i over every jet is capped at P, and that shadow is what the
# wall loses. Each jet stores its own share on J.absorbed. Temperature stops
# at T_v; power the particle cannot take is reflected, not given to the wall.
function heat_powder_beam!(model, domain)
    jets = _powder_jet_list(model.powder_jet)
    isempty(jets) && return nothing
    FT = eltype(first(jets).px)
    L = model.laser
    if L === nothing || !L.enabled || !(L.P > 0) || !(L.w > 0)
        for J in jets
            J.absorbed = zero(FT)
        end
        return nothing
    end
    laser_deposits_now(L, Int(domain.t)) || return nothing
    U = model.units
    cp = FT(U.cp)
    K = FT(U.K)
    dt = FT(U.s)
    if !(cp > 0 && K > 0 && dt > 0)
        for J in jets
            J.absorbed = zero(FT)
        end
        return nothing
    end
    raws = Vector{Vector{FT}}()
    live = PowderJet{FT}[]
    for J in jets
        raw = _powder_raw_shadow(J, L, U)
        if raw === nothing
            J.absorbed = zero(FT)
        else
            push!(raws, raw)
            push!(live, J)
        end
    end
    isempty(live) && return nothing
    P = FT(L.P)
    sraw = zero(FT)
    for raw in raws
        sraw += sum(raw)
    end
    scale = sraw > P ? P / sraw : one(FT)
    Tv = FT(domain.T_v)
    cap = Tv > 0
    η = fresnel_absorptance(one(FT), FT(L.n_re), FT(L.n_im))
    gap = FT(laser_hold_steps(L))
    for (J, raw) in zip(live, raws)
        shadowed = zero(FT)
        @inbounds for i in 1:J.nmax
            S = raw[i] * scale
            S > 0 || continue
            shadowed += S
            Pi = η * S
            m_kg = J.pm[i] * FT(U.kg)
            dT = Pi * dt * gap / (m_kg * cp * K)
            if cap
                room = Tv - J.pT[i]
                if dT > room
                    dT = room > 0 ? room : zero(FT)
                end
            end
            J.pT[i] += dT
        end
        J.absorbed = shadowed
    end
    return nothing
end

# Move one step, then deposit. `refreshed` is true when deposit_laser! just
# rewrote Q, so a debit from the previous step is already gone.
# Every jet is restored into one Q, then walked. Landings share the first
# jet's qhold so two nozzles that hit one cell mix once; the other qholds
# stay zero and the next step adds the shared debit back.
function advance_powder_jet!(model, domain, refreshed::Bool=false)
    jets = _powder_jet_list(model.powder_jet)
    isempty(jets) && return nothing
    FT = eltype(first(jets).px)
    moving = false
    restore = false
    for J in jets
        moving |= (J.enabled || any(J.alive))
        restore |= (!refreshed && !isempty(J.qhold))
    end
    if !moving && !restore
        return nothing
    end
    Qh = Array(domain.Q.data)
    for J in jets
        if length(J.qhold) != length(Qh)
            resize!(J.qhold, length(Qh))
            fill!(J.qhold, zero(FT))
        elseif !refreshed
            @inbounds for i in eachindex(Qh)
                Qh[i] += J.qhold[i]
            end
            fill!(J.qhold, zero(FT))
        else
            fill!(J.qhold, zero(FT))
        end
    end
    if !moving
        copyto!(domain.Q.data, Qh)
        return nothing
    end

    U = model.units
    τ_p = domain.τ_p
    Ts = FT(domain.Ts)
    Λ = FT(domain.Λ)
    γs = FT(domain.γ_s)
    γl = FT(domain.γ_l)
    Nx, Ny, Nz = Int(domain.Nx), Int(domain.Ny), Int(domain.Nz)
    flags = Array(domain.flags.data)
    ϕ = Array(domain.ϕ.data)
    mp = Array(domain.mp.data)
    mass = Array(domain.mass.data)
    Th = Array(domain.T.data)
    fsh = Array(domain.fs.data)
    ρh = Array(domain.ρ.data)
    qshared = first(jets).qhold
    captured = zero(FT)
    e_add = zero(FT)

    for J in jets
        if J.enabled && J.mdot > 0 && J.nparcels > 0
            m_step = FT(J.mdot * U.s / U.kg)
            Tspawn = J.Tfeed > zero(FT) ? J.Tfeed : FT(domain.T_p)
            _spawn_parcels!(J, m_step / FT(J.nparcels), Tspawn)
        end
        invv = J.v > 0 ? one(FT) / J.v : one(FT)
        @inbounds for i in 1:J.nmax
            J.alive[i] || continue
            dirx, diry, dirz = J.pvx[i]*invv, J.pvy[i]*invv, J.pvz[i]*invv
            nd = sqrt(dirx*dirx + diry*diry + dirz*dirz)
            if nd <= 0
                J.alive[i] = false
                continue
            end
            dirx /= nd; diry /= nd; dirz /= nd
            ox, oy, oz, live, dm, dE = _walk_parcel!(
                mp, mass, Qh, qshared, Th, fsh, ρh, flags, ϕ, τ_p,
                J.px[i], J.py[i], J.pz[i], dirx, diry, dirz,
                J.v, J.pm[i], J.pT[i], Ts, Λ, γs, γl, Nx, Ny, Nz,
            )
            J.px[i] = ox; J.py[i] = oy; J.pz[i] = oz
            J.alive[i] = live
            captured += dm
            e_add += dE
        end
    end

    @static if TEMPERATURE
        domain.E_powder += e_add
        domain.M_powder += captured
    end
    copyto!(domain.mp.data, mp)
    copyto!(domain.mass.data, mass)
    copyto!(domain.Q.data, Qh)
    return nothing
end
