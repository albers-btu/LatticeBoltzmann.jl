using Atomix

# Marangoni on TYPE_I (one-cell interface): F = σT (∇T - n(n·∇T)), n = ∇ϕ/|∇ϕ|.
@inline function marangoni_force(
    Tfield, ϕ, flags, σT::CType, x::Int, y::Int, z::Int, n::Int,
    Nx::Int, Ny::Int, Nz::Int, ::Type{CType}
) where {CType}
    xp = src_index(x, y, z, 1, 0, 0, Nx, Ny, Nz)
    xm = src_index(x, y, z, -1, 0, 0, Nx, Ny, Nz)
    yp = src_index(x, y, z, 0, 1, 0, Nx, Ny, Nz)
    ym = src_index(x, y, z, 0, -1, 0, Nx, Ny, Nz)
    zp = src_index(x, y, z, 0, 0, 1, Nx, Ny, Nz)
    zm = src_index(x, y, z, 0, 0, -1, Nx, Ny, Nz)
    dTx = axis_deriv(Tfield, flags, n, xp, xm, CType)
    dTy = axis_deriv(Tfield, flags, n, yp, ym, CType)
    dTz = axis_deriv(Tfield, flags, n, zp, zm, CType)
    half = CType(0.5)
    dϕx = half * (ϕ[xp] - ϕ[xm])
    dϕy = half * (ϕ[yp] - ϕ[ym])
    dϕz = half * (ϕ[zp] - ϕ[zm])
    mag = sqrt(dϕx * dϕx + dϕy * dϕy + dϕz * dϕz)
    if mag > eps(CType)
        inv = one(CType) / mag
        nx, ny, nz = dϕx * inv, dϕy * inv, dϕz * inv
        ndT = nx * dTx + ny * dTy + nz * dTz
        dTx -= nx * ndT
        dTy -= ny * ndT
        dTz -= nz * ndT
    end
    return σT * dTx, σT * dTy, σT * dTz
end

@inline function darcy_force(
    fsn::CType, ux::CType, uy::CType, uz::CType, ρn::CType, ν::CType, K0::CType
) where {CType}
    K0 <= zero(CType) && return zero(CType), zero(CType), zero(CType)
    fl = one(CType) - fsn
    # F = -drag u with feq at u + F/(2ρ) maps the population velocity to
    # u' = (1 - drag/ρ) u. drag = 2ρ stores a zero half-force velocity and
    # reflects the populations (u' = -u); that momentum advects heat while
    # the stored speed stays ~0. drag = ρ kills the populations in one step.
    maxd = ρn
    if fl < CType(1e-3)
        return -maxd * ux, -maxd * uy, -maxd * uz
    end
    K = K0 * (fl * fl * fl) / (fsn * fsn + CType(1e-8))
    drag = ν / K
    drag = ifelse(drag > maxd, maxd, drag)
    return -drag * ux, -drag * uy, -drag * uz
end

# Anisimov recoil: F = p_r n, p_r = 0.54 p_sat, n = ∇ϕ/|∇ϕ| (into liquid).
# Λ_v = 0 → off. |F| capped so Guo Δu stays O(0.1).
@inline function recoil_force(
    Tfield, ϕ, n::Int, x::Int, y::Int, z::Int,
    Nx::Int, Ny::Int, Nz::Int,
    Λ_v::CType, T_v::CType, p0::CType, β_v::CType, ::Type{CType}
) where {CType}
    Λ_v <= zero(CType) && return zero(CType), zero(CType), zero(CType)
    Tn = Tfield[n]
    pr = CType(0.54) * p_sat(Tn, T_v, p0, β_v)
    pr = ifelse(pr > CType(0.5), CType(0.5), pr)
    pr <= zero(CType) && return zero(CType), zero(CType), zero(CType)
    xp = src_index(x, y, z, 1, 0, 0, Nx, Ny, Nz)
    xm = src_index(x, y, z, -1, 0, 0, Nx, Ny, Nz)
    yp = src_index(x, y, z, 0, 1, 0, Nx, Ny, Nz)
    ym = src_index(x, y, z, 0, -1, 0, Nx, Ny, Nz)
    zp = src_index(x, y, z, 0, 0, 1, Nx, Ny, Nz)
    zm = src_index(x, y, z, 0, 0, -1, Nx, Ny, Nz)
    half = CType(0.5)
    dϕx = half * (ϕ[xp] - ϕ[xm])
    dϕy = half * (ϕ[yp] - ϕ[ym])
    dϕz = half * (ϕ[zp] - ϕ[zm])
    mag = sqrt(dϕx * dϕx + dϕy * dϕy + dϕz * dϕz)
    mag <= eps(CType) && return zero(CType), zero(CType), zero(CType)
    inv = one(CType) / mag
    return pr * dϕx * inv, pr * dϕy * inv, pr * dϕz * inv
end

@inline function axis_deriv(Tfield, flags, n, srcp, srcm, ::Type{CType}) where {CType}
    okp = is_thermal_liquid(flags[srcp])
    okm = is_thermal_liquid(flags[srcm])
    if okp && okm
        return CType(0.5) * (Tfield[srcp] - Tfield[srcm])
    elseif okp
        return Tfield[srcp] - Tfield[n]
    elseif okm
        return Tfield[n] - Tfield[srcm]
    else
        return zero(CType)
    end
end

@inline function is_thermal_liquid(flagsn::UInt8)
    return (flagsn & TYPE_BO) != TYPE_S && (flagsn & TYPE_SU) != TYPE_G
end

@inline function blend_phase(fsn::CType, solid::CType, liquid::CType) where {CType}
    return fsn * solid + (one(CType) - fsn) * liquid
end

# Never below 10% of the Tref value (or pmin). Linear k(T) about Tm goes
# negative at room T if kT is large; clamping to ~0 put ω on 2 and the
# pad melted / FSLBM ate the domain in a few tens of steps.
@inline function floor_prop(p0::CType, pmin::CType) where {CType}
    pabs = ifelse(p0 > zero(CType), p0, -p0)
    frac = CType(0.1) * pabs
    return ifelse(frac > pmin, frac, pmin)
end

# Phase property at T: p = p0 + pT (T - Tref), then fs-blend.
@inline function prop_fs_T(fsn::CType, p_s::CType, p_sT::CType, p_l::CType, p_lT::CType,
                           Tn::CType, Tref::CType, pmin::CType) where {CType}
    dT = Tn - Tref
    ps = p_s + p_sT * dT
    pl = p_l + p_lT * dT
    pslo = floor_prop(p_s, pmin)
    pllo = floor_prop(p_l, pmin)
    ps = ifelse(ps > pslo, ps, pslo)
    pl = ifelse(pl > pllo, pl, pllo)
    return blend_phase(fsn, ps, pl)
end

@inline function is_solid_fraction(fsn::CType) where {CType}
    return (one(CType) - fsn) < CType(1e-3)
end

@inline function acc_add!(Eacc, i::Int, val::CType) where {CType}
    val == zero(CType) && return nothing
    Atomix.@atomic Eacc[i] += val
    return nothing
end