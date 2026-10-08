using Atomix

# Marangoni on TYPE_I (one-cell interface): F = σT (∇T - n(n·∇T)), n = ∇ϕ/|∇ϕ|.
@inline function marangoni_force(
    Tfield, ϕ, flags, σT::CType, x::Int, y::Int, z::Int, n::Int,
    Nx::Int, Ny::Int, Nz::Int, ::Type{CType}
) where {CType}
    @static if DIM == 3
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
    elseif DIM == 2
        # (0,0,±1) on Nz = 1 wraps onto this cell; do not load it.
        xp = src_index(x, y, z, 1, 0, 0, Nx, Ny, Nz)
        xm = src_index(x, y, z, -1, 0, 0, Nx, Ny, Nz)
        yp = src_index(x, y, z, 0, 1, 0, Nx, Ny, Nz)
        ym = src_index(x, y, z, 0, -1, 0, Nx, Ny, Nz)
        dTx = axis_deriv(Tfield, flags, n, xp, xm, CType)
        dTy = axis_deriv(Tfield, flags, n, yp, ym, CType)
        dTz = zero(CType)
        half = CType(0.5)
        dϕx = half * (ϕ[xp] - ϕ[xm])
        dϕy = half * (ϕ[yp] - ϕ[ym])
        dϕz = zero(CType)
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
end

@inline function darcy_force(
    fsn::CType, ux::CType, uy::CType, uz::CType, ρn::CType, ν::CType, K0::CType
) where {CType}
    K0 <= zero(CType) && return zero(CType), zero(CType), zero(CType)
    fl = one(CType) - fsn
    maxd = CType(2) * ρn
    if fl < CType(1e-3)
        return -maxd * ux, -maxd * uy, -maxd * uz
    end
    K = K0 * (fl * fl * fl) / (fsn * fsn + CType(1e-8))
    drag = ν / K
    drag = ifelse(drag > maxd, maxd, drag)
    return -drag * ux, -drag * uy, -drag * uz
end

# Anisimov recoil pressure p_r = min(0.54 p_sat, 0.5). Same p_sat as
# evaporative_dT, including T < T_v. Λ_v = 0 → off.
# Hydrostatics use this as a gas density: p = ρ/3, so Δρ = 3 p_r.
# Adding p_r as a body force never builds that pressure. Each hydro substep
# then adds p_r/(2ρ), and the sum runs |u| onto c_s.
@inline function recoil_pressure(
    Tn::CType, Λ_v::CType, T_v::CType, p0::CType, β_v::CType
) where {CType}
    Λ_v <= zero(CType) && return zero(CType)
    pr = CType(0.54) * p_sat(Tn, T_v, p0, β_v)
    return ifelse(pr > CType(0.5), CType(0.5), pr)
end

@inline function gas_density_recoil(
    ρ_gas::CType, Tn::CType, Λ_v::CType, T_v::CType, p0::CType, β_v::CType
) where {CType}
    # Δρ = p_r / c_s². Boiling on the IN625 substep is ~0.20. The
    # interface viscosity floor is what keeps that from locking |u|
    # on c_s; this only drops a jump the floor has not been shown to carry.
    Δρ = CType(3) * recoil_pressure(Tn, Λ_v, T_v, p0, β_v)
    Δρ = ifelse(Δρ > CType(0.08), CType(0.08), Δρ)
    return clamp(ρ_gas + Δρ, CType(0.2), CType(2))
end

# Direction of recoil_pressure (into the liquid). The solver applies the
# pressure through gas_density_recoil, not through this vector.
@inline function recoil_force(
    Tfield, ϕ, n::Int, x::Int, y::Int, z::Int,
    Nx::Int, Ny::Int, Nz::Int,
    Λ_v::CType, T_v::CType, p0::CType, β_v::CType, ::Type{CType}
) where {CType}
    @static if DIM == 3
        Λ_v <= zero(CType) && return zero(CType), zero(CType), zero(CType)
        Tn = Tfield[n]
        pr = recoil_pressure(Tn, Λ_v, T_v, p0, β_v)
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
    elseif DIM == 2
        Λ_v <= zero(CType) && return zero(CType), zero(CType), zero(CType)
        Tn = Tfield[n]
        pr = recoil_pressure(Tn, Λ_v, T_v, p0, β_v)
        pr <= zero(CType) && return zero(CType), zero(CType), zero(CType)
        # (0,0,±1) on Nz = 1 wraps onto this cell; do not load it.
        xp = src_index(x, y, z, 1, 0, 0, Nx, Ny, Nz)
        xm = src_index(x, y, z, -1, 0, 0, Nx, Ny, Nz)
        yp = src_index(x, y, z, 0, 1, 0, Nx, Ny, Nz)
        ym = src_index(x, y, z, 0, -1, 0, Nx, Ny, Nz)
        half = CType(0.5)
        dϕx = half * (ϕ[xp] - ϕ[xm])
        dϕy = half * (ϕ[yp] - ϕ[ym])
        dϕz = zero(CType)
        mag = sqrt(dϕx * dϕx + dϕy * dϕy + dϕz * dϕz)
        mag <= eps(CType) && return zero(CType), zero(CType), zero(CType)
        inv = one(CType) / mag
        return pr * dϕx * inv, pr * dϕy * inv, pr * dϕz * inv
    end
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