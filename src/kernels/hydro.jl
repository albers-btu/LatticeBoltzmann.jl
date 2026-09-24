@inline function feq(wi, ρn, ux, uy, uz, uu, ci, ::Type{CType}) where {CType}
    cu = CType(ci[1])*ux + CType(ci[2])*uy + CType(ci[3])*uz
    return wi * ρn * (one(CType) + CType(3)*cu + CType(4.5)*cu*cu - uu)
end

@inline function srt(ω::CType, f::CType, wi::CType, ρn::CType, ux::CType, uy::CType, uz::CType, uu::CType, ci) where {CType}
    return (one(CType) - ω) * f + ω * feq(wi, ρn, ux, uy, uz, uu, ci, CType)
end

@inline function collide_pair(
    ωp::CType, ωm::CType, fp::CType, fm::CType, feqp::CType, feqm::CType
) where {CType}
    @static if TRT
        half = CType(0.5)
        fsum = fp + fm
        fdif = fp - fm
        esum = feqp + feqm
        edif = feqp - feqm
        return (
            fp + half * ωp * (esum - fsum) + half * ωm * (edif - fdif),
            fm + half * ωp * (esum - fsum) - half * ωm * (edif - fdif),
        )
    else
        return ((one(CType) - ωp) * fp + ωp * feqp,
                (one(CType) - ωp) * fm + ωp * feqm)
    end
end

@inline function omega_minus(ω::CType) where {CType}
    three_nu = one(CType) / ω - CType(0.5) # 3ν = τ⁺ − 1/2
    return one(CType) / (CType(0.1875) / three_nu + CType(0.5))
end

@inline function guo_fi(
    wi::CType, ux::CType, uy::CType, uz::CType,
    fx::CType, fy::CType, fz::CType, ci, ::Type{CType}
) where {CType}
    cu = CType(ci[1])*ux + CType(ci[2])*uy + CType(ci[3])*uz
    cF = CType(ci[1])*fx + CType(ci[2])*fy + CType(ci[3])*fz
    uF = -CType(1)/CType(3) * (ux*fx + uy*fy + uz*fz)
    return CType(9) * wi * (cF * (cu + CType(1)/CType(3)) + uF)
end

@inline function guo_pair(
    ωp::CType, ωm::CType, wp::CType, wm::CType,
    ux::CType, uy::CType, uz::CType, fx::CType, fy::CType, fz::CType,
    cp, cm, ::Type{CType}
) where {CType}
    Fip = guo_fi(wp, ux, uy, uz, fx, fy, fz, cp, CType)
    Fim = guo_fi(wm, ux, uy, uz, fx, fy, fz, cm, CType)
    return scale_force_pair(ωp, ωm, Fip, Fim)
end

@inline function guo_rest(
    ωp::CType, w0::CType, ux::CType, uy::CType, uz::CType,
    fx::CType, fy::CType, fz::CType, c0, ::Type{CType}
) where {CType}
    return scale_force_rest(ωp, guo_fi(w0, ux, uy, uz, fx, fy, fz, c0, CType))
end

@inline function scale_force_pair(ωp::CType, ωm::CType, Fip::CType, Fim::CType) where {CType}
    @static if TRT
        cp = CType(0.5) - CType(0.25)*ωp
        cm = CType(0.5) - CType(0.25)*ωm
        Fsum = Fip + Fim
        Fdif = Fip - Fim
        return (cp*Fsum + cm*Fdif, cp*Fsum - cm*Fdif)
    else
        s = one(CType) - CType(0.5)*ωp
        return (s*Fip, s*Fim)
    end
end

@inline function scale_force_rest(ωp::CType, Fi0::CType) where {CType}
    return (one(CType) - CType(0.5)*ωp) * Fi0
end

@inline function prescribed_hydro(
    ρn::CType, ux::CType, uy::CType, uz::CType,
    fx::CType, fy::CType, fz::CType
) where {CType}
    cs = CType(1) / sqrt(CType(3))
    if ρn <= zero(CType)
        return one(CType), zero(CType), zero(CType), zero(CType)
    end
    @static if APPLY_FORCE
        invρ = one(CType) / ρn
        # Guo's half force shift ρu* = Σfᵢ×cᵢ
        # u = u* × F / 2ρ
        ux = clamp(ux + fx * invρ * CType(0.5), -cs, cs)
        uy = clamp(uy + fy * invρ * CType(0.5), -cs, cs)
        uz = clamp(uz + fz * invρ * CType(0.5), -cs, cs)
    else
        ux = clamp(ux, -cs, cs)
        uy = clamp(uy, -cs, cs)
        uz = clamp(uz, -cs, cs)
    end
    return ρn, ux, uy, uz
end

@inline function store_feq!(
    fi, n, x, y, z, ρn,
    ux, uy, uz, w::NTuple{Q, CType}, c::NTuple{Q, SVector{3, Int}},
    N, Nx, Ny, Nz, t_odd::Val{odd}
) where {odd, Q, CType}
    uu = CType(1.5) * (ux*ux + uy*uy + uz*uz)
    fi[f_index(n, 1, N)] = eltype(fi)(feq(w[1], ρn, ux, uy, uz, uu, c[1], CType))
    NP = (Q - 1) ÷ 2
    for k in 1:NP
        i = 2k
        src = src_index(x, y, z, c[i][1], c[i][2], c[i][3], Nx, Ny, Nz)
        store_pair!(fi, n, src, i,
            feq(w[i], ρn, ux, uy, uz, uu, c[i], CType),
            feq(w[i + 1], ρn, ux, uy, uz, uu, c[i + 1], CType),
            t_odd, N)
    end
    return nothing
end

@inline function equilibrium_boundary!(
    t_odd::Val{odd}, fi, ρ, u, w, c, fx, fy, fz,
    N, Nx, Ny, Nz, n, x, y, z, ::Type{CType}
) where {odd, CType}
    ρn, ux, uy, uz = prescribed_hydro(ρ[n], u[n, 1], u[n, 2], u[n, 3], fx, fy, fz)
    store_feq!(fi, n, x, y, z, ρn, ux, uy, uz, w, c, N, Nx, Ny, Nz, t_odd)
    return nothing
end

@inline function moving_wall_pair(
    fp::CType, fm::CType, flags, u, src_p, src_m,
    wi::CType, cp, cm, flagsn::UInt8, ::Type{CType}
) where {CType}
    @static if MOVING_BOUNDARIES
        if (flagsn & TYPE_BO) == TYPE_MS
            w6 = CType(-6) * wi
            if (flags[src_m] & TYPE_BO) == TYPE_S
                fp += w6 * (CType(cm[1])*u[src_m, 1] + CType(cm[2])*u[src_m, 2] + CType(cm[3])*u[src_m, 3])
            end
            if (flags[src_p] & TYPE_BO) == TYPE_S
                fm += w6 * (CType(cp[1])*u[src_p, 1] + CType(cp[2])*u[src_p, 2] + CType(cp[3])*u[src_p, 3])
            end
        end
    end
    return fp, fm
end

# BGK is unstable at ω = 0 or 2. Stay off the wall.
@inline function clamp_omega(ω::CType) where {CType}
    lo = CType(0.05)
    hi = CType(1.95)
    return ifelse(ω > hi, hi, ifelse(ω < lo, lo, ω))
end

@inline function omega_from_nu(ν::CType) where {CType}
    return clamp_omega(one(CType) / (CType(3) * ν + CType(0.5)))
end