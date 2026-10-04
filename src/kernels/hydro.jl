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
    @static if DIM == 2
        uz = zero(CType)
    end
    return ρn, ux, uy, uz
end

@inline function store_feq!(
    fi, n, x, y, z, ρn,
    ux, uy, uz, w::NTuple{Q, CType}, c::NTuple{Q, SVector{3, Int}},
    N, Nx, Ny, Nz, t_odd::Val{odd}
) where {odd, Q, CType}
    @static if DIM == 2
        uz = zero(CType)
    end
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
    @static if KBC
        hi = CType(1.999) # KBC holds τ→1/2; TRT did not
    else
        hi = CType(1.95)
    end
    return ifelse(ω > hi, hi, ifelse(ω < lo, lo, ω))
end

# Deviatoric shear population. c_s²=1/3, w_i/(2 c_s⁴) = (9/2) w_i.
# Off-diagonal Π_αβ is stored once, so those terms carry a 2.
@inline function kbc_shear(wi::CType, cx::CType, cy::CType, cz::CType,
                           Πxx::CType, Πyy::CType, Πzz::CType,
                           Πxy::CType, Πxz::CType, Πyz::CType) where {CType}
    third = CType(1) / CType(3)
    qxx = cx * cx - third
    qyy = cy * cy - third
    qzz = cz * cz - third
    contr = qxx * Πxx + qyy * Πyy + qzz * Πzz +
            CType(2) * (cx * cy * Πxy + cx * cz * Πxz + cy * cz * Πyz)
    return wi * CType(4.5) * contr
end

# KBC: f* = feq + (1-β) s + (1-γβ) h, h = (f-feq) - s, β = ω(ν).
# γ from ⟨s|h⟩/⟨h|h⟩ with weight 1/feq. γ=1 is BGK.
@inline function kbc_store!(
    t_odd::Val{odd}, fi, fn1::CType, pairs, w::NTuple{Q, CType}, c::NTuple{Q, SVector{3, Int}},
    ρn::CType, ux::CType, uy::CType, uz::CType, uu::CType,
    fx::CType, fy::CType, fz::CType, ω::CType,
    N::Int, Nx::Int, Ny::Int, Nz::Int, n::Int, x::Int, y::Int, z::Int, ::Type{CType}
) where {odd, Q, CType}
    NP = (Q - 1) ÷ 2
    β = ω
    Πxx = Πyy = Πzz = Πxy = Πxz = Πyz = zero(CType)
    fe0 = feq(w[1], ρn, ux, uy, uz, uu, c[1], CType)
    for k in 1:NP
        i = 2k
        fp, fm = pairs[k]
        feqp = feq(w[i], ρn, ux, uy, uz, uu, c[i], CType)
        feqm = feq(w[i + 1], ρn, ux, uy, uz, uu, c[i + 1], CType)
        dp = fp - feqp
        dm = fm - feqm
        cxp = CType(c[i][1]); cyp = CType(c[i][2]); czp = CType(c[i][3])
        cxm = CType(c[i + 1][1]); cym = CType(c[i + 1][2]); czm = CType(c[i + 1][3])
        Πxx += dp * cxp * cxp + dm * cxm * cxm
        Πyy += dp * cyp * cyp + dm * cym * cym
        Πzz += dp * czp * czp + dm * czm * czm
        Πxy += dp * cxp * cyp + dm * cxm * cym
        Πxz += dp * cxp * czp + dm * cxm * czm
        Πyz += dp * cyp * czp + dm * cym * czm
    end
    # h = (f − feq) − s. Strip mass and momentum from h so γ cannot change them
    # (Guo lives in that sector). f* = BGK + (1−γ)β h⊥.
    s0 = kbc_shear(w[1], zero(CType), zero(CType), zero(CType), Πxx, Πyy, Πzz, Πxy, Πxz, Πyz)
    h0 = (fn1 - fe0) - s0
    m0 = h0
    mx = zero(CType); my = zero(CType); mz = zero(CType)
    for k in 1:NP
        i = 2k
        fp, fm = pairs[k]
        feqp = feq(w[i], ρn, ux, uy, uz, uu, c[i], CType)
        feqm = feq(w[i + 1], ρn, ux, uy, uz, uu, c[i + 1], CType)
        sp = kbc_shear(w[i], CType(c[i][1]), CType(c[i][2]), CType(c[i][3]), Πxx, Πyy, Πzz, Πxy, Πxz, Πyz)
        sm = kbc_shear(w[i + 1], CType(c[i + 1][1]), CType(c[i + 1][2]), CType(c[i + 1][3]), Πxx, Πyy, Πzz, Πxy, Πxz, Πyz)
        hp = (fp - feqp) - sp
        hm = (fm - feqm) - sm
        m0 += hp + hm
        mx += hp * CType(c[i][1]) + hm * CType(c[i + 1][1])
        my += hp * CType(c[i][2]) + hm * CType(c[i + 1][2])
        mz += hp * CType(c[i][3]) + hm * CType(c[i + 1][3])
    end
    sh = zero(CType)
    hh = zero(CType)
    h0p = kbc_hperp(w[1], h0, zero(CType), zero(CType), zero(CType), m0, mx, my, mz)
    fe0s = ifelse(fe0 > CType(1e-8), fe0, CType(1e-8))
    sh += s0 * h0p / fe0s
    hh += h0p * h0p / fe0s
    for k in 1:NP
        i = 2k
        fp, fm = pairs[k]
        feqp = feq(w[i], ρn, ux, uy, uz, uu, c[i], CType)
        feqm = feq(w[i + 1], ρn, ux, uy, uz, uu, c[i + 1], CType)
        sp = kbc_shear(w[i], CType(c[i][1]), CType(c[i][2]), CType(c[i][3]), Πxx, Πyy, Πzz, Πxy, Πxz, Πyz)
        sm = kbc_shear(w[i + 1], CType(c[i + 1][1]), CType(c[i + 1][2]), CType(c[i + 1][3]), Πxx, Πyy, Πzz, Πxy, Πxz, Πyz)
        hp = kbc_hperp(w[i], (fp - feqp) - sp, CType(c[i][1]), CType(c[i][2]), CType(c[i][3]), m0, mx, my, mz)
        hm = kbc_hperp(w[i + 1], (fm - feqm) - sm, CType(c[i + 1][1]), CType(c[i + 1][2]), CType(c[i + 1][3]), m0, mx, my, mz)
        fep = ifelse(feqp > CType(1e-8), feqp, CType(1e-8))
        fem = ifelse(feqm > CType(1e-8), feqm, CType(1e-8))
        sh += sp * hp / fep + sm * hm / fem
        hh += hp * hp / fep + hm * hm / fem
    end
    γ = kbc_gamma(sh, hh, β)
    omb = one(CType) - β
    ghost = (one(CType) - γ) * β
    gscale = one(CType) - CType(0.5) * β
    f0 = fe0 + omb * (fn1 - fe0) + ghost * h0p
    @static if APPLY_FORCE
        f0 += gscale * guo_fi(w[1], ux, uy, uz, fx, fy, fz, c[1], CType)
    end
    fi[f_index(n, 1, N)] = eltype(fi)(f0)
    for k in 1:NP
        i = 2k
        fp0, fm0 = pairs[k]
        feqp = feq(w[i], ρn, ux, uy, uz, uu, c[i], CType)
        feqm = feq(w[i + 1], ρn, ux, uy, uz, uu, c[i + 1], CType)
        sp = kbc_shear(w[i], CType(c[i][1]), CType(c[i][2]), CType(c[i][3]), Πxx, Πyy, Πzz, Πxy, Πxz, Πyz)
        sm = kbc_shear(w[i + 1], CType(c[i + 1][1]), CType(c[i + 1][2]), CType(c[i + 1][3]), Πxx, Πyy, Πzz, Πxy, Πxz, Πyz)
        hp = kbc_hperp(w[i], (fp0 - feqp) - sp, CType(c[i][1]), CType(c[i][2]), CType(c[i][3]), m0, mx, my, mz)
        hm = kbc_hperp(w[i + 1], (fm0 - feqm) - sm, CType(c[i + 1][1]), CType(c[i + 1][2]), CType(c[i + 1][3]), m0, mx, my, mz)
        fp = feqp + omb * (fp0 - feqp) + ghost * hp
        fm = feqm + omb * (fm0 - feqm) + ghost * hm
        @static if APPLY_FORCE
            fp += gscale * guo_fi(w[i], ux, uy, uz, fx, fy, fz, c[i], CType)
            fm += gscale * guo_fi(w[i + 1], ux, uy, uz, fx, fy, fz, c[i + 1], CType)
        end
        src = src_index(x, y, z, c[i][1], c[i][2], c[i][3], Nx, Ny, Nz)
        store_pair!(fi, n, src, i, fp, fm, t_odd, N)
    end
    return nothing
end

@inline function kbc_hperp(wi::CType, hi::CType, cx::CType, cy::CType, cz::CType,
                           m0::CType, mx::CType, my::CType, mz::CType) where {CType}
    return hi - wi * (m0 + CType(3) * (mx * cx + my * cy + mz * cz))
end

@inline function kbc_gamma(sh::CType, hh::CType, β::CType) where {CType}
    hh > CType(1e-14) || return one(CType)
    γ = (one(CType) / β) - ((CType(2) - β) / β) * (sh / hh)
    return clamp(γ, zero(CType), CType(2))
end

@inline function omega_from_nu(ν::CType) where {CType}
    return clamp_omega(one(CType) / (CType(3) * ν + CType(0.5)))
end