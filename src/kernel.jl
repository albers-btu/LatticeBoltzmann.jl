using KernelAbstractions
using Atomix

@inline function wrap_coord(x, dx, N)
    ifelse(dx == 0, x,
        ifelse(dx > 0, ifelse(x == N - 1, 0, x + 1),
                       ifelse(x == 0, N - 1, x - 1)))
end

@inline function src_index(x, y, z, cx, cy, cz, Nx, Ny, Nz)
    wrap_coord(x, cx, Nx) + 
    wrap_coord(y, cy, Ny) * Nx + 
    wrap_coord(z, cz, Nz) * Nx * Ny + 1
end

@inline load_pair(fi, n, src, i, ::Val{true}, N, ::Type{CType}) where {CType} =
    (CType(fi[f_index(n, i, N)]), CType(fi[f_index(src, i + 1, N)]))
@inline load_pair(fi, n, src, i, ::Val{false}, N, ::Type{CType}) where {CType} =
    (CType(fi[f_index(n, i + 1, N)]), CType(fi[f_index(src, i, N)]))

@inline function store_pair!(fi, n, src, i, f_plus, f_minus, ::Val{true}, N)
    fi[f_index(n, i, N)]       = eltype(fi)(f_minus)
    fi[f_index(src, i + 1, N)] = eltype(fi)(f_plus)
    return nothing
end
@inline function store_pair!(fi, n, src, i, f_plus, f_minus, ::Val{false}, N)
    fi[f_index(n, i + 1, N)]   = eltype(fi)(f_minus)
    fi[f_index(src, i, N)]     = eltype(fi)(f_plus)
    return nothing
end

@inline load_outgoing_pair(fi, n, src, i, ::Val{true}, N, ::Type{CType}) where {CType} =
    (CType(fi[f_index(src, i, N)]), CType(fi[f_index(n, i + 1, N)]))
@inline load_outgoing_pair(fi, n, src, i, ::Val{false}, N, ::Type{CType}) where {CType} =
    (CType(fi[f_index(src, i + 1, N)]), CType(fi[f_index(n, i, N)]))

@inline function store_reconstructed_pair!(fi, n, src, i, f_plus, f_minus, gas_plus, gas_minus, ::Val{true}, N)
    # odd: incoming slots are (n,i) and (src, i+1)
    gas_minus && (fi[f_index(n, i, N)]       = eltype(fi)(f_minus))
    gas_plus  && (fi[f_index(src, i + 1, N)] = eltype(fi)(f_plus))
    return nothing
end
@inline function store_reconstructed_pair!(fi, n, src, i, f_plus, f_minus, gas_plus, gas_minus, ::Val{false}, N)
    gas_minus && (fi[f_index(n, i + 1, N)] = eltype(fi)(f_minus))
    gas_plus  && (fi[f_index(src, i, N)]   = eltype(fi)(f_plus))
    return nothing
end

@inline function feq(wi, ρn, ux, uy, uz, uu, ci, ::Type{CType}) where {CType}
    cu = CType(ci[1])*ux + CType(ci[2])*uy + CType(ci[3])*uz
    return wi * ρn * (one(CType) + CType(3)*cu + CType(4.5)*cu*cu - uu)
end

@inline function calculate_phi(ρn::CType, massn::CType, flagsn::UInt8) where {CType}
    if (flagsn & TYPE_F) != 0x00
        return one(CType)
    elseif (flagsn & TYPE_I) != 0x00
        return ρn > 0 ? clamp(massn / ρn, zero(CType), one(CType)) : CType(0.5)
    else
        return zero(CType)
    end
end

@inline function srt(ω::CType, f::CType, wi::CType, ρn::CType, ux::CType, uy::CType, uz::CType, uu::CType, ci) where {CType}
    # cu = CType(ci[1])*ux + CType(ci[2])*uy + CType(ci[3])*uz
    # feq = wi * ρn * (one(CType) + CType(3.0)*cu + CType(4.5)*cu*cu - uu)
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

@inline function scale_force_rest(ωp::CType, Fi0::CType) where {CType}
    return (one(CType) - CType(0.5)*ωp) * Fi0
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

@inline function guo_rest(
    ωp::CType, w0::CType, ux::CType, uy::CType, uz::CType,
    fx::CType, fy::CType, fz::CType, c0, ::Type{CType}
) where {CType}
    return scale_force_rest(ωp, guo_fi(w0, ux, uy, uz, fx, fy, fz, c0, CType))
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

# D3Q7 thermal. Perturbation g' = g - w, T = 1 + Σg.
# ω_T = 1/(2α + 1/2). Boussinesq: F_eff = F - F β (T - T_avg).
# Volumetric Q: after SRT, add Δgeq(ΔT=Q) so ΣΔg = Q (lattice dT/step).
# TYPE_H: equivalent Dirichlet from Fourier + Robin, k = c_sT² (τ_T-1/2) = α/2.
@inline function geq_T_rest(Tn::CType) where {CType}
    return CType(0.25) * Tn - CType(0.25)
end
@inline function geq_T_axis(Tn::CType, ucomp::CType) where {CType}
    return CType(0.5) * Tn * ucomp + CType(0.125) * (Tn - one(CType))
end

# TYPE_S never collides. Uninitialized gi=0 is T=1 (lattice Tm). Keep geq(T[n],0)
# on solids so bounce-back is Dirichlet at the stored wall temperature.
@inline function store_geq_local!(gi, n, Tn::CType, N::Int) where {CType}
    gi[f_index(n, 1, N)] = eltype(gi)(geq_T_rest(Tn))
    gax = geq_T_axis(Tn, zero(CType))
    @inbounds for i in 2:7
        gi[f_index(n, i, N)] = eltype(gi)(gax)
    end
    return nothing
end

# AA streams only +c, so a min-side TYPE_S never collides the wall–fluid link.
# Reconstruct missing g only at TYPE_S|TYPE_T: Dirichlet ABB at T[wall].
# Plain TYPE_S and TYPE_G keep the AA populations (adiabatic bounce-back).
# Writing geq(T) into TYPE_G overwrites the streamed outgoing stored in the
# gas cell and acts as a heat sink (kills recoil / evaporation).
@inline function is_dirichlet_solid(fl::UInt8)
    return ((fl & TYPE_S) != 0x00) & ((fl & TYPE_T) != 0x00)
end
@inline function is_flux_solid(fl::UInt8)
    return ((fl & TYPE_S) != 0x00) & ((fl & TYPE_H) != 0x00) & ((fl & TYPE_T) == 0x00)
end

@inline function robin_wall_T(
    Tfluid::CType, Tinf::CType, hn::CType, qn::CType, ω_T::CType
) where {CType}
    kT = thermal_conductivity(ω_T)
    kT = ifelse(kT > CType(1e-12), kT, CType(1e-12))
    Bi = hn / kT
    return (Tfluid + qn / kT + Bi * Tinf) / (one(CType) + Bi)
end

@inline function reconstruct_g_boundaries!(
    t_odd::Val{odd}, gi, T, flags, hT, Qin,
    x::Int, y::Int, z::Int, n::Int,
    N::Int, Nx::Int, Ny::Int, Nz::Int, ::Type{CType},
    Eacc, fillc::CType, ω_T::CType
) where {odd, CType}
    @inbounds for (i, cx, cy, cz) in ((2, 1, 0, 0), (4, 0, 1, 0), (6, 0, 0, 1))
        srcp = src_index(x, y, z, cx, cy, cz, Nx, Ny, Nz)
        srcm = src_index(x, y, z, -cx, -cy, -cz, Nx, Ny, Nz)
        dir_p = is_dirichlet_solid(flags[srcp])
        dir_m = is_dirichlet_solid(flags[srcm])
        flux_p = is_flux_solid(flags[srcp])
        flux_m = is_flux_solid(flags[srcm])
        miss_p = dir_p | flux_p
        miss_m = dir_m | flux_m
        (miss_p | miss_m) || continue
        fp_in,  fm_in  = load_pair(gi, n, srcp, i, t_odd, N, CType)
        fp_out, fm_out = load_outgoing_pair(gi, n, srcp, i, t_odd, N, CType)
        rec_p = fp_out
        rec_m = fm_out
        if miss_p
            Tw = T[srcp]
            if flux_p
                hn = hT[srcp]; qn = Qin[srcp]
                if hn == zero(CType) && qn == zero(CType)
                    miss_p = false
                else
                    Tw = robin_wall_T(T[n], Tw, hn, qn, ω_T)
                end
            end
            if miss_p
                geg = geq_T_axis(Tw, zero(CType))
                rec_p = geg + geg - fp_out
                acc_add!(Eacc, EACC_WALL, fillc * (fm_in - rec_p))
            end
        end
        if miss_m
            Tw = T[srcm]
            if flux_m
                hn = hT[srcm]; qn = Qin[srcm]
                if hn == zero(CType) && qn == zero(CType)
                    miss_m = false
                else
                    Tw = robin_wall_T(T[n], Tw, hn, qn, ω_T)
                end
            end
            if miss_m
                geg = geq_T_axis(Tw, zero(CType))
                rec_m = geg + geg - fm_out
                acc_add!(Eacc, EACC_WALL, fillc * (fp_in - rec_m))
            end
        end
        (miss_p | miss_m) && store_reconstructed_pair!(gi, n, srcp, i, rec_p, rec_m, miss_p, miss_m, t_odd, N)
    end
    return nothing
end

@inline function thermal_conductivity(ω_T::CType) where {CType}
    return CType(0.25) * (one(CType) / ω_T - CType(0.5))
end

@inline function flux_neighbor_T(Tfield, flags, x, y, z, Nx, Ny, Nz, ::Type{CType}) where {CType}
    Tnb = zero(CType)
    cnt = 0
    src = src_index(x, y, z, -1, 0, 0, Nx, Ny, Nz)
    if (flags[src] & TYPE_S) != 0x00
        Tnb += Tfield[src_index(x, y, z, 1, 0, 0, Nx, Ny, Nz)]
        cnt += 1
    end
    src = src_index(x, y, z, 1, 0, 0, Nx, Ny, Nz)
    if (flags[src] & TYPE_S) != 0x00
        Tnb += Tfield[src_index(x, y, z, -1, 0, 0, Nx, Ny, Nz)]
        cnt += 1
    end
    src = src_index(x, y, z, 0, -1, 0, Nx, Ny, Nz)
    if (flags[src] & TYPE_S) != 0x00
        Tnb += Tfield[src_index(x, y, z, 0, 1, 0, Nx, Ny, Nz)]
        cnt += 1
    end
    src = src_index(x, y, z, 0, 1, 0, Nx, Ny, Nz)
    if (flags[src] & TYPE_S) != 0x00
        Tnb += Tfield[src_index(x, y, z, 0, -1, 0, Nx, Ny, Nz)]
        cnt += 1
    end
    src = src_index(x, y, z, 0, 0, -1, Nx, Ny, Nz)
    if (flags[src] & TYPE_S) != 0x00
        Tnb += Tfield[src_index(x, y, z, 0, 0, 1, Nx, Ny, Nz)]
        cnt += 1
    end
    src = src_index(x, y, z, 0, 0, 1, Nx, Ny, Nz)
    if (flags[src] & TYPE_S) != 0x00
        Tnb += Tfield[src_index(x, y, z, 0, 0, -1, Nx, Ny, Nz)]
        cnt += 1
    end
    return Tnb, cnt
end

@inline function is_thermal_liquid(flagsn::UInt8)
    return (flagsn & TYPE_BO) != TYPE_S && (flagsn & TYPE_SU) != TYPE_G
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

@inline function fs_from_T(Tn::CType, Ts::CType, Tl::CType) where {CType}
    Tn <= Ts && return one(CType)
    Tn >= Tl && return zero(CType)
    Δ = Tl - Ts
    Δ <= eps(CType) && return zero(CType)
    return (Tl - Tn) / Δ
end

# cp/cp_ref = 1 + γ (T - 1). γ = 0 → H_sens = T.
@inline function sensible_H(Tn::CType, γ::CType) where {CType}
    return Tn + γ * (CType(0.5) * Tn * Tn - Tn)
end

@inline function invert_sensible_H(H::CType, γ::CType) where {CType}
    ag = ifelse(γ > zero(CType), γ, -γ)
    ag < CType(1e-8) && return H
    oneγ = one(CType) - γ
    disc = oneγ * oneγ + CType(2) * γ * H
    disc = ifelse(disc > zero(CType), disc, zero(CType))
    return (γ - one(CType) + sqrt(disc)) / γ
end

# Specific enthalpy in lattice T: H = ∫(1+γ(θ-1)) dθ + Λ f_l.
@inline function cell_enthalpy(Tn::CType, fsn::CType, Λ::CType, γ::CType) where {CType}
    return sensible_H(Tn, γ) + Λ * (one(CType) - fsn)
end
@inline cell_enthalpy(Tn, fsn, Λ) = cell_enthalpy(Tn, fsn, Λ, zero(Tn))

# Invert H to (T, f_l). γ = 0 recovers H = T + Λ f_l.
@inline function invert_enthalpy(H::CType, Ts::CType, Tl::CType, Λ::CType, γ::CType) where {CType}
    ΔTm = Tl - Ts
    if ΔTm <= eps(CType)
        Tm = Ts
        Hm = sensible_H(Tm, γ)
        if H <= Hm
            return invert_sensible_H(H, γ), zero(CType)
        elseif H >= Hm + Λ
            return invert_sensible_H(H - Λ, γ), one(CType)
        else
            return Tm, (H - Hm) / Λ
        end
    end
    Hsol = sensible_H(Ts, γ)
    Hliq = sensible_H(Tl, γ) + Λ
    if H <= Hsol
        return invert_sensible_H(H, γ), zero(CType)
    elseif H >= Hliq
        return invert_sensible_H(H - Λ, γ), one(CType)
    else
        a = CType(0.5) * γ
        b = (one(CType) - γ) + Λ / ΔTm
        c = -(H + Λ * Ts / ΔTm)
        T = zero(CType)
        if ifelse(a > zero(CType), a, -a) < CType(1e-8)
            T = -c / b
        else
            disc = b * b - CType(4) * a * c
            disc = ifelse(disc > zero(CType), disc, zero(CType))
            T = (-b + sqrt(disc)) / (CType(2) * a)
        end
        fl = (T - Ts) / ΔTm
        return T, ifelse(fl < zero(CType), zero(CType), ifelse(fl > one(CType), one(CType), fl))
    end
end
@inline invert_enthalpy(H, Ts, Tl, Λ) = invert_enthalpy(H, Ts, Tl, Λ, zero(H))

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

@inline function is_solid_fraction(fsn::CType) where {CType}
    return (one(CType) - fsn) < CType(1e-3)
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

# BGK is unstable at ω = 0 or 2. Stay off the wall.
@inline function clamp_omega(ω::CType) where {CType}
    lo = CType(0.05)
    hi = CType(1.95)
    return ifelse(ω > hi, hi, ifelse(ω < lo, lo, ω))
end

@inline function omega_from_nu(ν::CType) where {CType}
    return clamp_omega(one(CType) / (CType(3) * ν + CType(0.5)))
end

@inline function omega_T_from_alpha(α::CType) where {CType}
    return clamp_omega(one(CType) / (CType(2) * α + CType(0.5)))
end

# Clausius–Clapeyron p_sat (lattice). p0 at T_v.
@inline function p_sat(Tn::CType, T_v::CType, p0::CType, β_v::CType) where {CType}
    if !(T_v > zero(CType) && Tn > zero(CType))
        return zero(CType)
    end
    x = -β_v * (one(CType) / Tn - one(CType) / T_v)
    x = clamp(x, CType(-20), CType(20))
    return p0 * exp(x)
end

# Hertz–Knudsen cooling as lattice dT/step. Zero for T ≤ T_v or Λ_v = 0.
# Capped so T cannot fall below T_v in one collide.
@inline function evaporative_dT(
    Tn::CType, Λ_v::CType, T_v::CType, C_hk::CType, p0::CType, β_v::CType
) where {CType}
    if !(Λ_v > zero(CType) && T_v > zero(CType) && Tn > T_v)
        return zero(CType)
    end
    ps = p_sat(Tn, T_v, p0, β_v)
    mdot = C_hk * ps / sqrt(Tn)
    Qe = mdot * Λ_v
    Qmax = Tn - T_v
    return ifelse(Qe > Qmax, Qmax, ifelse(Qe > zero(CType), Qe, zero(CType)))
end

# Mass flux consistent with the capped heat sink: ṁ = Qe / Λ_v.
@inline function evaporative_flux(
    Tn::CType, Λ_v::CType, T_v::CType, C_hk::CType, p0::CType, β_v::CType
) where {CType}
    Qe = evaporative_dT(Tn, Λ_v, T_v, C_hk, p0, β_v)
    mdot = (Λ_v > zero(CType) && Qe > zero(CType)) ? Qe / Λ_v : zero(CType)
    return Qe, mdot
end

# Net emission q = εσ(T^4-T_∞^4) as lattice dT/step in one interface cell.
# C_rad = 0 → off. Capped so T cannot cross T_inf in one collide.
@inline function radiation_dT(Tn::CType, C_rad::CType, T_inf::CType) where {CType}
    C_rad <= zero(CType) && return zero(CType)
    T2 = Tn * Tn
    I2 = T_inf * T_inf
    Qr = C_rad * (T2 * T2 - I2 * I2)
    Δ = Tn - T_inf
    (Qr > zero(CType) && Δ > zero(CType)) || return zero(CType)
    return ifelse(Qr > Δ, Δ, Qr)
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

@inline function acc_add!(Eacc, i::Int, val::CType) where {CType}
    val == zero(CType) && return nothing
    Atomix.@atomic Eacc[i] += val
    return nothing
end

@inline function collide_temperature!(
    t_odd::Val{odd}, gi, Tfield, Qin, hT, flags, flagsn, fs,
    ux::CType, uy::CType, uz::CType,
    fxn::CType, fyn::CType, fzn::CType,
    fx::CType, fy::CType, fz::CType,
    ω_T::CType, β::CType, T_avg::CType, Λ::CType, Ts::CType, Tl::CType,
    γ_s::CType, γ_l::CType,
    Λ_v::CType, T_v::CType, C_hk::CType, p0v::CType, β_v::CType,
    C_rad::CType, T_rad::CType,
    x::Int, y::Int, z::Int, Nx::Int, Ny::Int, Nz::Int, N::Int, n::Int,
    ::Type{CType}, Eacc, fill::CType
) where {odd, CType}
    srcx = src_index(x, y, z, 1, 0, 0, Nx, Ny, Nz)
    srcy = src_index(x, y, z, 0, 1, 0, Nx, Ny, Nz)
    srcz = src_index(x, y, z, 0, 0, 1, Nx, Ny, Nz)
    g0 = CType(gi[f_index(n, 1, N)])
    gpx, gmx = load_pair(gi, n, srcx, 2, t_odd, N, CType)
    gpy, gmy = load_pair(gi, n, srcy, 4, t_odd, N, CType)
    gpz, gmz = load_pair(gi, n, srcz, 6, t_odd, N, CType)
    Tfromg = g0 + gpx + gmx + gpy + gmy + gpz + gmz + one(CType)

    Qn_in = Qin[n]
    Qn = Qn_in
    dirichlet = (flagsn & TYPE_T) != 0x00
    use_flux = false
    write_geq = false
    Tn = zero(CType)
    mevap = zero(CType)
    fillc = fill < zero(CType) ? zero(CType) : fill
    γn = blend_phase(fs[n], γ_s, γ_l)

    if dirichlet
        Tn = Tfield[n]
        Qn = zero(CType)
        fs_old = fs[n]
        if Λ > zero(CType)
            fs[n] = fs_from_T(Tn, Ts, Tl)
        end
        dH = cell_enthalpy(Tfromg, fs_old, Λ, γn) - cell_enthalpy(Tn, fs[n], Λ, γn)
        acc_add!(Eacc, EACC_WALL, fillc * dH)
    elseif (flagsn & TYPE_H) != 0x00
        Tnb, cnt = flux_neighbor_T(Tfield, flags, x, y, z, Nx, Ny, Nz, CType)
        if cnt > 0
            kT = thermal_conductivity(ω_T)
            hn = hT[n]
            Tinf = Tfield[n]
            Bi = hn / kT
            Tn = (Tnb + Qn / kT + Bi * Tinf) / (one(CType) + Bi)
            use_flux = true
            Qn = zero(CType)
            if hn == zero(CType)
                Tfield[n] = Tn
            end
            dH = cell_enthalpy(Tfromg, fs[n], Λ, γn) - cell_enthalpy(Tn, fs[n], Λ, γn)
            acc_add!(Eacc, EACC_WALL, fillc * dH)
        end
    end

    if !dirichlet && !use_flux
        acc_add!(Eacc, EACC_Q, fillc * Qn_in)
        if Λ > zero(CType) || γn != zero(CType)
            H = cell_enthalpy(Tfromg, fs[n], Λ, γn) + Qn
            Tnew, fl = invert_enthalpy(H, Ts, Tl, Λ, γn)
            fs[n] = one(CType) - fl
            Tfield[n] = Tnew
            if fl > zero(CType) && fl < one(CType)
                Tn = Tnew
                Qn = zero(CType)
                write_geq = true
            else
                Qn = Tnew - Tfromg
                Tn = Tfromg
            end
        else
            Tn = Tfromg
            Tfield[n] = Tn + Qn
        end
    end

    @static if SURFACE
        if !dirichlet && !use_flux && (flagsn & TYPE_SU) == TYPE_I
            Te = Tfield[n]
            if Λ_v > zero(CType) && !is_solid_fraction(fs[n])
                Qe, mevap = evaporative_flux(Te, Λ_v, T_v, C_hk, p0v, β_v)
                Qn -= Qe
                Te -= Qe
                acc_add!(Eacc, EACC_EVAP, fillc * Qe)
            end
            if C_rad > zero(CType)
                Qr = radiation_dT(Te, C_rad, T_rad)
                Qn -= Qr
                Te -= Qr
                acc_add!(Eacc, EACC_RAD, fillc * Qr)
            end
            Tfield[n] = Te
            write_geq && (Tn = Te)
            if mevap > zero(CType)
                acc_add!(Eacc, EACC_EVAP, mevap * cell_enthalpy(Te, fs[n], Λ, γn))
            end
        end
    end

    ge0 = geq_T_rest(Tn)
    gexp, gexm = geq_T_axis(Tn, ux), geq_T_axis(Tn, -ux)
    geyp, geym = geq_T_axis(Tn, uy), geq_T_axis(Tn, -uy)
    gezp, gezm = geq_T_axis(Tn, uz), geq_T_axis(Tn, -uz)

    if dirichlet || use_flux || write_geq
        gi[f_index(n, 1, N)] = eltype(gi)(ge0)
        store_pair!(gi, n, srcx, 2, gexp, gexm, t_odd, N)
        store_pair!(gi, n, srcy, 4, geyp, geym, t_odd, N)
        store_pair!(gi, n, srcz, 6, gezp, gezm, t_odd, N)
    else
        om = one(CType) - ω_T
        gi[f_index(n, 1, N)] = eltype(gi)(om * g0 + ω_T * ge0 + CType(0.25) * Qn)
        store_pair!(gi, n, srcx, 2,
            om * gpx + ω_T * gexp + CType(0.5) * Qn * ux + CType(0.125) * Qn,
            om * gmx + ω_T * gexm + CType(0.5) * Qn * (-ux) + CType(0.125) * Qn,
            t_odd, N)
        store_pair!(gi, n, srcy, 4,
            om * gpy + ω_T * geyp + CType(0.5) * Qn * uy + CType(0.125) * Qn,
            om * gmy + ω_T * geym + CType(0.5) * Qn * (-uy) + CType(0.125) * Qn,
            t_odd, N)
        store_pair!(gi, n, srcz, 6,
            om * gpz + ω_T * gezp + CType(0.5) * Qn * uz + CType(0.125) * Qn,
            om * gmz + ω_T * gezm + CType(0.5) * Qn * (-uz) + CType(0.125) * Qn,
            t_odd, N)
    end

    Tmacro = dirichlet || use_flux ? Tn : Tn + Qn
    dT = Tmacro - T_avg
    fxn -= fx * β * dT
    fyn -= fy * β * dT
    fzn -= fz * β * dT
    return fxn, fyn, fzn, mevap
end

@inline function store_geq!(
    gi, n, x, y, z, Tn, ux, uy, uz, N, Nx, Ny, Nz, t_odd::Val{odd}, ::Type{CType}
) where {odd, CType}
    gi[f_index(n, 1, N)] = eltype(gi)(geq_T_rest(Tn))
    srcx = src_index(x, y, z, 1, 0, 0, Nx, Ny, Nz)
    srcy = src_index(x, y, z, 0, 1, 0, Nx, Ny, Nz)
    srcz = src_index(x, y, z, 0, 0, 1, Nx, Ny, Nz)
    store_pair!(gi, n, srcx, 2, geq_T_axis(Tn, ux), geq_T_axis(Tn, -ux), t_odd, N)
    store_pair!(gi, n, srcy, 4, geq_T_axis(Tn, uy), geq_T_axis(Tn, -uy), t_odd, N)
    store_pair!(gi, n, srcz, 6, geq_T_axis(Tn, uz), geq_T_axis(Tn, -uz), t_odd, N)
    return nothing
end

@static if !SURFACE

@kernel function initialize_kernel!(
    ρ, u, fi, flags,
    w::NTuple{Q, CType}, 
    c::NTuple{Q, SVector{3, Int}},
    N::Int, Nx::Int, Ny::Int, Nz::Int,
    gi, T
) where {Q, CType}
    n = @index(Global)
    @inbounds begin
        n0 = n - 1
        x = n0 % Nx
        y = (n0 ÷ Nx) % Ny
        z = n0 ÷ (Nx * Ny)
        ρn = ρ[n]
        ux, uy, uz = u[n, 1], u[n, 2], u[n, 3]
        flagsn = flags[n]

        if (flagsn & TYPE_BO) == TYPE_S
            @static if MOVING_BOUNDARIES
                only_s = true
                for i in 2:Q
                    src = src_index(x, y, z, c[i][1], c[i][2], c[i][3], Nx, Ny, Nz)
                    only_s &= (flags[src] & TYPE_BO) == TYPE_S
                end
                if only_s
                    u[n, 1] = zero(CType)
                    u[n, 2] = zero(CType)
                    u[n, 3] = zero(CType)
                end
            else
                u[n, 1] = zero(CType)
                u[n, 2] = zero(CType)
                u[n, 3] = zero(CType)
            end
            @static if TEMPERATURE
                store_geq_local!(gi, n, T[n], N)
            end
        else
            uu = CType(1.5) * (ux*ux + uy*uy + uz*uz)
            fi[f_index(n, 1, N)] = eltype(fi)(w[1] * ρn * (one(CType) - uu))

            for k in 1:((Q - 1) ÷ 2)
                i = 2k
                cp, cm = c[i], c[i + 1]
                cup = CType(cp[1])*ux + CType(cp[2])*uy + CType(cp[3])*uz
                cum = CType(cm[1])*ux + CType(cm[2])*uy + CType(cm[3])*uz
                feqp = w[i]     * ρn * (one(CType) + CType(3.0)*cup + CType(4.5)*cup*cup - uu)
                feqm = w[i + 1] * ρn * (one(CType) + CType(3.0)*cum + CType(4.5)*cum*cum - uu)
                src = src_index(x, y, z, cp[1], cp[2], cp[3], Nx, Ny, Nz)
                store_pair!(fi, n, src, i, feqp, feqm, Val(false), N)
            end
            @static if TEMPERATURE
                store_geq!(gi, n, x, y, z, T[n], ux, uy, uz, N, Nx, Ny, Nz, Val(false), CType)
            end
        end
    end
end

end

@static if SURFACE

@inline function average_neighbors_fluid(
    ρ, u, flags, x, y, z, c::NTuple{Q, SVector{3, Int}},
    Nx, Ny, Nz, ::Type{CType}
) where {Q, CType}
    ρt = zero(CType); uxt = zero(CType); uyt = zero(CType); uzt = zero(CType)
    cnt = zero(CType)

    for i in 2:Q
        src = src_index(x, y, z, c[i][1], c[i][2], c[i][3], Nx, Ny, Nz)
        if (flags[src] & TYPE_SU) == TYPE_F
            cnt += one(CType)
            ρt += ρ[src]
            uxt += u[src, 1]; uyt += u[src, 2]; uzt += u[src, 3]
        end
    end
    if cnt > 0
        return ρt/cnt, uxt/cnt, uyt/cnt, uzt/cnt
    else
        return one(CType), zero(CType), zero(CType), zero(CType)
    end
end

@inline function average_neighbors_non_gas(
    ρ, u, flags, x, y, z, c::NTuple{Q, SVector{3,Int}}, 
    Nx, Ny, Nz, ::Type{CType}
) where {Q, CType}
    ρt = zero(CType); uxt = zero(CType); uyt = zero(CType); uzt = zero(CType)
    cnt = zero(CType)

    for i in 2:Q
        src = src_index(x, y, z, c[i][1], c[i][2], c[i][3], Nx, Ny, Nz)
        su = flags[src] & (TYPE_SU | TYPE_S)
        if su == TYPE_F || su == TYPE_I || su == TYPE_IF
            cnt += one(CType)
            ρt += ρ[src]
            uxt += u[src, 1]; uyt += u[src, 2]; uzt += u[src, 3]
        end
    end
    if cnt > 0
        return ρt/cnt, uxt/cnt, uyt/cnt, uzt/cnt
    else
        return one(CType), zero(CType), zero(CType), zero(CType)
    end
end

@inline function average_neighbors_T(
    Tfield, flags, x, y, z, c::NTuple{Q, SVector{3, Int}},
    Nx, Ny, Nz, ::Type{CType}
) where {Q, CType}
    s = zero(CType)
    cnt = 0
    for i in 2:Q
        src = src_index(x, y, z, c[i][1], c[i][2], c[i][3], Nx, Ny, Nz)
        su = flags[src] & (TYPE_SU | TYPE_S)
        if su == TYPE_F || su == TYPE_I || su == TYPE_IF
            s += Tfield[src]
            cnt += 1
        end
    end
    return cnt > 0 ? s / CType(cnt) : one(CType)
end

@inline function average_neighbors_fs(
    fs, flags, x, y, z, c::NTuple{Q, SVector{3, Int}},
    Nx, Ny, Nz, ::Type{CType}
) where {Q, CType}
    s = zero(CType)
    cnt = 0
    for i in 2:Q
        src = src_index(x, y, z, c[i][1], c[i][2], c[i][3], Nx, Ny, Nz)
        su = flags[src] & (TYPE_SU | TYPE_S)
        if su == TYPE_F || su == TYPE_I || su == TYPE_IF
            s += fs[src]
            cnt += 1
        end
    end
    return cnt > 0 ? s / CType(cnt) : zero(CType)
end

@inline function initialize_body!(
    ρ, u, fi, flags, mass, massex, ϕ, gi, T, fs,
    w::NTuple{Q, CType},
    c::NTuple{Q, SVector{3, Int}},
    N::Int, Nx::Int, Ny::Int, Nz::Int, n
) where{Q, CType}
    n0 = n - 1
    x = n0 % Nx
    y = (n0 ÷ Nx) % Ny
    z = n0 ÷ (Nx * Ny)

    flagsn = flags[n]
    ρn = ρ[n]
    ux, uy, uz = u[n, 1], u[n, 2], u[n, 3]
    ϕn = ϕ[n]

    flagsj = ntuple(Val(Q - 1)) do k
        i = k + 1
        src = src_index(x, y, z, c[i][1], c[i][2], c[i][3], Nx, Ny, Nz)
        flags[src]
    end

    if (flagsn & (TYPE_S | TYPE_E | TYPE_T | TYPE_H | TYPE_F | TYPE_I)) == 0x00
        flagsn = (flagsn & ~TYPE_SU) | TYPE_G
    end

    if (flagsn & TYPE_SU) == TYPE_G
        to_interface = false
        for k in 1:(Q - 1)
            to_interface |= ((flagsj[k] & TYPE_SU) == TYPE_F)
        end
        if to_interface
            flagsn = (flagsn & ~TYPE_SU) | TYPE_I
            ϕn = CType(0.5)
            ρn, ux, uy, uz = average_neighbors_fluid(ρ, u, flags, x, y, z, c, Nx, Ny, Nz, CType)
            ρ[n] = ρn
            u[n, 1] = ux; u[n, 2] = uy; u[n, 3] = uz
            @static if TEMPERATURE
                fs[n] = average_neighbors_fs(fs, flags, x, y, z, c, Nx, Ny, Nz, CType)
            end
        end
    end

    if (flagsn & TYPE_BO) == TYPE_S
        @static if MOVING_BOUNDARIES
            only_s = true
            for k in 1:(Q - 1)
                only_s &= (flagsj[k] & TYPE_BO) == TYPE_S
            end
            if only_s
                u[n, 1] = zero(CType); u[n, 2] = zero(CType); u[n, 3] = zero(CType)
            end
        else
            u[n, 1] = zero(CType); u[n, 2] = zero(CType); u[n, 3] = zero(CType)
        end
        @static if TEMPERATURE
            store_geq_local!(gi, n, T[n], N)
        end
    elseif (flagsn & TYPE_SU) == TYPE_G
        u[n, 1] = zero(CType); u[n, 2] = zero(CType); u[n, 3] = zero(CType)
        ϕn = zero(CType)
    else
        if (flagsn & TYPE_SU) == TYPE_I && (ϕn < 0 || ϕn > 1)
            ϕn = CType(0.5)
        elseif (flagsn & TYPE_SU) == TYPE_F
            ϕn = one(CType)
        end
        store_feq!(fi, n, x, y, z, ρn, ux, uy, uz, w, c, N, Nx, Ny, Nz, Val(false))
        @static if TEMPERATURE
            store_geq!(gi, n, x, y, z, T[n], ux, uy, uz, N, Nx, Ny, Nz, Val(false), CType)
        end
    end

    @static if TEMPERATURE
        if (flagsn & TYPE_SU) == TYPE_G
            store_geq!(gi, n, x, y, z, T[n], zero(CType), zero(CType), zero(CType),
                       N, Nx, Ny, Nz, Val(false), CType)
        end
    end

    ϕ[n] = ϕn
    mass[n] = ϕn * ρ[n]
    massex[n] = zero(CType)
    flags[n] = flagsn
    return nothing
end

@kernel function initialize_kernel!(
    ρ, u, fi, flags,
    mass, massex, ϕ,
    w::NTuple{Q, CType},
    c::NTuple{Q, SVector{3, Int}},
    N::Int, Nx::Int, Ny::Int, Nz::Int,
    gi, T, fs
) where {Q, CType}
    n = @index(Global)
    @inbounds initialize_body!(ρ, u, fi, flags, mass, massex, ϕ, gi, T, fs, w, c, N, Nx, Ny, Nz, Int(n))
end

end # SURFACE

@static if !SURFACE

# generic fallback
@inline function stream_collide_body!(
    t_odd::Val{odd},
    flags, fi, ρ, u, F, gi, T, Qin, hT, fs,
    w::NTuple{Q, CType}, 
    c::NTuple{Q, SVector{3, Int}},
    ω::CType, fx::CType, fy::CType, fz::CType,
    ω_T::CType, β::CType, T_avg::CType, Λ::CType, Ts::CType, Tl::CType, K0::CType,
    α_s::CType, α_l::CType, α_sT::CType, α_lT::CType, γ_s::CType, γ_l::CType, ν_s::CType, ν_l::CType, ν_sT::CType, ν_lT::CType,
    Λ_v::CType, T_v::CType, C_hk::CType, p0v::CType, β_v::CType,
    C_rad::CType, T_rad::CType,
    N::Int, Nx::Int, Ny::Int, Nz::Int, n, Eacc
) where {odd, Q, CType}
    flagsn = flags[n]
    if (flagsn & TYPE_BO) == TYPE_S
        return nothing
    end
    n0 = n - 1
    x = n0 % Nx
    y = (n0 ÷ Nx) % Ny
    z = n0 ÷ (Nx * Ny)
    @static if EQUILIBRIUM_BOUNDARIES
        if (flagsn & TYPE_BO) == TYPE_E
            fxn, fyn, fzn = fx, fy, fz
            @static if FORCE_FIELD
                fxn += F[n, 1]; fyn += F[n, 2]; fzn += F[n, 3]
            end
            equilibrium_boundary!(t_odd, fi, ρ, u, w, c, fxn, fyn, fzn, N, Nx, Ny, Nz, n, x, y, z, CType)
            return nothing
        end
    end
    fn1 = CType(fi[f_index(n, 1, N)])
        NP  = (Q - 1) ÷ 2

        # pairs: (2, 3), (4, 5), ...
        pairs = ntuple(Val(NP)) do k
            i = 2k
            cp, cm = c[i], c[i + 1]
            srcp = src_index(x, y, z, cp[1], cp[2], cp[3], Nx, Ny, Nz)
            srcm = src_index(x, y, z, cm[1], cm[2], cm[3], Nx, Ny, Nz)
            fp, fm = load_pair(fi, n, srcp, i, t_odd, N, CType)
            moving_wall_pair(fp, fm, flags, u, srcp, srcm, w[i], cp, cm, flagsn, CType)
        end

        ρn = fn1
        ux = uy = uz = zero(CType)

        for k in 1:NP
            i = 2k
            fp, fm = pairs[k]
            ρn += fp + fm
            cp, cm = c[i], c[i + 1]
            ux += CType(cp[1]) * fp + CType(cm[1]) * fm
            uy += CType(cp[2]) * fp + CType(cm[2]) * fm
            uz += CType(cp[3]) * fp + CType(cm[3]) * fm
        end

        cs = CType(1) / sqrt(CType(3))
        fxn = fx; fyn = fy; fzn = fz
        @static if FORCE_FIELD
            fxn += F[n, 1]; fyn += F[n, 2]; fzn += F[n, 3]
        end
        if ρn <= zero(CType)
            ρn = one(CType)
            ux = zero(CType); uy = zero(CType); uz = zero(CType)
            fxn = zero(CType); fyn = zero(CType); fzn = zero(CType)
        else
            invρ = one(CType) / ρn
            ux *= invρ; uy *= invρ; uz *= invρ
            @static if TEMPERATURE
                ωTn = omega_T_from_alpha(prop_fs_T(fs[n], α_s, α_sT, α_l, α_lT, T[n], T_avg, CType(1e-6)))
                fxn, fyn, fzn, _ = collide_temperature!(
                    t_odd, gi, T, Qin, hT, flags, flagsn, fs, ux, uy, uz,
                    fxn, fyn, fzn, fx, fy, fz,
                    ωTn, β, T_avg, Λ, Ts, Tl, γ_s, γ_l, Λ_v, T_v, C_hk, p0v, β_v, C_rad, T_rad, x, y, z, Nx, Ny, Nz, N, n, CType, Eacc, one(CType))
            end
            @static if TEMPERATURE
                νc = prop_fs_T(fs[n], ν_s, ν_sT, ν_l, ν_lT, T[n], T_avg, CType(1e-8))
                ω = omega_from_nu(νc)
                dx, dy, dz = darcy_force(fs[n], ux, uy, uz, ρn, νc, K0)
                if K0 > zero(CType) && (one(CType) - fs[n]) < CType(1e-3)
                    fxn, fyn, fzn = dx, dy, dz
                else
                    fxn += dx; fyn += dy; fzn += dz
                end
            end
            @static if APPLY_FORCE
                ux += fxn * invρ * CType(0.5)
                uy += fyn * invρ * CType(0.5)
                uz += fzn * invρ * CType(0.5)
            end
            ux = clamp(ux, -cs, cs)
            uy = clamp(uy, -cs, cs)
            uz = clamp(uz, -cs, cs)
        end

        uu = CType(1.5) * (ux*ux + uy*uy + uz*uz)
        @static if TRT
            ωm = omega_minus(ω)
        else
            ωm = ω
        end

        Fi0 = zero(CType)
        @static if APPLY_FORCE
            Fi0 = guo_rest(ω, w[1], ux, uy, uz, fxn, fyn, fzn, c[1], CType)
        end
        fi[f_index(n, 1, N)] = eltype(fi)(
            (one(CType) - ω) * fn1 + ω * (w[1] * ρn * (one(CType) - uu)) + Fi0)

        for k in 1:NP
            i = 2k
            fp, fm = pairs[k]
            cp, cm = c[i], c[i + 1]
            cup = CType(cp[1])*ux + CType(cp[2])*uy + CType(cp[3])*uz
            cum = CType(cm[1])*ux + CType(cm[2])*uy + CType(cm[3])*uz
            feqp = w[i]     * ρn * (one(CType) + CType(3.0)*cup + CType(4.5)*cup*cup - uu)
            feqm = w[i + 1] * ρn * (one(CType) + CType(3.0)*cum + CType(4.5)*cum*cum - uu)
            fp_s, fm_s = collide_pair(ω, ωm, fp, fm, feqp, feqm)
            @static if APPLY_FORCE
                Fip, Fim = guo_pair(ω, ωm, w[i], w[i + 1], ux, uy, uz, fxn, fyn, fzn, cp, cm, CType)
                fp_s += Fip
                fm_s += Fim
            end
            src = src_index(x, y, z, cp[1], cp[2], cp[3], Nx, Ny, Nz)
            store_pair!(fi, n, src, i, fp_s, fm_s, t_odd, N)
        end
    return nothing
end

@inline function stream_collide_body!(
    t_odd::Val{odd},
    flags, fi, ρ, u, F, gi, T, Qin, hT, fs,
    w::NTuple{19, CType},
    c::NTuple{19, SVector{3,Int}},
    ω::CType, fx::CType, fy::CType, fz::CType,
    ω_T::CType, β::CType, T_avg::CType, Λ::CType, Ts::CType, Tl::CType, K0::CType,
    α_s::CType, α_l::CType, α_sT::CType, α_lT::CType, γ_s::CType, γ_l::CType, ν_s::CType, ν_l::CType, ν_sT::CType, ν_lT::CType,
    Λ_v::CType, T_v::CType, C_hk::CType, p0v::CType, β_v::CType,
    C_rad::CType, T_rad::CType,
    N::Int, Nx::Int, Ny::Int, Nz::Int, n, Eacc
) where {odd, CType}
    flagsn = flags[n]
    if (flagsn & TYPE_BO) == TYPE_S
        return nothing
    end

    n0 = n - 1
    x  = n0 % Nx
    y  = (n0 ÷ Nx) % Ny
    z  = n0 ÷ (Nx * Ny)

    @static if EQUILIBRIUM_BOUNDARIES
        if (flagsn & TYPE_BO) == TYPE_E
            fxn, fyn, fzn = fx, fy, fz
            @static if FORCE_FIELD
                fxn += F[n, 1]; fyn += F[n, 2]; fzn += F[n, 3]
            end
            equilibrium_boundary!(t_odd, fi, ρ, u, w, c, fxn, fyn, fzn, N, Nx, Ny, Nz, n, x, y, z, CType)
            return nothing
        end
    end

    fn1 = CType(fi[f_index(n, 1, N)])

    src2  = src_index(x, y, z, c[2][1],  c[2][2],  c[2][3],  Nx, Ny, Nz)
    src4  = src_index(x, y, z, c[4][1],  c[4][2],  c[4][3],  Nx, Ny, Nz)
    src6  = src_index(x, y, z, c[6][1],  c[6][2],  c[6][3],  Nx, Ny, Nz)
    src8  = src_index(x, y, z, c[8][1],  c[8][2],  c[8][3],  Nx, Ny, Nz)
    src10 = src_index(x, y, z, c[10][1], c[10][2], c[10][3], Nx, Ny, Nz)
    src12 = src_index(x, y, z, c[12][1], c[12][2], c[12][3], Nx, Ny, Nz)
    src14 = src_index(x, y, z, c[14][1], c[14][2], c[14][3], Nx, Ny, Nz)
    src16 = src_index(x, y, z, c[16][1], c[16][2], c[16][3], Nx, Ny, Nz)
    src18 = src_index(x, y, z, c[18][1], c[18][2], c[18][3], Nx, Ny, Nz)

    fp2,  fm3  = load_pair(fi, n, src2,  2,  t_odd, N, CType)
    fp4,  fm5  = load_pair(fi, n, src4,  4,  t_odd, N, CType)
    fp6,  fm7  = load_pair(fi, n, src6,  6,  t_odd, N, CType)
    fp8,  fm9  = load_pair(fi, n, src8,  8,  t_odd, N, CType)
    fp10, fm11 = load_pair(fi, n, src10, 10, t_odd, N, CType)
    fp12, fm13 = load_pair(fi, n, src12, 12, t_odd, N, CType)
    fp14, fm15 = load_pair(fi, n, src14, 14, t_odd, N, CType)
    fp16, fm17 = load_pair(fi, n, src16, 16, t_odd, N, CType)
    fp18, fm19 = load_pair(fi, n, src18, 18, t_odd, N, CType)

    @static if MOVING_BOUNDARIES
        src3  = src_index(x, y, z, c[3][1],  c[3][2],  c[3][3],  Nx, Ny, Nz)
        src5  = src_index(x, y, z, c[5][1],  c[5][2],  c[5][3],  Nx, Ny, Nz)
        src7  = src_index(x, y, z, c[7][1],  c[7][2],  c[7][3],  Nx, Ny, Nz)
        src9  = src_index(x, y, z, c[9][1],  c[9][2],  c[9][3],  Nx, Ny, Nz)
        src11 = src_index(x, y, z, c[11][1], c[11][2], c[11][3], Nx, Ny, Nz)
        src13 = src_index(x, y, z, c[13][1], c[13][2], c[13][3], Nx, Ny, Nz)
        src15 = src_index(x, y, z, c[15][1], c[15][2], c[15][3], Nx, Ny, Nz)
        src17 = src_index(x, y, z, c[17][1], c[17][2], c[17][3], Nx, Ny, Nz)
        src19 = src_index(x, y, z, c[19][1], c[19][2], c[19][3], Nx, Ny, Nz)
        fp2,  fm3  = moving_wall_pair(fp2,  fm3,  flags, u, src2,  src3,  w[2],  c[2],  c[3],  flagsn, CType)
        fp4,  fm5  = moving_wall_pair(fp4,  fm5,  flags, u, src4,  src5,  w[4],  c[4],  c[5],  flagsn, CType)
        fp6,  fm7  = moving_wall_pair(fp6,  fm7,  flags, u, src6,  src7,  w[6],  c[6],  c[7],  flagsn, CType)
        fp8,  fm9  = moving_wall_pair(fp8,  fm9,  flags, u, src8,  src9,  w[8],  c[8],  c[9],  flagsn, CType)
        fp10, fm11 = moving_wall_pair(fp10, fm11, flags, u, src10, src11, w[10], c[10], c[11], flagsn, CType)
        fp12, fm13 = moving_wall_pair(fp12, fm13, flags, u, src12, src13, w[12], c[12], c[13], flagsn, CType)
        fp14, fm15 = moving_wall_pair(fp14, fm15, flags, u, src14, src15, w[14], c[14], c[15], flagsn, CType)
        fp16, fm17 = moving_wall_pair(fp16, fm17, flags, u, src16, src17, w[16], c[16], c[17], flagsn, CType)
        fp18, fm19 = moving_wall_pair(fp18, fm19, flags, u, src18, src19, w[18], c[18], c[19], flagsn, CType)
    end

    ρn = fn1 + fp2 + fm3 + fp4 + fm5 + fp6 + fm7 + fp8 + fm9 +
         fp10 + fm11 + fp12 + fm13 + fp14 + fm15 + fp16 + fm17 + fp18 + fm19
    ux = CType(c[2][1])*fp2 + CType(c[3][1])*fm3 + CType(c[4][1])*fp4 + CType(c[5][1])*fm5 +
         CType(c[6][1])*fp6 + CType(c[7][1])*fm7 + CType(c[8][1])*fp8 + CType(c[9][1])*fm9 +
         CType(c[10][1])*fp10 + CType(c[11][1])*fm11 + CType(c[12][1])*fp12 + CType(c[13][1])*fm13 +
         CType(c[14][1])*fp14 + CType(c[15][1])*fm15 + CType(c[16][1])*fp16 + CType(c[17][1])*fm17 +
         CType(c[18][1])*fp18 + CType(c[19][1])*fm19
    uy = CType(c[2][2])*fp2 + CType(c[3][2])*fm3 + CType(c[4][2])*fp4 + CType(c[5][2])*fm5 +
         CType(c[6][2])*fp6 + CType(c[7][2])*fm7 + CType(c[8][2])*fp8 + CType(c[9][2])*fm9 +
         CType(c[10][2])*fp10 + CType(c[11][2])*fm11 + CType(c[12][2])*fp12 + CType(c[13][2])*fm13 +
         CType(c[14][2])*fp14 + CType(c[15][2])*fm15 + CType(c[16][2])*fp16 + CType(c[17][2])*fm17 +
         CType(c[18][2])*fp18 + CType(c[19][2])*fm19
    uz = CType(c[2][3])*fp2 + CType(c[3][3])*fm3 + CType(c[4][3])*fp4 + CType(c[5][3])*fm5 +
         CType(c[6][3])*fp6 + CType(c[7][3])*fm7 + CType(c[8][3])*fp8 + CType(c[9][3])*fm9 +
         CType(c[10][3])*fp10 + CType(c[11][3])*fm11 + CType(c[12][3])*fp12 + CType(c[13][3])*fm13 +
         CType(c[14][3])*fp14 + CType(c[15][3])*fm15 + CType(c[16][3])*fp16 + CType(c[17][3])*fm17 +
         CType(c[18][3])*fp18 + CType(c[19][3])*fm19

    cs = CType(1) / sqrt(CType(3))
    fxn = fx; fyn = fy; fzn = fz
    @static if FORCE_FIELD
        fxn += F[n, 1]; fyn += F[n, 2]; fzn += F[n, 3]
    end
    if ρn <= zero(CType)
        ρn = one(CType)
        ux = zero(CType); uy = zero(CType); uz = zero(CType)
        fxn = zero(CType); fyn = zero(CType); fzn = zero(CType)
    else
        invρ = one(CType) / ρn
        ux *= invρ; uy *= invρ; uz *= invρ
        @static if TEMPERATURE
            ωTn = omega_T_from_alpha(prop_fs_T(fs[n], α_s, α_sT, α_l, α_lT, T[n], T_avg, CType(1e-6)))
            fxn, fyn, fzn, _ = collide_temperature!(
                t_odd, gi, T, Qin, hT, flags, flagsn, fs, ux, uy, uz,
                fxn, fyn, fzn, fx, fy, fz,
                ωTn, β, T_avg, Λ, Ts, Tl, γ_s, γ_l, Λ_v, T_v, C_hk, p0v, β_v, C_rad, T_rad, x, y, z, Nx, Ny, Nz, N, n, CType, Eacc, one(CType))
        end
        @static if TEMPERATURE
            νc = prop_fs_T(fs[n], ν_s, ν_sT, ν_l, ν_lT, T[n], T_avg, CType(1e-8))
            ω = omega_from_nu(νc)
            dx, dy, dz = darcy_force(fs[n], ux, uy, uz, ρn, νc, K0)
            if K0 > zero(CType) && (one(CType) - fs[n]) < CType(1e-3)
                fxn, fyn, fzn = dx, dy, dz
            else
                fxn += dx; fyn += dy; fzn += dz
            end
        end
        @static if APPLY_FORCE
            ux += fxn * invρ * CType(0.5)
            uy += fyn * invρ * CType(0.5)
            uz += fzn * invρ * CType(0.5)
        end
        ux = clamp(ux, -cs, cs)
        uy = clamp(uy, -cs, cs)
        uz = clamp(uz, -cs, cs)
    end
    uu = CType(1.5) * (ux*ux + uy*uy + uz*uz)

    @static if TRT
        ωm = omega_minus(ω)
    else
        ωm = ω
    end

    Fi0 = zero(CType)
    @static if APPLY_FORCE
        Fi0 = guo_rest(ω, w[1], ux, uy, uz, fxn, fyn, fzn, c[1], CType)
    end
    fi[f_index(n, 1, N)] = eltype(fi)(
        (one(CType) - ω) * fn1 + ω * (w[1] * ρn * (one(CType) - uu)) + Fi0)


    let feqp = feq(w[2], ρn, ux, uy, uz, uu, c[2], CType)
        feqm = feq(w[3], ρn, ux, uy, uz, uu, c[3], CType)
        fp_s, fm_s = collide_pair(ω, ωm, fp2, fm3, feqp, feqm)
        @static if APPLY_FORCE
            Fip, Fim = guo_pair(ω, ωm, w[2], w[3], ux, uy, uz, fxn, fyn, fzn, c[2], c[3], CType)
            fp_s += Fip
            fm_s += Fim
        end
        store_pair!(fi, n, src2, 2, fp_s, fm_s, t_odd, N)
    end
    let feqp = feq(w[4], ρn, ux, uy, uz, uu, c[4], CType)
        feqm = feq(w[5], ρn, ux, uy, uz, uu, c[5], CType)
        fp_s, fm_s = collide_pair(ω, ωm, fp4, fm5, feqp, feqm)
        @static if APPLY_FORCE
            Fip, Fim = guo_pair(ω, ωm, w[4], w[5], ux, uy, uz, fxn, fyn, fzn, c[4], c[5], CType)
            fp_s += Fip
            fm_s += Fim
        end
        store_pair!(fi, n, src4, 4, fp_s, fm_s, t_odd, N)
    end
    let feqp = feq(w[6], ρn, ux, uy, uz, uu, c[6], CType)
        feqm = feq(w[7], ρn, ux, uy, uz, uu, c[7], CType)
        fp_s, fm_s = collide_pair(ω, ωm, fp6, fm7, feqp, feqm)
        @static if APPLY_FORCE
            Fip, Fim = guo_pair(ω, ωm, w[6], w[7], ux, uy, uz, fxn, fyn, fzn, c[6], c[7], CType)
            fp_s += Fip
            fm_s += Fim
        end
        store_pair!(fi, n, src6, 6, fp_s, fm_s, t_odd, N)
    end
    let feqp = feq(w[8], ρn, ux, uy, uz, uu, c[8], CType)
        feqm = feq(w[9], ρn, ux, uy, uz, uu, c[9], CType)
        fp_s, fm_s = collide_pair(ω, ωm, fp8, fm9, feqp, feqm)
        @static if APPLY_FORCE
            Fip, Fim = guo_pair(ω, ωm, w[8], w[9], ux, uy, uz, fxn, fyn, fzn, c[8], c[9], CType)
            fp_s += Fip
            fm_s += Fim
        end
        store_pair!(fi, n, src8, 8, fp_s, fm_s, t_odd, N)
    end
    let feqp = feq(w[10], ρn, ux, uy, uz, uu, c[10], CType)
        feqm = feq(w[11], ρn, ux, uy, uz, uu, c[11], CType)
        fp_s, fm_s = collide_pair(ω, ωm, fp10, fm11, feqp, feqm)
        @static if APPLY_FORCE
            Fip, Fim = guo_pair(ω, ωm, w[10], w[11], ux, uy, uz, fxn, fyn, fzn, c[10], c[11], CType)
            fp_s += Fip
            fm_s += Fim
        end
        store_pair!(fi, n, src10, 10, fp_s, fm_s, t_odd, N)
    end
    let feqp = feq(w[12], ρn, ux, uy, uz, uu, c[12], CType)
        feqm = feq(w[13], ρn, ux, uy, uz, uu, c[13], CType)
        fp_s, fm_s = collide_pair(ω, ωm, fp12, fm13, feqp, feqm)
        @static if APPLY_FORCE
            Fip, Fim = guo_pair(ω, ωm, w[12], w[13], ux, uy, uz, fxn, fyn, fzn, c[12], c[13], CType)
            fp_s += Fip
            fm_s += Fim
        end
        store_pair!(fi, n, src12, 12, fp_s, fm_s, t_odd, N)
    end
    let feqp = feq(w[14], ρn, ux, uy, uz, uu, c[14], CType)
        feqm = feq(w[15], ρn, ux, uy, uz, uu, c[15], CType)
        fp_s, fm_s = collide_pair(ω, ωm, fp14, fm15, feqp, feqm)
        @static if APPLY_FORCE
            Fip, Fim = guo_pair(ω, ωm, w[14], w[15], ux, uy, uz, fxn, fyn, fzn, c[14], c[15], CType)
            fp_s += Fip
            fm_s += Fim
        end
        store_pair!(fi, n, src14, 14, fp_s, fm_s, t_odd, N)
    end
    let feqp = feq(w[16], ρn, ux, uy, uz, uu, c[16], CType)
        feqm = feq(w[17], ρn, ux, uy, uz, uu, c[17], CType)
        fp_s, fm_s = collide_pair(ω, ωm, fp16, fm17, feqp, feqm)
        @static if APPLY_FORCE
            Fip, Fim = guo_pair(ω, ωm, w[16], w[17], ux, uy, uz, fxn, fyn, fzn, c[16], c[17], CType)
            fp_s += Fip
            fm_s += Fim
        end
        store_pair!(fi, n, src16, 16, fp_s, fm_s, t_odd, N)
    end
    let feqp = feq(w[18], ρn, ux, uy, uz, uu, c[18], CType)
        feqm = feq(w[19], ρn, ux, uy, uz, uu, c[19], CType)
        fp_s, fm_s = collide_pair(ω, ωm, fp18, fm19, feqp, feqm)
        @static if APPLY_FORCE
            Fip, Fim = guo_pair(ω, ωm, w[18], w[19], ux, uy, uz, fxn, fyn, fzn, c[18], c[19], CType)
            fp_s += Fip
            fm_s += Fim
        end
        store_pair!(fi, n, src18, 18, fp_s, fm_s, t_odd, N)
    end

    return nothing
end

@kernel function stream_collide_even_kernel!(
    @Const(flags), fi, ρ, u, F, gi, T, Qin, hT, fs,
    w::NTuple{Q, CType}, 
    c::NTuple{Q, SVector{3, Int}},
    ω::CType, fx::CType, fy::CType, fz::CType,
    ω_T::CType, β::CType, T_avg::CType, Λ::CType, Ts::CType, Tl::CType, K0::CType,
    α_s::CType, α_l::CType, α_sT::CType, α_lT::CType, γ_s::CType, γ_l::CType, ν_s::CType, ν_l::CType, ν_sT::CType, ν_lT::CType,
    Λ_v::CType, T_v::CType, C_hk::CType, p0v::CType, β_v::CType,
    C_rad::CType, T_rad::CType,
    N::Int, Nx::Int, Ny::Int, Nz::Int, Eacc
) where {Q, CType}
    n = @index(Global)
    @inbounds stream_collide_body!(Val(false), flags, fi, ρ, u, F, gi, T, Qin, hT, fs, w, c, ω, fx, fy, fz, ω_T, β, T_avg, Λ, Ts, Tl, K0, α_s, α_l, α_sT, α_lT, γ_s, γ_l, ν_s, ν_l, ν_sT, ν_lT, Λ_v, T_v, C_hk, p0v, β_v, C_rad, T_rad, N, Nx, Ny, Nz, Int(n), Eacc)
end

@kernel function stream_collide_odd_kernel!(
    @Const(flags), fi, ρ, u, F, gi, T, Qin, hT, fs,
    w::NTuple{Q, CType}, 
    c::NTuple{Q, SVector{3, Int}},
    ω::CType, fx::CType, fy::CType, fz::CType,
    ω_T::CType, β::CType, T_avg::CType, Λ::CType, Ts::CType, Tl::CType, K0::CType,
    α_s::CType, α_l::CType, α_sT::CType, α_lT::CType, γ_s::CType, γ_l::CType, ν_s::CType, ν_l::CType, ν_sT::CType, ν_lT::CType,
    Λ_v::CType, T_v::CType, C_hk::CType, p0v::CType, β_v::CType,
    C_rad::CType, T_rad::CType,
    N::Int, Nx::Int, Ny::Int, Nz::Int, Eacc
) where {Q, CType}
    n = @index(Global)
    @inbounds stream_collide_body!(Val(true), flags, fi, ρ, u, F, gi, T, Qin, hT, fs, w, c, ω, fx, fy, fz, ω_T, β, T_avg, Λ, Ts, Tl, K0, α_s, α_l, α_sT, α_lT, γ_s, γ_l, ν_s, ν_l, ν_sT, ν_lT, Λ_v, T_v, C_hk, p0v, β_v, C_rad, T_rad, N, Nx, Ny, Nz, Int(n), Eacc)
end

end

@static if SURFACE

@inline function stream_collide_surface_body!(
    t_odd::Val{odd},
    flags, fi, ρ, u, F, mass, gi, T, Qin, hT, ϕ, fs, msrc, mp,
    w::NTuple{Q, CType},
    c::NTuple{Q, SVector{3, Int}},
    ω::CType, fx::CType, fy::CType, fz::CType,
    ω_T::CType, β::CType, T_avg::CType, σT::CType, Λ::CType, Ts::CType, Tl::CType, K0::CType,
    α_s::CType, α_l::CType, α_sT::CType, α_lT::CType, γ_s::CType, γ_l::CType, ν_s::CType, ν_l::CType, ν_sT::CType, ν_lT::CType,
    Λ_v::CType, T_v::CType, C_hk::CType, p0v::CType, β_v::CType,
    C_rad::CType, T_rad::CType, τ_p::CType, T_p::CType,
    N::Int, Nx::Int, Ny::Int, Nz::Int, n, Eacc, Macc
) where {odd, Q, CType}
    flagsn = flags[n]
    if (flagsn & TYPE_BO) == TYPE_S || (flagsn & TYPE_SU) == TYPE_G
        return nothing
    end

    n0 = n - 1
    x = n0 % Nx; y = (n0 ÷ Nx) % Ny; z = n0 ÷ (Nx * Ny)
    NP = (Q - 1) ÷ 2

    @static if EQUILIBRIUM_BOUNDARIES
        if (flagsn & TYPE_BO) == TYPE_E
            fxn, fyn, fzn = fx, fy, fz
            @static if FORCE_FIELD
                fxn += F[n, 1]; fyn += F[n, 2]; fzn += F[n, 3]
            end
            equilibrium_boundary!(t_odd, fi, ρ, u, w, c, fxn, fyn, fzn, N, Nx, Ny, Nz, n, x, y, z, CType)
            return nothing
        end
    end

    fn1 = CType(fi[f_index(n, 1, N)])
    pairs = ntuple(Val(NP)) do k
        i = 2k
        cp, cm = c[i], c[i + 1]
        srcp = src_index(x, y, z, cp[1], cp[2], cp[3], Nx, Ny, Nz)
        srcm = src_index(x, y, z, cm[1], cm[2], cm[3], Nx, Ny, Nz)
        fp, fm = load_pair(fi, n, srcp, i, t_odd, N, CType)
        moving_wall_pair(fp, fm, flags, u, srcp, srcm, w[i], cp, cm, flagsn, CType)
    end

    ρn = fn1
    ux = zero(CType); uy = zero(CType); uz = zero(CType)
    for k in 1:NP
        i = 2k
        fp, fm = pairs[k]
        ρn += fp + fm
        ux += CType(c[i][1])*fp + CType(c[i+1][1])*fm
        uy += CType(c[i][2])*fp + CType(c[i+1][2])*fm
        uz += CType(c[i][3])*fp + CType(c[i+1][3])*fm
    end

    cs = CType(1) / sqrt(CType(3))
    fxn = fx; fyn = fy; fzn = fz
    @static if FORCE_FIELD
        fxn += F[n, 1]; fyn += F[n, 2]; fzn += F[n, 3]
    end
    if ρn <= zero(CType)
        ρn = one(CType)
        ux = zero(CType); uy = zero(CType); uz = zero(CType)
        fxn = zero(CType); fyn = zero(CType); fzn = zero(CType)
    else
        invρ = one(CType) / ρn
        ux *= invρ; uy *= invρ; uz *= invρ
        @static if TEMPERATURE
            ωTn = omega_T_from_alpha(prop_fs_T(fs[n], α_s, α_sT, α_l, α_lT, T[n], T_avg, CType(1e-6)))
            debit = zero(CType)
            fillc = ϕ[n]
            fillc = ifelse(fillc > zero(CType), fillc, zero(CType))
            if τ_p > zero(CType)
                mp_src = msrc[n] * ρn
                mpn = mp[n] + mp_src
                acc_add!(Eacc, EACC_POWDER, mp_src * sensible_H(T_p, γ_s))
                acc_add!(Macc, MACC_POWDER, mp_src)
                Tpred = T[n] + Qin[n]
                if Tpred >= Ts
                    dm = mpn < ρn ? mpn : ρn
                    mpn -= dm
                    mass[n] += dm
                    debit = (dm / ρn) * (max(Tpred - T_p, zero(CType)) + Λ)
                    Qin[n] -= debit
                else
                    mpd = mpn * exp(-one(CType) / τ_p)
                    acc_add!(Eacc, EACC_POWDER, (mpd - mpn) * sensible_H(T_p, γ_s))
                    acc_add!(Macc, MACC_POWDER, mpd - mpn)
                    mpn = mpd
                end
                mp[n] = ifelse(mpn > CType(1e-12), mpn, zero(CType))
            else
                Sn = msrc[n]
                if is_solid_fraction(fs[n])
                    Sn = zero(CType)
                end
                mass[n] += Sn * ρn
                acc_add!(Eacc, EACC_POWDER, Sn * ρn * sensible_H(T_p, γ_s))
                acc_add!(Macc, MACC_POWDER, Sn * ρn)
            end
            fxn, fyn, fzn, mevap = collide_temperature!(
                t_odd, gi, T, Qin, hT, flags, flagsn, fs, ux, uy, uz,
                fxn, fyn, fzn, fx, fy, fz,
                ωTn, β, T_avg, Λ, Ts, Tl, γ_s, γ_l, Λ_v, T_v, C_hk, p0v, β_v, C_rad, T_rad, x, y, z, Nx, Ny, Nz, N, n, CType, Eacc, fillc)
            debit != zero(CType) && (Qin[n] += debit)
            if mevap > zero(CType)
                mass[n] -= mevap * ρn
                acc_add!(Macc, MACC_EVAP, mevap * ρn)
            end
            if (flagsn & TYPE_SU) == TYPE_I && !is_solid_fraction(fs[n])
                if σT != zero(CType)
                    mx, my, mz = marangoni_force(T, ϕ, flags, σT, x, y, z, n, Nx, Ny, Nz, CType)
                    fxn += mx; fyn += my; fzn += mz
                end
                if Λ_v > zero(CType)
                    rx, ry, rz = recoil_force(T, ϕ, n, x, y, z, Nx, Ny, Nz, Λ_v, T_v, p0v, β_v, CType)
                    fxn += rx; fyn += ry; fzn += rz
                end
            end
        end
        @static if TEMPERATURE
            νc = prop_fs_T(fs[n], ν_s, ν_sT, ν_l, ν_lT, T[n], T_avg, CType(1e-8))
            ω = omega_from_nu(νc)
            dx, dy, dz = darcy_force(fs[n], ux, uy, uz, ρn, νc, K0)
            if K0 > zero(CType) && (one(CType) - fs[n]) < CType(1e-3)
                fxn, fyn, fzn = dx, dy, dz
            else
                fxn += dx; fyn += dy; fzn += dz
            end
        end
        @static if APPLY_FORCE
            ux += fxn * invρ * CType(0.5)
            uy += fyn * invρ * CType(0.5)
            uz += fzn * invρ * CType(0.5)
        end
        ux = clamp(ux, -cs, cs)
        uy = clamp(uy, -cs, cs)
        uz = clamp(uz, -cs, cs)
    end

    @static if UPDATE_FIELDS
        ρ[n] = ρn
        u[n, 1] = ux; u[n, 2] = uy; u[n, 3] = uz
    end

    if (flagsn & TYPE_SU) == TYPE_I
        frozen_i = false
        @static if TEMPERATURE
            frozen_i = is_solid_fraction(fs[n])
        end
        if !frozen_i
            noF = true; noG = true
            for i in 2:Q
                src = src_index(x, y, z, c[i][1], c[i][2], c[i][3], Nx, Ny, Nz)
                suj = flags[src] & TYPE_SU
                noF &= suj != TYPE_F
                noG &= suj != TYPE_G
            end
            massn = mass[n]
            if massn > ρn || noG
                flags[n] = (flagsn & ~TYPE_SU) | TYPE_IF
            elseif massn < 0 || noF
                flags[n] = (flagsn & ~TYPE_SU) | TYPE_IG
            end
        end
    end

    uu = CType(1.5) * (ux*ux + uy*uy + uz*uz)
    @static if TRT
        ωm = omega_minus(ω)
    else
        ωm = ω
    end

    Fi0 = zero(CType)
    @static if APPLY_FORCE
        Fi0 = guo_rest(ω, w[1], ux, uy, uz, fxn, fyn, fzn, c[1], CType)
    end
    fi[f_index(n, 1, N)] = eltype(fi)(srt(ω, fn1, w[1], ρn, ux, uy, uz, uu, c[1]) + Fi0)

    for k in 1:NP
        i = 2k
        fp, fm = pairs[k]
        feqp = feq(w[i],     ρn, ux, uy, uz, uu, c[i],     CType)
        feqm = feq(w[i + 1], ρn, ux, uy, uz, uu, c[i + 1], CType)
        fp_s, fm_s = collide_pair(ω, ωm, fp, fm, feqp, feqm)
        @static if APPLY_FORCE
            Fip, Fim = guo_pair(ω, ωm, w[i], w[i + 1], ux, uy, uz, fxn, fyn, fzn, c[i], c[i + 1], CType)
            fp_s += Fip
            fm_s += Fim
        end
        src = src_index(x, y, z, c[i][1], c[i][2], c[i][3], Nx, Ny, Nz)
        store_pair!(fi, n, src, i, fp_s, fm_s, t_odd, N)
    end
    return nothing
end

@kernel function stream_collide_even_kernel!(
    flags, fi, ρ, u, F, mass, gi, T, Qin, hT, ϕ, fs, msrc, mp,
    w::NTuple{Q, CType}, c::NTuple{Q, SVector{3, Int}},
    ω::CType, fx::CType, fy::CType, fz::CType,
    ω_T::CType, β::CType, T_avg::CType, σT::CType, Λ::CType, Ts::CType, Tl::CType, K0::CType,
    α_s::CType, α_l::CType, α_sT::CType, α_lT::CType, γ_s::CType, γ_l::CType, ν_s::CType, ν_l::CType, ν_sT::CType, ν_lT::CType,
    Λ_v::CType, T_v::CType, C_hk::CType, p0v::CType, β_v::CType,
    C_rad::CType, T_rad::CType, τ_p::CType, T_p::CType,
    N::Int, Nx::Int, Ny::Int, Nz::Int, Eacc, Macc
) where {Q, CType}
    n = @index(Global)
    @inbounds stream_collide_surface_body!(Val(false), flags, fi, ρ, u, F, mass, gi, T, Qin, hT, ϕ, fs, msrc, mp, w, c, ω, fx, fy, fz, ω_T, β, T_avg, σT, Λ, Ts, Tl, K0, α_s, α_l, α_sT, α_lT, γ_s, γ_l, ν_s, ν_l, ν_sT, ν_lT, Λ_v, T_v, C_hk, p0v, β_v, C_rad, T_rad, τ_p, T_p, N, Nx, Ny, Nz, Int(n), Eacc, Macc)
end

@kernel function stream_collide_odd_kernel!(
    flags, fi, ρ, u, F, mass, gi, T, Qin, hT, ϕ, fs, msrc, mp,
    w::NTuple{Q, CType}, c::NTuple{Q, SVector{3, Int}},
    ω::CType, fx::CType, fy::CType, fz::CType,
    ω_T::CType, β::CType, T_avg::CType, σT::CType, Λ::CType, Ts::CType, Tl::CType, K0::CType,
    α_s::CType, α_l::CType, α_sT::CType, α_lT::CType, γ_s::CType, γ_l::CType, ν_s::CType, ν_l::CType, ν_sT::CType, ν_lT::CType,
    Λ_v::CType, T_v::CType, C_hk::CType, p0v::CType, β_v::CType,
    C_rad::CType, T_rad::CType, τ_p::CType, T_p::CType,
    N::Int, Nx::Int, Ny::Int, Nz::Int, Eacc, Macc
) where {Q, CType}
    n = @index(Global)
    @inbounds stream_collide_surface_body!(Val(true), flags, fi, ρ, u, F, mass, gi, T, Qin, hT, ϕ, fs, msrc, mp, w, c, ω, fx, fy, fz, ω_T, β, T_avg, σT, Λ, Ts, Tl, K0, α_s, α_l, α_sT, α_lT, γ_s, γ_l, ν_s, ν_l, ν_sT, ν_lT, Λ_v, T_v, C_hk, p0v, β_v, C_rad, T_rad, τ_p, T_p, N, Nx, Ny, Nz, Int(n), Eacc, Macc)
end

# Loose powder on TYPE_G: feed + decay. Never becomes metal (no hydro DDF).
@kernel function powder_gas_kernel!(
    flags, mp, msrc, ρ, τ_p::CType, T_p::CType, γ_s::CType, Eacc, Macc, N::Int
) where {CType}
    n = @index(Global)
    @inbounds begin
        if τ_p > zero(CType)
            fl = flags[n]
            if (fl & TYPE_BO) != TYPE_S && (fl & TYPE_SU) == TYPE_G
                ρn = ρ[n]
                ρn = ifelse(ρn > zero(CType), ρn, one(CType))
                mp_src = msrc[n] * ρn
                mpn = mp[n] + mp_src
                acc_add!(Eacc, EACC_POWDER, mp_src * sensible_H(T_p, γ_s))
                acc_add!(Macc, MACC_POWDER, mp_src)
                mpd = mpn * exp(-one(CType) / τ_p)
                acc_add!(Eacc, EACC_POWDER, (mpd - mpn) * sensible_H(T_p, γ_s))
                acc_add!(Macc, MACC_POWDER, mpd - mpn)
                mp[n] = ifelse(mpd > CType(1e-12), mpd, zero(CType))
            end
        end
    end
end

end

@inline function moments_body!(
    t_odd::Val{odd},
    ρ, u, flags, fi, gi, T,
    w::NTuple{Q, CType},
    c::NTuple{Q, SVector{3, Int}},
    N::Int, Nx::Int, Ny::Int, Nz::Int, n
) where {odd, Q, CType}
    flagsn = flags[n]
    su = flagsn & TYPE_SU

    skip = (flagsn & TYPE_BO) == TYPE_S
    @static if SURFACE
        skip |= (su == TYPE_G) | (su == TYPE_IG)
    end
    @static if EQUILIBRIUM_BOUNDARIES
        if (flagsn & TYPE_BO) == TYPE_E
            return nothing
        end
    end
    if skip
        if (flagsn & TYPE_BO) == TYPE_S
            @static if !MOVING_BOUNDARIES
                ρ[n] = one(CType)
                u[n, 1] = zero(CType)
                u[n, 2] = zero(CType)
                u[n, 3] = zero(CType)
            end
        else
            ρ[n] = one(CType)
            u[n, 1] = zero(CType)
            u[n, 2] = zero(CType)
            u[n, 3] = zero(CType)
        end
        return nothing
    end

    @static if UPDATE_FIELDS
        return nothing
    end

    n0 = n - 1
    x = n0 % Nx
    y = (n0 ÷ Nx) % Ny
    z = n0 ÷ (Nx * Ny)

    fn1 = CType(fi[f_index(n, 1, N)])
    NP = (Q - 1) ÷ 2

    pairs = ntuple(Val(NP)) do k
        i = 2k
        cp = c[i]
        src = src_index(x, y, z, cp[1], cp[2], cp[3], Nx, Ny, Nz)
        load_pair(fi, n, src, i, t_odd, N, CType)
    end

    ρn = fn1
    ux = uy = uz = zero(CType)

    for k in 1:NP
        i = 2k
        fp, fm = pairs[k]
        ρn += fp + fm
        ux += CType(c[i][1])*fp + CType(c[i+1][1])*fm
        uy += CType(c[i][2])*fp + CType(c[i+1][2])*fm
        uz += CType(c[i][3])*fp + CType(c[i+1][3])*fm
    end

    invρ = one(CType) / ρn
    ρ[n] = ρn
    u[n, 1] = ux * invρ
    u[n, 2] = uy * invρ
    u[n, 3] = uz * invρ
    @static if TEMPERATURE
        if (flagsn & (TYPE_T | TYPE_H)) == 0x00
            srcx = src_index(x, y, z, 1, 0, 0, Nx, Ny, Nz)
            srcy = src_index(x, y, z, 0, 1, 0, Nx, Ny, Nz)
            srcz = src_index(x, y, z, 0, 0, 1, Nx, Ny, Nz)
            g0 = CType(gi[f_index(n, 1, N)])
            gpx, gmx = load_pair(gi, n, srcx, 2, t_odd, N, CType)
            gpy, gmy = load_pair(gi, n, srcy, 4, t_odd, N, CType)
            gpz, gmz = load_pair(gi, n, srcz, 6, t_odd, N, CType)
            T[n] = g0 + gpx + gmx + gpy + gmy + gpz + gmz + one(CType)
        end
    end
    return nothing
end

@kernel function moments_even_kernel!(
    ρ, u, @Const(flags), fi, gi, T, w::NTuple{Q, CType},
    c, N, Nx, Ny, Nz
) where {Q, CType}
    n = @index(Global)
    @inbounds moments_body!(Val(false), ρ, u, flags, fi, gi, T, w, c, N, Nx, Ny, Nz, Int(n))
end

@kernel function moments_odd_kernel!(
    ρ, u, @Const(flags), fi, gi, T, w::NTuple{Q, CType},
    c, N, Nx, Ny, Nz
) where {Q, CType}
    n = @index(Global)
    @inbounds moments_body!(Val(true), ρ, u, flags, fi, gi, T, w, c, N, Nx, Ny, Nz, Int(n))
end

@static if MOVING_BOUNDARIES
@kernel function update_moving_boundaries_kernel!(
    u, flags, c::NTuple{Q, SVector{3, Int}},
    N::Int, Nx::Int, Ny::Int, Nz::Int
) where {Q}
    n = @index(Global)
    @inbounds begin
        flagsn = flags[n]
        bo = flagsn & TYPE_BO
        if bo != TYPE_S && bo != TYPE_E
            n0 = n - 1
            x = n0 % Nx
            y = (n0 ÷ Nx) % Ny
            z = n0 ÷ (Nx * Ny)
            moving = false
            for i in 2:Q
                src = src_index(x, y, z, c[i][1], c[i][2], c[i][3], Nx, Ny, Nz)
                if (flags[src] & TYPE_BO) == TYPE_S
                    moving |= (u[src, 1] != zero(eltype(u))) |
                              (u[src, 2] != zero(eltype(u))) |
                              (u[src, 3] != zero(eltype(u)))
                end
            end
            flags[n] = moving ? (flagsn | TYPE_MS) : (flagsn & ~TYPE_MS)
        end
    end
end
end

@static if FORCE_FIELD
@inline function update_force_field_body!(
    t_odd::Val{odd}, flags, fi, F::AbstractArray{CType},
    c::NTuple{Q, SVector{3, Int}},
    N::Int, Nx::Int, Ny::Int, Nz::Int, n
) where {odd, Q, CType}
    if (flags[n] & TYPE_BO) != TYPE_S
        return nothing
    end
    n0 = n - 1
    x = n0 % Nx; y = (n0 ÷ Nx) % Ny; z = n0 ÷ (Nx * Ny)
    NP = (Q - 1) ÷ 2
    fn1 = CType(fi[f_index(n, 1, N)])
    mx = zero(CType); my = zero(CType); mz = zero(CType)
    for k in 1:NP
        i = 2k
        src = src_index(x, y, z, c[i][1], c[i][2], c[i][3], Nx, Ny, Nz)
        fp, fm = load_pair(fi, n, src, i, t_odd, N, CType)
        mx += CType(c[i][1])*fp + CType(c[i+1][1])*fm
        my += CType(c[i][2])*fp + CType(c[i+1][2])*fm
        mz += CType(c[i][3])*fp + CType(c[i+1][3])*fm
    end
    two = CType(2)
    F[n, 1] = two * mx
    F[n, 2] = two * my
    F[n, 3] = two * mz
    return nothing
end

@kernel function update_force_field_even_kernel!(
    @Const(flags), fi, F, c, N, Nx, Ny, Nz
)
    n = @index(Global)
    @inbounds update_force_field_body!(Val(false), flags, fi, F, c, N, Nx, Ny, Nz, Int(n))
end

@kernel function update_force_field_odd_kernel!(
    @Const(flags), fi, F, c, N, Nx, Ny, Nz
)
    n = @index(Global)
    @inbounds update_force_field_body!(Val(true), flags, fi, F, c, N, Nx, Ny, Nz, Int(n))
end

@kernel function reset_force_field_kernel!(F)
    n = @index(Global)
    @inbounds begin
        F[n, 1] = zero(eltype(F))
        F[n, 2] = zero(eltype(F))
        F[n, 3] = zero(eltype(F))
    end
end
end

@static if SURFACE

@kernel function surface_1_kernel!(
    flags, c::NTuple{Q, SVector{3, Int}},
    N::Int, Nx::Int, Ny::Int, Nz::Int
) where {Q}
    n = @index(Global)
    @inbounds begin
        sus = flags[n] & (TYPE_SU | TYPE_S)
        if sus == TYPE_IF
            n0 = n - 1
            x = n0 % Nx; y = (n0 ÷ Nx) % Ny; z = n0 ÷ (Nx * Ny)
            for i in 2:Q
                j = src_index(x, y, z, c[i][1], c[i][2], c[i][3], Nx, Ny, Nz)
                fj = flags[j]
                suj = fj & (TYPE_SU | TYPE_S)
                rest = fj & ~TYPE_SU
                if suj == TYPE_IG
                    flags[j] = rest | TYPE_I
                elseif suj == TYPE_G
                    flags[j] = rest | TYPE_GI
                end
            end
        end
    end
end

@inline function surface_2_body!(
    t_odd::Val{odd},
    fi, ρ, u, flags, gi, T, fs,
    w::NTuple{Q, CType}, c::NTuple{Q, SVector{3, Int}},
    N::Int, Nx::Int, Ny::Int, Nz::Int, n
) where {odd, Q, CType}
    sus = flags[n] & (TYPE_SU | TYPE_S)
    n0 = n - 1
    x = n0 % Nx; y = (n0 ÷ Nx) % Ny; z = n0 ÷ (Nx * Ny)

    if sus == TYPE_GI
        ρn, ux, uy, uz = average_neighbors_non_gas(ρ, u, flags, x, y, z, c, Nx, Ny, Nz, CType)
        store_feq!(fi, n, x, y, z, ρn, ux, uy, uz, w, c, N, Nx, Ny, Nz, t_odd)
        @static if TEMPERATURE
            Tn = average_neighbors_T(T, flags, x, y, z, c, Nx, Ny, Nz, CType)
            T[n] = Tn
            store_geq!(gi, n, x, y, z, Tn, ux, uy, uz, N, Nx, Ny, Nz, t_odd, CType)
            fs[n] = average_neighbors_fs(fs, flags, x, y, z, c, Nx, Ny, Nz, CType)
        end
        return nothing
    elseif sus == TYPE_IG
        for i in 2:Q
            j = src_index(x, y, z, c[i][1], c[i][2], c[i][3], Nx, Ny, Nz)
            fj = flags[j]
            suj = fj & (TYPE_SU | TYPE_S)
            rest = fj & ~TYPE_SU
            if suj == TYPE_F || suj == TYPE_IF
                flags[j] = rest | TYPE_I
            end
        end
    end
    return nothing
end

@kernel function surface_2_even_kernel!(fi, @Const(ρ), @Const(u), flags, gi, T, fs, w::NTuple{Q,CType}, c, N, Nx, Ny, Nz) where {Q, CType}
    n = @index(Global)
    @inbounds surface_2_body!(Val(false), fi, ρ, u, flags, gi, T, fs, w, c, N, Nx, Ny, Nz, Int(n))
end
@kernel function surface_2_odd_kernel!(fi, @Const(ρ), @Const(u), flags, gi, T, fs, w::NTuple{Q,CType}, c, N, Nx, Ny, Nz) where {Q, CType}
    n = @index(Global)
    @inbounds surface_2_body!(Val(true), fi, ρ, u, flags, gi, T, fs, w, c, N, Nx, Ny, Nz, Int(n))
end

@kernel function surface_3_kernel!(
    ρ, flags, mass, massex, ϕ, fs,
    c::NTuple{Q, SVector{3, Int}},
    N::Int, Nx::Int, Ny::Int, Nz::Int
) where {Q}
    n = @index(Global)
    @inbounds begin
        flagsn = flags[n]
        sus = flagsn & (TYPE_SU | TYPE_S)
        if (sus & TYPE_S) == 0x00
            CType = eltype(ρ)

            frozen = false
            @static if TEMPERATURE
                frozen = is_solid_fraction(fs[n])
            end

            ρn = ρ[n]
            massn = mass[n]
            massexn = zero(CType)
            ϕn = zero(CType)

            if frozen && (sus == TYPE_F || sus == TYPE_I)
                massexn = zero(CType)
                ϕn = sus == TYPE_F ? one(CType) : calculate_phi(ρn, massn, TYPE_I)
            elseif sus == TYPE_F
                massexn = massn - ρn
                massn = ρn
                ϕn = one(CType)
            elseif sus == TYPE_I
                massexn = massn > ρn ? massn - ρn : massn < 0 ? massn : zero(CType)
                massn = clamp(massn, zero(CType), ρn)
                ϕn = calculate_phi(ρn, massn, TYPE_I)
            elseif sus == TYPE_G
                massexn = massn
                massn = zero(CType)
                ϕn = zero(CType)
            elseif sus == TYPE_IF
                flags[n] = (flagsn & ~TYPE_SU) | TYPE_F
                massexn = massn - ρn
                massn = ρn
                ϕn = one(CType)
            elseif sus == TYPE_IG
                flags[n] = (flagsn & ~TYPE_SU) | TYPE_G
                massexn = massn
                massn = zero(CType)
                ϕn = zero(CType)
            elseif sus == TYPE_GI
                flags[n] = (flagsn & ~TYPE_SU) | TYPE_I
                massexn = massn > ρn ? massn - ρn : massn < 0 ? massn : zero(CType)
                massn = clamp(massn, zero(CType), ρn)
                ϕn = calculate_phi(ρn, massn, TYPE_I)
            end

            n0 = n - 1
            x = n0 % Nx; y = (n0 ÷ Nx) % Ny; z = n0 ÷ (Nx * Ny)
            counter = 0
            for i in 2:Q
                j = src_index(x, y, z, c[i][1], c[i][2], c[i][3], Nx, Ny, Nz)
                suj = flags[j] & (TYPE_SU | TYPE_S)
                liquid = suj == TYPE_F || suj == TYPE_I || suj == TYPE_IF || suj == TYPE_GI
                @static if TEMPERATURE
                    liquid = liquid && !is_solid_fraction(fs[j])
                end
                counter += Int(liquid)
            end
            if frozen
                massexn = zero(CType)
            elseif counter == 0
                massn += massexn
                massexn = zero(CType)
            else
                massexn /= CType(counter)
            end
            mass[n] = massn
            massex[n] = massexn
            ϕ[n] = ϕn
        end
    end
end

end

# Dissolved gas: FTCS diffusion + upwind advection, Henry c = k_H p on TYPE_I.
# nflux = ϕ (c* − c_H) / k_H is the lattice amount added to the bubble.
# ci[:,1] (index n) holds the diffused c* between the two kernels.
@static if SURFACE && TEMPERATURE

@inline function _liquid_c(c, flags, j, c0::CType) where {CType}
    fl = flags[j]
    ((fl & TYPE_BO) == TYPE_S || (fl & TYPE_SU) == TYPE_G) ? c0 : CType(c[j])
end

@kernel function dissolved_diffuse_kernel!(
    cstar, @Const(c), @Const(flags), @Const(u), D::CType, Nx::Int, Ny::Int, Nz::Int
) where {CType}
    n = @index(Global)
    @inbounds begin
        flagsn = flags[n]
        su = flagsn & TYPE_SU
        if (flagsn & TYPE_BO) == TYPE_S || su == TYPE_G
            cstar[n] = c[n]
        else
            n0 = n - 1
            x = n0 % Nx
            y = (n0 ÷ Nx) % Ny
            z = n0 ÷ (Nx * Ny)
            c0 = CType(c[n])
            xp = src_index(x, y, z, 1, 0, 0, Nx, Ny, Nz)
            xm = src_index(x, y, z, -1, 0, 0, Nx, Ny, Nz)
            yp = src_index(x, y, z, 0, 1, 0, Nx, Ny, Nz)
            ym = src_index(x, y, z, 0, -1, 0, Nx, Ny, Nz)
            zp = src_index(x, y, z, 0, 0, 1, Nx, Ny, Nz)
            zm = src_index(x, y, z, 0, 0, -1, Nx, Ny, Nz)
            cxp = _liquid_c(c, flags, xp, c0)
            cxm = _liquid_c(c, flags, xm, c0)
            cyp = _liquid_c(c, flags, yp, c0)
            cym = _liquid_c(c, flags, ym, c0)
            czp = _liquid_c(c, flags, zp, c0)
            czm = _liquid_c(c, flags, zm, c0)
            lap = cxp + cxm + cyp + cym + czp + czm - CType(6) * c0
            ux = u[n, 1]; uy = u[n, 2]; uz = u[n, 3]
            adv = ux * (ux > 0 ? c0 - cxm : cxp - c0) +
                  uy * (uy > 0 ? c0 - cym : cyp - c0) +
                  uz * (uz > 0 ? c0 - czm : czp - c0)
            cstar[n] = c0 + D * lap - adv
        end
    end
end

@kernel function dissolved_henry_kernel!(
    c, nflux, @Const(cstar), @Const(flags), @Const(ϕ), @Const(pgas), k_H::CType
) where {CType}
    n = @index(Global)
    @inbounds begin
        flagsn = flags[n]
        su = flagsn & TYPE_SU
        if (flagsn & TYPE_BO) == TYPE_S || su == TYPE_G
            nflux[n] = zero(CType)
        else
            cs = CType(cstar[n])
            henry = (k_H > zero(CType)) & (su == TYPE_I)
            if henry
                cH = k_H * pgas[n]
                cH = ifelse(cH > CType(1e-8), cH, CType(1e-8))
                fillc = ϕ[n]
                fillc = ifelse(fillc > zero(CType),
                    ifelse(fillc < one(CType), fillc, one(CType)), zero(CType))
                nflux[n] = fillc * (cs - cH) / k_H
                c[n] = cH
            else
                nflux[n] = zero(CType)
                c[n] = cs
            end
        end
    end
end

end