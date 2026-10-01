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

@inline function is_dirichlet_solid(fl::UInt8)
    return ((fl & TYPE_S) != 0x00) & ((fl & TYPE_T) != 0x00)
end

# Coldest adjacent Dirichlet wall. At ω_T near 2 a frozen interface cell
# against the mold picks up an odd-even mode and its temperature falls
# tens of kelvin below the wall. The wall is the cold boundary, so the
# node is not allowed under that temperature.
@inline function coldest_dirichlet(Tfield, flags, x, y, z, Nx, Ny, Nz, ::Type{CType}) where {CType}
    flo = zero(CType)
    found = false
    src = src_index(x, y, z, 1, 0, 0, Nx, Ny, Nz)
    found, flo = _wall_min(found, flo, Tfield, flags, src, CType)
    src = src_index(x, y, z, -1, 0, 0, Nx, Ny, Nz)
    found, flo = _wall_min(found, flo, Tfield, flags, src, CType)
    src = src_index(x, y, z, 0, 1, 0, Nx, Ny, Nz)
    found, flo = _wall_min(found, flo, Tfield, flags, src, CType)
    src = src_index(x, y, z, 0, -1, 0, Nx, Ny, Nz)
    found, flo = _wall_min(found, flo, Tfield, flags, src, CType)
    src = src_index(x, y, z, 0, 0, 1, Nx, Ny, Nz)
    found, flo = _wall_min(found, flo, Tfield, flags, src, CType)
    src = src_index(x, y, z, 0, 0, -1, Nx, Ny, Nz)
    found, flo = _wall_min(found, flo, Tfield, flags, src, CType)
    return found, flo
end

@inline function _wall_min(found::Bool, flo::CType, Tfield, flags, src, ::Type{CType}) where {CType}
    is_dirichlet_solid(flags[src]) || return found, flo
    Tw = CType(Tfield[src])
    if !found || Tw < flo
        return true, Tw
    end
    return true, flo
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
        # Pore and headspace gas do not collide, so one AA parity reads a
        # stale population and the melt heats. Rebuild that link at the
        # interface temperature: g_in = 2 geq(T) - g_out. Copying g_out
        # itself was a sink. This exchange is not a wall flux.
        gas_p = (flags[srcp] & TYPE_SU) == TYPE_G
        gas_m = (flags[srcm] & TYPE_SU) == TYPE_G
        miss_p = dir_p | flux_p | gas_p
        miss_m = dir_m | flux_m | gas_m
        (miss_p | miss_m) || continue
        fp_in,  fm_in  = load_pair(gi, n, srcp, i, t_odd, N, CType)
        fp_out, fm_out = load_outgoing_pair(gi, n, srcp, i, t_odd, N, CType)
        rec_p = zero(CType)
        rec_m = zero(CType)
        if miss_p
            Tw = gas_p ? T[n] : T[srcp]
            if flux_p && !gas_p
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
                gas_p || acc_add!(Eacc, EACC_WALL, fillc * (fm_in - rec_p))
            end
        end
        if miss_m
            Tw = gas_m ? T[n] : T[srcm]
            if flux_m && !gas_m
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
                gas_m || acc_add!(Eacc, EACC_WALL, fillc * (fp_in - rec_m))
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

# Using first order Taylor approximation
# cₚ(T) ≈ cₚ,ref + dcₚ/dT at Tₘ (T - Tₘ)
# divide by cₚ,ref
# multiply the right term by Tₘ/Tₘ
# cp/cp_ref = 1 + γ (T - 1)
# γ = 0 leads to H_sens = T
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
# Calculates the total enthalpy of a single cell by adding the sensible enthalpy to
# the latent heat of melting
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

@inline function omega_T_from_alpha(α::CType) where {CType}
    return clamp_omega(one(CType) / (CType(2) * α + CType(0.5)))
end

@inline function fs_from_T(Tn::CType, Ts::CType, Tl::CType) where {CType}
    Tn <= Ts && return one(CType)
    Tn >= Tl && return zero(CType)
    Δ = Tl - Ts
    Δ <= eps(CType) && return zero(CType)
    return (Tl - Tn) / Δ
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
            walled, Twmin = coldest_dirichlet(Tfield, flags, x, y, z, Nx, Ny, Nz, CType)
            if walled && Tnew < Twmin
                acc_add!(Eacc, EACC_WALL, fillc * (Twmin - Tnew))
                Tnew = Twmin
            end
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