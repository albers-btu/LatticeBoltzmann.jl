# D3Q7 thermal. Perturbation g' = g - w, T = 1 + Σg.
# ω_T = 1/(2α + 1/2). Boussinesq: F_eff = F - F β (T - T_avg).
# Volumetric Q: after SRT, add Δgeq(ΔT=Q) so ΣΔg = Q (lattice dT/step).
# TYPE_H: equivalent Dirichlet from Fourier + Robin, k = c_sT² (τ_T-1/2) = α/2.
@inline function geq_T_rest(Tn::CType) where {CType}
    @static if DIM == 3
        return CType(0.25) * Tn - CType(0.25)
    else
        # D2Q5 w0 = 1/3. g0 = w0 (T - 1).
        return (one(CType) / CType(3)) * Tn - (one(CType) / CType(3))
    end
end
@inline function geq_T_axis(Tn::CType, ucomp::CType) where {CType}
    @static if DIM == 3
        return CType(0.5) * Tn * ucomp + CType(0.125) * (Tn - one(CType))
    else
        # D2Q5 axis weight 1/6. g = (1/6)(T - 1) + (1/2) T u_comp.
        return CType(0.5) * Tn * ucomp + (one(CType) / CType(6)) * (Tn - one(CType))
    end
end

# TYPE_S never collides. Uninitialized gi=0 is T=1 (lattice Tm). Keep geq(T[n],0)
# on solids so bounce-back is Dirichlet at the stored wall temperature.
@inline function store_geq_local!(gi, n, Tn::CType, N::Int) where {CType}
    gi[f_index(n, 1, N)] = eltype(gi)(geq_T_rest(Tn))
    gax = geq_T_axis(Tn, zero(CType))
    # Not a store_pair! site: one population per index. 2:5 covers both
    # D2Q5 axis pairs. Indices 2 and 4 alone would leave 3 and 5 at 0.
    @static if DIM == 3
        @inbounds for i in 2:7
            gi[f_index(n, i, N)] = eltype(gi)(gax)
        end
    else
        @inbounds for i in 2:5
            gi[f_index(n, i, N)] = eltype(gi)(gax)
        end
    end
    return nothing
end

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
    # Pair 2 writes populations 2 and 3, pair 4 writes 4 and 5. No pair 6 in 2D.
    @static if DIM == 3
        thermal_axes = ((2, 1, 0, 0), (4, 0, 1, 0), (6, 0, 0, 1))
    else
        thermal_axes = ((2, 1, 0, 0), (4, 0, 1, 0))
    end
    @inbounds for (i, cx, cy, cz) in thermal_axes
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
                # |g| past the cap is a mirrored runaway, not a wall population.
                rec_p = bound_population(geg + geg - fp_out, geg)
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
                rec_m = bound_population(geg + geg - fm_out, geg)
                acc_add!(Eacc, EACC_WALL, fillc * (fp_in - rec_m))
            end
        end
        (miss_p | miss_m) && store_reconstructed_pair!(gi, n, srcp, i, rec_p, rec_m, miss_p, miss_m, t_odd, N)
    end
    return nothing
end

@inline function reconstruct_g_boundaries!(
    g_odd::Bool, gi, T, flags, hT, Qin,
    x::Int, y::Int, z::Int, n::Int,
    N::Int, Nx::Int, Ny::Int, Nz::Int, ::Type{CType},
    Eacc, fillc::CType, ω_T::CType
) where {CType}
    if g_odd
        reconstruct_g_boundaries!(Val(true), gi, T, flags, hT, Qin, x, y, z, n, N, Nx, Ny, Nz, CType, Eacc, fillc, ω_T)
    else
        reconstruct_g_boundaries!(Val(false), gi, T, flags, hT, Qin, x, y, z, n, N, Nx, Ny, Nz, CType, Eacc, fillc, ω_T)
    end
    return nothing
end

# Gas and a plain solid (no Dirichlet, no Robin) never collide, so nothing
# streams back. The population that arrived from the fluid stays; the missing
# side is set so the pair sums to 2·geq(T, 0). That sum is the equilibrium
# pair at any normal speed, and it does not depend on the arrived value, so a
# recoil spike cannot multiply T. collide_temperature! also drops that normal
# speed when writing geq, so the spike is not stored into the fluid neighbor.
@inline function thermal_adiabatic_face(fl::UInt8)
    plain = ((fl & TYPE_S) != 0x00) & ((fl & TYPE_T) == 0x00) & ((fl & TYPE_H) == 0x00)
    return plain | ((fl & TYPE_SU) == TYPE_G)
end

@inline function reconstruct_g_adiabatic!(
    t_odd::Val{odd}, gi, Tn::CType, flags,
    x::Int, y::Int, z::Int, n::Int,
    N::Int, Nx::Int, Ny::Int, Nz::Int, ::Type{CType}
) where {odd, CType}
    # geq(T, +u) + geq(T, −u) = 2 geq(T, 0) on one axis. The missing side is
    # that sum minus the population that actually arrived, so a recoil spike
    # in the arrived link cannot change this cell's temperature.
    rest = geq_T_axis(Tn, zero(CType))
    pair0 = rest + rest
    @static if DIM == 3
        thermal_axes = ((2, 1, 0, 0), (4, 0, 1, 0), (6, 0, 0, 1))
    else
        thermal_axes = ((2, 1, 0, 0), (4, 0, 1, 0))
    end
    @inbounds for (i, cx, cy, cz) in thermal_axes
        srcp = src_index(x, y, z, cx, cy, cz, Nx, Ny, Nz)
        srcm = src_index(x, y, z, -cx, -cy, -cz, Nx, Ny, Nz)
        miss_p = thermal_adiabatic_face(flags[srcp])
        miss_m = thermal_adiabatic_face(flags[srcm])
        (miss_p | miss_m) || continue
        gp, gm = load_pair(gi, n, srcp, i, t_odd, N, CType)
        # First stored value is what this parity's load reads as g₋.
        if miss_p && miss_m
            rec_m = rest
            rec_p = rest
        elseif miss_p
            rec_m = pair0 - gp
            rec_p = gm
        else
            rec_m = gp
            rec_p = pair0 - gm
        end
        rec_m = bound_population(rec_m, rest)
        rec_p = bound_population(rec_p, rest)
        store_reconstructed_pair!(gi, n, srcp, i, rec_m, rec_p, miss_p, miss_m, t_odd, N)
    end
    return nothing
end

# Drop the normal speed on an axis whose neighbor does not collide. geq's
# ±u_n terms would otherwise be stored into the fluid and heat it by ~T·u_n.
@inline function thermal_face_velocity(ucomp::CType, flags, srcp::Int, srcm::Int) where {CType}
    if thermal_adiabatic_face(flags[srcp]) || thermal_adiabatic_face(flags[srcm])
        return zero(CType)
    end
    return ucomp
end

@inline function reconstruct_g_adiabatic!(
    g_odd::Bool, gi, Tn::CType, flags,
    x::Int, y::Int, z::Int, n::Int,
    N::Int, Nx::Int, Ny::Int, Nz::Int, ::Type{CType}
) where {CType}
    if g_odd
        reconstruct_g_adiabatic!(Val(true), gi, Tn, flags, x, y, z, n, N, Nx, Ny, Nz, CType)
    else
        reconstruct_g_adiabatic!(Val(false), gi, Tn, flags, x, y, z, n, N, Nx, Ny, Nz, CType)
    end
    return nothing
end

@inline function thermal_conductivity(ω_T::CType) where {CType}
    @static if DIM == 3
        return CType(0.25) * (one(CType) / ω_T - CType(0.5))
    else
        # k = cs² (1/ω - 1/2) with cs² = 1/3. Do not use the D3Q7 1/4.
        return (one(CType) / CType(3)) * (one(CType) / ω_T - CType(0.5))
    end
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
    @static if DIM == 3
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

# Hertz–Knudsen cooling as lattice dT/step. Λ_v = 0 or T ≤ 0 → off.
# p_sat is the Clausius–Clapeyron pressure recoil uses, including T < T_v.
# Above T_v the raw flux is exponential. A cap that stops exactly at T_v
# pins the cell there: the next laser increment crosses T_v and is removed
# only back to T_v, so recoil stays at p_sat(T_v) for every later hydro
# substep. The step may therefore land below T_v, by at most the boiling
# flux C_hk p0 Λ_v / √T_v on top of the superheat. Below T_v the only
# limit is T ≥ 0.
@inline function evaporative_dT(
    Tn::CType, Λ_v::CType, T_v::CType, C_hk::CType, p0::CType, β_v::CType
) where {CType}
    if !(Λ_v > zero(CType) && T_v > zero(CType) && Tn > zero(CType))
        return zero(CType)
    end
    ps = p_sat(Tn, T_v, p0, β_v)
    mdot = C_hk * ps / sqrt(Tn)
    Qe = mdot * Λ_v
    if Tn > T_v
        Qboil = C_hk * p0 / sqrt(T_v) * Λ_v
        Qmax = (Tn - T_v) + Qboil
    else
        Qmax = Tn
    end
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
    @static if DIM == 3
        return clamp_omega(one(CType) / (CType(2) * α + CType(0.5)))
    else
        # domain.α = 2χ and χ = cs² (1/ω - 1/2), cs² = 1/3, so 1/ω = (3/2)α + 1/2.
        return clamp_omega(one(CType) / ((CType(3) / CType(2)) * α + CType(0.5)))
    end
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
    g0 = CType(gi[f_index(n, 1, N)])
    gpx, gmx = load_pair(gi, n, srcx, 2, t_odd, N, CType)
    gpy, gmy = load_pair(gi, n, srcy, 4, t_odd, N, CType)
    @static if DIM == 3
        srcz = src_index(x, y, z, 0, 0, 1, Nx, Ny, Nz)
        gpz, gmz = load_pair(gi, n, srcz, 6, t_odd, N, CType)
        Tfromg = g0 + gpx + gmx + gpy + gmy + gpz + gmz + one(CType)
    else
        Tfromg = g0 + gpx + gmx + gpy + gmy + one(CType)
    end

    # SRT would keep a non-finite or runaway g. |g|≤2 covers geq at T≲5 and |u_α|≤c_s.
    g_bad = !population_ok(g0) | !population_ok(gpx) | !population_ok(gmx) |
            !population_ok(gpy) | !population_ok(gmy)
    @static if DIM == 3
        g_bad = g_bad | !population_ok(gpz) | !population_ok(gmz)
    end
    if g_bad
        Tw = Tfield[n]
        Tfromg = isfinite(Tw) ? Tw : Ts
    end

    Qn_in = Qin[n]
    Qn = Qn_in
    if !isfinite(Qn)
        Qn = zero(CType)
        Qn_in = zero(CType)
    end
    dirichlet = (flagsn & TYPE_T) != 0x00
    use_flux = false
    write_geq = g_bad
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

    # Local copies only. The caller still collides hydro at the real velocity.
    ux = thermal_face_velocity(ux, flags, srcx, src_index(x, y, z, -1, 0, 0, Nx, Ny, Nz))
    uy = thermal_face_velocity(uy, flags, srcy, src_index(x, y, z, 0, -1, 0, Nx, Ny, Nz))
    @static if DIM == 3
        uz = thermal_face_velocity(uz, flags, srcz, src_index(x, y, z, 0, 0, -1, Nx, Ny, Nz))
    end
    ge0 = geq_T_rest(Tn)
    gexp, gexm = geq_T_axis(Tn, ux), geq_T_axis(Tn, -ux)
    geyp, geym = geq_T_axis(Tn, uy), geq_T_axis(Tn, -uy)
    @static if DIM == 3
        gezp, gezm = geq_T_axis(Tn, uz), geq_T_axis(Tn, -uz)
    end

    if dirichlet || use_flux || write_geq
        gi[f_index(n, 1, N)] = eltype(gi)(ge0)
        store_pair!(gi, n, srcx, 2, gexp, gexm, t_odd, N)
        store_pair!(gi, n, srcy, 4, geyp, geym, t_odd, N)
        @static if DIM == 3
            store_pair!(gi, n, srcz, 6, gezp, gezm, t_odd, N)
        end
    else
        om = one(CType) - ω_T
        @static if DIM == 3
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
        else
            # Δg0 = Q/3, Δg_axis = Q/6 + (Q u_comp)/2. Σ Δg = Q.
            gi[f_index(n, 1, N)] = eltype(gi)(om * g0 + ω_T * ge0 + (one(CType) / CType(3)) * Qn)
            store_pair!(gi, n, srcx, 2,
                om * gpx + ω_T * gexp + CType(0.5) * Qn * ux + (one(CType) / CType(6)) * Qn,
                om * gmx + ω_T * gexm + CType(0.5) * Qn * (-ux) + (one(CType) / CType(6)) * Qn,
                t_odd, N)
            store_pair!(gi, n, srcy, 4,
                om * gpy + ω_T * geyp + CType(0.5) * Qn * uy + (one(CType) / CType(6)) * Qn,
                om * gmy + ω_T * geym + CType(0.5) * Qn * (-uy) + (one(CType) / CType(6)) * Qn,
                t_odd, N)
        end
    end

    Tmacro = dirichlet || use_flux ? Tn : Tn + Qn
    dT = Tmacro - T_avg
    fxn -= fx * β * dT
    fyn -= fy * β * dT
    @static if DIM == 3
        fzn -= fz * β * dT
    end
    return fxn, fyn, fzn, mevap
end

@inline function collide_temperature!(
    g_odd::Bool, gi, Tfield, Qin, hT, flags, flagsn, fs,
    ux::CType, uy::CType, uz::CType,
    fxn::CType, fyn::CType, fzn::CType,
    fx::CType, fy::CType, fz::CType,
    ω_T::CType, β::CType, T_avg::CType, Λ::CType, Ts::CType, Tl::CType,
    γ_s::CType, γ_l::CType,
    Λ_v::CType, T_v::CType, C_hk::CType, p0v::CType, β_v::CType,
    C_rad::CType, T_rad::CType,
    x::Int, y::Int, z::Int, Nx::Int, Ny::Int, Nz::Int, N::Int, n::Int,
    ::Type{CType}, Eacc, fill::CType
) where {CType}
    if g_odd
        return collide_temperature!(
            Val(true), gi, Tfield, Qin, hT, flags, flagsn, fs,
            ux, uy, uz, fxn, fyn, fzn, fx, fy, fz,
            ω_T, β, T_avg, Λ, Ts, Tl, γ_s, γ_l,
            Λ_v, T_v, C_hk, p0v, β_v, C_rad, T_rad,
            x, y, z, Nx, Ny, Nz, N, n, CType, Eacc, fill)
    else
        return collide_temperature!(
            Val(false), gi, Tfield, Qin, hT, flags, flagsn, fs,
            ux, uy, uz, fxn, fyn, fzn, fx, fy, fz,
            ω_T, β, T_avg, Λ, Ts, Tl, γ_s, γ_l,
            Λ_v, T_v, C_hk, p0v, β_v, C_rad, T_rad,
            x, y, z, Nx, Ny, Nz, N, n, CType, Eacc, fill)
    end
end

@inline function store_geq!(
    gi, n, x, y, z, Tn, ux, uy, uz, N, Nx, Ny, Nz, t_odd::Val{odd}, ::Type{CType},
    ::Val{swap}=Val(false)
) where {odd, CType, swap}
    gi[f_index(n, 1, N)] = eltype(gi)(geq_T_rest(Tn))
    srcx = src_index(x, y, z, 1, 0, 0, Nx, Ny, Nz)
    srcy = src_index(x, y, z, 0, 1, 0, Nx, Ny, Nz)
    gpx, gmx = geq_T_axis(Tn, ux), geq_T_axis(Tn, -ux)
    gpy, gmy = geq_T_axis(Tn, uy), geq_T_axis(Tn, -uy)
    # Same init correction as store_feq!: even load would otherwise see −u.
    if swap
        gpx, gmx = gmx, gpx
        gpy, gmy = gmy, gpy
    end
    store_pair!(gi, n, srcx, 2, gpx, gmx, t_odd, N)
    store_pair!(gi, n, srcy, 4, gpy, gmy, t_odd, N)
    @static if DIM == 3
        srcz = src_index(x, y, z, 0, 0, 1, Nx, Ny, Nz)
        gpz, gmz = geq_T_axis(Tn, uz), geq_T_axis(Tn, -uz)
        if swap
            gpz, gmz = gmz, gpz
        end
        store_pair!(gi, n, srcz, 6, gpz, gmz, t_odd, N)
    end
    return nothing
end

@inline function store_geq!(
    gi, n, x, y, z, Tn, ux, uy, uz, N, Nx, Ny, Nz, g_odd::Bool, ::Type{CType}
) where {CType}
    if g_odd
        store_geq!(gi, n, x, y, z, Tn, ux, uy, uz, N, Nx, Ny, Nz, Val(true), CType)
    else
        store_geq!(gi, n, x, y, z, Tn, ux, uy, uz, N, Nx, Ny, Nz, Val(false), CType)
    end
    return nothing
end
