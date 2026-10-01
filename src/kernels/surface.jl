using KernelAbstractions

@inline function calculate_phi(ρn::CType, massn::CType, flagsn::UInt8) where {CType}
    su = flagsn & TYPE_SU
    if su == TYPE_F || su == TYPE_IF
        return one(CType)
    elseif su == TYPE_I || su == TYPE_GI
        return ρn > 0 ? clamp(massn / ρn, zero(CType), one(CType)) : CType(0.5)
    else
        return zero(CType)
    end
end

@static if SURFACE

# This function is the setup, which is called before the collide pass. It
# updates the mass and massex (mass excess) and sets up / reconstructs 
# the fᵢ/gᵢ. surface_0 only runs on TYPE_F (fluid) and TYPE_I (interface) 
# and skips TYPE_S (solid) and TYPE_G (gas).
@static if !FOAM

@inline function surface_0_body!(
    t_odd::Val{odd},
    fi, ρ, u, flags, mass, massex, ϕ, T, fs, gi,
    w::NTuple{Q, CType},
    c::NTuple{Q, SVector{3, Int}},
    fx::CType, fy::CType, fz::CType, σ::CType, σT::CType, Tσ::CType,
    Λ_v::CType, T_v::CType, p0v::CType, β_v::CType,
    N::Int, Nx::Int, Ny::Int, Nz::Int, n, Eacc, hT, Qin, ω_T::CType
) where {odd, Q, CType}
    flagsn = flags[n]
    bo = flagsn & TYPE_BO
    su = flagsn & TYPE_SU
    (bo == TYPE_S || su == TYPE_G) && return nothing

    n0 = n - 1
    x = n0 % Nx
    y = (n0 ÷ Nx) % Ny
    z = n0 ÷ (Nx * Ny)

    solidified = false
    @static if TEMPERATURE
        solidified = is_solid_fraction(fs[n])
    end

    # Collect last steps mass excess, skip if the cell is solidified, since
    # it cant take any more mass if solidifed.
    massn = mass[n]
    if !solidified
        for i in 2:Q
            src = src_index(x, y, z, c[i][1], c[i][2], c[i][3], Nx, Ny, Nz)
            massn += massex[src]
        end
    end

    # Number of ± velocity pairs
    NP = (Q - 1) ÷ 2

    if su == TYPE_F
        # Calculate net mass gain/loss
        if !solidified
            for k in 1:NP
                i = 2k # plus velocities
                src = src_index(x, y, z, c[i][1], c[i][2], c[i][3], Nx, Ny, Nz)
                fp_in,  fm_in  = load_pair(fi, n, src, i, t_odd, N, CType)
                fp_out, fm_out = load_outgoing_pair(fi, n, src, i, t_odd, N, CType)
                massn += (fp_in - fp_out) + (fm_in - fm_out)
            end
        end
        mass[n] = massn

        @static if TEMPERATURE
            fillc = ϕ[n]
            fillc = ifelse(fillc > zero(CType), fillc, zero(CType))
            reconstruct_g_boundaries!(t_odd, gi, T, flags, hT, Qin, x, y, z, n, N, Nx, Ny, Nz, CType, Eacc, fillc, ω_T)
        end
        return nothing
    end

    if su != TYPE_I
        mass[n] = massn
        return nothing
    end

    # Applies only to the interface (TYPE_F)
    cs = CType(1) / sqrt(CType(3))
    @static if EQUILIBRIUM_BOUNDARIES
        eq = (flagsn & TYPE_BO) == TYPE_E
    else
        eq = false
    end

    # This big ifelse block calculates the velocities and density for a
    # given cell n.
    if solidified
        ρn = ρ[n]
        ρn = ρn > zero(CType) ? ρn : one(CType)
        ux = zero(CType); uy = zero(CType); uz = zero(CType)
        uxg = zero(CType); uyg = zero(CType); uzg = zero(CType)
        ρ_gas = one(CType)
        ϕn = calculate_phi(ρn, massn, flagsn)
    elseif eq
        ρn, ux, uy, uz = prescribed_hydro(ρ[n], u[n, 1], u[n, 2], u[n, 3], fx, fy, fz)
        ϕn = calculate_phi(ρn, massn, flagsn)
        σn = σ
        @static if TEMPERATURE
            σn = σ + σT * (T[n] - Tσ)
            σn = ifelse(σn > zero(CType), σn, zero(CType))
        end
        ρ_gas = gas_density_plic(σn, ϕ, ϕn, x, y, z, Nx, Ny, Nz)
        uxg, uyg, uzg = ux, uy, uz
    else
        ρn = CType(fi[f_index(n, 1, N)])
        ux = zero(CType); uy = zero(CType); uz = zero(CType)
        for k in 1:NP
            i = 2k
            src = src_index(x, y, z, c[i][1], c[i][2], c[i][3], Nx, Ny, Nz)
            fp_out, fm_out = load_outgoing_pair(fi, n, src, i, t_odd, N, CType)
            ρn += fp_out + fm_out
            ux += CType(c[i][1])*fp_out + CType(c[i+1][1])*fm_out
            uy += CType(c[i][2])*fp_out + CType(c[i+1][2])*fm_out
            uz += CType(c[i][3])*fp_out + CType(c[i+1][3])*fm_out
        end
        if ρn <= zero(CType)
            ρn = one(CType)
            ux = zero(CType); uy = zero(CType); uz = zero(CType)
        else
            invρ = one(CType) / ρn
            ux *= invρ; uy *= invρ; uz *= invρ
            ux = clamp(ux, -cs, cs); uy = clamp(uy, -cs, cs); uz = clamp(uz, -cs, cs)
        end
        ϕn = calculate_phi(ρn, massn, flagsn)
        σn = σ
        @static if TEMPERATURE
            σn = σ + σT * (T[n] - Tσ)
            σn = ifelse(σn > zero(CType), σn, zero(CType))
        end
        ρ_gas = gas_density_plic(σn, ϕ, ϕn, x, y, z, Nx, Ny, Nz)
        @static if VOLUME_FORCE
            uxg = clamp(ux + fx / (CType(2) * ρn), -cs, cs)
            uyg = clamp(uy + fy / (CType(2) * ρn), -cs, cs)
            uzg = clamp(uz + fz / (CType(2) * ρn), -cs, cs)
        else
            uxg, uyg, uzg = ux, uy, uz
        end
        @static if TEMPERATURE
            inv2ρ = one(CType) / (CType(2) * ρn)
            if σT != zero(CType)
                mx, my, mz = marangoni_force(T, ϕ, flags, σT, x, y, z, n, Nx, Ny, Nz, CType)
                uxg = clamp(uxg + mx * inv2ρ, -cs, cs)
                uyg = clamp(uyg + my * inv2ρ, -cs, cs)
                uzg = clamp(uzg + mz * inv2ρ, -cs, cs)
            end
            if Λ_v > zero(CType)
                rx, ry, rz = recoil_force(T, ϕ, n, x, y, z, Nx, Ny, Nz, Λ_v, T_v, p0v, β_v, CType)
                uxg = clamp(uxg + rx * inv2ρ, -cs, cs)
                uyg = clamp(uyg + ry * inv2ρ, -cs, cs)
                uzg = clamp(uzg + rz * inv2ρ, -cs, cs)
            end
        end
    end
    uug = CType(1.5) * (uxg*uxg + uyg*uyg + uzg*uzg)

    for k in 1:NP
        i = 2k
        cp, cm = c[i], c[i + 1]
        srcp = src_index(x, y, z, cp[1], cp[2], cp[3], Nx, Ny, Nz)
        srcm = src_index(x, y, z, cm[1], cm[2], cm[3], Nx, Ny, Nz)
        sup = flags[srcp] & TYPE_SU
        sum_ = flags[srcm] & TYPE_SU
        ϕp = ϕ[srcp]; ϕm = ϕ[srcm]

        fp_in,  fm_in  = load_pair(fi, n, srcp, i, t_odd, N, CType)
        fp_out, fm_out = load_outgoing_pair(fi, n, srcp, i, t_odd, N, CType)

        if !solidified
            if (sup & (TYPE_F | TYPE_I)) != 0x00
                fluxp = fm_in - fp_out
                massn += sup == TYPE_F ? fluxp : CType(0.5) * (ϕp + ϕn) * fluxp
            end
            if (sum_ & (TYPE_F | TYPE_I)) != 0x00
                fluxm = fp_in - fm_out
                massn += sum_ == TYPE_F ? fluxm : CType(0.5) * (ϕm + ϕn) * fluxm
            end
        end

        fegp = feq(w[i],     ρ_gas, uxg, uyg, uzg, uug, cp, CType)
        fegm = feq(w[i + 1], ρ_gas, uxg, uyg, uzg, uug, cm, CType)
        fp_rec = fegm - fm_out + fegp
        fm_rec = fegp - fp_out + fegm
        store_reconstructed_pair!(
            fi, n, srcp, i, fm_rec, fp_rec,
            sup == TYPE_G, sum_ == TYPE_G, t_odd, N)
    end
    mass[n] = massn
    @static if TEMPERATURE
        fillc = ϕn
        fillc = ifelse(fillc > zero(CType), fillc, zero(CType))
        reconstruct_g_boundaries!(t_odd, gi, T, flags, hT, Qin, x, y, z, n, N, Nx, Ny, Nz, CType, Eacc, fillc, ω_T)
    end
    return nothing
end

@kernel function surface_0_even_kernel!(
    fi, @Const(ρ), @Const(u), @Const(flags), mass, @Const(massex), @Const(ϕ), T, fs, gi,
    w::NTuple{Q, CType}, c::NTuple{Q, SVector{3, Int}},
    fx::CType, fy::CType, fz::CType, σ::CType, σT::CType, Tσ::CType,
    Λ_v::CType, T_v::CType, p0v::CType, β_v::CType,
    N::Int, Nx::Int, Ny::Int, Nz::Int, Eacc, hT, Qin, ω_T::CType
) where {Q, CType}
    n = @index(Global)
    @inbounds surface_0_body!(Val(false), fi, ρ, u, flags, mass, massex, ϕ, T, fs, gi, w, c, fx, fy, fz, σ, σT, Tσ, Λ_v, T_v, p0v, β_v, N, Nx, Ny, Nz, Int(n), Eacc, hT, Qin, ω_T)
end

@kernel function surface_0_odd_kernel!(
    fi, @Const(ρ), @Const(u), @Const(flags), mass, @Const(massex), @Const(ϕ), T, fs, gi,
    w::NTuple{Q, CType}, c::NTuple{Q, SVector{3, Int}},
    fx::CType, fy::CType, fz::CType, σ::CType, σT::CType, Tσ::CType,
    Λ_v::CType, T_v::CType, p0v::CType, β_v::CType,
    N::Int, Nx::Int, Ny::Int, Nz::Int, Eacc, hT, Qin, ω_T::CType
) where {Q, CType}
    n = @index(Global)
    @inbounds surface_0_body!(Val(true), fi, ρ, u, flags, mass, massex, ϕ, T, fs, gi, w, c, fx, fy, fz, σ, σT, Tσ, Λ_v, T_v, p0v, β_v, N, Nx, Ny, Nz, Int(n), Eacc, hT, Qin, ω_T)
end

end

@static if FOAM

@inline function surface_0_body!(
    t_odd::Val{odd},
    fi, ρ, u, flags, mass, massex, ϕ, T, fs, gi,
    w::NTuple{Q, CType},
    c::NTuple{Q, SVector{3, Int}},
    fx::CType, fy::CType, fz::CType, σ::CType, σT::CType, Tσ::CType,
    Λ_v::CType, T_v::CType, p0v::CType, β_v::CType,
    N::Int, Nx::Int, Ny::Int, Nz::Int, n, Eacc, hT, Qin, ω_T::CType,
    ci, flux, ρb, Pi, k_H::CType, D::CType, tag, cfield
) where {odd, Q, CType}
    flagsn = flags[n]
    bo = flagsn & TYPE_BO
    su = flagsn & TYPE_SU
    (bo == TYPE_S || su == TYPE_G) && return nothing

    n0 = n - 1
    x = n0 % Nx
    y = (n0 ÷ Nx) % Ny
    z = n0 ÷ (Nx * Ny)

    solidified = false
    @static if TEMPERATURE
        solidified = is_solid_fraction(fs[n])
    end

    # Collect last steps mass excess, skip if the cell is solidified, since
    # it cant take any more mass if solidifed.
    massn = mass[n]
    if !solidified
        for i in 2:Q
            src = src_index(x, y, z, c[i][1], c[i][2], c[i][3], Nx, Ny, Nz)
            massn += massex[src]
        end
    end

    # Number of ± velocity pairs
    NP = (Q - 1) ÷ 2

    if su == TYPE_F
        # Calculate net mass gain/loss
        if !solidified
            for k in 1:NP
                i = 2k # plus velocities
                src = src_index(x, y, z, c[i][1], c[i][2], c[i][3], Nx, Ny, Nz)
                fp_in,  fm_in  = load_pair(fi, n, src, i, t_odd, N, CType)
                fp_out, fm_out = load_outgoing_pair(fi, n, src, i, t_odd, N, CType)
                massn += (fp_in - fp_out) + (fm_in - fm_out)
            end
        end
        mass[n] = massn

        @static if TEMPERATURE
            fillc = ϕ[n]
            fillc = ifelse(fillc > zero(CType), fillc, zero(CType))
            reconstruct_g_boundaries!(t_odd, gi, T, flags, hT, Qin, x, y, z, n, N, Nx, Ny, Nz, CType, Eacc, fillc, ω_T)
        end
        return nothing
    end

    if su != TYPE_I
        mass[n] = massn
        return nothing
    end

    # Applies only to the interface (TYPE_F)
    cs = CType(1) / sqrt(CType(3))
    @static if EQUILIBRIUM_BOUNDARIES
        eq = (flagsn & TYPE_BO) == TYPE_E
    else
        eq = false
    end

    # This big ifelse block calculates the velocities and density for a
    # given cell n.
    if solidified
        ρn = ρ[n]
        ρn = ρn > zero(CType) ? ρn : one(CType)
        ux = zero(CType); uy = zero(CType); uz = zero(CType)
        uxg = zero(CType); uyg = zero(CType); uzg = zero(CType)
        # Hold the fill, and keep this bubble's gas pressure. Ambient ρ = 1
        # on a half-frozen shell next to a neighbor that still uses ρb
        # is a pressure jump and spikes |u|.
        ϕn = ϕ[n]
        σn = σ
        ρ_gas = gas_density_plic(σn, ϕ, ϕn, x, y, z, Nx, Ny, Nz, ρb[n], Pi[n])
    elseif eq
        ρn, ux, uy, uz = prescribed_hydro(ρ[n], u[n, 1], u[n, 2], u[n, 3], fx, fy, fz)
        ϕn = calculate_phi(ρn, massn, flagsn)
        σn = σ
        @static if TEMPERATURE
            σn = σ + σT * (T[n] - Tσ)
            σn = ifelse(σn > zero(CType), σn, zero(CType))
        end
        ρ_gas = gas_density_plic(σn, ϕ, ϕn, x, y, z, Nx, Ny, Nz, ρb[n], Pi[n])
        uxg, uyg, uzg = ux, uy, uz
    else
        ρn = CType(fi[f_index(n, 1, N)])
        ux = zero(CType); uy = zero(CType); uz = zero(CType)
        for k in 1:NP
            i = 2k
            src = src_index(x, y, z, c[i][1], c[i][2], c[i][3], Nx, Ny, Nz)
            fp_out, fm_out = load_outgoing_pair(fi, n, src, i, t_odd, N, CType)
            ρn += fp_out + fm_out
            ux += CType(c[i][1])*fp_out + CType(c[i+1][1])*fm_out
            uy += CType(c[i][2])*fp_out + CType(c[i+1][2])*fm_out
            uz += CType(c[i][3])*fp_out + CType(c[i+1][3])*fm_out
        end
        if ρn <= zero(CType)
            ρn = one(CType)
            ux = zero(CType); uy = zero(CType); uz = zero(CType)
        else
            invρ = one(CType) / ρn
            ux *= invρ; uy *= invρ; uz *= invρ
            ux = clamp(ux, -cs, cs); uy = clamp(uy, -cs, cs); uz = clamp(uz, -cs, cs)
        end
        ϕn = calculate_phi(ρn, massn, flagsn)
        σn = σ
        @static if TEMPERATURE
            σn = σ + σT * (T[n] - Tσ)
            σn = ifelse(σn > zero(CType), σn, zero(CType))
        end
        ρ_gas = gas_density_plic(σn, ϕ, ϕn, x, y, z, Nx, Ny, Nz, ρb[n], Pi[n])
        @static if VOLUME_FORCE
            uxg = clamp(ux + fx / (CType(2) * ρn), -cs, cs)
            uyg = clamp(uy + fy / (CType(2) * ρn), -cs, cs)
            uzg = clamp(uz + fz / (CType(2) * ρn), -cs, cs)
        else
            uxg, uyg, uzg = ux, uy, uz
        end
        @static if TEMPERATURE
            inv2ρ = one(CType) / (CType(2) * ρn)
            if σT != zero(CType)
                mx, my, mz = marangoni_force(T, ϕ, flags, σT, x, y, z, n, Nx, Ny, Nz, CType)
                uxg = clamp(uxg + mx * inv2ρ, -cs, cs)
                uyg = clamp(uyg + my * inv2ρ, -cs, cs)
                uzg = clamp(uzg + mz * inv2ρ, -cs, cs)
            end
            if Λ_v > zero(CType)
                rx, ry, rz = recoil_force(T, ϕ, n, x, y, z, Nx, Ny, Nz, Λ_v, T_v, p0v, β_v, CType)
                uxg = clamp(uxg + rx * inv2ρ, -cs, cs)
                uyg = clamp(uyg + ry * inv2ρ, -cs, cs)
                uzg = clamp(uzg + rz * inv2ρ, -cs, cs)
            end
        end
    end
    uug = CType(1.5) * (uxg*uxg + uyg*uyg + uzg*uzg)

    for k in 1:NP
        i = 2k
        cp, cm = c[i], c[i + 1]
        srcp = src_index(x, y, z, cp[1], cp[2], cp[3], Nx, Ny, Nz)
        srcm = src_index(x, y, z, cm[1], cm[2], cm[3], Nx, Ny, Nz)
        sup = flags[srcp] & TYPE_SU
        sum_ = flags[srcm] & TYPE_SU
        ϕp = ϕ[srcp]; ϕm = ϕ[srcm]

        fp_in,  fm_in  = load_pair(fi, n, srcp, i, t_odd, N, CType)
        fp_out, fm_out = load_outgoing_pair(fi, n, srcp, i, t_odd, N, CType)

        if !solidified
            if (sup & (TYPE_F | TYPE_I)) != 0x00
                fluxp = fm_in - fp_out
                massn += sup == TYPE_F ? fluxp : CType(0.5) * (ϕp + ϕn) * fluxp
            end
            if (sum_ & (TYPE_F | TYPE_I)) != 0x00
                fluxm = fp_in - fm_out
                massn += sum_ == TYPE_F ? fluxm : CType(0.5) * (ϕm + ϕn) * fluxm
            end
        end

        fegp = feq(w[i],     ρ_gas, uxg, uyg, uzg, uug, cp, CType)
        fegm = feq(w[i + 1], ρ_gas, uxg, uyg, uzg, uug, cm, CType)
        fp_rec = fegm - fm_out + fegp
        fm_rec = fegp - fp_out + fegm
        store_reconstructed_pair!(
            fi, n, srcp, i, fm_rec, fp_rec,
            sup == TYPE_G, sum_ == TYPE_G, t_odd, N)
    end
    # D3Q7 axes are hydro indices 2, 4, 6. Do not walk the D3Q19 pairs.
    # 5th store argument is the minus reconstruction, 6th the plus.
    # A pore uses Henry's law, c_H = k_H ρ_b / 3. The free surface
    # (tag −1) is no-flux at the neighboring liquid concentration.
    # The interface cell itself is a depleted boundary, so its own c
    # is not the bath value.
    atm = tag[n] == Int32(-1)
    if !solidified && D > zero(CType) && (atm || k_H > zero(CType))
        c_H = k_H * CType(ρb[n]) * (CType(1) / CType(3))
        if atm
            c_sum = zero(CType)
            n_sum = zero(CType)
            for k2 in 1:3
                i2 = 2k2
                cp2, cm2 = c[i2], c[i2 + 1]
                jp = src_index(x, y, z, cp2[1], cp2[2], cp2[3], Nx, Ny, Nz)
                jm = src_index(x, y, z, cm2[1], cm2[2], cm2[3], Nx, Ny, Nz)
                if (flags[jp] & TYPE_SU) == TYPE_F
                    c_sum += CType(cfield[jp])
                    n_sum += one(CType)
                end
                if (flags[jm] & TYPE_SU) == TYPE_F
                    c_sum += CType(cfield[jm])
                    n_sum += one(CType)
                end
            end
            c_H = n_sum > zero(CType) ? c_sum / n_sum : CType(cfield[n])
        end
        # The unknown population is replaced by Henry anti-bounce-back.
        # (old − new) is staged in flux. The host then spreads that sum over
        # every TYPE_F rest, so the Dirichlet sink stays and the bath does
        # not lose it. Eq. 27, afterwards, is the only pore credit while any
        # liquid cell remains. With no TYPE_F cell the staged drop stays on
        # the pore: a bare film has nowhere else to put the solute. Tag −1
        # is not a pore.
        tag_n = tag[n]
        book = !atm && tag_n > Int32(0) && tag_n <= Int32(MAX_BUBBLES)
        restored = zero(CType)
        for k in 1:3
            i = 2k
            cp, cm = c[i], c[i + 1]
            srcp = src_index(x, y, z, cp[1], cp[2], cp[3], Nx, Ny, Nz)
            srcm = src_index(x, y, z, cm[1], cm[2], cm[3], Nx, Ny, Nz)
            sup = flags[srcp] & TYPE_SU
            sum_ = flags[srcm] & TYPE_SU
            (sup == TYPE_G || sum_ == TYPE_G) || continue
            u_ax = CType(cp[1]) * uxg + CType(cp[2]) * uyg + CType(cp[3]) * uzg
            ceq_p = ceq_axis(c_H, u_ax)
            ceq_m = ceq_axis(c_H, -u_ax)
            fp_out, fm_out = load_outgoing_pair(ci, n, srcp, i, t_odd, N, CType)
            cp_rec = ceq_m - fm_out + ceq_p
            cm_rec = ceq_p - fp_out + ceq_m
            if book
                # Even gas-plus writes src slot i; odd writes src slot i+1.
                # Even gas-minus writes n slot i+1; odd writes n slot i.
                if sup == TYPE_G
                    slot = ifelse(odd, f_index(srcp, i + 1, N), f_index(srcp, i, N))
                    restored += CType(ci[slot]) - cm_rec
                end
                if sum_ == TYPE_G
                    slot = ifelse(odd, f_index(n, i, N), f_index(n, i + 1, N))
                    restored += CType(ci[slot]) - cp_rec
                end
            end
            store_reconstructed_pair!(
                ci, n, srcp, i, cm_rec, cp_rec,
                sup == TYPE_G, sum_ == TYPE_G, t_odd, N)
        end
        if book && restored != zero(CType)
            acc_add!(flux, Int(tag_n), restored)
        end
    end
    mass[n] = massn
    @static if TEMPERATURE
        fillc = ϕn
        fillc = ifelse(fillc > zero(CType), fillc, zero(CType))
        reconstruct_g_boundaries!(t_odd, gi, T, flags, hT, Qin, x, y, z, n, N, Nx, Ny, Nz, CType, Eacc, fillc, ω_T)
    end
    return nothing
end

@kernel function surface_0_even_kernel!(
    fi, @Const(ρ), @Const(u), @Const(flags), mass, @Const(massex), @Const(ϕ), T, fs, gi,
    w::NTuple{Q, CType}, c::NTuple{Q, SVector{3, Int}},
    fx::CType, fy::CType, fz::CType, σ::CType, σT::CType, Tσ::CType,
    Λ_v::CType, T_v::CType, p0v::CType, β_v::CType,
    N::Int, Nx::Int, Ny::Int, Nz::Int, Eacc, hT, Qin, ω_T::CType,
    ci, flux, ρb, Pi, k_H::CType, D::CType, @Const(tag), @Const(cfield)
) where {Q, CType}
    n = @index(Global)
    @inbounds surface_0_body!(Val(false), fi, ρ, u, flags, mass, massex, ϕ, T, fs, gi, w, c, fx, fy, fz, σ, σT, Tσ, Λ_v, T_v, p0v, β_v, N, Nx, Ny, Nz, Int(n), Eacc, hT, Qin, ω_T, ci, flux, ρb, Pi, k_H, D, tag, cfield)
end

@kernel function surface_0_odd_kernel!(
    fi, @Const(ρ), @Const(u), @Const(flags), mass, @Const(massex), @Const(ϕ), T, fs, gi,
    w::NTuple{Q, CType}, c::NTuple{Q, SVector{3, Int}},
    fx::CType, fy::CType, fz::CType, σ::CType, σT::CType, Tσ::CType,
    Λ_v::CType, T_v::CType, p0v::CType, β_v::CType,
    N::Int, Nx::Int, Ny::Int, Nz::Int, Eacc, hT, Qin, ω_T::CType,
    ci, flux, ρb, Pi, k_H::CType, D::CType, @Const(tag), @Const(cfield)
) where {Q, CType}
    n = @index(Global)
    @inbounds surface_0_body!(Val(true), fi, ρ, u, flags, mass, massex, ϕ, T, fs, gi, w, c, fx, fy, fz, σ, σT, Tσ, Λ_v, T_v, p0v, β_v, N, Nx, Ny, Nz, Int(n), Eacc, hT, Qin, ω_T, ci, flux, ρb, Pi, k_H, D, tag, cfield)
end

end

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
            # Gas does not advect heat. geq(T, u) on this cell is what the
            # melt reads as the missing population; a nonzero u was a heat
            # source on the pore (Tmax climbed for hundreds of steps).
            store_geq!(gi, n, x, y, z, Tn, zero(CType), zero(CType), zero(CType), N, Nx, Ny, Nz, t_odd, CType)
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
                # A solid cell keeps the fill it froze with. Rewriting ϕ from
                # mass/ρ lets a density fluctuation move the bubble volume.
                massexn = zero(CType)
                ϕn = sus == TYPE_F ? one(CType) : ϕ[n]
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

end # SURFACE