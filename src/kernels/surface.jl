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
            @static if DIM == 3
                uz += CType(c[i][3])*fp_out + CType(c[i+1][3])*fm_out
            end
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
            @static if DIM == 3
                uzg = clamp(uz + fz / (CType(2) * ρn), -cs, cs)
            elseif DIM == 2
                uzg = zero(CType)
            end
        else
            uxg, uyg, uzg = ux, uy, uz
        end
        @static if TEMPERATURE
            inv2ρ = one(CType) / (CType(2) * ρn)
            if σT != zero(CType)
                mx, my, mz = marangoni_force(T, ϕ, flags, σT, x, y, z, n, Nx, Ny, Nz, CType)
                uxg = clamp(uxg + mx * inv2ρ, -cs, cs)
                uyg = clamp(uyg + my * inv2ρ, -cs, cs)
                @static if DIM == 3
                    uzg = clamp(uzg + mz * inv2ρ, -cs, cs)
                end
            end
            if Λ_v > zero(CType)
                rx, ry, rz = recoil_force(T, ϕ, n, x, y, z, Nx, Ny, Nz, Λ_v, T_v, p0v, β_v, CType)
                uxg = clamp(uxg + rx * inv2ρ, -cs, cs)
                uyg = clamp(uyg + ry * inv2ρ, -cs, cs)
                @static if DIM == 3
                    uzg = clamp(uzg + rz * inv2ρ, -cs, cs)
                end
            end
        end
    end
    # D2Q9 has cz = 0, but uzg still enters uug. Do not let fz back in.
    @static if DIM == 2
        uz = zero(CType)
        uzg = zero(CType)
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

end # SURFACE