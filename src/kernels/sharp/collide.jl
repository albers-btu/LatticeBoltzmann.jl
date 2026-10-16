using KernelAbstractions

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
    C_rad::CType, T_rad::CType, τ_p::CType, T_p::CType, T_stick::CType,
    ::Val{thermal}, g_odd::Bool,
    N::Int, Nx::Int, Ny::Int, Nz::Int, n, Eacc, Macc
) where {odd, Q, CType, thermal}
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
        fp, fm = load_bb_pair(fi, flags, n, srcp, srcm, i, t_odd, N, CType)
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
          if thermal
            ωTn = omega_T_from_alpha(prop_fs_T(fs[n], α_s, α_sT, α_l, α_lT, T[n], T_avg, CType(1e-6)))
            debit = zero(CType)
            fillc = ϕ[n]
            fillc = ifelse(fillc > zero(CType), fillc, zero(CType))
            if τ_p > zero(CType)
                # Molten parcels join mass on the host. mp is the cold remainder.
                mp_src = msrc[n] * ρn
                mpn = mp[n] + mp_src
                acc_add!(Eacc, EACC_POWDER, mp_src * sensible_H(T_p, γ_s))
                acc_add!(Macc, MACC_POWDER, mp_src)
                Tpred = T[n] + Qin[n]
                if Tpred >= Ts
                    # mp is cold solid. Mix it. (T − T_p + Λ) charged against ρ
                    # puts a full cell of powder below H = 0 (about −550 K).
                    dm = mpn < ρn ? mpn : ρn
                    mpn -= dm
                    m_store = mass[n]
                    m_now = m_store > zero(CType) ? m_store : ρn
                    mass[n] = m_store + dm
                    fsn = fs[n]
                    γc = blend_phase(fsn, γ_s, γ_l)
                    h_now = cell_enthalpy(T[n], fsn, Λ, γc) + Qin[n]
                    h_p = sensible_H(T_p, γ_s)
                    m_new = m_now + dm
                    h_mix = (m_now * h_now + dm * h_p) / m_new
                    debit = h_now - h_mix
                    Qin[n] -= debit
                elseif Tpred < T_stick
                    # Cold solid sheds. Hot solid holds mp until the cell melts.
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
                g_odd, gi, T, Qin, hT, flags, flagsn, fs, ux, uy, uz,
                fxn, fyn, fzn, fx, fy, fz,
                ωTn, β, T_avg, Λ, Ts, Tl, γ_s, γ_l, Λ_v, T_v, C_hk, p0v, β_v, C_rad, T_rad, x, y, z, Nx, Ny, Nz, N, n, CType, Eacc, fillc)
            debit != zero(CType) && (Qin[n] += debit)
            if mevap > zero(CType)
                mass[n] -= mevap * ρn
                acc_add!(Macc, MACC_EVAP, mevap * ρn)
            end
          else
            dT = T[n] - T_avg
            fxn -= fx * β * dT
            fyn -= fy * β * dT
            @static if DIM == 3
            fzn -= fz * β * dT
            end
          end
            if (flagsn & TYPE_SU) == TYPE_I && !is_solid_fraction(fs[n])
                if σT != zero(CType)
                    mx, my, mz = marangoni_force(T, ϕ, flags, σT, x, y, z, n, Nx, Ny, Nz, CType)
                    fxn += mx; fyn += my; fzn += mz
                end
            end
        end
        @static if TEMPERATURE
            νc = prop_fs_T(fs[n], ν_s, ν_sT, ν_l, ν_lT, T[n], T_avg, CType(1e-8))
            # Substep ν is ~0.001 (ω ≈ 2), so the anti-bounce does not
            # damp. A liquid interface cell is held at ν ≥ 0.05 (ω ≈ 1.54).
            # At 0.02 the same spot still climbed from |u|≈0.08 to 0.56.
            if (flagsn & TYPE_SU) == TYPE_I && !is_solid_fraction(fs[n])
                νc = ifelse(νc > CType(0.05), νc, CType(0.05))
            end
            ω = omega_from_nu(νc)
            dx, dy, dz = darcy_force(fs[n], ux, uy, uz, ρn, νc, K0)
            if K0 > zero(CType) && (one(CType) - fs[n]) < CType(1e-3)
                fxn, fyn, fzn = dx, dy, dz
            else
                fxn += dx; fyn += dy; fzn += dz
            end
        end
        @static if TEMPERATURE
            # Momentum after Guo. Substep ν never carries Marangoni down to
            # the wall, so a component walks onto ±c_s and stays there.
            # Replace the force so a liquid interface streams at |u| ≤ 0.2.
            if (flagsn & TYPE_SU) == TYPE_I && !is_solid_fraction(fs[n])
                umax = CType(0.2)
                umx = ux + fxn * invρ
                umy = uy + fyn * invρ
                umz = uz + fzn * invρ
                usq = umx * umx + umy * umy + umz * umz
                if usq > umax * umax
                    s = umax / sqrt(usq)
                    fxn = ρn * (s * umx - ux)
                    fyn = ρn * (s * umy - uy)
                    fzn = ρn * (s * umz - uz)
                end
            end
        end
        @static if APPLY_FORCE
            ux += fxn * invρ * CType(0.5)
            uy += fyn * invρ * CType(0.5)
            uz += fzn * invρ * CType(0.5)
        end
        @static if DIM == 2
            fzn = zero(CType)
            uz = zero(CType)
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
        massn = mass[n]
        # A solid interface that powder has filled past one cell still opens
        # the gas neighbor. A solid cell at or below ρ does not.
        if !frozen_i || massn > ρn
            noF = true; noG = true
            for i in 2:Q
                src = src_index(x, y, z, c[i][1], c[i][2], c[i][3], Nx, Ny, Nz)
                suj = flags[src] & TYPE_SU
                noF &= suj != TYPE_F
                noG &= suj != TYPE_G
            end
            if massn > ρn || noG
                flags[n] = (flagsn & ~TYPE_SU) | TYPE_IF
            elseif massn < 0 || (noF && !touches_metal(flags, x, y, z, c, Nx, Ny, Nz))
                # A new cell on an interface parent has no TYPE_F neighbor.
                # Any metal neighbor keeps a non-negative cell alive.
                flags[n] = (flagsn & ~TYPE_SU) | TYPE_IG
            end
        end
    end

    @static if DIM == 2
        uz = zero(CType)
    end
    uu = CType(1.5) * (ux*ux + uy*uy + uz*uz)
    @static if KBC
        kbc_store!(t_odd, fi, fn1, pairs, w, c, ρn, ux, uy, uz, uu, fxn, fyn, fzn, ω,
                   N, Nx, Ny, Nz, n, x, y, z, CType)
    else
        @static if TRT
            ωm = omega_minus(ω)
        else
            ωm = ω
        end
        Fi0 = zero(CType)
        @static if APPLY_FORCE
            Fi0 = guo_rest(ω, w[1], ux, uy, uz, fxn, fyn, fzn, c[1], CType)
        end
        fe0 = feq(w[1], ρn, ux, uy, uz, uu, c[1], CType)
        f0 = srt(ω, fn1, w[1], ρn, ux, uy, uz, uu, c[1]) + Fi0
        fi[f_index(n, 1, N)] = eltype(fi)(bound_population(f0, fe0))
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
            fp_s, fm_s = bound_pair(fp_s, fm_s, feqp, feqm)
            src = src_index(x, y, z, c[i][1], c[i][2], c[i][3], Nx, Ny, Nz)
            store_pair!(fi, n, src, i, fp_s, fm_s, t_odd, N)
        end
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
    C_rad::CType, T_rad::CType, τ_p::CType, T_p::CType, T_stick::CType, ::Val{thermal}, g_odd::Bool,
    N::Int, Nx::Int, Ny::Int, Nz::Int, Eacc, Macc
) where {Q, CType, thermal}
    n = @index(Global)
    @inbounds stream_collide_surface_body!(Val(false), flags, fi, ρ, u, F, mass, gi, T, Qin, hT, ϕ, fs, msrc, mp, w, c, ω, fx, fy, fz, ω_T, β, T_avg, σT, Λ, Ts, Tl, K0, α_s, α_l, α_sT, α_lT, γ_s, γ_l, ν_s, ν_l, ν_sT, ν_lT, Λ_v, T_v, C_hk, p0v, β_v, C_rad, T_rad, τ_p, T_p, T_stick, Val(thermal), g_odd, N, Nx, Ny, Nz, Int(n), Eacc, Macc)
end

@kernel function stream_collide_odd_kernel!(
    flags, fi, ρ, u, F, mass, gi, T, Qin, hT, ϕ, fs, msrc, mp,
    w::NTuple{Q, CType}, c::NTuple{Q, SVector{3, Int}},
    ω::CType, fx::CType, fy::CType, fz::CType,
    ω_T::CType, β::CType, T_avg::CType, σT::CType, Λ::CType, Ts::CType, Tl::CType, K0::CType,
    α_s::CType, α_l::CType, α_sT::CType, α_lT::CType, γ_s::CType, γ_l::CType, ν_s::CType, ν_l::CType, ν_sT::CType, ν_lT::CType,
    Λ_v::CType, T_v::CType, C_hk::CType, p0v::CType, β_v::CType,
    C_rad::CType, T_rad::CType, τ_p::CType, T_p::CType, T_stick::CType, ::Val{thermal}, g_odd::Bool,
    N::Int, Nx::Int, Ny::Int, Nz::Int, Eacc, Macc
) where {Q, CType, thermal}
    n = @index(Global)
    @inbounds stream_collide_surface_body!(Val(true), flags, fi, ρ, u, F, mass, gi, T, Qin, hT, ϕ, fs, msrc, mp, w, c, ω, fx, fy, fz, ω_T, β, T_avg, σT, Λ, Ts, Tl, K0, α_s, α_l, α_sT, α_lT, γ_s, γ_l, ν_s, ν_l, ν_sT, ν_lT, Λ_v, T_v, C_hk, p0v, β_v, C_rad, T_rad, τ_p, T_p, T_stick, Val(thermal), g_odd, N, Nx, Ny, Nz, Int(n), Eacc, Macc)
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

end # SURFACE