using KernelAbstractions

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
    do_thermal::Bool, g_odd::Bool,
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
            fp, fm = load_bb_pair(fi, flags, n, srcp, srcm, i, t_odd, N, CType)
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
                if do_thermal
                    ωTn = omega_T_from_alpha(prop_fs_T(fs[n], α_s, α_sT, α_l, α_lT, T[n], T_avg, CType(1e-6)))
                    fxn, fyn, fzn, _ = collide_temperature!(
                        g_odd, gi, T, Qin, hT, flags, flagsn, fs, ux, uy, uz,
                        fxn, fyn, fzn, fx, fy, fz,
                        ωTn, β, T_avg, Λ, Ts, Tl, γ_s, γ_l, Λ_v, T_v, C_hk, p0v, β_v, C_rad, T_rad, x, y, z, Nx, Ny, Nz, N, n, CType, Eacc, one(CType))
                else
                    dT = T[n] - T_avg
                    fxn -= fx * β * dT
                    fyn -= fy * β * dT
                    @static if DIM == 3
                        fzn -= fz * β * dT
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
            @static if DIM == 2
                fzn = zero(CType)
                uz = zero(CType)
            end
            ux = clamp(ux, -cs, cs)
            uy = clamp(uy, -cs, cs)
            uz = clamp(uz, -cs, cs)
        end

        @static if DIM == 2
            uz = zero(CType)
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
    do_thermal::Bool, g_odd::Bool,
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
    src3  = src_index(x, y, z, c[3][1],  c[3][2],  c[3][3],  Nx, Ny, Nz)
    src5  = src_index(x, y, z, c[5][1],  c[5][2],  c[5][3],  Nx, Ny, Nz)
    src7  = src_index(x, y, z, c[7][1],  c[7][2],  c[7][3],  Nx, Ny, Nz)
    src9  = src_index(x, y, z, c[9][1],  c[9][2],  c[9][3],  Nx, Ny, Nz)
    src11 = src_index(x, y, z, c[11][1], c[11][2], c[11][3], Nx, Ny, Nz)
    src13 = src_index(x, y, z, c[13][1], c[13][2], c[13][3], Nx, Ny, Nz)
    src15 = src_index(x, y, z, c[15][1], c[15][2], c[15][3], Nx, Ny, Nz)
    src17 = src_index(x, y, z, c[17][1], c[17][2], c[17][3], Nx, Ny, Nz)
    src19 = src_index(x, y, z, c[19][1], c[19][2], c[19][3], Nx, Ny, Nz)

    fp2,  fm3  = load_bb_pair(fi, flags, n, src2,  src3,  2,  t_odd, N, CType)
    fp4,  fm5  = load_bb_pair(fi, flags, n, src4,  src5,  4,  t_odd, N, CType)
    fp6,  fm7  = load_bb_pair(fi, flags, n, src6,  src7,  6,  t_odd, N, CType)
    fp8,  fm9  = load_bb_pair(fi, flags, n, src8,  src9,  8,  t_odd, N, CType)
    fp10, fm11 = load_bb_pair(fi, flags, n, src10, src11, 10, t_odd, N, CType)
    fp12, fm13 = load_bb_pair(fi, flags, n, src12, src13, 12, t_odd, N, CType)
    fp14, fm15 = load_bb_pair(fi, flags, n, src14, src15, 14, t_odd, N, CType)
    fp16, fm17 = load_bb_pair(fi, flags, n, src16, src17, 16, t_odd, N, CType)
    fp18, fm19 = load_bb_pair(fi, flags, n, src18, src19, 18, t_odd, N, CType)

    @static if MOVING_BOUNDARIES
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
            if do_thermal
                ωTn = omega_T_from_alpha(prop_fs_T(fs[n], α_s, α_sT, α_l, α_lT, T[n], T_avg, CType(1e-6)))
                fxn, fyn, fzn, _ = collide_temperature!(
                    g_odd, gi, T, Qin, hT, flags, flagsn, fs, ux, uy, uz,
                    fxn, fyn, fzn, fx, fy, fz,
                    ωTn, β, T_avg, Λ, Ts, Tl, γ_s, γ_l, Λ_v, T_v, C_hk, p0v, β_v, C_rad, T_rad, x, y, z, Nx, Ny, Nz, N, n, CType, Eacc, one(CType))
            else
                dT = T[n] - T_avg
                fxn -= fx * β * dT
                fyn -= fy * β * dT
                fzn -= fz * β * dT
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
    do_thermal::Bool, g_odd::Bool,
    N::Int, Nx::Int, Ny::Int, Nz::Int, Eacc
) where {Q, CType}
    n = @index(Global)
    @inbounds stream_collide_body!(Val(false), flags, fi, ρ, u, F, gi, T, Qin, hT, fs, w, c, ω, fx, fy, fz, ω_T, β, T_avg, Λ, Ts, Tl, K0, α_s, α_l, α_sT, α_lT, γ_s, γ_l, ν_s, ν_l, ν_sT, ν_lT, Λ_v, T_v, C_hk, p0v, β_v, C_rad, T_rad, do_thermal, g_odd, N, Nx, Ny, Nz, Int(n), Eacc)
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
    do_thermal::Bool, g_odd::Bool,
    N::Int, Nx::Int, Ny::Int, Nz::Int, Eacc
) where {Q, CType}
    n = @index(Global)
    @inbounds stream_collide_body!(Val(true), flags, fi, ρ, u, F, gi, T, Qin, hT, fs, w, c, ω, fx, fy, fz, ω_T, β, T_avg, Λ, Ts, Tl, K0, α_s, α_l, α_sT, α_lT, γ_s, γ_l, ν_s, ν_l, ν_sT, ν_lT, Λ_v, T_v, C_hk, p0v, β_v, C_rad, T_rad, do_thermal, g_odd, N, Nx, Ny, Nz, Int(n), Eacc)
end

end # not SURFACE

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
    do_thermal::Bool, g_odd::Bool,
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
          if do_thermal
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
    C_rad::CType, T_rad::CType, τ_p::CType, T_p::CType, do_thermal::Bool, g_odd::Bool,
    N::Int, Nx::Int, Ny::Int, Nz::Int, Eacc, Macc
) where {Q, CType}
    n = @index(Global)
    @inbounds stream_collide_surface_body!(Val(false), flags, fi, ρ, u, F, mass, gi, T, Qin, hT, ϕ, fs, msrc, mp, w, c, ω, fx, fy, fz, ω_T, β, T_avg, σT, Λ, Ts, Tl, K0, α_s, α_l, α_sT, α_lT, γ_s, γ_l, ν_s, ν_l, ν_sT, ν_lT, Λ_v, T_v, C_hk, p0v, β_v, C_rad, T_rad, τ_p, T_p, do_thermal, g_odd, N, Nx, Ny, Nz, Int(n), Eacc, Macc)
end

@kernel function stream_collide_odd_kernel!(
    flags, fi, ρ, u, F, mass, gi, T, Qin, hT, ϕ, fs, msrc, mp,
    w::NTuple{Q, CType}, c::NTuple{Q, SVector{3, Int}},
    ω::CType, fx::CType, fy::CType, fz::CType,
    ω_T::CType, β::CType, T_avg::CType, σT::CType, Λ::CType, Ts::CType, Tl::CType, K0::CType,
    α_s::CType, α_l::CType, α_sT::CType, α_lT::CType, γ_s::CType, γ_l::CType, ν_s::CType, ν_l::CType, ν_sT::CType, ν_lT::CType,
    Λ_v::CType, T_v::CType, C_hk::CType, p0v::CType, β_v::CType,
    C_rad::CType, T_rad::CType, τ_p::CType, T_p::CType, do_thermal::Bool, g_odd::Bool,
    N::Int, Nx::Int, Ny::Int, Nz::Int, Eacc, Macc
) where {Q, CType}
    n = @index(Global)
    @inbounds stream_collide_surface_body!(Val(true), flags, fi, ρ, u, F, mass, gi, T, Qin, hT, ϕ, fs, msrc, mp, w, c, ω, fx, fy, fz, ω_T, β, T_avg, σT, Λ, Ts, Tl, K0, α_s, α_l, α_sT, α_lT, γ_s, γ_l, ν_s, ν_l, ν_sT, ν_lT, Λ_v, T_v, C_hk, p0v, β_v, C_rad, T_rad, τ_p, T_p, do_thermal, g_odd, N, Nx, Ny, Nz, Int(n), Eacc, Macc)
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