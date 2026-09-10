using KernelAbstractions

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

@static if !SURFACE

@kernel function initialize_kernel!(
    ρ, u, fi, flags,
    w::NTuple{Q, CType}, 
    c::NTuple{Q, SVector{3, Int}},
    N::Int, Nx::Int, Ny::Int, Nz::Int
) where {Q, CType}
    n = @index(Global)
    @inbounds begin
        n0 = n - 1
        ρn = ρ[n]
        ux, uy, uz = u[n, 1], u[n, 2], u[n, 3]

        if (flags[n] & TYPE_S) == TYPE_S
            u[n, 1] = zero(CType)
            u[n, 2] = zero(CType)
            u[n, 3] = zero(CType)
        else
            uu = CType(1.5) * (ux*ux + uy*uy + uz*uz)
            fi[f_index(n, 1, N)] = eltype(fi)(w[1] * ρn * (one(CType) - uu))

            x = n0 % Nx
            y = (n0 ÷ Nx) % Ny
            z = n0 ÷ (Nx * Ny)

            for k in 1:((Q - 1) ÷ 2)
                i = 2k
                cp, cm = c[i], c[i + 1]
                cup = CType(cp[1])*ux + CType(cp[2])*uy + CType(cp[3])*uz
                cum = CType(cm[1])*ux + CType(cm[2])*uy + CType(cm[3])*uz
                feqp = w[i]     * ρn * (one(CType) + CType(3.0)*cup + CType(4.5)*cup*cup - uu)
                feqm = w[i + 1] * ρn * (one(CType) + CType(3.0)*cum + CType(4.5)*cum*cum - uu)
                src = src_index(x, y, z, cp[1], cp[2], cp[3], Nx, Ny, Nz)
                store_pair!(fi, n, src, i, feqp, feqm, Val(true), N)
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
            feq(w[i+1], ρn, ux, uy, uz, uu, c[i+1], CType),
            t_odd, N)
    end
    return nothing
end

@inline function initialize_body!(
    ρ, u, fi, flags, mass, massex, ϕ,
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

    if (flagsn & (TYPE_S | TYPE_E | TYPE_T | TYPE_F | TYPE_I)) == 0x00
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
        end
    end

    if (flagsn & TYPE_S) == TYPE_S
        u[n, 1] = zero(CType); u[n, 2] = zero(CType); u[n, 3] = zero(CType)
    elseif (flagsn & TYPE_SU) == TYPE_G
        u[n, 1] = zero(CType); u[n, 2] = zero(CType); u[n, 3] = zero(CType)
        ϕn = zero(CType)
    else
        if (flagsn & TYPE_SU) == TYPE_I && (ϕn < 0 || ϕn > 1)
            ϕn = CType(0.5)
        elseif (flagsn & TYPE_SU) == TYPE_F
            ϕn = one(CType)
        end
        store_feq!(fi, n, x, y, z, ρn, ux, uy, uz, w, c, N, Nx, Ny, Nz, Val(true))
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
    N::Int, Nx::Int, Ny::Int, Nz::Int
) where {Q, CType}
    n = @index(Global)
    @inbounds initialize_body!(ρ, u, fi, flags, mass, massex, ϕ, w, c, N, Nx, Ny, Nz, Int(n))
end

end # SURFACE

@static if !SURFACE

# generic fallback
@inline function stream_collide_body!(
    t_odd::Val{odd},
    flags, fi,
    w::NTuple{Q, CType}, 
    c::NTuple{Q, SVector{3, Int}},
    ω::CType, fx::CType, fy::CType, fz::CType,
    N::Int, Nx::Int, Ny::Int, Nz::Int, n
) where {odd, Q, CType}
    if (flags[n] & TYPE_S) != TYPE_S
        n0 = n - 1

        x = n0 % Nx
        y = (n0 ÷ Nx) % Ny
        z = n0 ÷ (Nx * Ny)

        fn1 = CType(fi[f_index(n, 1, N)])
        NP  = (Q - 1) ÷ 2

        # pairs: (2, 3), (4, 5), ...
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
            cp, cm = c[i], c[i + 1]
            ux += CType(cp[1]) * fp + CType(cm[1]) * fm
            uy += CType(cp[2]) * fp + CType(cm[2]) * fm
            uz += CType(cp[3]) * fp + CType(cm[3]) * fm
        end

        cs = CType(1) / sqrt(CType(3))
        fxn = fx; fyn = fy; fzn = fz
        if ρn <= zero(CType)
            ρn = one(CType)
            ux = zero(CType); uy = zero(CType); uz = zero(CType)
            fxn = zero(CType); fyn = zero(CType); fzn = zero(CType)
        else
            invρ = one(CType) / ρn
            ux *= invρ; uy *= invρ; uz *= invρ
            @static if VOLUME_FORCE
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
        @static if VOLUME_FORCE
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
            @static if VOLUME_FORCE
                Fip, Fim = guo_pair(ω, ωm, w[i], w[i + 1], ux, uy, uz, fxn, fyn, fzn, cp, cm, CType)
                fp_s += Fip
                fm_s += Fim
            end
            src = src_index(x, y, z, cp[1], cp[2], cp[3], Nx, Ny, Nz)
            store_pair!(fi, n, src, i, fp_s, fm_s, t_odd, N)
        end
    end
    return nothing
end

@inline function stream_collide_body!(
    t_odd::Val{odd},
    flags, fi,
    w::NTuple{19, CType},
    c::NTuple{19, SVector{3,Int}},
    ω::CType, fx::CType, fy::CType, fz::CType,
    N::Int, Nx::Int, Ny::Int, Nz::Int, n,
) where {odd, CType}
    if (flags[n] & TYPE_S) == TYPE_S
        return nothing
    end

    n0 = n - 1
    x  = n0 % Nx
    y  = (n0 ÷ Nx) % Ny
    z  = n0 ÷ (Nx * Ny)

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
    if ρn <= zero(CType)
        ρn = one(CType)
        ux = zero(CType); uy = zero(CType); uz = zero(CType)
        fxn = zero(CType); fyn = zero(CType); fzn = zero(CType)
    else
        invρ = one(CType) / ρn
        ux *= invρ; uy *= invρ; uz *= invρ
        @static if VOLUME_FORCE
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
    @static if VOLUME_FORCE
        Fi0 = guo_rest(ω, w[1], ux, uy, uz, fxn, fyn, fzn, c[1], CType)
    end
    fi[f_index(n, 1, N)] = eltype(fi)(
        (one(CType) - ω) * fn1 + ω * (w[1] * ρn * (one(CType) - uu)) + Fi0)


    let feqp = feq(w[2], ρn, ux, uy, uz, uu, c[2], CType)
        feqm = feq(w[3], ρn, ux, uy, uz, uu, c[3], CType)
        fp_s, fm_s = collide_pair(ω, ωm, fp2, fm3, feqp, feqm)
        @static if VOLUME_FORCE
            Fip, Fim = guo_pair(ω, ωm, w[2], w[3], ux, uy, uz, fxn, fyn, fzn, c[2], c[3], CType)
            fp_s += Fip
            fm_s += Fim
        end
        store_pair!(fi, n, src2, 2, fp_s, fm_s, t_odd, N)
    end
    let feqp = feq(w[4], ρn, ux, uy, uz, uu, c[4], CType)
        feqm = feq(w[5], ρn, ux, uy, uz, uu, c[5], CType)
        fp_s, fm_s = collide_pair(ω, ωm, fp4, fm5, feqp, feqm)
        @static if VOLUME_FORCE
            Fip, Fim = guo_pair(ω, ωm, w[4], w[5], ux, uy, uz, fxn, fyn, fzn, c[4], c[5], CType)
            fp_s += Fip
            fm_s += Fim
        end
        store_pair!(fi, n, src4, 4, fp_s, fm_s, t_odd, N)
    end
    let feqp = feq(w[6], ρn, ux, uy, uz, uu, c[6], CType)
        feqm = feq(w[7], ρn, ux, uy, uz, uu, c[7], CType)
        fp_s, fm_s = collide_pair(ω, ωm, fp6, fm7, feqp, feqm)
        @static if VOLUME_FORCE
            Fip, Fim = guo_pair(ω, ωm, w[6], w[7], ux, uy, uz, fxn, fyn, fzn, c[6], c[7], CType)
            fp_s += Fip
            fm_s += Fim
        end
        store_pair!(fi, n, src6, 6, fp_s, fm_s, t_odd, N)
    end
    let feqp = feq(w[8], ρn, ux, uy, uz, uu, c[8], CType)
        feqm = feq(w[9], ρn, ux, uy, uz, uu, c[9], CType)
        fp_s, fm_s = collide_pair(ω, ωm, fp8, fm9, feqp, feqm)
        @static if VOLUME_FORCE
            Fip, Fim = guo_pair(ω, ωm, w[8], w[9], ux, uy, uz, fxn, fyn, fzn, c[8], c[9], CType)
            fp_s += Fip
            fm_s += Fim
        end
        store_pair!(fi, n, src8, 8, fp_s, fm_s, t_odd, N)
    end
    let feqp = feq(w[10], ρn, ux, uy, uz, uu, c[10], CType)
        feqm = feq(w[11], ρn, ux, uy, uz, uu, c[11], CType)
        fp_s, fm_s = collide_pair(ω, ωm, fp10, fm11, feqp, feqm)
        @static if VOLUME_FORCE
            Fip, Fim = guo_pair(ω, ωm, w[10], w[11], ux, uy, uz, fxn, fyn, fzn, c[10], c[11], CType)
            fp_s += Fip
            fm_s += Fim
        end
        store_pair!(fi, n, src10, 10, fp_s, fm_s, t_odd, N)
    end
    let feqp = feq(w[12], ρn, ux, uy, uz, uu, c[12], CType)
        feqm = feq(w[13], ρn, ux, uy, uz, uu, c[13], CType)
        fp_s, fm_s = collide_pair(ω, ωm, fp12, fm13, feqp, feqm)
        @static if VOLUME_FORCE
            Fip, Fim = guo_pair(ω, ωm, w[12], w[13], ux, uy, uz, fxn, fyn, fzn, c[12], c[13], CType)
            fp_s += Fip
            fm_s += Fim
        end
        store_pair!(fi, n, src12, 12, fp_s, fm_s, t_odd, N)
    end
    let feqp = feq(w[14], ρn, ux, uy, uz, uu, c[14], CType)
        feqm = feq(w[15], ρn, ux, uy, uz, uu, c[15], CType)
        fp_s, fm_s = collide_pair(ω, ωm, fp14, fm15, feqp, feqm)
        @static if VOLUME_FORCE
            Fip, Fim = guo_pair(ω, ωm, w[14], w[15], ux, uy, uz, fxn, fyn, fzn, c[14], c[15], CType)
            fp_s += Fip
            fm_s += Fim
        end
        store_pair!(fi, n, src14, 14, fp_s, fm_s, t_odd, N)
    end
    let feqp = feq(w[16], ρn, ux, uy, uz, uu, c[16], CType)
        feqm = feq(w[17], ρn, ux, uy, uz, uu, c[17], CType)
        fp_s, fm_s = collide_pair(ω, ωm, fp16, fm17, feqp, feqm)
        @static if VOLUME_FORCE
            Fip, Fim = guo_pair(ω, ωm, w[16], w[17], ux, uy, uz, fxn, fyn, fzn, c[16], c[17], CType)
            fp_s += Fip
            fm_s += Fim
        end
        store_pair!(fi, n, src16, 16, fp_s, fm_s, t_odd, N)
    end
    let feqp = feq(w[18], ρn, ux, uy, uz, uu, c[18], CType)
        feqm = feq(w[19], ρn, ux, uy, uz, uu, c[19], CType)
        fp_s, fm_s = collide_pair(ω, ωm, fp18, fm19, feqp, feqm)
        @static if VOLUME_FORCE
            Fip, Fim = guo_pair(ω, ωm, w[18], w[19], ux, uy, uz, fxn, fyn, fzn, c[18], c[19], CType)
            fp_s += Fip
            fm_s += Fim
        end
        store_pair!(fi, n, src18, 18, fp_s, fm_s, t_odd, N)
    end

    return nothing
end

@kernel function stream_collide_even_kernel!(
    @Const(flags), fi,
    w::NTuple{Q, CType}, 
    c::NTuple{Q, SVector{3, Int}},
    ω::CType, fx::CType, fy::CType, fz::CType,
    N::Int, Nx::Int, Ny::Int, Nz::Int,
) where {Q, CType}
    n = @index(Global)
    @inbounds stream_collide_body!(Val(false), flags, fi, w, c, ω, fx, fy, fz, N, Nx, Ny, Nz, Int(n))
end

@kernel function stream_collide_odd_kernel!(
    @Const(flags), fi,
    w::NTuple{Q, CType}, 
    c::NTuple{Q, SVector{3, Int}},
    ω::CType, fx::CType, fy::CType, fz::CType,
    N::Int, Nx::Int, Ny::Int, Nz::Int,
) where {Q, CType}
    n = @index(Global)
    @inbounds stream_collide_body!(Val(true), flags, fi, w, c, ω, fx, fy, fz, N, Nx, Ny, Nz, Int(n))
end

end

@static if SURFACE

@inline function stream_collide_surface_body!(
    t_odd::Val{odd},
    flags, fi, ρ, u, mass,
    w::NTuple{Q, CType},
    c::NTuple{Q, SVector{3, Int}},
    ω::CType, fx::CType, fy::CType, fz::CType,
    N::Int, Nx::Int, Ny::Int, Nz::Int, n
) where {odd, Q, CType}
    flagsn = flags[n]
    if (flagsn & TYPE_BO) == TYPE_S || (flagsn & TYPE_SU) == TYPE_G
        return nothing
    end

    n0 = n - 1
    x = n0 % Nx; y = (n0 ÷ Nx) % Ny; z = n0 ÷ (Nx * Ny)
    NP = (Q - 1) ÷ 2

    fn1 = CType(fi[f_index(n, 1, N)])
    pairs = ntuple(Val(NP)) do k
        i = 2k
        cp = c[i]
        src = src_index(x, y, z, cp[1], cp[2], cp[3], Nx, Ny, Nz)
        load_pair(fi, n, src, i, t_odd, N, CType)
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
    if ρn <= zero(CType)
        ρn = one(CType)
        ux = zero(CType); uy = zero(CType); uz = zero(CType)
        fxn = zero(CType); fyn = zero(CType); fzn = zero(CType)
    else
        invρ = one(CType) / ρn
        ux *= invρ; uy *= invρ; uz *= invρ
        @static if VOLUME_FORCE
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

    uu = CType(1.5) * (ux*ux + uy*uy + uz*uz)
    @static if TRT
        ωm = omega_minus(ω)
    else
        ωm = ω
    end

    Fi0 = zero(CType)
    @static if VOLUME_FORCE
        Fi0 = guo_rest(ω, w[1], ux, uy, uz, fxn, fyn, fzn, c[1], CType)
    end
    fi[f_index(n, 1, N)] = eltype(fi)(srt(ω, fn1, w[1], ρn, ux, uy, uz, uu, c[1]) + Fi0)

    for k in 1:NP
        i = 2k
        fp, fm = pairs[k]
        feqp = feq(w[i],     ρn, ux, uy, uz, uu, c[i],     CType)
        feqm = feq(w[i + 1], ρn, ux, uy, uz, uu, c[i + 1], CType)
        fp_s, fm_s = collide_pair(ω, ωm, fp, fm, feqp, feqm)
        @static if VOLUME_FORCE
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
    flags, fi, ρ, u, mass,
    w::NTuple{Q, CType}, c::NTuple{Q, SVector{3, Int}},
    ω::CType, fx::CType, fy::CType, fz::CType,
    N::Int, Nx::Int, Ny::Int, Nz::Int
) where {Q, CType}
    n = @index(Global)
    @inbounds stream_collide_surface_body!(Val(false), flags, fi, ρ, u, mass, w, c, ω, fx, fy, fz, N, Nx, Ny, Nz, Int(n))
end

@kernel function stream_collide_odd_kernel!(
    flags, fi, ρ, u, mass,
    w::NTuple{Q, CType}, c::NTuple{Q, SVector{3, Int}},
    ω::CType, fx::CType, fy::CType, fz::CType,
    N::Int, Nx::Int, Ny::Int, Nz::Int
) where {Q, CType}
    n = @index(Global)
    @inbounds stream_collide_surface_body!(Val(true), flags, fi, ρ, u, mass, w, c, ω, fx, fy, fz, N, Nx, Ny, Nz, Int(n))
end

end

@inline function moments_body!(
    t_odd::Val{odd},
    ρ, u, flags, fi,
    w::NTuple{Q, CType},
    c::NTuple{Q, SVector{3, Int}},
    N::Int, Nx::Int, Ny::Int, Nz::Int, n
) where {odd, Q, CType}
    flagsn = flags[n]
    su = flagsn & TYPE_SU

    skip = (flagsn & TYPE_S) == TYPE_S
    @static if SURFACE
        skip |= (su == TYPE_G) | (su == TYPE_IG)
    end
    if skip
        ρ[n] = one(CType)
        u[n, 1] = zero(CType)
        u[n, 2] = zero(CType)
        u[n, 3] = zero(CType)
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
    return nothing
end

@kernel function moments_even_kernel!(
    ρ, u, @Const(flags), fi, w::NTuple{Q, CType},
    c, N, Nx, Ny, Nz
) where {Q, CType}
    n = @index(Global)
    @inbounds moments_body!(Val(false), ρ, u, flags, fi, w, c, N, Nx, Ny, Nz, Int(n))
end

@kernel function moments_odd_kernel!(
    ρ, u, @Const(flags), fi, w::NTuple{Q, CType},
    c, N, Nx, Ny, Nz
) where {Q, CType}
    n = @index(Global)
    @inbounds moments_body!(Val(true), ρ, u, flags, fi, w, c, N, Nx, Ny, Nz, Int(n))
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
    fi, ρ, u, flags,
    w::NTuple{Q, CType}, c::NTuple{Q, SVector{3, Int}},
    N::Int, Nx::Int, Ny::Int, Nz::Int, n
) where {odd, Q, CType}
    sus = flags[n] & (TYPE_SU | TYPE_S)
    n0 = n - 1
    x = n0 % Nx; y = (n0 ÷ Nx) % Ny; z = n0 ÷ (Nx * Ny)

    if sus == TYPE_GI
        ρn, ux, uy, uz = average_neighbors_non_gas(ρ, u, flags, x, y, z, c, Nx, Ny, Nz, CType)
        store_feq!(fi, n, x, y, z, ρn, ux, uy, uz, w, c, N, Nx, Ny, Nz, t_odd)
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

@kernel function surface_2_even_kernel!(fi, @Const(ρ), @Const(u), flags, w::NTuple{Q,CType}, c, N, Nx, Ny, Nz) where {Q, CType}
    n = @index(Global)
    @inbounds surface_2_body!(Val(false), fi, ρ, u, flags, w, c, N, Nx, Ny, Nz, Int(n))
end
@kernel function surface_2_odd_kernel!(fi, @Const(ρ), @Const(u), flags, w::NTuple{Q,CType}, c, N, Nx, Ny, Nz) where {Q, CType}
    n = @index(Global)
    @inbounds surface_2_body!(Val(true), fi, ρ, u, flags, w, c, N, Nx, Ny, Nz, Int(n))
end

@kernel function surface_3_kernel!(
    ρ, flags, mass, massex, ϕ,
    c::NTuple{Q, SVector{3, Int}},
    N::Int, Nx::Int, Ny::Int, Nz::Int
) where {Q}
    n = @index(Global)
    @inbounds begin
        flagsn = flags[n]
        sus = flagsn & (TYPE_SU | TYPE_S)
        if (sus & TYPE_S) == 0x00
            CType = eltype(ρ)

            ρn = ρ[n]
            massn = mass[n]
            massexn = zero(CType)
            ϕn = zero(CType)

            if sus == TYPE_F
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
                counter += Int(suj == TYPE_F || suj == TYPE_I || suj == TYPE_IF || suj == TYPE_GI)
            end
            if counter == 0
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