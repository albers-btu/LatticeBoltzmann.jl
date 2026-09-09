using KernelAbstractions

@inline function wrap_coord(x, dx, N)
    ifelse(dx == 0, x,
        ifelse(dx > 0, ifelse(x == N - 1, 0, x + 1),
                       ifelse(x == 0, N - 1, x - 1)))
end

@inline function src_index(x, y, z, cx, cy, cz, Nx, Ny, Nz)
      wrap_coord(x, cx, Nx)
    + wrap_coord(y, cy, Ny) * Nx
    + wrap_coord(z, cz, Nz) * Nx * Ny + 1
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
            u[n, 3] = zero(CType);
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

# generic fallback
@inline function stream_collide_body!(
    t_odd::Val{odd},
    flags, fi,
    w::NTuple{Q, CType}, 
    c::NTuple{Q, SVector{3, Int}},
    ω::CType,
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

        invρ = one(CType) / ρn
        ux *= invρ; uy *= invρ; uz *= invρ
        uu = CType(1.5) * (ux*ux + uy*uy + uz*uz)

        # SRT: f* = f - ω(f - feq)
        fi[f_index(n, 1, N)] = eltype(fi)((one(CType) - ω) * fn1 + ω * (w[1] * ρn * (one(CType) - uu)))

        for k in 1:NP
            i = 2k
            fp, fm = pairs[k]
            cp, cm = c[i], c[i + 1]
            cup = CType(cp[1])*ux + CType(cp[2])*uy + CType(cp[3])*uz
            cum = CType(cm[1])*ux + CType(cm[2])*uy + CType(cm[3])*uz
            feqp = w[i]     * ρn * (one(CType) + CType(3.0)*cup + CType(4.5)*cup*cup - uu)
            feqm = w[i + 1] * ρn * (one(CType) + CType(3.0)*cum + CType(4.5)*cum*cum - uu)
            src = src_index(x, y, z, cp[1], cp[2], cp[3], Nx, Ny, Nz)
            store_pair!(fi, n, src, i,
                (one(CType) - ω) * fp + ω * feqp,
                (one(CType) - ω) * fm + ω * feqm,
                t_odd, N)
        end
    end
    return nothing
end

@inline function stream_collide_body!(
    t_odd::Val{odd},
    flags, fi,
    w::NTuple{19, CType},
    c::NTuple{19, SVector{3,Int}},
    ω::CType, N::Int, Nx::Int, Ny::Int, Nz::Int, n,
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

    invρ = one(CType) / ρn
    ux *= invρ; uy *= invρ; uz *= invρ
    uu = CType(1.5) * (ux*ux + uy*uy + uz*uz)

    fi[f_index(n, 1, N)] = eltype(fi)((one(CType) - ω) * fn1 + ω * (w[1] * ρn * (one(CType) - uu)))

    store_pair!(fi, n, src2,  2,  srt(ω, fp2,  w[2],  ρn, ux, uy, uz, uu, c[2]),  srt(ω, fm3,  w[3],  ρn, ux, uy, uz, uu, c[3]),  t_odd, N)
    store_pair!(fi, n, src4,  4,  srt(ω, fp4,  w[4],  ρn, ux, uy, uz, uu, c[4]),  srt(ω, fm5,  w[5],  ρn, ux, uy, uz, uu, c[5]),  t_odd, N)
    store_pair!(fi, n, src6,  6,  srt(ω, fp6,  w[6],  ρn, ux, uy, uz, uu, c[6]),  srt(ω, fm7,  w[7],  ρn, ux, uy, uz, uu, c[7]),  t_odd, N)
    store_pair!(fi, n, src8,  8,  srt(ω, fp8,  w[8],  ρn, ux, uy, uz, uu, c[8]),  srt(ω, fm9,  w[9],  ρn, ux, uy, uz, uu, c[9]),  t_odd, N)
    store_pair!(fi, n, src10, 10, srt(ω, fp10, w[10], ρn, ux, uy, uz, uu, c[10]), srt(ω, fm11, w[11], ρn, ux, uy, uz, uu, c[11]), t_odd, N)
    store_pair!(fi, n, src12, 12, srt(ω, fp12, w[12], ρn, ux, uy, uz, uu, c[12]), srt(ω, fm13, w[13], ρn, ux, uy, uz, uu, c[13]), t_odd, N)
    store_pair!(fi, n, src14, 14, srt(ω, fp14, w[14], ρn, ux, uy, uz, uu, c[14]), srt(ω, fm15, w[15], ρn, ux, uy, uz, uu, c[15]), t_odd, N)
    store_pair!(fi, n, src16, 16, srt(ω, fp16, w[16], ρn, ux, uy, uz, uu, c[16]), srt(ω, fm17, w[17], ρn, ux, uy, uz, uu, c[17]), t_odd, N)
    store_pair!(fi, n, src18, 18, srt(ω, fp18, w[18], ρn, ux, uy, uz, uu, c[18]), srt(ω, fm19, w[19], ρn, ux, uy, uz, uu, c[19]), t_odd, N)
    return nothing
end

@inline function srt(ω::CType, f::CType, wi::CType, ρn::CType, ux::CType, uy::CType, uz::CType, uu::CType, ci) where {CType}
    cu = CType(ci[1])*ux + CType(ci[2])*uy + CType(ci[3])*uz
    feq = wi * ρn * (one(CType) + CType(3.0)*cu + CType(4.5)*cu*cu - uu)
    return (one(CType) - ω) * f + ω * feq
end

@kernel function stream_collide_even_kernel!(
    flags, fi,
    w::NTuple{Q, CType}, 
    c::NTuple{Q, SVector{3, Int}},
    ω::CType,
    N::Int, Nx::Int, Ny::Int, Nz::Int,
) where {Q, CType}
    n = @index(Global)
    @inbounds stream_collide_body!(Val(false), flags, fi, w, c, ω, N, Nx, Ny, Nz, Int(n))
end

@kernel function stream_collide_odd_kernel!(
    flags, fi,
    w::NTuple{Q, CType}, 
    c::NTuple{Q, SVector{3, Int}},
    ω::CType,
    N::Int, Nx::Int, Ny::Int, Nz::Int,
) where {Q, CType}
    n = @index(Global)
    @inbounds stream_collide_body!(Val(true), flags, fi, w, c, ω, N, Nx, Ny, Nz, Int(n))
end