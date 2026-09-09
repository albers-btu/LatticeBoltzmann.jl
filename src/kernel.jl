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

@inline load_pair(fi, n, src, i, ::Val{true}, N) =
    (fi[f_index(n, i, N)], fi[f_index(src, i + 1, N)])

@inline load_pair(fi, n, src, i, ::Val{false}, N) =
    (fi[f_index(n, i + 1, N)], fi[f_index(src, i, N)])

@inline function store_pair!(fi, n, src, i, f_plus, f_minus, ::Val{true}, N)
        fi[f_index(n, i, N)]       = f_minus
        fi[f_index(src, i + 1, N)] = f_plus
    return nothing
end
@inline function store_pair!(fi, n, src, i, f_plus, f_minus, ::Val{false}, N)
        fi[f_index(n, i + 1, N)]   = f_minus
        fi[f_index(src, i, N)]     = f_plus
    return nothing
end

@kernel function initialize_kernel!(
    ρ, u, fi, flags,
    w::NTuple{Q, DType}, 
    c::NTuple{Q, SVector{3, Int}},
    N::Int, Nx::Int, Ny::Int, Nz::Int
) where {Q, DType}
    n = @index(Global)
    @inbounds begin
        n0 = n - 1
        ρn = ρ[n]
        ux, uy, uz = u[n, 1], u[n, 2], u[n, 3]

        if (flags[n] & TYPE_S) == TYPE_S
            u[n, 1] = zero(DType)
            u[n, 2] = zero(DType)
            u[n, 3] = zero(DType);
        else
            uu = DType(1.5) * (ux*ux + uy*uy + uz*uz)
            fi[f_index(n, 1, N)] = w[1] * ρn * (one(DType) - uu)

            x = n0 % Nx
            y = (n0 ÷ Nx) % Ny
            z = n0 ÷ (Nx * Ny)

            for k in 1:((Q - 1) ÷ 2)
                i = 2k
                cp, cm = c[i], c[i + 1]
                cup = DType(cp[1])*ux + DType(cp[2])*uy + DType(cp[3])*uz
                cum = DType(cm[1])*ux + DType(cm[2])*uy + DType(cm[3])*uz
                feqp = w[i]     * ρn * (one(DType) + DType(3.0)*cup + DType(4.5)*cup*cup - uu)
                feqm = w[i + 1] * ρn * (one(DType) + DType(3.0)*cum + DType(4.5)*cum*cum - uu)
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
    w::NTuple{Q, DType}, 
    c::NTuple{Q, SVector{3, Int}},
    ω::DType,
    N::Int, Nx::Int, Ny::Int, Nz::Int, n
) where {odd, Q, DType}
    if (flags[n] & TYPE_S) != TYPE_S
        n0 = n - 1

        x = n0 % Nx
        y = (n0 ÷ Nx) % Ny
        z = n0 ÷ (Nx * Ny)

        fn1 = fi[f_index(n, 1, N)]
        NP  = (Q - 1) ÷ 2

        # pairs: (2, 3), (4, 5), ...
        pairs = ntuple(Val(NP)) do k
            i = 2k
            cp = c[i]
            src = src_index(x, y, z, cp[1], cp[2], cp[3], Nx, Ny, Nz)
            load_pair(fi, n, src, i, t_odd, N)
        end

        ρn = fn1
        ux = uy = uz = zero(DType)

        for k in 1:NP
            i = 2k
            fp, fm = pairs[k]
            ρn += fp + fm
            cp, cm = c[i], c[i + 1]
            ux += DType(cp[1]) * fp + DType(cm[1]) * fm
            uy += DType(cp[2]) * fp + DType(cm[2]) * fm
            uz += DType(cp[3]) * fp + DType(cm[3]) * fm
        end

        invρ = one(DType) / ρn
        ux *= invρ; uy *= invρ; uz *= invρ
        uu = DType(1.5) * (ux*ux + uy*uy + uz*uz)

        # SRT: f* = f - ω(f - feq)
        fi[f_index(n, 1, N)] = (one(DType) - ω) * fn1 + ω * (w[1] * ρn * (one(DType) - uu))

        for k in 1:NP
            i = 2k
            fp, fm = pairs[k]
            cp, cm = c[i], c[i + 1]
            cup = DType(cp[1])*ux + DType(cp[2])*uy + DType(cp[3])*uz
            cum = DType(cm[1])*ux + DType(cm[2])*uy + DType(cm[3])*uz
            feqp = w[i]     * ρn * (one(DType) + DType(3.0)*cup + DType(4.5)*cup*cup - uu)
            feqm = w[i + 1] * ρn * (one(DType) + DType(3.0)*cum + DType(4.5)*cum*cum - uu)
            src = src_index(x, y, z, cp[1], cp[2], cp[3], Nx, Ny, Nz)
            store_pair!(fi, n, src, i,
                (one(DType) - ω) * fp + ω * feqp,
                (one(DType) - ω) * fm + ω * feqm,
                t_odd, N)
        end
    end
    return nothing
end

@inline function stream_collide_body!(
    t_odd::Val{odd},
    flags, fi,
    w::NTuple{19, DType},
    c::NTuple{19, SVector{3,Int}},
    ω::DType, N::Int, Nx::Int, Ny::Int, Nz::Int, n,
) where {odd, DType}
    if (flags[n] & TYPE_S) == TYPE_S
        return nothing
    end

    n0 = n - 1
    x  = n0 % Nx
    y  = (n0 ÷ Nx) % Ny
    z  = n0 ÷ (Nx * Ny)

    fn1 = fi[f_index(n, 1, N)]

    src2  = src_index(x, y, z, c[2][1],  c[2][2],  c[2][3],  Nx, Ny, Nz)
    src4  = src_index(x, y, z, c[4][1],  c[4][2],  c[4][3],  Nx, Ny, Nz)
    src6  = src_index(x, y, z, c[6][1],  c[6][2],  c[6][3],  Nx, Ny, Nz)
    src8  = src_index(x, y, z, c[8][1],  c[8][2],  c[8][3],  Nx, Ny, Nz)
    src10 = src_index(x, y, z, c[10][1], c[10][2], c[10][3], Nx, Ny, Nz)
    src12 = src_index(x, y, z, c[12][1], c[12][2], c[12][3], Nx, Ny, Nz)
    src14 = src_index(x, y, z, c[14][1], c[14][2], c[14][3], Nx, Ny, Nz)
    src16 = src_index(x, y, z, c[16][1], c[16][2], c[16][3], Nx, Ny, Nz)
    src18 = src_index(x, y, z, c[18][1], c[18][2], c[18][3], Nx, Ny, Nz)

    fp2,  fm3  = load_pair(fi, n, src2,  2,  t_odd, N)
    fp4,  fm5  = load_pair(fi, n, src4,  4,  t_odd, N)
    fp6,  fm7  = load_pair(fi, n, src6,  6,  t_odd, N)
    fp8,  fm9  = load_pair(fi, n, src8,  8,  t_odd, N)
    fp10, fm11 = load_pair(fi, n, src10, 10, t_odd, N)
    fp12, fm13 = load_pair(fi, n, src12, 12, t_odd, N)
    fp14, fm15 = load_pair(fi, n, src14, 14, t_odd, N)
    fp16, fm17 = load_pair(fi, n, src16, 16, t_odd, N)
    fp18, fm19 = load_pair(fi, n, src18, 18, t_odd, N)

    ρn = fn1 + fp2 + fm3 + fp4 + fm5 + fp6 + fm7 + fp8 + fm9 +
         fp10 + fm11 + fp12 + fm13 + fp14 + fm15 + fp16 + fm17 + fp18 + fm19
    ux = DType(c[2][1])*fp2 + DType(c[3][1])*fm3 + DType(c[4][1])*fp4 + DType(c[5][1])*fm5 +
         DType(c[6][1])*fp6 + DType(c[7][1])*fm7 + DType(c[8][1])*fp8 + DType(c[9][1])*fm9 +
         DType(c[10][1])*fp10 + DType(c[11][1])*fm11 + DType(c[12][1])*fp12 + DType(c[13][1])*fm13 +
         DType(c[14][1])*fp14 + DType(c[15][1])*fm15 + DType(c[16][1])*fp16 + DType(c[17][1])*fm17 +
         DType(c[18][1])*fp18 + DType(c[19][1])*fm19
    uy = DType(c[2][2])*fp2 + DType(c[3][2])*fm3 + DType(c[4][2])*fp4 + DType(c[5][2])*fm5 +
         DType(c[6][2])*fp6 + DType(c[7][2])*fm7 + DType(c[8][2])*fp8 + DType(c[9][2])*fm9 +
         DType(c[10][2])*fp10 + DType(c[11][2])*fm11 + DType(c[12][2])*fp12 + DType(c[13][2])*fm13 +
         DType(c[14][2])*fp14 + DType(c[15][2])*fm15 + DType(c[16][2])*fp16 + DType(c[17][2])*fm17 +
         DType(c[18][2])*fp18 + DType(c[19][2])*fm19
    uz = DType(c[2][3])*fp2 + DType(c[3][3])*fm3 + DType(c[4][3])*fp4 + DType(c[5][3])*fm5 +
         DType(c[6][3])*fp6 + DType(c[7][3])*fm7 + DType(c[8][3])*fp8 + DType(c[9][3])*fm9 +
         DType(c[10][3])*fp10 + DType(c[11][3])*fm11 + DType(c[12][3])*fp12 + DType(c[13][3])*fm13 +
         DType(c[14][3])*fp14 + DType(c[15][3])*fm15 + DType(c[16][3])*fp16 + DType(c[17][3])*fm17 +
         DType(c[18][3])*fp18 + DType(c[19][3])*fm19

    invρ = one(DType) / ρn
    ux *= invρ; uy *= invρ; uz *= invρ
    uu = DType(1.5) * (ux*ux + uy*uy + uz*uz)

    fi[f_index(n, 1, N)] = (one(DType) - ω) * fn1 + ω * (w[1] * ρn * (one(DType) - uu))

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

@inline function srt(ω::DType, f::DType, wi::DType, ρn::DType, ux::DType, uy::DType, uz::DType, uu::DType, ci) where {DType}
    cu = DType(ci[1])*ux + DType(ci[2])*uy + DType(ci[3])*uz
    feq = wi * ρn * (one(DType) + DType(3.0)*cu + DType(4.5)*cu*cu - uu)
    return (one(DType) - ω) * f + ω * feq
end

@kernel function stream_collide_even_kernel!(
    flags, fi,
    w::NTuple{Q, DType}, 
    c::NTuple{Q, SVector{3, Int}},
    ω::DType,
    N::Int, Nx::Int, Ny::Int, Nz::Int,
) where {Q, DType}
    n = @index(Global)
    @inbounds stream_collide_body!(Val(false), flags, fi, w, c, ω, N, Nx, Ny, Nz, Int(n))
end

@kernel function stream_collide_odd_kernel!(
    flags, fi,
    w::NTuple{Q, DType}, 
    c::NTuple{Q, SVector{3, Int}},
    ω::DType,
    N::Int, Nx::Int, Ny::Int, Nz::Int,
) where {Q, DType}
    n = @index(Global)
    @inbounds stream_collide_body!(Val(true), flags, fi, w, c, ω, N, Nx, Ny, Nz, Int(n))
end