using KernelAbstractions

@inline function opp(i::Int)
    i == 1 ? 1 : iseven(i) ? i + 1 : i - 1
end

@inline function wrap_coord(x, dx, N)
    ifelse(dx == 0, x,
        ifelse(dx > 0, ifelse(x == N - 1, 0, x + 1),
                       ifelse(x == 0, N - 1, x - 1)))
end

@inline function load_pair(fi, n, src, i, t_odd::Bool, N)
    if t_odd
        fi[f_index(n, i, N)], fi[f_index(src, i + 1, N)]
    else
        fi[f_index(n, i + 1, N)], fi[f_index(src, i, N)]
    end
end

@inline function store_pair!(fi, n, src, i, f_plus, f_minus, t_odd::Bool, N)
    if t_odd
        fi[f_index(n, i, N)]       = f_minus
        fi[f_index(src, i + 1, N)] = f_plus
    else
        fi[f_index(n, i + 1, N)]   = f_minus
        fi[f_index(src, i, N)]     = f_plus
    end
    return nothing
end

@inline function neighbor_n(n0, cx, cy, cz, Nx, Ny, Nz)
    # n0 0-based
    t  = n0
    x  = t % Nx
    y  = (t ÷ Nx) % Ny
    z  = t ÷ (Nx  * Ny)
    x2 = wrap_coord(x, cx, Nx)
    y2 = wrap_coord(y, cy, Ny)
    z2 = wrap_coord(z, cz, Nz)
    return x2 + y2 * Nx + z2 * Nx * Ny
end

@kernel function initialize_kernel!(
    ρ, u, fi, flags,
    w::NTuple{Q, Float32}, 
    c::NTuple{Q, SVector{3, Int}},
    Nx::Int, Ny::Int, Nz::Int
) where {Q}
    n = @index(Global)
    @inbounds begin
        N = Nx * Ny * Nz
        n0 = n - 1
        ρn = ρ[n]
        ux, uy, uz = u[n, 1], u[n, 2], u[n, 3]

        if (flags[n] & TYPE_S) == TYPE_S
            ux = uy = uz = 0.0f0
            u[n, 1] = ux; u[n, 2] = uy; u[n, 3] = uz;
        end

        uu = 1.5f0 * (ux*ux + uy*uy + uz*uz)
        fi[f_index(n, 1, N)] = w[1] * ρn * (1.0f0 - uu)

        for k in 1:((Q - 1) ÷ 2)
            i = 2k
            cp, cm = c[i], c[i + 1]
            cup = Float32(cp[1])*ux + Float32(cp[2])*uy + Float32(cp[3])*uz
            cum = Float32(cm[1])*ux + Float32(cm[2])*uy + Float32(cm[3])*uz
            feqp = w[i]     * ρn * (1.0f0 + 3.0f0*cup + 4.5f0*cup*cup - uu)
            feqm = w[i + 1] * ρn * (1.0f0 + 3.0f0*cum + 4.5f0*cum*cum - uu)
            src = neighbor_n(n0, cp[1], cp[2], cp[3], Nx, Ny, Nz) + 1
            store_pair!(fi, n, src, i, feqp, feqm, true, N) # t_odd = true
        end
    end
end

@kernel function stream_collide_kernel!(
    flags, fi,
    w::NTuple{Q, Float32}, 
    c::NTuple{Q, SVector{3, Int}},
    ω::Float32,
    Nx::Int, Ny::Int, Nz::Int,
    t_odd::Bool
) where {Q}
    n = @index(Global)
    @inbounds begin
        if (flags[n] & TYPE_S) != TYPE_S
            N = Nx * Ny * Nz
            n0 = n - 1

            fn1 = fi[f_index(n, 1, N)]
            # pairs: (2, 3), (4, 5), ...
            pairs = ntuple(Val((Q - 1) ÷ 2)) do k
                i = 2k
                ci = c[i]
                src = neighbor_n(n0, ci[1], ci[2], ci[3], Nx, Ny, Nz) + 1
                load_pair(fi, n, src, i, t_odd, N)
            end

            ρn = fn1
            ux = uy = uz = 0.0f0

            for k in 1:length(pairs)
                i = 2k
                fp, fm = pairs[k]
                ρn += fp + fm
                cp, cm = c[i], c[i + 1]
                ux += Float32(cp[1]) * fp + Float32(cm[1]) * fm
                uy += Float32(cp[2]) * fp + Float32(cm[2]) * fm
                uz += Float32(cp[3]) * fp + Float32(cm[3]) * fm
            end

            invρ = 1.0f0 / ρn
            ux *= invρ; uy *= invρ; uz *= invρ
            uu = 1.5f0 * (ux*ux + uy*uy + uz*uz)

            # SRT: f* = f - ω(f - feq)
            feq1 = w[1] * ρn * (1.0f0 - uu)
            fi[f_index(n, 1, N)] = (1.0f0 - ω) * fn1 + ω * feq1

            for k in 1:length(pairs)
                i = 2k
                fp, fm = pairs[k]
                cp, cm = c[i], c[i + 1]
                cup = Float32(cp[1])*ux + Float32(cp[2])*uy + Float32(cp[3])*uz
                cum = Float32(cm[1])*ux + Float32(cm[2])*uy + Float32(cm[3])*uz
                feqp = w[i]     * ρn * (1.0f0 + 3.0f0*cup + 4.5f0*cup*cup - uu)
                feqm = w[i + 1] * ρn * (1.0f0 + 3.0f0*cum + 4.5f0*cum*cum - uu)
                src = neighbor_n(n0, cp[1], cp[2], cp[3], Nx, Ny, Nz) + 1
                store_pair!(fi, n, src, i,
                    (1.0f0 - ω) * fp + ω * feqp,
                    (1.0f0 - ω) * fm + ω * feqm,
                    t_odd, N)
            end
        end
    end
end