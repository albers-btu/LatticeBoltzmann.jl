using KernelAbstractions

@inline function opp(i::Int)
    i == 1 ? 1 : iseven(i) ? i + 1 : i - 1
end

@inline function wrap_coord(x, dx, N)
    ifelse(dx == 0, x,
        ifelse(dx > 0, ifelse(x == N - 1, 0, x + 1),
                       ifelse(x == 0, N - 1, x - 1)))
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

@kernel function initialize_kernel!(ρ, u, fi, flags, w, c, N)
    n = @index(Global)
    @inbounds begin
        ρn = ρ[n]
        ux, uy, uz = u[n, 1], u[n, 2], u[n, 3]
        if (flags[n] & TYPE_S) == TYPE_S
            ux = uy = uz = 0.0f0
            u[n, 1] = ux;
            u[n, 2] = uy;
            u[n, 3] = uz;
        end
        uu = 1.5f0 * (ux*ux + uy*uy + uz*uz)
        for i in 1:length(w)
            ci = c[i]
            cu = Float32(ci[1])*ux + Float32(ci[2])*uy + Float32(ci[3])*uz
            fi[f_index(n, i, N)] = w[i] * ρn * (1.0f0 + 3.0f0 * cu + 4.5f0 * cu * cu - uu)
        end
    end
end

@kernel function stream_collide_kernel!(
    flags,
    fi, fo,
    w::NTuple{Q, Float32}, 
    c::NTuple{Q, SVector{3, Int}},
    ω::Float32,
    Nx::Int, Ny::Int, Nz::Int
) where {Q}
    n = @index(Global)
    @inbounds begin
        if (flags[n] & TYPE_S) != TYPE_S
            N = Nx * Ny * Nz
            n0 = n - 1
            ρn = ux = uy = uz = 0.0f0

            # pull
            fn = ntuple(i -> begin
                ci = c[i]
                src0 = neighbor_n(n0, -ci[1], -ci[2], -ci[3], Nx, Ny, Nz)
                src = src0 + 1
                if (flags[src] & TYPE_S) == TYPE_S
                    fi[f_index(n, opp(i), N)]
                else
                    fi[f_index(src, i, N)]
                end
            end, Val(Q))

            for i in 1:Q
                fi_i = fn[i]
                ρn += fi_i
                ci = c[i]
                ux += Float32(ci[1]) * fi_i
                uy += Float32(ci[2]) * fi_i
                uz += Float32(ci[3]) * fi_i
            end
            invρ = 1.0f0 / ρn
            ux *= invρ; uy *= invρ; uz *= invρ

            # SRT: f* = f - ω(f - feq)
            uu = 1.5f0 * (ux*ux + uy*uy + uz*uz)
            for i in 1:Q
                ci = c[i]
                cu = Float32(ci[1])*ux + Float32(ci[2])*uy + Float32(ci[3])*uz
                feq = w[i] * ρn * (1.0f0 + 3.0f0*cu + 4.5f0*cu*cu - uu)
                fo[f_index(n, i, N)] = (1.0f0 - ω) * fn[i] + ω * feq
            end
        end
    end
end