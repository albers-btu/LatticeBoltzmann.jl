using KernelAbstractions

@inline function opp(i::Int)
    i == 1 ? 1 : iseven(i) ? i + 1 : i - 1
end

@inline function neighbor_index(x, y, z, ci, Nx, Ny, Nz)
    # pull from x - c_i, periodic
    xs = mod1(x - ci[1], Nx)
    ys = mod1(y - ci[2], Ny)
    zs = mod1(z - ci[3], Nz)
    return xs + (ys - 1) * Nx + (zs - 1) * Nx * Ny
end

@kernel function initialize_kernel!(ρ, u, fi, flags, w, c)
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
            fi[n, i] = w[i] * ρn * (1.0f0 + 3.0f0 * cu + 4.5f0 * cu * cu - uu)
        end
    end
end

@kernel function stream_collide_kernel!(
    ρ, u, fi, fo, flags,
    w::NTuple{Q, Float32}, 
    c::NTuple{Q, SVector{3, Int}},
    τ::Float32,
    fx::Float32, fy::Float32, fz::Float32,
    Nx::Int, Ny::Int, Nz::Int) where
    {Q}

    n = @index(Global)
    @inbounds begin
        if (flags[n] & TYPE_S) != TYPE_S
            t = n - 1
            x = t % Nx + 1
            y = (t ÷ Nx) % Ny + 1
            z = t ÷ (Nx * Ny) + 1

            # pull
            fn = ntuple(Val(Q)) do i
                ci = c[i]
                src = neighbor_index(x, y, z, ci, Nx, Ny, Nz)
                if (flags[src] & TYPE_S) == TYPE_S
                    fi[n, opp(i)] # bounce-backend
                else
                    fi[src, i]
                end
            end

            # moments
            ρn = 0.0f0
            ux = 0.0f0
            uy = 0.0f0
            uz = 0.0f0
            for i in 1:Q
                fi_i = fn[i]
                ρn += fi_i
                ci = c[i]
                ux += Float32(ci[1]) * fi_i
                uy += Float32(ci[2]) * fi_i
                uz += Float32(ci[3]) * fi_i
            end
            invρ = 1.0f0 / ρn
            ux *= invρ
            uy *= invρ
            uz *= invρ

            ρ[n]    = ρn
            u[n, 1] = ux
            u[n, 2] = uy
            u[n, 3] = uz

            # SRT: f* = f - (1/τ)(f - feq)
            ω = 1.0f0 / τ
            uu = 1.5f0 * (ux*ux + uy*uy + uz*uz)
            for i in 1:Q
                ci = c[i]
                cu = Float32(ci[1])*ux + Float32(ci[2])*uy + Float32(ci[3])*uz
                feq = w[i] * ρn * (1.0f0 + 3.0f0*cu + 4.5f0*cu*cu - uu)
                fo[n, i] = (1.0f0 - ω) * fn[i] + ω * feq
            end
        end
    end
end