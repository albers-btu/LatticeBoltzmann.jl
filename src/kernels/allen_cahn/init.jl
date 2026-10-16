using KernelAbstractions

@static if !SURFACE

@inline function initialize_closed!(
    ρ, u, fi, flags,
    w::NTuple{Q, CType},
    c::NTuple{Q, SVector{3, Int}},
    N::Int, Nx::Int, Ny::Int, Nz::Int,
    gi, T,
    n::Int,
) where {Q, CType}
    n0 = n - 1
    x = n0 % Nx
    y = (n0 ÷ Nx) % Ny
    z = n0 ÷ (Nx * Ny)
    ρn = ρ[n]
    ux, uy, uz = u[n, 1], u[n, 2], u[n, 3]
    flagsn = flags[n]

    if (flagsn & TYPE_BO) == TYPE_S
        @static if MOVING_BOUNDARIES
            only_s = true
            for i in 2:Q
                src = src_index(x, y, z, c[i][1], c[i][2], c[i][3], Nx, Ny, Nz)
                only_s &= (flags[src] & TYPE_BO) == TYPE_S
            end
            if only_s
                u[n, 1] = zero(CType)
                u[n, 2] = zero(CType)
                u[n, 3] = zero(CType)
            end
        else
            u[n, 1] = zero(CType)
            u[n, 2] = zero(CType)
            u[n, 3] = zero(CType)
        end
        @static if TEMPERATURE
            store_geq_local!(gi, n, T[n], N)
        end
        ρbb = ρn > zero(CType) ? ρn : one(CType)
        store_feq!(fi, n, x, y, z, ρbb, zero(CType), zero(CType), zero(CType),
                   w, c, N, Nx, Ny, Nz, Val(false), Val(true))
    else
        @static if DIM == 2
            uz = zero(CType)
        end
        uu = CType(1.5) * (ux*ux + uy*uy + uz*uz)
        fi[f_index(n, 1, N)] = eltype(fi)(w[1] * ρn * (one(CType) - uu))

        for k in 1:((Q - 1) ÷ 2)
            i = 2k
            cp, cm = c[i], c[i + 1]
            cup = CType(cp[1])*ux + CType(cp[2])*uy + CType(cp[3])*uz
            cum = CType(cm[1])*ux + CType(cm[2])*uy + CType(cm[3])*uz
            feqp = w[i]     * ρn * (one(CType) + CType(3.0)*cup + CType(4.5)*cup*cup - uu)
            feqm = w[i + 1] * ρn * (one(CType) + CType(3.0)*cum + CType(4.5)*cum*cum - uu)
            src = src_index(x, y, z, cp[1], cp[2], cp[3], Nx, Ny, Nz)
            # Even store writes + into the slot the first (even) load reads as −.
            store_pair!(fi, n, src, i, feqm, feqp, Val(false), N)
        end
        @static if TEMPERATURE
            store_geq!(gi, n, x, y, z, T[n], ux, uy, uz, N, Nx, Ny, Nz, Val(false), CType, Val(true))
        end
    end
    return nothing
end

@static if ALLEN_CAHN
@inline function store_hi_equilibrium!(
    hi, phi, u, flags,
    w::NTuple{Q, CType}, c::NTuple{Q, SVector{3, Int}},
    Mphi::CType, W::CType,
    N::Int, Nx::Int, Ny::Int, Nz::Int,
    n::Int,
) where {Q, CType}
    n0 = n - 1
    x = n0 % Nx
    y = (n0 ÷ Nx) % Ny
    z = n0 ÷ (Nx * Ny)
    φ = CType(phi[n])
    hi[f_index(n, 1, N)] = eltype(hi)(w[1] * φ)
    if (flags[n] & TYPE_BO) == TYPE_S
        hux = zero(CType)
        huy = zero(CType)
        huz = zero(CType)
    else
        hux = CType(u[n, 1])
        huy = CType(u[n, 2])
        huz = CType(u[n, 3])
        @static if DIM == 2
            huz = zero(CType)
        end
    end
    # Same CAC equilibrium as ac_collide!: j_ct = M (4/W) φ(1−φ) n.
    gx, gy, gz = grad_phi(phi, x, y, z, w, c, Nx, Ny, Nz)
    gnorm = sqrt(gx * gx + gy * gy + gz * gz)
    invn = gnorm > CType(1.0e-8) ? one(CType) / gnorm : zero(CType)
    nx = gx * invn
    ny = gy * invn
    nz = gz * invn
    φc = ifelse(φ < zero(CType), zero(CType), ifelse(φ > one(CType), one(CType), φ))
    jct = Mphi * (CType(4) / W) * φc * (one(CType) - φc)
    for k in 1:((Q - 1) ÷ 2)
        i = 2k
        cp, cm = c[i], c[i + 1]
        cup = CType(cp[1]) * hux + CType(cp[2]) * huy + CType(cp[3]) * huz
        cum = CType(cm[1]) * hux + CType(cm[2]) * huy + CType(cm[3]) * huz
        cnp = CType(cp[1]) * nx + CType(cp[2]) * ny + CType(cp[3]) * nz
        cnm = CType(cm[1]) * nx + CType(cm[2]) * ny + CType(cm[3]) * nz
        heqp = w[i] * φ * (one(CType) + CType(3) * cup) + CType(3) * w[i] * jct * cnp
        heqm = w[i + 1] * φ * (one(CType) + CType(3) * cum) + CType(3) * w[i + 1] * jct * cnm
        src = src_index(x, y, z, cp[1], cp[2], cp[3], Nx, Ny, Nz)
        # Even store writes + into the slot the first (even) load reads as −.
        store_pair!(hi, n, src, i, heqm, heqp, Val(false), N)
    end
    return nothing
end

@kernel function initialize_kernel!(
    ρ, u, fi, flags,
    w::NTuple{Q, CType},
    c::NTuple{Q, SVector{3, Int}},
    N::Int, Nx::Int, Ny::Int, Nz::Int,
    gi, T,
    hi, phi,
    Mphi::CType, W::CType,
) where {Q, CType}
    n = @index(Global)
    @inbounds begin
        initialize_closed!(ρ, u, fi, flags, w, c, N, Nx, Ny, Nz, gi, T, Int(n))
        store_hi_equilibrium!(hi, phi, u, flags, w, c, Mphi, W, N, Nx, Ny, Nz, Int(n))
    end
end
else
@kernel function initialize_kernel!(
    ρ, u, fi, flags,
    w::NTuple{Q, CType},
    c::NTuple{Q, SVector{3, Int}},
    N::Int, Nx::Int, Ny::Int, Nz::Int,
    gi, T,
) where {Q, CType}
    n = @index(Global)
    @inbounds initialize_closed!(ρ, u, fi, flags, w, c, N, Nx, Ny, Nz, gi, T, Int(n))
end
end

end # not SURFACE