using KernelAbstractions

@inline function moments_body!(
    t_odd::Val{odd},
    ρ, u, flags, fi, gi, T,
    w::NTuple{Q, CType},
    c::NTuple{Q, SVector{3, Int}},
    N::Int, Nx::Int, Ny::Int, Nz::Int, n
) where {odd, Q, CType}
    flagsn = flags[n]
    su = flagsn & TYPE_SU

    skip = (flagsn & TYPE_BO) == TYPE_S
    @static if SURFACE
        skip |= (su == TYPE_G) | (su == TYPE_IG)
    end
    @static if EQUILIBRIUM_BOUNDARIES
        if (flagsn & TYPE_BO) == TYPE_E
            return nothing
        end
    end
    if skip
        if (flagsn & TYPE_BO) == TYPE_S
            @static if !MOVING_BOUNDARIES
                ρ[n] = one(CType)
                u[n, 1] = zero(CType)
                u[n, 2] = zero(CType)
                u[n, 3] = zero(CType)
            end
        else
            ρ[n] = one(CType)
            u[n, 1] = zero(CType)
            u[n, 2] = zero(CType)
            u[n, 3] = zero(CType)
        end
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
    @static if TEMPERATURE
        if (flagsn & (TYPE_T | TYPE_H)) == 0x00
            srcx = src_index(x, y, z, 1, 0, 0, Nx, Ny, Nz)
            srcy = src_index(x, y, z, 0, 1, 0, Nx, Ny, Nz)
            srcz = src_index(x, y, z, 0, 0, 1, Nx, Ny, Nz)
            g0 = CType(gi[f_index(n, 1, N)])
            gpx, gmx = load_pair(gi, n, srcx, 2, t_odd, N, CType)
            gpy, gmy = load_pair(gi, n, srcy, 4, t_odd, N, CType)
            gpz, gmz = load_pair(gi, n, srcz, 6, t_odd, N, CType)
            T[n] = g0 + gpx + gmx + gpy + gmy + gpz + gmz + one(CType)
        end
    end
    return nothing
end

@kernel function moments_even_kernel!(
    ρ, u, @Const(flags), fi, gi, T, w::NTuple{Q, CType},
    c, N, Nx, Ny, Nz
) where {Q, CType}
    n = @index(Global)
    @inbounds moments_body!(Val(false), ρ, u, flags, fi, gi, T, w, c, N, Nx, Ny, Nz, Int(n))
end

@kernel function moments_odd_kernel!(
    ρ, u, @Const(flags), fi, gi, T, w::NTuple{Q, CType},
    c, N, Nx, Ny, Nz
) where {Q, CType}
    n = @index(Global)
    @inbounds moments_body!(Val(true), ρ, u, flags, fi, gi, T, w, c, N, Nx, Ny, Nz, Int(n))
end

@static if MOVING_BOUNDARIES

@kernel function update_moving_boundaries_kernel!(
    u, flags, c::NTuple{Q, SVector{3, Int}},
    N::Int, Nx::Int, Ny::Int, Nz::Int
) where {Q}
    n = @index(Global)
    @inbounds begin
        flagsn = flags[n]
        bo = flagsn & TYPE_BO
        if bo != TYPE_S && bo != TYPE_E
            n0 = n - 1
            x = n0 % Nx
            y = (n0 ÷ Nx) % Ny
            z = n0 ÷ (Nx * Ny)
            moving = false
            for i in 2:Q
                src = src_index(x, y, z, c[i][1], c[i][2], c[i][3], Nx, Ny, Nz)
                if (flags[src] & TYPE_BO) == TYPE_S
                    moving |= (u[src, 1] != zero(eltype(u))) |
                              (u[src, 2] != zero(eltype(u))) |
                              (u[src, 3] != zero(eltype(u)))
                end
            end
            flags[n] = moving ? (flagsn | TYPE_MS) : (flagsn & ~TYPE_MS)
        end
    end
end

end # MOVING_BOUNDARIES

@static if FORCE_FIELD

@inline function update_force_field_body!(
    t_odd::Val{odd}, flags, fi, F::AbstractArray{CType},
    c::NTuple{Q, SVector{3, Int}},
    N::Int, Nx::Int, Ny::Int, Nz::Int, n
) where {odd, Q, CType}
    if (flags[n] & TYPE_BO) != TYPE_S
        return nothing
    end
    n0 = n - 1
    x = n0 % Nx; y = (n0 ÷ Nx) % Ny; z = n0 ÷ (Nx * Ny)
    NP = (Q - 1) ÷ 2
    fn1 = CType(fi[f_index(n, 1, N)])
    mx = zero(CType); my = zero(CType); mz = zero(CType)
    for k in 1:NP
        i = 2k
        src = src_index(x, y, z, c[i][1], c[i][2], c[i][3], Nx, Ny, Nz)
        fp, fm = load_pair(fi, n, src, i, t_odd, N, CType)
        mx += CType(c[i][1])*fp + CType(c[i+1][1])*fm
        my += CType(c[i][2])*fp + CType(c[i+1][2])*fm
        mz += CType(c[i][3])*fp + CType(c[i+1][3])*fm
    end
    two = CType(2)
    F[n, 1] = two * mx
    F[n, 2] = two * my
    F[n, 3] = two * mz
    return nothing
end

@kernel function update_force_field_even_kernel!(
    @Const(flags), fi, F, c, N, Nx, Ny, Nz
)
    n = @index(Global)
    @inbounds update_force_field_body!(Val(false), flags, fi, F, c, N, Nx, Ny, Nz, Int(n))
end

@kernel function update_force_field_odd_kernel!(
    @Const(flags), fi, F, c, N, Nx, Ny, Nz
)
    n = @index(Global)
    @inbounds update_force_field_body!(Val(true), flags, fi, F, c, N, Nx, Ny, Nz, Int(n))
end

@kernel function reset_force_field_kernel!(F)
    n = @index(Global)
    @inbounds begin
        F[n, 1] = zero(eltype(F))
        F[n, 2] = zero(eltype(F))
        F[n, 3] = zero(eltype(F))
    end
end

end # FORCE_FIELD