using KernelAbstractions

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

@inline function average_neighbors_T(
    Tfield, flags, x, y, z, c::NTuple{Q, SVector{3, Int}},
    Nx, Ny, Nz, ::Type{CType}
) where {Q, CType}
    s = zero(CType)
    cnt = 0
    for i in 2:Q
        src = src_index(x, y, z, c[i][1], c[i][2], c[i][3], Nx, Ny, Nz)
        su = flags[src] & (TYPE_SU | TYPE_S)
        if su == TYPE_F || su == TYPE_I || su == TYPE_IF
            s += Tfield[src]
            cnt += 1
        end
    end
    return cnt > 0 ? s / CType(cnt) : one(CType)
end

@inline function average_neighbors_fs(
    fs, flags, x, y, z, c::NTuple{Q, SVector{3, Int}},
    Nx, Ny, Nz, ::Type{CType}
) where {Q, CType}
    s = zero(CType)
    cnt = 0
    for i in 2:Q
        src = src_index(x, y, z, c[i][1], c[i][2], c[i][3], Nx, Ny, Nz)
        su = flags[src] & (TYPE_SU | TYPE_S)
        if su == TYPE_F || su == TYPE_I || su == TYPE_IF
            s += fs[src]
            cnt += 1
        end
    end
    return cnt > 0 ? s / CType(cnt) : zero(CType)
end

@inline function initialize_body!(
    ρ, u, fi, flags, mass, massex, ϕ, gi, T, fs,
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

    if (flagsn & (TYPE_S | TYPE_E | TYPE_T | TYPE_H | TYPE_F | TYPE_I)) == 0x00
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
            @static if DIM == 2
                uz = zero(CType)
                u[n, 3] = zero(CType)
            end
            @static if TEMPERATURE
                fs[n] = average_neighbors_fs(fs, flags, x, y, z, c, Nx, Ny, Nz, CType)
            end
        end
    end

    if (flagsn & TYPE_BO) == TYPE_S
        @static if MOVING_BOUNDARIES
            only_s = true
            for k in 1:(Q - 1)
                only_s &= (flagsj[k] & TYPE_BO) == TYPE_S
            end
            if only_s
                u[n, 1] = zero(CType); u[n, 2] = zero(CType); u[n, 3] = zero(CType)
            end
        else
            u[n, 1] = zero(CType); u[n, 2] = zero(CType); u[n, 3] = zero(CType)
        end
        @static if TEMPERATURE
            store_geq_local!(gi, n, T[n], N)
        end
        ρbb = ρn > zero(CType) ? ρn : one(CType)
        store_feq!(fi, n, x, y, z, ρbb, zero(CType), zero(CType), zero(CType),
                   w, c, N, Nx, Ny, Nz, Val(false), Val(true))
    elseif (flagsn & TYPE_SU) == TYPE_G
        u[n, 1] = zero(CType); u[n, 2] = zero(CType); u[n, 3] = zero(CType)
        ϕn = zero(CType)
    else
        if (flagsn & TYPE_SU) == TYPE_I && (ϕn < 0 || ϕn > 1)
            ϕn = CType(0.5)
        elseif (flagsn & TYPE_SU) == TYPE_F
            ϕn = one(CType)
        end
        @static if DIM == 2
            uz = zero(CType)
        end
        store_feq!(fi, n, x, y, z, ρn, ux, uy, uz, w, c, N, Nx, Ny, Nz, Val(false), Val(true))
        @static if TEMPERATURE
            store_geq!(gi, n, x, y, z, T[n], ux, uy, uz, N, Nx, Ny, Nz, Val(false), CType, Val(true))
        end
    end

    @static if TEMPERATURE
        if (flagsn & TYPE_SU) == TYPE_G
            store_geq!(gi, n, x, y, z, T[n], zero(CType), zero(CType), zero(CType),
                       N, Nx, Ny, Nz, Val(false), CType, Val(true))
        end
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
    N::Int, Nx::Int, Ny::Int, Nz::Int,
    gi, T, fs
) where {Q, CType}
    n = @index(Global)
    @inbounds initialize_body!(ρ, u, fi, flags, mass, massex, ϕ, gi, T, fs, w, c, N, Nx, Ny, Nz, Int(n))
end

end # SURFACE


@static if !SURFACE

@kernel function initialize_kernel!(
    ρ, u, fi, flags,
    w::NTuple{Q, CType}, 
    c::NTuple{Q, SVector{3, Int}},
    N::Int, Nx::Int, Ny::Int, Nz::Int,
    gi, T
) where {Q, CType}
    n = @index(Global)
    @inbounds begin
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
    end
end

end # not SURFACE