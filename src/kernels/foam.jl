using KernelAbstractions

@static if FOAM

@inline function concentration_body!(::Val{odd}, ::Int) where {odd}
    return nothing
end

@kernel function concentration_even_kernel!()
    n = @index(Global)
    @inbounds concentration_body!(Val(false), Int(n))
end

@kernel function concentration_odd_kernel!()
    n = @index(Global)
    @inbounds concentration_body!(Val(true), Int(n))
end

# Tag 0 is the liquid film, not a stop. s sums one full Woo tDelta per crossing.
@inline function disjoining_body!(
    ϕ, flags, tag, Pi, k_Π::CType, Nx::Int, Ny::Int, Nz::Int, n::Int
) where {CType}
    (flags[n] & TYPE_SU) != TYPE_I && return nothing
    tagn = tag[n]
    tagn <= Int32(0) && return nothing

    Pi[n] = zero(CType)

    n0 = n - 1
    x = n0 % Nx
    y = (n0 ÷ Nx) % Ny
    z = n0 ÷ (Nx * Ny)
    ϕn = CType(ϕ[n])
    phij = gather_phi_d3q27(ϕ, ϕn, x, y, z, Nx, Ny, Nz)
    nϕ = calculate_normal_py(phij)
    n2 = nϕ[1] * nϕ[1] + nϕ[2] * nϕ[2] + nϕ[3] * nϕ[3]
    n2 <= zero(CType) && return nothing

    dirx = -nϕ[1]
    diry = -nϕ[2]
    dirz = -nϕ[3]
    δself = abs(plic_cube(ϕn, nϕ))
    tDeltaX = one(CType) / abs(dirx)
    tDeltaY = one(CType) / abs(diry)
    tDeltaZ = one(CType) / abs(dirz)
    tMaxX = CType(0.5) * tDeltaX
    tMaxY = CType(0.5) * tDeltaY
    tMaxZ = CType(0.5) * tDeltaZ
    stepX = dirx > zero(CType) ? 1 : -1
    stepY = diry > zero(CType) ? 1 : -1
    stepZ = dirz > zero(CType) ? 1 : -1

    xj, yj, zj = x, y, z
    s = zero(CType)
    for _crossing in 1:4
        along_x = tMaxX < tMaxY && tMaxX < tMaxZ
        along_y = !along_x && tMaxY < tMaxZ
        if along_x
            s += tDeltaX
            tMaxX += tDeltaX
            j = src_index(xj, yj, zj, stepX, 0, 0, Nx, Ny, Nz)
        elseif along_y
            s += tDeltaY
            tMaxY += tDeltaY
            j = src_index(xj, yj, zj, 0, stepY, 0, Nx, Ny, Nz)
        else
            s += tDeltaZ
            tMaxZ += tDeltaZ
            j = src_index(xj, yj, zj, 0, 0, stepZ, Nx, Ny, Nz)
        end
        !(s < CType(4)) && return nothing
        j0 = j - 1
        xj = j0 % Nx
        yj = (j0 ÷ Nx) % Ny
        zj = j0 ÷ (Nx * Ny)

        fl = flags[j]
        (fl & TYPE_S) != 0x00 && return nothing
        tg = tag[j]
        if tg == Int32(-1)
            return nothing
        elseif tg != tagn
            su = fl & TYPE_SU
            if tg > Int32(0) && (su == TYPE_I || su == TYPE_G)
                δo = su == TYPE_I ? abs(plic_cube(CType(ϕ[j]), nϕ)) : zero(CType)
                d = s - δself - δo
                d < zero(CType) && (d = zero(CType))
                d < CType(4) && (Pi[n] = k_Π * (CType(4) - d))
                return nothing
            end
        end
    end
    return nothing
end

@kernel function disjoining_kernel!(
    @Const(ϕ), @Const(flags), @Const(tag), Pi, k_Π::CType, Nx::Int, Ny::Int, Nz::Int
) where {CType}
    n = @index(Global)
    @inbounds disjoining_body!(ϕ, flags, tag, Pi, k_Π, Nx, Ny, Nz, Int(n))
end

@inline function ϕ_correction_body!(::Int)
    return nothing
end

@kernel function ϕ_correction_kernel!()
    n = @index(Global)
    @inbounds ϕ_correction_body!(Int(n))
end

end
