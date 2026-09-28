using KernelAbstractions

@static if FOAM

@inline ceq_rest(c::CType) where {CType} = CType(0.25) * c

@inline ceq_axis(c::CType, ucomp::CType) where {CType} =
    CType(0.125) * c * (one(CType) + ucomp * CType(4))

@inline function omega_c_from_D(D::CType) where {CType}
    return one(CType) / (CType(4) * D + CType(0.5))
end

@inline function _axis_src(k::Int, x, y, z, Nx, Ny, Nz)
    cx = ifelse(k == 1, 1, 0)
    cy = ifelse(k == 2, 1, 0)
    cz = ifelse(k == 3, 1, 0)
    return src_index(x, y, z, cx, cy, cz, Nx, Ny, Nz)
end

@inline function _ucomp(k::Int, ux::CType, uy::CType, uz::CType) where {CType}
    return ifelse(k == 1, ux, ifelse(k == 2, uy, uz))
end

@inline function store_ceq!(
    ci, n, x, y, z, cn::CType, ux::CType, uy::CType, uz::CType,
    N::Int, Nx::Int, Ny::Int, Nz::Int, t_odd::Val{odd},
) where {odd, CType}
    ci[f_index(n, 1, N)] = eltype(ci)(ceq_rest(cn))
    for k in 1:3
        i = 2k
        src = _axis_src(k, x, y, z, Nx, Ny, Nz)
        ucomp = _ucomp(k, ux, uy, uz)
        store_pair!(ci, n, src, i, ceq_axis(cn, ucomp), ceq_axis(cn, -ucomp), t_odd, N)
    end
    return nothing
end

@inline function concentration_init_body!(
    ci, cfield, flags, u, c0::CType,
    N::Int, Nx::Int, Ny::Int, Nz::Int, n,
) where {CType}
    su = flags[n] & TYPE_SU
    if !(su == TYPE_F || su == TYPE_I || su == TYPE_IF || su == TYPE_IG)
        return nothing
    end
    n0 = n - 1
    x = n0 % Nx
    y = (n0 ÷ Nx) % Ny
    z = n0 ÷ (Nx * Ny)
    # Val(false): the first step is even.
    store_ceq!(ci, n, x, y, z, c0,
               CType(u[n, 1]), CType(u[n, 2]), CType(u[n, 3]),
               N, Nx, Ny, Nz, Val(false))
    cfield[n] = c0
    return nothing
end

@kernel function concentration_init_kernel!(
    ci, cfield, @Const(flags), @Const(u), c0::CType,
    N::Int, Nx::Int, Ny::Int, Nz::Int,
) where {CType}
    n = @index(Global)
    @inbounds concentration_init_body!(ci, cfield, flags, u, c0, N, Nx, Ny, Nz, Int(n))
end

@inline function concentration_body!(
    t_odd::Val{odd}, ci, cfield, flags, u, tag, flux,
    ωc::CType, q::CType,
    N::Int, Nx::Int, Ny::Int, Nz::Int, n,
) where {odd, CType}
    flagsn = flags[n]
    if (flagsn & TYPE_BO) == TYPE_S || (flagsn & TYPE_SU) == TYPE_G
        return nothing
    end
    su = flagsn & TYPE_SU
    if !(su == TYPE_F || su == TYPE_I || su == TYPE_IF || su == TYPE_IG)
        return nothing
    end

    n0 = n - 1
    x = n0 % Nx
    y = (n0 ÷ Nx) % Ny
    z = n0 ÷ (Nx * Ny)
    ux = CType(u[n, 1])
    uy = CType(u[n, 2])
    uz = CType(u[n, 3])

    # Rest slot does not swap. Δ is the interface-to-liquid population flux.
    crest = CType(ci[f_index(n, 1, N)])
    cn = crest
    Δ = zero(CType)
    interface = su == TYPE_I || su == TYPE_IF || su == TYPE_IG
    for k in 1:3
        i = 2k
        cx = ifelse(k == 1, 1, 0)
        cy = ifelse(k == 2, 1, 0)
        cz = ifelse(k == 3, 1, 0)
        srcp = src_index(x, y, z, cx, cy, cz, Nx, Ny, Nz)
        srcm = src_index(x, y, z, -cx, -cy, -cz, Nx, Ny, Nz)
        fp_in, fm_in = load_pair(ci, n, srcp, i, t_odd, N, CType)
        fp_out, fm_out = load_outgoing_pair(ci, n, srcp, i, t_odd, N, CType)
        cn += fp_in + fm_in
        # TYPE_IF is not TYPE_F. No ϕ weighting.
        if interface && (flags[srcp] & TYPE_SU) == TYPE_F
            Δ += fm_in - fp_out
        end
        if interface && (flags[srcm] & TYPE_SU) == TYPE_F
            Δ += fp_in - fm_out
        end
    end
    cfield[n] = cn
    tag_n = tag[n]
    if tag_n > Int32(0) && tag_n <= Int32(MAX_BUBBLES) && Δ != zero(CType)
        acc_add!(flux, Int(tag_n), Δ)
    end

    # TYPE_IF is not pure TYPE_F.
    qn = su == TYPE_F ? q : zero(CType)
    om = one(CType) - ωc
    ci[f_index(n, 1, N)] = eltype(ci)(om * crest + ωc * ceq_rest(cn) + CType(0.25) * qn)
    for k in 1:3
        i = 2k
        src = _axis_src(k, x, y, z, Nx, Ny, Nz)
        fp, fm = load_pair(ci, n, src, i, t_odd, N, CType)
        ucomp = _ucomp(k, ux, uy, uz)
        store_pair!(ci, n, src, i,
            om * fp + ωc * ceq_axis(cn, ucomp) + CType(0.125) * qn,
            om * fm + ωc * ceq_axis(cn, -ucomp) + CType(0.125) * qn,
            t_odd, N)
    end
    return nothing
end

@kernel function concentration_even_kernel!(
    ci, cfield, @Const(flags), @Const(u), @Const(tag), flux,
    ωc::CType, q::CType,
    N::Int, Nx::Int, Ny::Int, Nz::Int,
) where {CType}
    n = @index(Global)
    @inbounds concentration_body!(Val(false), ci, cfield, flags, u, tag, flux, ωc, q, N, Nx, Ny, Nz, Int(n))
end

@kernel function concentration_odd_kernel!(
    ci, cfield, @Const(flags), @Const(u), @Const(tag), flux,
    ωc::CType, q::CType,
    N::Int, Nx::Int, Ny::Int, Nz::Int,
) where {CType}
    n = @index(Global)
    @inbounds concentration_body!(Val(true), ci, cfield, flags, u, tag, flux, ωc, q, N, Nx, Ny, Nz, Int(n))
end

# Tag 0 is the liquid film, not a stop. s sums one full Woo tDelta per crossing.
@inline function disjoining_body!(
    ϕ, flags, tag, Pi, k_Π::CType, Nx::Int, Ny::Int, Nz::Int, n::Int
) where {CType}
    # Drop last step's Π before the eligibility return. surface_0 reads every TYPE_I.
    Pi[n] = zero(CType)
    (flags[n] & TYPE_SU) != TYPE_I && return nothing
    tagn = tag[n]
    tagn <= Int32(0) && return nothing

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

# δ = -c Δϕ. A tag-0 TYPE_I was liquid at upload and was converted on the
# D3Q19 stencil; credit the minimum positive tag in that same neighborhood.
@inline function ϕ_correction_body!(
    ϕ, ϕ_old, cfield, flags, tag, flux,
    c::NTuple{Q, SVector{3, Int}},
    Nx::Int, Ny::Int, Nz::Int, n::Int,
) where {Q}
    ϕn = ϕ[n]
    if (flags[n] & TYPE_SU) == TYPE_I
        CType = eltype(cfield)
        δ = -CType(cfield[n]) * (CType(ϕn) - CType(ϕ_old[n]))
        if δ != zero(CType)
            tagn = tag[n]
            if tagn > Int32(0)
                tagn <= Int32(MAX_BUBBLES) && acc_add!(flux, Int(tagn), δ)
            elseif tagn == Int32(0)
                n0 = n - 1
                x = n0 % Nx
                y = (n0 ÷ Nx) % Ny
                z = n0 ÷ (Nx * Ny)
                jtag = Int32(0)
                for i in 2:Q
                    src = src_index(x, y, z, c[i][1], c[i][2], c[i][3], Nx, Ny, Nz)
                    ts = tag[src]
                    if ts > Int32(0) && (jtag == Int32(0) || ts < jtag)
                        jtag = ts
                    end
                end
                if jtag > Int32(0) && jtag <= Int32(MAX_BUBBLES)
                    acc_add!(flux, Int(jtag), δ)
                end
            end
        end
    end
    ϕ_old[n] = ϕn
    return nothing
end

@kernel function ϕ_correction_kernel!(
    @Const(ϕ), ϕ_old, @Const(cfield), @Const(flags), @Const(tag), flux,
    c::NTuple{Q, SVector{3, Int}},
    Nx::Int, Ny::Int, Nz::Int,
) where {Q}
    n = @index(Global)
    @inbounds ϕ_correction_body!(ϕ, ϕ_old, cfield, flags, tag, flux, c, Nx, Ny, Nz, Int(n))
end

end
