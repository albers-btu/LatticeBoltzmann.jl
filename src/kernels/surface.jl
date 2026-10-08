using KernelAbstractions

@inline function calculate_phi(ρn::CType, massn::CType, flagsn::UInt8) where {CType}
    su = flagsn & TYPE_SU
    if su == TYPE_F || su == TYPE_IF
        return one(CType)
    elseif su == TYPE_I || su == TYPE_GI
        return ρn > 0 ? clamp(massn / ρn, zero(CType), one(CType)) : CType(0.5)
    else
        return zero(CType)
    end
end

@static if SURFACE

# This function is the setup, which is called before the collide pass. It
# updates the mass and massex (mass excess) and sets up / reconstructs 
# the fᵢ/gᵢ. surface_0 only runs on TYPE_F (fluid) and TYPE_I (interface) 
# and skips TYPE_S (solid) and TYPE_G (gas).
@inline function surface_0_body!(
    t_odd::Val{odd},
    fi, ρ, u, flags, mass, massex, ϕ, T, fs, gi,
    w::NTuple{Q, CType},
    c::NTuple{Q, SVector{3, Int}},
    fx::CType, fy::CType, fz::CType, σ::CType, σT::CType, Tσ::CType,
    Λ_v::CType, T_v::CType, p0v::CType, β_v::CType,
    N::Int, Nx::Int, Ny::Int, Nz::Int, n, Eacc, hT, Qin, ω_T::CType,
    ::Val{thermal}, g_odd::Bool
) where {odd, Q, CType, thermal}
    flagsn = flags[n]
    bo = flagsn & TYPE_BO
    su = flagsn & TYPE_SU
    (bo == TYPE_S || su == TYPE_G) && return nothing

    n0 = n - 1
    x = n0 % Nx
    y = (n0 ÷ Nx) % Ny
    z = n0 ÷ (Nx * Ny)

    solidified = false
    @static if TEMPERATURE
        solidified = is_solid_fraction(fs[n])
    end

    # massex[src] is that donor's total give. Pull this cell's share of it.
    # ϕ is unchanged since the donor's surface_3, so the weights match.
    massn = mass[n]
    for i in 2:Q
        cx, cy, cz = c[i][1], c[i][2], c[i][3]
        src = src_index(x, y, z, cx, cy, cz, Nx, Ny, Nz)
        mx = CType(massex[src])
        mx == zero(CType) && continue
        xd = wrap_coord(x, cx, Nx)
        yd = wrap_coord(y, cy, Ny)
        zd = wrap_coord(z, cz, Nz)
        massn += mx * donor_share(ϕ, flags, fs, xd, yd, zd, -cx, -cy, -cz, c, Nx, Ny, Nz, mx, CType)
    end

    # Number of ± velocity pairs
    NP = (Q - 1) ÷ 2

    if su == TYPE_F
        # Calculate net mass gain/loss
        if !solidified
            for k in 1:NP
                i = 2k # plus velocities
                cp, cm = c[i], c[i + 1]
                srcp = src_index(x, y, z, cp[1], cp[2], cp[3], Nx, Ny, Nz)
                srcm = src_index(x, y, z, cm[1], cm[2], cm[3], Nx, Ny, Nz)
                fp_in,  fm_in  = load_bb_pair(fi, flags, n, srcp, srcm, i, t_odd, N, CType)
                fp_out, fm_out = load_outgoing_pair(fi, n, srcp, i, t_odd, N, CType)
                massn += (fp_in - fp_out) + (fm_in - fm_out)
            end
        end
        mass[n] = massn

        @static if TEMPERATURE
            if thermal
                fillc = ϕ[n]
                fillc = ifelse(fillc > zero(CType), fillc, zero(CType))
                reconstruct_g_adiabatic!(g_odd, gi, CType(T[n]), flags, x, y, z, n, N, Nx, Ny, Nz, CType)
                reconstruct_g_boundaries!(g_odd, gi, T, flags, hT, Qin, x, y, z, n, N, Nx, Ny, Nz, CType, Eacc, fillc, ω_T)
            end
        end
        return nothing
    end

    if su != TYPE_I
        mass[n] = massn
        return nothing
    end

    # Applies only to the interface (TYPE_F)
    cs = CType(1) / sqrt(CType(3))
    @static if EQUILIBRIUM_BOUNDARIES
        eq = (flagsn & TYPE_BO) == TYPE_E
    else
        eq = false
    end

    # This big ifelse block calculates the velocities and density for a
    # given cell n.
    if solidified
        ρn = ρ[n]
        ρn = ρn > zero(CType) ? ρn : one(CType)
        ux = zero(CType); uy = zero(CType); uz = zero(CType)
        uxg = zero(CType); uyg = zero(CType); uzg = zero(CType)
        ρ_csf = one(CType)
        ρ_gas = one(CType)
        ϕn = calculate_phi(ρn, massn, flagsn)
    elseif eq
        ρn, ux, uy, uz = prescribed_hydro(ρ[n], u[n, 1], u[n, 2], u[n, 3], fx, fy, fz)
        ϕn = calculate_phi(ρn, massn, flagsn)
        σn = σ
        @static if TEMPERATURE
            σn = σ + σT * (T[n] - Tσ)
            σn = ifelse(σn > zero(CType), σn, zero(CType))
        end
        ρ_csf = gas_density_plic(σn, ϕ, ϕn, x, y, z, Nx, Ny, Nz)
        ρ_gas = ρ_csf
        @static if TEMPERATURE
            # Unsupported interface (no bulk neighbor) keeps the capillary
            # density. Recoil there launches the cell instead of denting a pool.
            if recoil_has_bulk(flags, x, y, z, c, Nx, Ny, Nz)
                ρ_gas = gas_density_recoil(ρ_csf, CType(T[n]), Λ_v, T_v, p0v, β_v)
            end
        end
        uxg, uyg, uzg = ux, uy, uz
    else
        ρn = CType(fi[f_index(n, 1, N)])
        ux = zero(CType); uy = zero(CType); uz = zero(CType)
        for k in 1:NP
            i = 2k
            src = src_index(x, y, z, c[i][1], c[i][2], c[i][3], Nx, Ny, Nz)
            fp_out, fm_out = load_outgoing_pair(fi, n, src, i, t_odd, N, CType)
            ρn += fp_out + fm_out
            ux += CType(c[i][1])*fp_out + CType(c[i+1][1])*fm_out
            uy += CType(c[i][2])*fp_out + CType(c[i+1][2])*fm_out
            @static if DIM == 3
                uz += CType(c[i][3])*fp_out + CType(c[i+1][3])*fm_out
            end
        end
        if ρn <= zero(CType)
            ρn = one(CType)
            ux = zero(CType); uy = zero(CType); uz = zero(CType)
        else
            invρ = one(CType) / ρn
            ux *= invρ; uy *= invρ; uz *= invρ
            ux = clamp(ux, -cs, cs); uy = clamp(uy, -cs, cs); uz = clamp(uz, -cs, cs)
        end
        ϕn = calculate_phi(ρn, massn, flagsn)
        σn = σ
        @static if TEMPERATURE
            σn = σ + σT * (T[n] - Tσ)
            σn = ifelse(σn > zero(CType), σn, zero(CType))
        end
        ρ_csf = gas_density_plic(σn, ϕ, ϕn, x, y, z, Nx, Ny, Nz)
        ρ_gas = ρ_csf
        @static if TEMPERATURE
            # Unsupported interface (no bulk neighbor) keeps the capillary
            # density. Recoil there launches the cell instead of denting a pool.
            if recoil_has_bulk(flags, x, y, z, c, Nx, Ny, Nz)
                ρ_gas = gas_density_recoil(ρ_csf, CType(T[n]), Λ_v, T_v, p0v, β_v)
            end
        end
        @static if VOLUME_FORCE
            uxg = clamp(ux + fx / (CType(2) * ρn), -cs, cs)
            uyg = clamp(uy + fy / (CType(2) * ρn), -cs, cs)
            @static if DIM == 3
                uzg = clamp(uz + fz / (CType(2) * ρn), -cs, cs)
            elseif DIM == 2
                uzg = zero(CType)
            end
        else
            uxg, uyg, uzg = ux, uy, uz
        end
        @static if TEMPERATURE
            inv2ρ = one(CType) / (CType(2) * ρn)
            if σT != zero(CType)
                mx, my, mz = marangoni_force(T, ϕ, flags, σT, x, y, z, n, Nx, Ny, Nz, CType)
                uxg = clamp(uxg + mx * inv2ρ, -cs, cs)
                uyg = clamp(uyg + my * inv2ρ, -cs, cs)
                @static if DIM == 3
                    uzg = clamp(uzg + mz * inv2ρ, -cs, cs)
                end
            end
        end
    end
    # D2Q9 has cz = 0, but uzg still enters uug. Do not let fz back in.
    @static if DIM == 2
        uz = zero(CType)
        uzg = zero(CType)
    end
    uug = CType(1.5) * (uxg*uxg + uyg*uyg + uzg*uzg)

    for k in 1:NP
        i = 2k
        cp, cm = c[i], c[i + 1]
        srcp = src_index(x, y, z, cp[1], cp[2], cp[3], Nx, Ny, Nz)
        srcm = src_index(x, y, z, cm[1], cm[2], cm[3], Nx, Ny, Nz)
        sup = flags[srcp] & TYPE_SU
        sum_ = flags[srcm] & TYPE_SU
        ϕp = ϕ[srcp]; ϕm = ϕ[srcm]

        fp_in,  fm_in  = load_pair(fi, n, srcp, i, t_odd, N, CType)
        fp_out, fm_out = load_outgoing_pair(fi, n, srcp, i, t_odd, N, CType)

        if !solidified
            # A solidified neighbor skips its own mass update. Taking the
            # flux here would delete it: the frozen plate never credits it back.
            take_p = (sup & (TYPE_F | TYPE_I)) != 0x00
            take_m = (sum_ & (TYPE_F | TYPE_I)) != 0x00
            @static if TEMPERATURE
                take_p = take_p & !is_solid_fraction(fs[srcp])
                take_m = take_m & !is_solid_fraction(fs[srcm])
            end
            if take_p
                fluxp = fm_in - fp_out
                massn += sup == TYPE_F ? fluxp : CType(0.5) * (ϕp + ϕn) * fluxp
            end
            if take_m
                fluxm = fp_in - fm_out
                massn += sum_ == TYPE_F ? fluxm : CType(0.5) * (ϕm + ϕn) * fluxm
            end
        end

        # Anti-bounce at the capillary density. Recoil is only the extra
        # equilibrium pressure: mirroring f_out at ρ_gas copies a fast
        # outgoing link back in and the substep sum runs |u| onto c_s.
        fegp = feq(w[i],     ρ_csf, uxg, uyg, uzg, uug, cp, CType)
        fegm = feq(w[i + 1], ρ_csf, uxg, uyg, uzg, uug, cm, CType)
        fp_rec = fegm - fm_out + fegp
        fm_rec = fegp - fp_out + fegm
        if ρ_gas != ρ_csf
            fp_rec += feq(w[i],     ρ_gas, uxg, uyg, uzg, uug, cp, CType) - fegp
            fm_rec += feq(w[i + 1], ρ_gas, uxg, uyg, uzg, uug, cm, CType) - fegm
        end
        fp_rec = bound_population(fp_rec, fegp)
        fm_rec = bound_population(fm_rec, fegm)
        store_reconstructed_pair!(
            fi, n, srcp, i, fm_rec, fp_rec,
            sup == TYPE_G, sum_ == TYPE_G, t_odd, N)
    end
    mass[n] = massn
    @static if TEMPERATURE
        if thermal
            fillc = ϕn
            fillc = ifelse(fillc > zero(CType), fillc, zero(CType))
            reconstruct_g_adiabatic!(g_odd, gi, CType(T[n]), flags, x, y, z, n, N, Nx, Ny, Nz, CType)
            reconstruct_g_boundaries!(g_odd, gi, T, flags, hT, Qin, x, y, z, n, N, Nx, Ny, Nz, CType, Eacc, fillc, ω_T)
        end
    end
    return nothing
end

@kernel function surface_0_even_kernel!(
    fi, @Const(ρ), @Const(u), @Const(flags), mass, @Const(massex), @Const(ϕ), T, fs, gi,
    w::NTuple{Q, CType}, c::NTuple{Q, SVector{3, Int}},
    fx::CType, fy::CType, fz::CType, σ::CType, σT::CType, Tσ::CType,
    Λ_v::CType, T_v::CType, p0v::CType, β_v::CType,
    N::Int, Nx::Int, Ny::Int, Nz::Int, Eacc, hT, Qin, ω_T::CType,
    ::Val{thermal}, g_odd::Bool
) where {Q, CType, thermal}
    n = @index(Global)
    @inbounds surface_0_body!(Val(false), fi, ρ, u, flags, mass, massex, ϕ, T, fs, gi, w, c, fx, fy, fz, σ, σT, Tσ, Λ_v, T_v, p0v, β_v, N, Nx, Ny, Nz, Int(n), Eacc, hT, Qin, ω_T, Val(thermal), g_odd)
end

@kernel function surface_0_odd_kernel!(
    fi, @Const(ρ), @Const(u), @Const(flags), mass, @Const(massex), @Const(ϕ), T, fs, gi,
    w::NTuple{Q, CType}, c::NTuple{Q, SVector{3, Int}},
    fx::CType, fy::CType, fz::CType, σ::CType, σT::CType, Tσ::CType,
    Λ_v::CType, T_v::CType, p0v::CType, β_v::CType,
    N::Int, Nx::Int, Ny::Int, Nz::Int, Eacc, hT, Qin, ω_T::CType,
    ::Val{thermal}, g_odd::Bool
) where {Q, CType, thermal}
    n = @index(Global)
    @inbounds surface_0_body!(Val(true), fi, ρ, u, flags, mass, massex, ϕ, T, fs, gi, w, c, fx, fy, fz, σ, σT, Tσ, Λ_v, T_v, p0v, β_v, N, Nx, Ny, Nz, Int(n), Eacc, hT, Qin, ω_T, Val(thermal), g_odd)
end

# True when j will run surface_0 and add the donor's massex.
@inline function liquid_receiver(flags, fs, j::Int)
    suj = flags[j] & (TYPE_SU | TYPE_S)
    liquid = suj == TYPE_F || suj == TYPE_I || suj == TYPE_IF || suj == TYPE_GI
    @static if TEMPERATURE
        liquid = liquid && !is_solid_fraction(fs[j])
    end
    return liquid
end

# F/I/IF/GI, solidified included. These cells can hold surplus.
@inline function metal_holder(flags, j::Int)
    su = flags[j] & (TYPE_SU | TYPE_S)
    return su == TYPE_F || su == TYPE_I || su == TYPE_IF || su == TYPE_GI
end

@inline function touches_metal(flags, x::Int, y::Int, z::Int, c::NTuple{Q, SVector{3, Int}}, Nx::Int, Ny::Int, Nz::Int) where {Q}
    for i in 2:Q
        j = src_index(x, y, z, c[i][1], c[i][2], c[i][3], Nx, Ny, Nz)
        metal_holder(flags, j) && return true
    end
    return false
end

# Outward unit normal −∇ϕ/|∇ϕ|. ok is false when the gradient vanishes.
@inline function outward_normal(ϕ, x::Int, y::Int, z::Int, Nx::Int, Ny::Int, Nz::Int, ::Type{CType}) where {CType}
    half = CType(0.5)
    @static if DIM == 3
        xp = src_index(x, y, z,  1, 0, 0, Nx, Ny, Nz)
        xm = src_index(x, y, z, -1, 0, 0, Nx, Ny, Nz)
        yp = src_index(x, y, z,  0, 1, 0, Nx, Ny, Nz)
        ym = src_index(x, y, z,  0,-1, 0, Nx, Ny, Nz)
        zp = src_index(x, y, z,  0, 0, 1, Nx, Ny, Nz)
        zm = src_index(x, y, z,  0, 0,-1, Nx, Ny, Nz)
        dϕx = half * (ϕ[xp] - ϕ[xm])
        dϕy = half * (ϕ[yp] - ϕ[ym])
        dϕz = half * (ϕ[zp] - ϕ[zm])
    else
        xp = src_index(x, y, z,  1, 0, 0, Nx, Ny, Nz)
        xm = src_index(x, y, z, -1, 0, 0, Nx, Ny, Nz)
        yp = src_index(x, y, z,  0, 1, 0, Nx, Ny, Nz)
        ym = src_index(x, y, z,  0,-1, 0, Nx, Ny, Nz)
        dϕx = half * (ϕ[xp] - ϕ[xm])
        dϕy = half * (ϕ[yp] - ϕ[ym])
        dϕz = zero(CType)
    end
    mag = sqrt(dϕx * dϕx + dϕy * dϕy + dϕz * dϕz)
    if mag > eps(CType)
        invm = -one(CType) / mag
        return dϕx * invm, dϕy * invm, dϕz * invm, true
    end
    return zero(CType), zero(CType), zero(CType), false
end

# max(n · ĉ, 0), ĉ = c/|c|. Unit directions, so a flat top weights +z above the diagonals.
@inline function normal_weight(nx::CType, ny::CType, nz::CType, cx::Int, cy::Int, cz::Int, ::Type{CType}) where {CType}
    l2 = cx * cx + cy * cy + cz * cz
    l2 == 0 && return zero(CType)
    invl = one(CType) / sqrt(CType(l2))
    dot = (nx * CType(cx) + ny * CType(cy) + nz * CType(cz)) * invl
    return dot > zero(CType) ? dot : zero(CType)
end

# True when this cell has bulk fluid or a wall on a face. A free-surface
# cell sitting on the plate is backed. A cell with gas on both sides is not.
@inline function _face_backed(flags, x::Int, y::Int, z::Int, dx::Int, dy::Int, dz::Int, Nx::Int, Ny::Int, Nz::Int)
    j = src_index(x, y, z, dx, dy, dz, Nx, Ny, Nz)
    fj = flags[j]
    su = fj & (TYPE_SU | TYPE_S)
    return su == TYPE_F || (fj & TYPE_BO) == TYPE_S
end

@inline function backed_by_bulk(flags, x::Int, y::Int, z::Int, Nx::Int, Ny::Int, Nz::Int)
    _face_backed(flags, x, y, z,  1, 0, 0, Nx, Ny, Nz) && return true
    _face_backed(flags, x, y, z, -1, 0, 0, Nx, Ny, Nz) && return true
    _face_backed(flags, x, y, z,  0, 1, 0, Nx, Ny, Nz) && return true
    _face_backed(flags, x, y, z,  0,-1, 0, Nx, Ny, Nz) && return true
    @static if DIM == 3
        _face_backed(flags, x, y, z,  0, 0,  1, Nx, Ny, Nz) && return true
        _face_backed(flags, x, y, z,  0, 0, -1, Nx, Ny, Nz) && return true
    end
    return false
end

# True when this cube face can still take volume: gas, an empty flag, or
# metal stored below one cell. A wall and a full metal cell cannot.
@inline function face_has_capacity(ϕ, flags, x::Int, y::Int, z::Int, dx::Int, dy::Int, dz::Int, Nx::Int, Ny::Int, Nz::Int, ::Type{CType}) where {CType}
    j = src_index(x, y, z, dx, dy, dz, Nx, Ny, Nz)
    fj = flags[j]
    (fj & TYPE_BO) == TYPE_S && return false
    if metal_holder(flags, j)
        return ϕ[j] < one(CType) - CType(1.0e-3)
    end
    return true
end

@inline function any_face_capacity(ϕ, flags, x::Int, y::Int, z::Int, Nx::Int, Ny::Int, Nz::Int, ::Type{CType}) where {CType}
    face_has_capacity(ϕ, flags, x, y, z,  1, 0, 0, Nx, Ny, Nz, CType) && return true
    face_has_capacity(ϕ, flags, x, y, z, -1, 0, 0, Nx, Ny, Nz, CType) && return true
    face_has_capacity(ϕ, flags, x, y, z,  0, 1, 0, Nx, Ny, Nz, CType) && return true
    face_has_capacity(ϕ, flags, x, y, z,  0,-1, 0, Nx, Ny, Nz, CType) && return true
    @static if DIM == 3
        face_has_capacity(ϕ, flags, x, y, z,  0, 0,  1, Nx, Ny, Nz, CType) && return true
        face_has_capacity(ϕ, flags, x, y, z,  0, 0, -1, Nx, Ny, Nz, CType) && return true
    end
    return false
end

# Distance in cells to the first unfilled cell on this axis, without wrapping
# and without crossing a wall. (0, 0) means there is none. The mass still
# moves only into the direct neighbor; the distance picks the axis.
@inline function axis_unfilled(
    ϕ, flags, ϕn::CType,
    x::Int, y::Int, z::Int, cx::Int, cy::Int, cz::Int,
    Nx::Int, Ny::Int, Nz::Int, ::Type{CType}
) where {CType}
    xi = x
    yi = y
    zi = z
    for dist in 1:48
        xi += cx; yi += cy; zi += cz
        (xi < 0 || xi >= Nx || yi < 0 || yi >= Ny || zi < 0 || zi >= Nz) && return zero(CType), 0
        j = xi + yi * Nx + zi * Nx * Ny + 1
        fj = flags[j]
        (fj & TYPE_BO) == TYPE_S && return zero(CType), 0
        opening = ϕn - ϕ[j]
        opening > CType(1.0e-3) && return opening, dist
        metal_holder(flags, j) || return zero(CType), 0
    end
    return zero(CType), 0
end

@inline function _nearest_axis(dmin::Int, ϕ, flags, ϕn::CType, x::Int, y::Int, z::Int, cx::Int, cy::Int, cz::Int, Nx::Int, Ny::Int, Nz::Int, ::Type{CType}) where {CType}
    j = src_index(x, y, z, cx, cy, cz, Nx, Ny, Nz)
    metal_holder(flags, j) || return dmin
    ϕ[j] >= one(CType) - CType(1.0e-3) || return dmin
    _, dist = axis_unfilled(ϕ, flags, ϕn, x, y, z, cx, cy, cz, Nx, Ny, Nz, CType)
    dist == 0 && return dmin
    return (dmin == 0 || dist < dmin) ? dist : dmin
end

@inline function nearest_opening_dist(ϕ, flags, ϕn::CType, x::Int, y::Int, z::Int, Nx::Int, Ny::Int, Nz::Int, ::Type{CType}) where {CType}
    dmin = 0
    dmin = _nearest_axis(dmin, ϕ, flags, ϕn, x, y, z,  1, 0, 0, Nx, Ny, Nz, CType)
    dmin = _nearest_axis(dmin, ϕ, flags, ϕn, x, y, z, -1, 0, 0, Nx, Ny, Nz, CType)
    dmin = _nearest_axis(dmin, ϕ, flags, ϕn, x, y, z,  0, 1, 0, Nx, Ny, Nz, CType)
    dmin = _nearest_axis(dmin, ϕ, flags, ϕn, x, y, z,  0,-1, 0, Nx, Ny, Nz, CType)
    @static if DIM == 3
        dmin = _nearest_axis(dmin, ϕ, flags, ϕn, x, y, z,  0, 0,  1, Nx, Ny, Nz, CType)
        dmin = _nearest_axis(dmin, ϕ, flags, ϕn, x, y, z,  0, 0, -1, Nx, Ny, Nz, CType)
    end
    return dmin
end

# Aperture of one cube face. Incompressible metal fills a cell up to φ = 1,
# so m ≤ ρ and the surplus δm = max(m − ρ, 0) is volume that has to occupy
# another cell. It leaves through a face, never through a corner.
# One free surface has n = −∇φ/|∇φ|, from the centered difference
# (φ(ê) − φ(−ê))/2, and the face weight is max(n·ĉ, 0). That difference is
# zero when both neighbors hold the same fill. If either neighbor is bulk
# or a wall, the surface has a back side and the weight stays max(n·ĉ, 0),
# so a flat top still puts the excess on +z. If neither neighbor is backed,
# the cell is a sheet on that axis and each open face weighs φ − φ_nb.
# A neighbor already at φ = 1 has no free volume, so its weight is zero
# while any face still has room. When every face is full, the surplus steps
# one cell along the nearest axis that reaches an unfilled cell. The direct
# neighbor receives it and passes it on; the opening itself is not skipped.
# A diagonal link is not a face of the cube. Handing the surplus to the
# diagonal leaves the orthogonal face gas, and the next metal cell then
# shares only a corner with the donor.
@inline function aperture_weight(
    ϕ, flags, ϕn::CType, nx::CType, ny::CType, nz::CType, normal_ok::Bool, hop::Bool,
    x::Int, y::Int, z::Int, cx::Int, cy::Int, cz::Int,
    Nx::Int, Ny::Int, Nz::Int, ::Type{CType}
) where {CType}
    l2 = cx * cx + cy * cy + cz * cz
    l2 == 1 || return zero(CType)
    j = src_index(x, y, z, cx, cy, cz, Nx, Ny, Nz)
    jo = src_index(x, y, z, -cx, -cy, -cz, Nx, Ny, Nz)
    δ = ϕ[j] - ϕ[jo]
    tol = CType(1.0e-4)
    if abs(δ) <= tol
        xj = wrap_coord(x,  cx, Nx)
        yj = wrap_coord(y,  cy, Ny)
        zj = wrap_coord(z,  cz, Nz)
        xo = wrap_coord(x, -cx, Nx)
        yo = wrap_coord(y, -cy, Ny)
        zo = wrap_coord(z, -cz, Nz)
        # Bulk or a wall on either side means this axis has a back.
        sheet = !backed_by_bulk(flags, xj, yj, zj, Nx, Ny, Nz) &&
                !backed_by_bulk(flags, xo, yo, zo, Nx, Ny, Nz)
        drop = ϕn - ϕ[j]
        if sheet && drop > zero(CType)
            return drop
        end
    end
    # Full metal is not a sink. A buried solid cell has no flux, so the extra
    # volume is placed one cell toward the nearest unfilled cell. Buried
    # liquid is shared across liquid neighbors in donor_share.
    if metal_holder(flags, j) && ϕ[j] >= one(CType) - CType(1.0e-3)
        if hop && !any_face_capacity(ϕ, flags, x, y, z, Nx, Ny, Nz, CType)
            opening, dist = axis_unfilled(ϕ, flags, ϕn, x, y, z, cx, cy, cz, Nx, Ny, Nz, CType)
            dmin = nearest_opening_dist(ϕ, flags, ϕn, x, y, z, Nx, Ny, Nz, CType)
            if dist > 0 && dist == dmin && opening > zero(CType)
                return opening
            end
        end
        return zero(CType)
    end
    return normal_ok ? normal_weight(nx, ny, nz, cx, cy, cz, CType) : zero(CType)
end

@inline function weight_sum_metal(
    ϕ, flags, x::Int, y::Int, z::Int,
    nx::CType, ny::CType, nz::CType, normal_ok::Bool, hop::Bool,
    c::NTuple{Q, SVector{3, Int}}, Nx::Int, Ny::Int, Nz::Int, ::Type{CType}
) where {Q, CType}
    ϕn = ϕ[x + y * Nx + z * Nx * Ny + 1]
    s = zero(CType)
    for i in 2:Q
        cx, cy, cz = c[i][1], c[i][2], c[i][3]
        j = src_index(x, y, z, cx, cy, cz, Nx, Ny, Nz)
        metal_holder(flags, j) || continue
        s += aperture_weight(ϕ, flags, ϕn, nx, ny, nz, normal_ok, hop, x, y, z, cx, cy, cz, Nx, Ny, Nz, CType)
    end
    return s
end

@inline function count_liquid_receivers(flags, fs, x::Int, y::Int, z::Int, c::NTuple{Q, SVector{3, Int}}, Nx::Int, Ny::Int, Nz::Int) where {Q}
    cnt = 0
    for i in 2:Q
        j = src_index(x, y, z, c[i][1], c[i][2], c[i][3], Nx, Ny, Nz)
        cnt += Int(liquid_receiver(flags, fs, j))
    end
    return cnt
end

# Share of the donor's massex that the cell in direction (rx,ry,rz) receives.
# Positive give is split by face aperture. While the leading gas face still
# outranks the metal in front of it, publish_surplus leaves the surplus on
# the donor, liquid and solid alike, until that face holds half a cell.
# A buried solid cell passes one cell toward the nearest unfilled face.
# Buried liquid has no flux of its own: the extra volume is shared across
# the liquid neighbors. Gas is not a receiver until surface_1 has made it metal.
@inline function donor_share(
    ϕ, flags, fs,
    xd::Int, yd::Int, zd::Int,
    rx::Int, ry::Int, rz::Int,
    c::NTuple{Q, SVector{3, Int}},
    Nx::Int, Ny::Int, Nz::Int,
    mx::CType, ::Type{CType}
) where {Q, CType}
    mx == zero(CType) && return zero(CType)
    nd = xd + yd * Nx + zd * Nx * Ny + 1
    su = flags[nd] & (TYPE_SU | TYPE_S)
    jself = src_index(xd, yd, zd, rx, ry, rz, Nx, Ny, Nz)
    use_normal = false
    nx = zero(CType)
    ny = zero(CType)
    nz = zero(CType)
    frozen_donor = false
    @static if TEMPERATURE
        frozen_donor = is_solid_fraction(fs[nd])
    end
    if mx > zero(CType) && (su == TYPE_I || su == TYPE_IF || su == TYPE_F)
        nx, ny, nz, use_normal = outward_normal(ϕ, xd, yd, zd, Nx, Ny, Nz, CType)
    end
    wsum = weight_sum_metal(ϕ, flags, xd, yd, zd, nx, ny, nz, use_normal, frozen_donor, c, Nx, Ny, Nz, CType)
    if wsum > zero(CType)
        metal_holder(flags, jself) || return zero(CType)
        ϕd = ϕ[nd]
        wself = aperture_weight(ϕ, flags, ϕd, nx, ny, nz, use_normal, frozen_donor, xd, yd, zd, rx, ry, rz, Nx, Ny, Nz, CType)
        wself <= zero(CType) && return zero(CType)
        return wself / wsum
    end
    # No metal lies on an exposed face. A solidified interface does not flow.
    # Bulk and gas still pass massex on to whatever liquid can take it.
    if frozen_donor && (su == TYPE_I || su == TYPE_IF)
        return zero(CType)
    end
    liquid_receiver(flags, fs, jself) || return zero(CType)
    cnt = count_liquid_receivers(flags, fs, xd, yd, zd, c, Nx, Ny, Nz)
    cnt == 0 && return zero(CType)
    return one(CType) / CType(cnt)
end

# Gas cell of largest aperture, plus the aperture already on metal.
# Several gas faces can share that largest aperture. jbest is one of them.
@inline function leading_gas(ϕ, flags, x::Int, y::Int, z::Int, c::NTuple{Q, SVector{3, Int}}, Nx::Int, Ny::Int, Nz::Int, hop::Bool, ::Type{CType}) where {Q, CType}
    jbest = 0
    wbest = zero(CType)
    wmetal = zero(CType)
    nx, ny, nz, ok = outward_normal(ϕ, x, y, z, Nx, Ny, Nz, CType)
    ϕn = ϕ[x + y * Nx + z * Nx * Ny + 1]
    for i in 2:Q
        cx, cy, cz = c[i][1], c[i][2], c[i][3]
        w = aperture_weight(ϕ, flags, ϕn, nx, ny, nz, ok, hop, x, y, z, cx, cy, cz, Nx, Ny, Nz, CType)
        w <= zero(CType) && continue
        j = src_index(x, y, z, cx, cy, cz, Nx, Ny, Nz)
        su = flags[j] & (TYPE_SU | TYPE_S)
        if su == TYPE_G
            if w > wbest
                wbest = w
                jbest = j
            end
        elseif metal_holder(flags, j)
            wmetal += w
        end
    end
    return jbest, wbest, wmetal
end

@inline function has_gas_neighbor(flags, x::Int, y::Int, z::Int, c::NTuple{Q, SVector{3, Int}}, Nx::Int, Ny::Int, Nz::Int) where {Q}
    for i in 2:Q
        j = src_index(x, y, z, c[i][1], c[i][2], c[i][3], Nx, Ny, Nz)
        if (flags[j] & (TYPE_SU | TYPE_S)) == TYPE_G
            return true
        end
    end
    return false
end

# Recoil is the gas pressure on a surface that has bulk metal behind it.
# A filament made only of interface cells has nothing to support that
# pressure: once a neck opens, the underside pushes the free piece up to c_s.
# TYPE_F (liquid or solid bulk) or a wall supplies that backing. Another
# interface cell does not.
@inline function recoil_has_bulk(flags, x::Int, y::Int, z::Int, c::NTuple{Q, SVector{3, Int}}, Nx::Int, Ny::Int, Nz::Int) where {Q}
    for i in 2:Q
        j = src_index(x, y, z, c[i][1], c[i][2], c[i][3], Nx, Ny, Nz)
        fj = flags[j]
        su = fj & (TYPE_SU | TYPE_S)
        if su == TYPE_F || (fj & TYPE_BO) == TYPE_S
            return true
        end
    end
    return false
end

# True when some neighbor will pull massex. A frozen metal cell does, when
# the face aperture points at it; liquid receivers do as well.
@inline function surplus_is_claimed(ϕ, flags, fs, x::Int, y::Int, z::Int, c::NTuple{Q, SVector{3, Int}}, Nx::Int, Ny::Int, Nz::Int, ::Type{CType}) where {Q, CType}
    count_liquid_receivers(flags, fs, x, y, z, c, Nx, Ny, Nz) > 0 && return true
    nx, ny, nz, ok = outward_normal(ϕ, x, y, z, Nx, Ny, Nz, CType)
    n = x + y * Nx + z * Nx * Ny + 1
    hop = false
    @static if TEMPERATURE
        hop = is_solid_fraction(fs[n])
    end
    return weight_sum_metal(ϕ, flags, x, y, z, nx, ny, nz, ok, hop, c, Nx, Ny, Nz, CType) > zero(CType)
end

# At most one cell of surplus per step. While the leading gas face still
# outranks metal in front of it, the surplus stays here. surface_1 opens
# that face once its share is half a cell, and the same rule does it for
# liquid and for solid. The opened cell is then metal with room and takes
# the give, including the underside of a lip. A full neighbor receives
# nothing unless it is the one-cell step from a buried solid cell toward
# the nearest unfilled cell.
@inline function publish_surplus(massn::CType, ρn::CType, ϕ, flags, fs, x::Int, y::Int, z::Int, c::NTuple{Q, SVector{3, Int}}, Nx::Int, Ny::Int, Nz::Int, frozen::Bool, ::Type{CType}) where {Q, CType}
    give = min(massn - ρn, ρn)
    give <= zero(CType) && return massn, zero(CType)
    _, wbest, wmetal = leading_gas(ϕ, flags, x, y, z, c, Nx, Ny, Nz, frozen, CType)
    if wbest > wmetal && wbest > zero(CType)
        return massn, zero(CType)
    end
    nx, ny, nz, ok = outward_normal(ϕ, x, y, z, Nx, Ny, Nz, CType)
    if weight_sum_metal(ϕ, flags, x, y, z, nx, ny, nz, ok, frozen, c, Nx, Ny, Nz, CType) > zero(CType)
        return massn - give, give
    end
    if !frozen && count_liquid_receivers(flags, fs, x, y, z, c, Nx, Ny, Nz) > 0
        return massn - give, give
    end
    return massn, zero(CType)
end

# Bulk holds at most ρ. Up to one ρ of surplus leaves this step; a surplus
# nobody claims stays in the cell. Streaming of a solid is still zero: this
# is the volume constraint, not advection.
@inline function place_surplus(massn::CType, ρn::CType, ϕ, flags, fs, x::Int, y::Int, z::Int, c::NTuple{Q, SVector{3, Int}}, Nx::Int, Ny::Int, Nz::Int, ::Type{CType}) where {Q, CType}
    extra = massn - ρn
    extra > zero(CType) || return massn, zero(CType)
    shift = extra < ρn ? extra : ρn
    kept = massn - shift
    if surplus_is_claimed(ϕ, flags, fs, x, y, z, c, Nx, Ny, Nz, CType)
        return kept, shift
    end
    return massn, zero(CType)
end

# Equal-split massex has nowhere to go. Put it back on the cell.
@inline function park_if_unclaimed(massn::CType, massexn::CType, flags, fs, x::Int, y::Int, z::Int, c::NTuple{Q, SVector{3, Int}}, Nx::Int, Ny::Int, Nz::Int) where {Q, CType}
    if massexn != zero(CType) && count_liquid_receivers(flags, fs, x, y, z, c, Nx, Ny, Nz) == 0
        return massn + massexn, zero(CType)
    end
    return massn, massexn
end

@kernel function surface_1_kernel!(
    flags, ϕ, mass, ρ,
    c::NTuple{Q, SVector{3, Int}},
    N::Int, Nx::Int, Ny::Int, Nz::Int
) where {Q}
    n = @index(Global)
    @inbounds begin
        sus = flags[n] & (TYPE_SU | TYPE_S)
        if sus == TYPE_IF
            CType = eltype(ρ)
            n0 = n - 1
            x = n0 % Nx; y = (n0 ÷ Nx) % Ny; z = n0 ÷ (Nx * Ny)
            for i in 2:Q
                j = src_index(x, y, z, c[i][1], c[i][2], c[i][3], Nx, Ny, Nz)
                fj = flags[j]
                if (fj & (TYPE_SU | TYPE_S)) == TYPE_IG
                    flags[j] = (fj & ~TYPE_SU) | TYPE_I
                end
            end
            ρn = ρ[n]
            massn = mass[n]
            if ρn > zero(CType) && massn > ρn
                _, wbest, wmetal = leading_gas(ϕ, flags, x, y, z, c, Nx, Ny, Nz, false, CType)
                # Every gas face that shares the largest aperture advances.
                # A flat top has a unique +z maximum. A normal along a diagonal
                # ties the two cube faces and opens both; the diagonal cell is
                # not a face, so it stays gas. A lip with gas above and below
                # ties the underside with the forward face.
                # The share is weighed against metal already in front, and the
                # face opens only at half a cell.
                if wbest > wmetal && wbest > zero(CType)
                    give = min(massn - ρn, ρn)
                    birth = give * wbest / (wbest + wmetal)
                    # Half a cell opens the face. A few ulps under that, from
                    # ρ drifting off 1, is still half a cell.
                    if birth + CType(1.0e-4) * ρn >= CType(0.5) * ρn
                        nx, ny, nz, ok = outward_normal(ϕ, x, y, z, Nx, Ny, Nz, CType)
                        ϕn = ϕ[n]
                        tol = wbest * CType(1.0e-4)
                        for i in 2:Q
                            cx, cy, cz = c[i][1], c[i][2], c[i][3]
                            w = aperture_weight(ϕ, flags, ϕn, nx, ny, nz, ok, false, x, y, z, cx, cy, cz, Nx, Ny, Nz, CType)
                            w + tol < wbest && continue
                            j = src_index(x, y, z, cx, cy, cz, Nx, Ny, Nz)
                            fj = flags[j]
                            if (fj & (TYPE_SU | TYPE_S)) == TYPE_G
                                flags[j] = (fj & ~TYPE_SU) | TYPE_GI
                            end
                        end
                    end
                end
            end
        end
    end
end

# Liquid neighbors only. A cell born on top of solid metal must not copy fs = 1.
@inline function average_molten_neighbors(
    Tfield, fs, flags, x::Int, y::Int, z::Int,
    c::NTuple{Q, SVector{3, Int}},
    Nx::Int, Ny::Int, Nz::Int, ::Type{CType}
) where {Q, CType}
    sT = zero(CType)
    sfs = zero(CType)
    cnt = 0
    for i in 2:Q
        src = src_index(x, y, z, c[i][1], c[i][2], c[i][3], Nx, Ny, Nz)
        su = flags[src] & (TYPE_SU | TYPE_S)
        if (su == TYPE_F || su == TYPE_I || su == TYPE_IF) && !is_solid_fraction(fs[src])
            sT += Tfield[src]
            sfs += fs[src]
            cnt += 1
        end
    end
    return sT, sfs, cnt
end

@inline function surface_2_body!(
    t_odd::Val{odd},
    fi, ρ, u, flags, gi, T, fs,
    w::NTuple{Q, CType}, c::NTuple{Q, SVector{3, Int}},
    Ts::CType,
    N::Int, Nx::Int, Ny::Int, Nz::Int, n, g_store_odd::Bool
) where {odd, Q, CType}
    sus = flags[n] & (TYPE_SU | TYPE_S)
    n0 = n - 1
    x = n0 % Nx; y = (n0 ÷ Nx) % Ny; z = n0 ÷ (Nx * Ny)

    if sus == TYPE_GI
        ρn, ux, uy, uz = average_neighbors_non_gas(ρ, u, flags, x, y, z, c, Nx, Ny, Nz, CType)
        ux, uy, uz = cap_interface_velocity(ux, uy, uz)
        # store(P) is what a neighbor's load(!P) pulls. This cell's own next
        # load reads the other parity, and that read swaps + and −.
        store_feq!(fi, n, x, y, z, ρn, ux, uy, uz, w, c, N, Nx, Ny, Nz, t_odd)
        store_feq!(fi, n, x, y, z, ρn, ux, uy, uz, w, c, N, Nx, Ny, Nz, Val(!odd), Val(true))
        @static if TEMPERATURE
            sT, sfs, cnt = average_molten_neighbors(T, fs, flags, x, y, z, c, Nx, Ny, Nz, CType)
            if cnt > 0
                Tn = sT / CType(cnt)
                fs[n] = sfs / CType(cnt)
            else
                # No liquid neighbor: the new metal is the melt, at the solidus.
                # fs = 0 so the next thermal collide books Λ and does not freeze.
                Tn = Ts
                fs[n] = zero(CType)
            end
            T[n] = Tn
            store_geq!(gi, n, x, y, z, Tn, ux, uy, uz, N, Nx, Ny, Nz, g_store_odd, CType)
            if g_store_odd
                store_geq!(gi, n, x, y, z, Tn, ux, uy, uz, N, Nx, Ny, Nz, Val(false), CType, Val(true))
            else
                store_geq!(gi, n, x, y, z, Tn, ux, uy, uz, N, Nx, Ny, Nz, Val(true), CType, Val(true))
            end
        end
        return nothing
    elseif sus == TYPE_IG
        for i in 2:Q
            j = src_index(x, y, z, c[i][1], c[i][2], c[i][3], Nx, Ny, Nz)
            fj = flags[j]
            suj = fj & (TYPE_SU | TYPE_S)
            rest = fj & ~TYPE_SU
            if suj == TYPE_F || suj == TYPE_IF
                flags[j] = rest | TYPE_I
            end
        end
    end
    return nothing
end

@kernel function surface_2_even_kernel!(fi, @Const(ρ), @Const(u), flags, gi, T, fs, w::NTuple{Q,CType}, c, Ts::CType, N, Nx, Ny, Nz, g_store_odd::Bool) where {Q, CType}
    n = @index(Global)
    @inbounds surface_2_body!(Val(false), fi, ρ, u, flags, gi, T, fs, w, c, Ts, N, Nx, Ny, Nz, Int(n), g_store_odd)
end
@kernel function surface_2_odd_kernel!(fi, @Const(ρ), @Const(u), flags, gi, T, fs, w::NTuple{Q,CType}, c, Ts::CType, N, Nx, Ny, Nz, g_store_odd::Bool) where {Q, CType}
    n = @index(Global)
    @inbounds surface_2_body!(Val(true), fi, ρ, u, flags, gi, T, fs, w, c, Ts, N, Nx, Ny, Nz, Int(n), g_store_odd)
end

@kernel function surface_3_kernel!(
    ρ, flags, mass, massex, ϕ, fs,
    c::NTuple{Q, SVector{3, Int}},
    N::Int, Nx::Int, Ny::Int, Nz::Int
) where {Q}
    n = @index(Global)
    @inbounds begin
        flagsn = flags[n]
        sus = flagsn & (TYPE_SU | TYPE_S)
        if (sus & TYPE_S) == 0x00
            CType = eltype(ρ)

            frozen = false
            @static if TEMPERATURE
                frozen = is_solid_fraction(fs[n])
            end

            ρn = ρ[n]
            massn = mass[n]
            massexn = zero(CType)
            ϕn = zero(CType)
            over = massn > ρn

            n0 = n - 1
            x = n0 % Nx; y = (n0 ÷ Nx) % Ny; z = n0 ÷ (Nx * Ny)

            if sus == TYPE_F
                massn, massexn = place_surplus(massn, ρn, ϕ, flags, fs, x, y, z, c, Nx, Ny, Nz, CType)
                ϕn = one(CType)
            elseif sus == TYPE_I || sus == TYPE_IF
                gas_left = has_gas_neighbor(flags, x, y, z, c, Nx, Ny, Nz)
                # Liquid sheds m − ρ as it becomes bulk. A solid cell does not,
                # so it stays an interface until that volume has left.
                if gas_left || (over && frozen)
                    if sus == TYPE_IF
                        flags[n] = (flagsn & ~TYPE_SU) | TYPE_I
                    end
                    if over
                        massn, massexn = publish_surplus(massn, ρn, ϕ, flags, fs, x, y, z, c, Nx, Ny, Nz, frozen, CType)
                        ϕn = one(CType)
                    else
                        # A modest deficit is shared and the cell stays an
                        # interface. Deleting it here is what opens holes while
                        # surplus is still waiting to stack.
                        if massn < zero(CType)
                            massexn = massn
                            massn = zero(CType)
                            massn, massexn = park_if_unclaimed(massn, massexn, flags, fs, x, y, z, c, Nx, Ny, Nz)
                        else
                            massexn = zero(CType)
                        end
                        ϕn = calculate_phi(ρn, massn, TYPE_I)
                    end
                else
                    flags[n] = (flagsn & ~TYPE_SU) | TYPE_F
                    massn, massexn = place_surplus(massn, ρn, ϕ, flags, fs, x, y, z, c, Nx, Ny, Nz, CType)
                    ϕn = one(CType)
                end
            elseif sus == TYPE_G
                massexn = massn
                massn = zero(CType)
                ϕn = zero(CType)
                massn, massexn = park_if_unclaimed(massn, massexn, flags, fs, x, y, z, c, Nx, Ny, Nz)
            elseif sus == TYPE_IG
                flags[n] = (flagsn & ~TYPE_SU) | TYPE_G
                massexn = massn
                massn = zero(CType)
                ϕn = zero(CType)
                massn, massexn = park_if_unclaimed(massn, massexn, flags, fs, x, y, z, c, Nx, Ny, Nz)
            elseif sus == TYPE_GI
                flags[n] = (flagsn & ~TYPE_SU) | TYPE_I
                if over
                    massn, massexn = publish_surplus(massn, ρn, ϕ, flags, fs, x, y, z, c, Nx, Ny, Nz, frozen, CType)
                    ϕn = one(CType)
                else
                    massexn = massn < zero(CType) ? massn : zero(CType)
                    massn = clamp(massn, zero(CType), ρn)
                    massn, massexn = park_if_unclaimed(massn, massexn, flags, fs, x, y, z, c, Nx, Ny, Nz)
                    ϕn = calculate_phi(ρn, massn, TYPE_I)
                end
            end
            mass[n] = massn
            massex[n] = massexn
            ϕ[n] = ϕn
        end
    end
end

end # SURFACE