using KernelAbstractions

@static if FOAM

# Absolute dissolved-gas equilibrium: c = Σ c_i.
# Not geq_T, which subtracts w_i so Σ g = T − 1.
# w0 = 1/4, w_axis = 1/8, c_sT² = 1/4. D3Q19 model.weights are not used.
@inline ceq_rest(c::CType) where {CType} = CType(0.25) * c

@inline ceq_axis(c::CType, ucomp::CType) where {CType} =
    CType(0.125) * c * (one(CType) + ucomp * CType(4))

# D = (1/4) (1/ω − 1/2). Not omega_T_from_alpha (that takes 2α) and not clamp_omega.
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
    # First step! is even, so the slots match store_feq!(..., Val(false)).
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
    t_odd::Val{odd}, ci, cfield, flags, u,
    ωc::CType, q::CType,
    N::Int, Nx::Int, Ny::Int, Nz::Int, n,
) where {odd, CType}
    flagsn = flags[n]
    if (flagsn & TYPE_BO) == TYPE_S || (flagsn & TYPE_SU) == TYPE_G
        return nothing
    end
    su = flagsn & TYPE_SU
    # TYPE_GI is not a concentration state: surface_1 writes it after this kernel.
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

    crest = CType(ci[f_index(n, 1, N)])
    cn = crest
    for k in 1:3
        i = 2k
        src = _axis_src(k, x, y, z, Nx, Ny, Nz)
        fp, fm = load_pair(ci, n, src, i, t_odd, N, CType)
        cn += fp + fm
    end
    cfield[n] = cn

    # w0 q + 6 w_axis q = q, and only on pure liquid. TYPE_IF carries both F and I.
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
    ci, cfield, @Const(flags), @Const(u),
    ωc::CType, q::CType,
    N::Int, Nx::Int, Ny::Int, Nz::Int,
) where {CType}
    n = @index(Global)
    @inbounds concentration_body!(Val(false), ci, cfield, flags, u, ωc, q, N, Nx, Ny, Nz, Int(n))
end

@kernel function concentration_odd_kernel!(
    ci, cfield, @Const(flags), @Const(u),
    ωc::CType, q::CType,
    N::Int, Nx::Int, Ny::Int, Nz::Int,
) where {CType}
    n = @index(Global)
    @inbounds concentration_body!(Val(true), ci, cfield, flags, u, ωc, q, N, Nx, Ny, Nz, Int(n))
end

@inline function disjoining_body!(::Int)
    return nothing
end

@kernel function disjoining_kernel!()
    n = @index(Global)
    @inbounds disjoining_body!(Int(n))
end

@inline function ϕ_correction_body!(::Int)
    return nothing
end

@kernel function ϕ_correction_kernel!()
    n = @index(Global)
    @inbounds ϕ_correction_body!(Int(n))
end

end
