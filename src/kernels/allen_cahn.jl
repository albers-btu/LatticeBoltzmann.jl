# Factors 3 and 6 are 1/cs² and 2/cs² with cs² = 1/3.

@inline function grad_phi(
    phi, x::Int, y::Int, z::Int,
    w::NTuple{Q, CType}, c::NTuple{Q, SVector{3, Int}},
    Nx::Int, Ny::Int, Nz::Int,
) where {Q, CType}
    gx = zero(CType)
    gy = zero(CType)
    gz = zero(CType)
    for i in 2:Q
        ci = c[i]
        src = src_index(x, y, z, ci[1], ci[2], ci[3], Nx, Ny, Nz)
        p = CType(phi[src])
        wi = w[i]
        gx += wi * CType(ci[1]) * p
        gy += wi * CType(ci[2]) * p
        gz += wi * CType(ci[3]) * p
    end
    three = CType(3)
    return three * gx, three * gy, three * gz
end

@inline function laplacian_phi(
    phi, x::Int, y::Int, z::Int,
    w::NTuple{Q, CType}, c::NTuple{Q, SVector{3, Int}},
    Nx::Int, Ny::Int, Nz::Int,
) where {Q, CType}
    n = src_index(x, y, z, 0, 0, 0, Nx, Ny, Nz)
    pn = CType(phi[n])
    acc = zero(CType)
    for i in 2:Q
        ci = c[i]
        src = src_index(x, y, z, ci[1], ci[2], ci[3], Nx, Ny, Nz)
        acc += w[i] * (CType(phi[src]) - pn)
    end
    return CType(6) * acc
end

@inline function mu_phi(phi::CType, lap::CType, σ::CType, W::CType) where {CType}
    return (CType(3) / CType(2)) * σ * (
        (CType(16) / W) * phi * (one(CType) - phi) * (one(CType) - CType(2) * phi) - W * lap
    )
end

# Even/odd EsotericPull of hi. One outer step, same parity as gi.
# h_eq uses the streamed sum. ∇φ is read from the phi snapshot, then phi is overwritten.
@static if ALLEN_CAHN

@inline function ac_collide!(
    hi, phi, phi_w, u, flags,
    w::NTuple{Q, CType},
    c::NTuple{Q, SVector{3, Int}},
    Mphi::CType, W::CType,
    N::Int, Nx::Int, Ny::Int, Nz::Int,
    n::Int, t_odd::Val{ODD},
) where {Q, CType, ODD}
    flagsn = flags[n]
    if (flagsn & TYPE_BO) == TYPE_S
        return nothing
    end
    n0 = n - 1
    x = n0 % Nx
    y = (n0 ÷ Nx) % Ny
    z = n0 ÷ (Nx * Ny)
    ux = CType(u[n, 1])
    uy = CType(u[n, 2])
    uz = CType(u[n, 3])
    @static if DIM == 2
        uz = zero(CType)
    end
    gx, gy, gz = grad_phi(phi, x, y, z, w, c, Nx, Ny, Nz)
    h0s = CType(hi[f_index(n, 1, N)])
    sum_pairs = zero(CType)
    np = (Q - 1) ÷ 2
    for k in 1:np
        i = 2k
        cp = c[i]
        cm = c[i + 1]
        src_p = src_index(x, y, z, cp[1], cp[2], cp[3], Nx, Ny, Nz)
        src_m = src_index(x, y, z, cm[1], cm[2], cm[3], Nx, Ny, Nz)
        fp, fm = load_bb_pair(hi, flags, n, src_p, src_m, i, t_odd, N, CType)
        sum_pairs += fp + fm
    end
    φ = h0s + sum_pairs
    h0 = w[1] * φ
    hi[f_index(n, 1, N)] = eltype(hi)(h0)
    acc = h0
    ωφ = one(CType) / (CType(3) * Mphi + CType(0.5))
    gnorm = sqrt(gx * gx + gy * gy + gz * gz)
    # BGK already diffuses. The parenthesis is zero on the target, so it cannot cancel that.
    # Shipped term is compression ∝ ωφ Mphi (4φ(1-φ)/W), with an empirical (8/7).
    sharpen = gnorm < CType(1e-8) ? zero(CType) :
        (CType(8) / CType(7)) * ωφ * Mphi * (CType(4) * φ * (one(CType) - φ) / (W * gnorm))
    for k in 1:np
        i = 2k
        cp = c[i]
        cm = c[i + 1]
        src_p = src_index(x, y, z, cp[1], cp[2], cp[3], Nx, Ny, Nz)
        src_m = src_index(x, y, z, cm[1], cm[2], cm[3], Nx, Ny, Nz)
        fp, fm = load_bb_pair(hi, flags, n, src_p, src_m, i, t_odd, N, CType)
        cup = CType(cp[1]) * ux + CType(cp[2]) * uy + CType(cp[3]) * uz
        cum = CType(cm[1]) * ux + CType(cm[2]) * uy + CType(cm[3]) * uz
        heqp = w[i] * φ * (one(CType) + CType(3) * cup)
        heqm = w[i + 1] * φ * (one(CType) + CType(3) * cum)
        hp = fp + ωφ * (heqp - fp)
        hm = fm + ωφ * (heqm - fm)
        hp += CType(3) * w[i] * sharpen * (CType(cp[1]) * gx + CType(cp[2]) * gy + CType(cp[3]) * gz)
        hm += CType(3) * w[i + 1] * sharpen * (CType(cm[1]) * gx + CType(cm[2]) * gy + CType(cm[3]) * gz)
        store_pair!(hi, n, src_p, i, hp, hm, t_odd, N)
        acc += hp + hm
    end
    phi_w[n] = eltype(phi_w)(acc)
    return nothing
end

@kernel function allen_cahn_odd_kernel!(
    hi, phi, phi_w, u, flags,
    w::NTuple{Q, CType},
    c::NTuple{Q, SVector{3, Int}},
    Mphi::CType, W::CType,
    N::Int, Nx::Int, Ny::Int, Nz::Int,
) where {Q, CType}
    n = @index(Global)
    @inbounds ac_collide!(hi, phi, phi_w, u, flags, w, c, Mphi, W, N, Nx, Ny, Nz, Int(n), Val(true))
end

@kernel function allen_cahn_even_kernel!(
    hi, phi, phi_w, u, flags,
    w::NTuple{Q, CType},
    c::NTuple{Q, SVector{3, Int}},
    Mphi::CType, W::CType,
    N::Int, Nx::Int, Ny::Int, Nz::Int,
) where {Q, CType}
    n = @index(Global)
    @inbounds ac_collide!(hi, phi, phi_w, u, flags, w, c, Mphi, W, N, Nx, Ny, Nz, Int(n), Val(false))
end

# F = μ ∇φ + ρ(φ) (fx, fy, fz). Hydro fx, fy, fz stay zero so gravity is not applied twice.
@inline function capillary_force!(
    F, phi,
    w::NTuple{Q, CType}, c::NTuple{Q, SVector{3, Int}},
    σ::CType, W::CType,
    fx::CType, fy::CType, fz::CType,
    rho_a::CType, rho_b::CType,
    Nx::Int, Ny::Int, Nz::Int, n::Int,
) where {Q, CType}
    n0 = n - 1
    x = n0 % Nx
    y = (n0 ÷ Nx) % Ny
    z = n0 ÷ (Nx * Ny)
    gx, gy, gz = grad_phi(phi, x, y, z, w, c, Nx, Ny, Nz)
    lap = laplacian_phi(phi, x, y, z, w, c, Nx, Ny, Nz)
    φn = CType(phi[n])
    μ = mu_phi(φn, lap, σ, W)
    ρφ = rho_a + φn * (rho_b - rho_a)
    F[n, 1] = eltype(F)(μ * gx + ρφ * fx)
    F[n, 2] = eltype(F)(μ * gy + ρφ * fy)
    F[n, 3] = eltype(F)(μ * gz + ρφ * fz)
    return nothing
end

@kernel function capillary_force_kernel!(
    F, phi,
    w::NTuple{Q, CType},
    c::NTuple{Q, SVector{3, Int}},
    σ::CType, W::CType,
    fx::CType, fy::CType, fz::CType,
    rho_a::CType, rho_b::CType,
    Nx::Int, Ny::Int, Nz::Int,
) where {Q, CType}
    n = @index(Global)
    @inbounds capillary_force!(F, phi, w, c, σ, W, fx, fy, fz, rho_a, rho_b, Nx, Ny, Nz, Int(n))
end

end # ALLEN_CAHN
