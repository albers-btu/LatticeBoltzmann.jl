# Gaussian laser with PLIC hits, Fresnel absorption, and multiple reflections.
# Deposits absorbed power into Q (lattice dT/step). No third-party optic package.
#
# PLIC n = calculate_normal_py points metal → gas. Front-face hit: −dir · n > 0.
# Optional `skin` spreads each hit along −n into metal (numerical absorption depth).

mutable struct Laser{T<:AbstractFloat}
    enabled::Bool
    P::T                 # incident power [W]
    w::T                 # 1/e² radius [lattice cells]
    x::T                 # beam axis, 1-based cell coords
    y::T
    z::T                 # launch plane z
    dx::T
    dy::T
    dz::T
    n_re::T              # Re(ñ) of metal (316L ~ 1.07 µm)
    n_im::T
    nrays::Int           # nrays × nrays bundle
    max_bounce::Int
    every::Int
    skin::Int            # cells into metal to spread a hit; 1 → interface only
    ox::Vector{T}        # ray offsets in x
    oy::Vector{T}
    Pray::Vector{T}      # incident power per ray [W]
end

# Unpolarized Fresnel absorptance, vacuum → metal ñ = n + i k.
# Real arithmetic so the same code runs on CPU and CUDA.
@inline function fresnel_absorptance(cosθ::T, n_re::T, n_im::T) where {T}
    c = clamp(cosθ, zero(T), one(T))
    s2 = max(zero(T), one(T) - c * c)
    a = n_re * n_re - n_im * n_im - s2
    b = T(2) * n_re * n_im
    r = sqrt(a * a + b * b)
    ur = sqrt(max(zero(T), T(0.5) * (r + a)))
    ui = copysign(sqrt(max(zero(T), T(0.5) * (r - a))), b)
    Rs = ((c - ur) * (c - ur) + ui * ui) / ((c + ur) * (c + ur) + ui * ui)
    n2r = n_re * n_re - n_im * n_im
    n2i = T(2) * n_re * n_im
    ar = n2r * c - ur
    ai = n2i * c - ui
    br = n2r * c + ur
    bi = n2i * c + ui
    Rp = (ar * ar + ai * ai) / (br * br + bi * bi)
    R = T(0.5) * (Rs + Rp)
    return one(T) - clamp(R, zero(T), one(T))
end

function _build_ray_bundle(P::T, w::T, nrays::Int) where {T}
    nrays < 1 && return T[], T[], T[]
    half = T(2) * w
    Δ = (T(2) * half) / T(nrays)
    ox = T[]
    oy = T[]
    wt = T[]
    I0 = T(2) * P / (T(π) * w * w)
    @inbounds for j in 1:nrays, i in 1:nrays
        x = -half + (T(i) - T(0.5)) * Δ
        y = -half + (T(j) - T(0.5)) * Δ
        r2 = x * x + y * y
        W = I0 * exp(-T(2) * r2 / (w * w)) * Δ * Δ
        if W > T(1e-8) * P
            push!(ox, x)
            push!(oy, y)
            push!(wt, W)
        end
    end
    s = sum(wt)
    if s > 0
        @inbounds for i in eachindex(wt)
            wt[i] *= P / s
        end
    end
    return ox, oy, wt
end

function Laser{T}(;
    P = zero(T),
    w = one(T),
    x = one(T),
    y = one(T),
    z = one(T),
    dir = (zero(T), zero(T), -one(T)),
    n_re = T(3.27),
    n_im = T(4.48),
    nrays = 17,
    max_bounce = 8,
    every = 1,
    skin = 1,
    enabled = true,
) where {T<:AbstractFloat}
    d = SVector{3,T}(T(dir[1]), T(dir[2]), T(dir[3]))
    nrm = sqrt(d[1]*d[1] + d[2]*d[2] + d[3]*d[3])
    nrm <= 0 && (d = SVector{3,T}(zero(T), zero(T), -one(T)); nrm = one(T))
    d = d / nrm
    ox, oy, Pray = _build_ray_bundle(T(P), T(w), Int(nrays))
    return Laser{T}(enabled, T(P), T(w), T(x), T(y), T(z), d[1], d[2], d[3],
                    T(n_re), T(n_im), Int(nrays), Int(max_bounce), Int(every),
                    max(1, Int(skin)), ox, oy, Pray)
end

function Laser(U::Units{T};
    P, w,
    x = one(T), y = one(T), z = one(T),
    dir = (0, 0, -1),
    n_re = 3.27, n_im = 4.48,
    nrays = 17, max_bounce = 8, every = 1, skin = 1,
    enabled = true,
) where {T}
    Pw = P isa Quantity ? T(ustrip(u"W", P)) : T(P)
    wl = w isa Quantity ? T(ustrip(u"m", w) / U.m) : T(w)
    return Laser{T}(; P=Pw, w=wl, x=T(x), y=T(y), z=T(z), dir=dir,
                    n_re=T(n_re), n_im=T(n_im), nrays=nrays,
                    max_bounce=max_bounce, every=every, skin=skin, enabled=enabled)
end

function set_laser_position!(L::Laser{T}, x, y, z=L.z) where {T}
    L.x = T(x)
    L.y = T(y)
    L.z = T(z)
    return L
end

@inline function _add_q!(Q, n::Int, dq)
    @inbounds Q[n] += dq
    return nothing
end

# Parker–Youngs n points metal → gas. Plane: n · (r − c) = plic_cube(ϕ, n).
@inline function plic_hit(ϕ0::T, phij, ox::T, oy::T, oz::T, dirx::T, diry::T, dirz::T,
                          cx::T, cy::T, cz::T) where {T}
    nϕ = calculate_normal_py(phij)
    n2 = nϕ[1]*nϕ[1] + nϕ[2]*nϕ[2] + nϕ[3]*nϕ[3]
    n2 <= eps(T) && return false, zero(T), nϕ
    dpl = plic_cube(ϕ0, nϕ)
    nd = nϕ[1]*dirx + nϕ[2]*diry + nϕ[3]*dirz
    abs(nd) <= T(1e-8) && return false, zero(T), nϕ
    t = (nϕ[1]*(cx - ox) + nϕ[2]*(cy - oy) + nϕ[3]*(cz - oz) + dpl) / nd
    t <= T(1e-6) && return false, t, nϕ
    hx = ox + t * dirx
    hy = oy + t * diry
    hz = oz + t * dirz
    inside = abs(hx - cx) <= T(0.5) + T(1e-4) &&
             abs(hy - cy) <= T(0.5) + T(1e-4) &&
             abs(hz - cz) <= T(0.5) + T(1e-4)
    return inside, t, nϕ
end

@inline function _deposit_along_normal!(
    Q, flags, Pabs_q::T, skin::Int,
    ix::Int, iy::Int, iz::Int,
    nx::T, ny::T, nz::T,
    Nx::Int, Ny::Int, Nz::Int,
) where {T}
    nskin = max(1, skin)
    dq = Pabs_q / T(nskin)
    px = T(ix)
    py = T(iy)
    pz = T(iz)
    inx = -nx
    iny = -ny
    inz = -nz
    @inbounds for _k in 1:nskin
        jx = floor(Int, px + T(0.5))
        jy = floor(Int, py + T(0.5))
        jz = floor(Int, pz + T(0.5))
        (jx < 1 || jx > Nx || jy < 1 || jy > Ny || jz < 1 || jz > Nz) && return nothing
        nn = jx + (jy - 1) * Nx + (jz - 1) * Nx * Ny
        fl = flags[nn]
        su = fl & TYPE_SU
        if su == TYPE_I || su == TYPE_F
            _add_q!(Q, nn, dq)
        else
            return nothing
        end
        px += inx
        py += iny
        pz += inz
    end
    return nothing
end

@inline function _walk_laser_ray!(
    Q, flags, ϕ,
    o0x::T, o0y::T, o0z::T, dx::T, dy::T, dz::T, Pray::T,
    n_re::T, n_im::T, max_bounce::Int, skin::Int, qfac::T,
    Nx::Int, Ny::Int, Nz::Int,
) where {T}
    ox, oy, oz = o0x, o0y, o0z
    dirx, diry, dirz = dx, dy, dz
    Pleft = Pray
    bounces = 0
    max_step = Nx + Ny + Nz + 16
    epsn = T(1e-4)
    @inbounds for _ in 1:max_step
        Pleft < T(1e-8) * Pray && break
        ix = floor(Int, ox + T(0.5))
        iy = floor(Int, oy + T(0.5))
        iz = floor(Int, oz + T(0.5))
        (ix < 1 || ix > Nx || iy < 1 || iy > Ny || iz < 1 || iz > Nz) && break
        n = ix + (iy - 1) * Nx + (iz - 1) * Nx * Ny
        fl = flags[n]
        su = fl & TYPE_SU
        if (fl & TYPE_BO) == TYPE_S
            break
        elseif su == TYPE_I
            ϕ0 = T(ϕ[n])
            phij = gather_phi_d3q27(ϕ, ϕ0, ix - 1, iy - 1, iz - 1, Nx, Ny, Nz)
            hit, t, nϕ = plic_hit(ϕ0, phij, ox, oy, oz, dirx, diry, dirz,
                                  T(ix), T(iy), T(iz))
            noutx, nouty, noutz = nϕ[1], nϕ[2], nϕ[3]
            cθ = -(dirx * noutx + diry * nouty + dirz * noutz)
            if hit && cθ > zero(T)
                A = clamp(fresnel_absorptance(cθ, n_re, n_im), zero(T), one(T))
                Pabs = A * Pleft
                _deposit_along_normal!(Q, flags, Pabs * qfac, skin,
                                       ix, iy, iz, noutx, nouty, noutz,
                                       Nx, Ny, Nz)
                Pleft -= Pabs
                bounces += 1
                (bounces >= max_bounce || Pleft < T(1e-8) * Pray) && break
                hx = ox + t * dirx
                hy = oy + t * diry
                hz = oz + t * dirz
                dn = T(2) * (dirx * noutx + diry * nouty + dirz * noutz)
                dirx -= dn * noutx
                diry -= dn * nouty
                dirz -= dn * noutz
                invd = T(1) / max(sqrt(dirx*dirx + diry*diry + dirz*dirz), epsn)
                dirx *= invd; diry *= invd; dirz *= invd
                ox = hx + epsn * dirx
                oy = hy + epsn * diry
                oz = hz + epsn * dirz
                continue
            end
        elseif su == TYPE_F
            _add_q!(Q, n, Pleft * qfac)
            break
        end
        tMaxX = dirx > 0 ? (T(ix) + T(0.5) - ox) / dirx :
                dirx < 0 ? (ox - (T(ix) - T(0.5))) / (-dirx) : T(Inf)
        tMaxY = diry > 0 ? (T(iy) + T(0.5) - oy) / diry :
                diry < 0 ? (oy - (T(iy) - T(0.5))) / (-diry) : T(Inf)
        tMaxZ = dirz > 0 ? (T(iz) + T(0.5) - oz) / dirz :
                dirz < 0 ? (oz - (T(iz) - T(0.5))) / (-dirz) : T(Inf)
        tstep = min(max(tMaxX, zero(T)), max(tMaxY, zero(T)), max(tMaxZ, zero(T))) + T(1e-5)
        ox += tstep * dirx
        oy += tstep * diry
        oz += tstep * dirz
    end
    return nothing
end

function laser_qfac(U::Units{T}, ρlat=one(T)) where {T}
    return T(U.s / (si_ρ(U, ρlat) * U.cp * U.K * U.m^3))
end

function deposit_laser!(model, domain)
    L = model.laser
    (L === nothing || !L.enabled || L.P <= 0) && return nothing
    t = Int(domain.t)
    L.every > 1 && (t % L.every != 0) && t != 0 && return nothing
    Q = domain.Q.data
    fill!(Q, zero(eltype(Q)))
    nray = length(L.Pray)
    nray == 0 && return nothing
    qfac = laser_qfac(model.units)
    Nx, Ny, Nz = Int(domain.Nx), Int(domain.Ny), Int(domain.Nz)
    flags = Array(domain.flags.data)
    ϕ = Array(domain.ϕ.data)
    Qh = zeros(eltype(Q), length(Q))
    @inbounds for rid in 1:nray
        _walk_laser_ray!(
            Qh, flags, ϕ,
            L.x + L.ox[rid], L.y + L.oy[rid], L.z,
            L.dx, L.dy, L.dz, L.Pray[rid],
            L.n_re, L.n_im, L.max_bounce, L.skin, qfac,
            Nx, Ny, Nz,
        )
    end
    copyto!(Q, Qh)
    return nothing
end
