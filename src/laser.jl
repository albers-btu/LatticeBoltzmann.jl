# Gaussian laser with PLIC hits, Fresnel absorption and multiple reflections.
mutable struct Laser{T<:AbstractFloat}
    enabled::Bool
    P::T                 # Incident power in W
    w::T                 # 1/e² radius in cells (unit)
    x::T                 # X position of ray bundle in cell coords
    y::T                 # Y position of ray bundle in cell coords
    z::T                 # Z position of ray bundle in cell coords
    dx::T                # X direction of beam in cell coords
    dy::T                # Y direction of beam in cell coords
    dz::T                # Z direction of beam in cell coords
    n_re::T              # Real      part of complex refrective index (316L ~ 1.07 µm)
    n_im::T              # Imaginary part of complex refrective index
    nrays::Int           # nrays × nrays bundle
    max_bounce::Int      # Maximum number of ray bounces
    every::Int           # Deposit laser every n steps 
    skin::Int            # Cells into metal to spread a hit; 1 leads to interface only
    ox::Vector{T}        # Ray offsets in x
    oy::Vector{T}        # Ray offsets in y
    Pray::Vector{T}      # Incident power per ray in W
end

# Unpolarized Fresnel absorptance, vacuum to metal ñ = n + i k.
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

# Returns the ray bundle with given incident power per ray.
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

# Default Laser
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

# SI Wrapper of default Laser
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

# Add heat into cell n
@inline function _add_q!(Q, n::Int, dq)
    dq == 0 && return nothing
    @inbounds Q[n] += dq
    return nothing
end

# Parker–Youngs n points metal to gas.
# Returns a boolean for inside the this voxels cube, distance of origin
# along the ray in cells, and the normal of the cube.
@inline function plic_hit(
    ϕ0::T, phij,
    ox::T, oy::T, oz::T,
    dirx::T, diry::T, dirz::T,
    cx::T, cy::T, cz::T
) where {T}
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

# Spread the heat energy into the cells along ray with a depth of skin cells.
@inline function _deposit_along_normal!(
    Q, flags, Pabs_q::T, skin::Int,
    ix::Int, iy::Int, iz::Int,
    nx::T, ny::T, nz::T,
    Nx::Int, Ny::Int, Nz::Int,
) where {T}
    Pabs_q == 0 && return nothing
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

# Start with a single ray at origin o, direction d, and power Pray. Walk
# Ray with DDA until power is gone, wall is hit, or out of boundary.
@inline function _walk_laser_ray!(
    Q, flags, ϕ,
    o0x::T, o0y::T, o0z::T, dx::T, dy::T, dz::T, Pray::T,
    n_re::T, n_im::T, max_bounce::Int, skin::Int, qfac::T,
    Nx::Int, Ny::Int, Nz::Int,
    path=nothing,
) where {T}
    ox, oy, oz = o0x, o0y, o0z
    dirx, diry, dirz = dx, dy, dz
    Pleft = Pray
    bounces = 0
    max_step = Nx + Ny + Nz + 16
    epsn = T(1e-4)
    _raypoint!(path, ox, oy, oz, Pleft)
    @inbounds for _ in 1:max_step
        Pleft < T(1e-8) * Pray && break
        ix = floor(Int, ox + T(0.5))
        iy = floor(Int, oy + T(0.5))
        iz = floor(Int, oz + T(0.5))
        if ix < 1 || ix > Nx || iy < 1 || iy > Ny || iz < 1 || iz > Nz
            _raypoint!(path, ox, oy, oz, Pleft)
            break
        end
        n = ix + (iy - 1) * Nx + (iz - 1) * Nx * Ny
        fl = flags[n]
        su = fl & TYPE_SU
        if (fl & TYPE_BO) == TYPE_S
            _raypoint!(path, ox, oy, oz, Pleft)
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
                hx = ox + t * dirx
                hy = oy + t * diry
                hz = oz + t * dirz
                _raypoint!(path, hx, hy, hz, Pleft)
                (bounces >= max_bounce || Pleft < T(1e-8) * Pray) && break
                dn = T(2) * (dirx * noutx + diry * nouty + dirz * noutz)
                dirx -= dn * noutx
                diry -= dn * nouty
                dirz -= dn * noutz
                invd = T(1) / max(sqrt(dirx*dirx + diry*diry + dirz*dirz), epsn)
                dirx *= invd; diry *= invd; dirz *= invd
                ox = hx + epsn * dirx
                oy = hy + epsn * diry
                oz = hz + epsn * dirz
                _raypoint!(path, ox, oy, oz, Pleft)
                continue
            end
        elseif su == TYPE_F
            _add_q!(Q, n, Pleft * qfac)
            _raypoint!(path, ox, oy, oz, zero(T))
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
        _raypoint!(path, ox, oy, oz, Pleft)
    end
    return nothing
end

_raypoint!(::Nothing, args...) = nothing
function _raypoint!(path::Vector, x, y, z, p)
    if !isempty(path)
        lx, ly, lz = path[end][1], path[end][2], path[end][3]
        (x - lx)^2 + (y - ly)^2 + (z - lz)^2 < 1e-10 && return nothing
    end
    push!(path, (Float64(x), Float64(y), Float64(z), Float64(p)))
    return nothing
end

# Bundle offsets (ox, oy) live in the plane normal to the beam.
# A downward beam keeps ox along x and oy along y.
function _ray_origin(L, rid)
    T = eltype(L.x)
    dx, dy, dz = L.dx, L.dy, L.dz
    ox, oy = L.ox[rid], L.oy[rid]
    if abs(dz) >= abs(dx) && abs(dz) >= abs(dy)
        return L.x + ox, L.y + oy, L.z
    end
    ax, ay, az = abs(dx), abs(dy), abs(dz)
    hx, hy, hz = ax <= ay && ax <= az ? (one(T), zero(T), zero(T)) :
                 ay <= az ? (zero(T), one(T), zero(T)) : (zero(T), zero(T), one(T))
    e1x = dy * hz - dz * hy
    e1y = dz * hx - dx * hz
    e1z = dx * hy - dy * hx
    n1 = sqrt(e1x * e1x + e1y * e1y + e1z * e1z)
    n1 = max(n1, T(1e-12))
    e1x /= n1; e1y /= n1; e1z /= n1
    e2x = dy * e1z - dz * e1y
    e2y = dz * e1x - dx * e1z
    e2z = dx * e1y - dy * e1x
    n2 = sqrt(e2x * e2x + e2y * e2y + e2z * e2z)
    n2 = max(n2, T(1e-12))
    e2x /= n2; e2y /= n2; e2z /= n2
    return L.x + ox * e1x + oy * e2x,
           L.y + ox * e1y + oy * e2y,
           L.z + ox * e1z + oy * e2z
end

# Retrace the current bundle on the host. Each entry is one ray of (x,y,z,P_left)
# in cell coordinates. Does not deposit heat.
function trace_laser_rays(model, domain)
    L = model.laser
    (L === nothing || !L.enabled || L.P <= 0 || isempty(L.Pray)) && return Vector{NTuple{4,Float64}}[]
    Nx, Ny, Nz = Int(domain.Nx), Int(domain.Ny), Int(domain.Nz)
    flags = Array(domain.flags.data)
    ϕ = Array(domain.ϕ.data)
    Q = zeros(Float32, 1)
    rays = Vector{NTuple{4,Float64}}[]
    T = eltype(L.x)
    @inbounds for rid in eachindex(L.Pray)
        path = NTuple{4,Float64}[]
        rx, ry, rz = _ray_origin(L, rid)
        _walk_laser_ray!(
            Q, flags, ϕ,
            rx, ry, rz,
            L.dx, L.dy, L.dz, L.Pray[rid],
            L.n_re, L.n_im, L.max_bounce, L.skin, zero(T),
            Nx, Ny, Nz, path,
        )
        length(path) >= 2 && push!(rays, path)
    end
    return rays
end

# Converts from input power (Watts) to lattice Q.
function laser_qfac(U::Units{T}, ρ_lattice=one(T)) where {T}
    return T(U.s / (si_ρ(U, ρ_lattice) * U.cp * U.K * U.m^3))
end

# Deposit laser energy for this domain. Is called from step function.
function deposit_laser!(model, domain)
    L = model.laser
    (L === nothing || !L.enabled || L.P <= 0) && return nothing
    t = Int(domain.t)
    L.every > 1 && (t % L.every != 0) && t != 0 && return nothing
    Q = domain.Q.data
    nray = length(L.Pray)
    if nray == 0
        fill!(Q, zero(eltype(Q)))
        return nothing
    end
    qfac = laser_qfac(model.units)
    Nx, Ny, Nz = Int(domain.Nx), Int(domain.Ny), Int(domain.Nz)
    flags = Array(domain.flags.data)
    ϕ = Array(domain.ϕ.data)
    Qh = zeros(eltype(Q), length(Q))
    @inbounds for rid in 1:nray
        rx, ry, rz = _ray_origin(L, rid)
        _walk_laser_ray!(
            Qh, flags, ϕ,
            rx, ry, rz,
            L.dx, L.dy, L.dz, L.Pray[rid],
            L.n_re, L.n_im, L.max_bounce, L.skin, qfac,
            Nx, Ny, Nz,
        )
    end
    copyto!(Q, Qh)
    return nothing
end
