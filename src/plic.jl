using StaticArrays

const D3Q27_C = ntuple(i -> VELOCITIES[:D3Q27][i], 27)

@inline _sq(x) = x * x
@inline _cb(x) = x * x * x

@inline function _dot(a::SVector{3,T}, b::SVector{3,T}) where {T}
    return a[1]*b[1] + a[2]*b[2] + a[3]*b[3]
end

@inline function _cross(a::SVector{3,T}, b::SVector{3,T}) where {T}
    return SVector{3,T}(
        a[2]*b[3] - a[3]*b[2],
        a[3]*b[1] - a[1]*b[3],
        a[1]*b[2] - a[2]*b[1],
    )
end

@inline function _normalize(v::SVector{3,T}) where {T}
    n2 = _dot(v, v)
    n2 <= zero(T) && return SVector{3,T}(zero(T), zero(T), zero(T))
    s = sqrt(n2)
    return SVector{3,T}(v[1] / s, v[2] / s, v[3] / s)
end

@inline function calculate_normal_py(phij::NTuple{27,T}) where {T}
    nx = T(4)*(phij[3]-phij[2]) +
         T(2)*(phij[9]-phij[8] + phij[11]-phij[10] + phij[15]-phij[14] + phij[17]-phij[16]) +
         (phij[21]-phij[20] + phij[23]-phij[22] + phij[25]-phij[24] + phij[26]-phij[27])
    ny = T(4)*(phij[5]-phij[4]) +
         T(2)*(phij[9]-phij[8] + phij[13]-phij[12] + phij[14]-phij[15] + phij[19]-phij[18]) +
         (phij[21]-phij[20] + phij[23]-phij[22] + phij[24]-phij[25] + phij[27]-phij[26])
    nz = T(4)*(phij[7]-phij[6]) +
         T(2)*(phij[11]-phij[10] + phij[13]-phij[12] + phij[16]-phij[17] + phij[18]-phij[19]) +
         (phij[21]-phij[20] + phij[22]-phij[23] + phij[25]-phij[24] + phij[27]-phij[26])
    return _normalize(SVector{3,T}(nx, ny, nz))
end

@inline function plic_cube_reduced(V::T, n1::T, n2::T, n3::T) where {T}
    n12 = n1 + n2
    n3V = n3 * V
    n12 <= T(2) * n3V && return n3V + T(0.5) * n12
    sqn1 = _sq(n1)
    n26 = T(6) * n2
    v1 = sqn1 / n26
    if v1 <= n3V && n3V < v1 + T(0.5) * (n2 - n1)
        return T(0.5) * (n1 + sqrt(sqn1 + T(8) * n2 * (n3V - v1)))
    end
    V6 = n1 * n26 * n3V
    n3V < v1 && return cbrt(V6)
    v3 = n3 < n12 ?
        (_sq(n3)*(T(3)*n12 - n3) + sqn1*(n1 - T(3)*n3) + _sq(n2)*(n2 - T(3)*n3)) / (n1 * n26) :
        T(0.5) * n12
    sqn12 = sqn1 + _sq(n2)
    V6cbn12 = V6 - _cb(n1) - _cb(n2)
    case34 = n3V < v3
    a = case34 ? V6cbn12 : T(0.5) * (V6cbn12 - _cb(n3))
    b = case34 ? sqn12 : T(0.5) * (sqn12 + _sq(n3))
    c = case34 ? n12 : T(0.5)
    t2 = _sq(c) - b
    t2 <= zero(T) && return c
    t = sqrt(t2)
    t <= zero(T) && return c
    arg = (_cb(c) - T(0.5)*a - T(1.5)*b*c) / _cb(t)
    arg = clamp(arg, -one(T), one(T))
    return c - T(2) * t * sin(T(1)/T(3) * asin(arg))
end

@inline function plic_cube(V0::T, n::SVector{3,T}) where {T}
    ax, ay, az = abs(n[1]), abs(n[2]), abs(n[3])
    l = ax + ay + az
    l <= zero(T) && return zero(T)
    V = T(0.5) - abs(V0 - T(0.5))
    n1 = min(ax, ay, az) / l
    n3 = max(ax, ay, az) / l
    n2 = max(one(T) - n1 - n3, zero(T))
    d = plic_cube_reduced(V, n1, n2, n3)
    return l * copysign(T(0.5) - d, V0 - T(0.5))
end

@inline function lu_solve5!(M::MVector{25,T}, x::MVector{5,T}, b::MVector{5,T}, Nsol::Int) where {T}
    N = 5
    @inbounds for i in 1:Nsol
        diag = M[N*(i - 1) + i]
        abs(diag) <= eps(T) && return false
        for j in (i + 1):Nsol
            M[N*(j - 1) + i] /= diag
            mij = M[N*(j - 1) + i]
            for k in (i + 1):Nsol
                M[N*(j - 1) + k] -= mij * M[N*(i - 1) + k]
            end
        end
    end
    @inbounds for i in 1:Nsol
        xi = b[i]
        for k in 1:(i - 1)
            xi -= M[N*(i - 1) + k] * x[k]
        end
        x[i] = xi
    end
    @inbounds for i in Nsol:-1:1
        xi = x[i]
        for k in (i + 1):Nsol
            xi -= M[N*(i - 1) + k] * x[k]
        end
        diag = M[N*(i - 1) + i]
        abs(diag) <= eps(T) && return false
        x[i] = xi / diag
    end
    return true
end

@inline function gather_phi_d3q27(
    ϕ, ϕ0::T, x::Int, y::Int, z::Int, Nx::Int, Ny::Int, Nz::Int
) where {T}
    return ntuple(Val(27)) do i
        if i == 1
            ϕ0
        else
            ci = D3Q27_C[i]
            T(ϕ[src_index(x, y, z, ci[1], ci[2], ci[3], Nx, Ny, Nz)])
        end
    end
end

@inline function calculate_curvature(phij::NTuple{27,T}) where {T}
    bz = calculate_normal_py(phij)
    _dot(bz, bz) <= zero(T) && return zero(T)

    rn = SVector{3,T}(T(0.56270900), T(0.32704452), T(0.75921047))
    by = _normalize(_cross(bz, rn))
    _dot(by, by) <= zero(T) && return zero(T)
    bx = _cross(by, bz)

    center_offset = plic_cube(phij[1], bz)

    M = zero(MVector{25,T})
    b = zero(MVector{5,T})
    xsol = zero(MVector{5,T})
    number = 0

    @inbounds for i in 2:27
        ϕi = phij[i]
        if ϕi > zero(T) && ϕi < one(T)
            ei = SVector{3,T}(T(D3Q27_C[i][1]), T(D3Q27_C[i][2]), T(D3Q27_C[i][3]))
            offset = plic_cube(ϕi, bz) - center_offset
            px = _dot(ei, bx)
            py = _dot(ei, by)
            pz = _dot(ei, bz) + offset
            x2 = px * px
            y2 = py * py
            x3 = x2 * px
            y3 = y2 * py
            M[1]  += x2 * x2
            M[2]  += x2 * y2
            M[3]  += x3 * py
            M[4]  += x3
            M[5]  += x2 * py
            b[1]  += x2 * pz
            M[7]  += y2 * y2
            M[8]  += px * y3
            M[9]  += px * y2
            M[10] += y3
            b[2]  += y2 * pz
            M[13] += x2 * y2
            M[14] += x2 * py
            M[15] += px * y2
            b[3]  += px * py * pz
            M[19] += x2
            M[20] += px * py
            b[4]  += px * pz
            M[25] += y2
            b[5]  += py * pz
            number += 1
        end
    end

    M[6]  = M[2]
    M[11] = M[3]
    M[12] = M[8]
    M[16] = M[4]
    M[17] = M[9]
    M[18] = M[14]
    M[21] = M[5]
    M[22] = M[10]
    M[23] = M[15]
    M[24] = M[20]

    Nsol = number >= 5 ? 5 : number
    Nsol <= 0 && return zero(T)
    ok = lu_solve5!(M, xsol, b, Nsol)
    !ok && return zero(T)

    A, B, C, H, I = xsol[1], xsol[2], xsol[3], xsol[4], xsol[5]
    den = H*H + I*I + one(T)
    den <= zero(T) && return zero(T)
    K = (A*(I*I + one(T)) + B*(H*H + one(T)) - C*H*I) * _cb(T(1) / sqrt(den))
    !isfinite(K) && return zero(T)
    return clamp(K, -one(T), one(T))
end

@inline function gas_density_plic(σ::T, ϕ, ϕ0::T, x, y, z, Nx, Ny, Nz) where {T}
    σ == zero(T) && return one(T)
    phij = gather_phi_d3q27(ϕ, ϕ0, x, y, z, Nx, Ny, Nz)
    κ = calculate_curvature(phij)
    return one(T) - T(6) * σ * κ
end
