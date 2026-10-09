using Test
using LatticeBoltzmann
using KernelAbstractions

ac_index(i, Nx, Ny) = DIM == 2 ? (i + 1) : (i * Nx * Ny + 1)

function ac_crossing(phi, Nx, Ny, Nz)
    naxis = DIM == 2 ? Nx : Nz
    center = naxis / 2
    best = center
    bestd = Inf
    for i in 0:naxis-2
        p0 = Float64(phi[ac_index(i, Nx, Ny)])
        p1 = Float64(phi[ac_index(i + 1, Nx, Ny)])
        dp = p1 - p0
        dp == 0 && continue
        if (p0 - 0.5) * (p1 - 0.5) <= 0
            pos = i + (0.5 - p0) / dp
            d = abs(pos - center)
            if d < bestd
                bestd = d
                best = pos
            end
        end
    end
    return best
end

function ac_midpoint(phi, Nx, Ny, Nz)
    naxis = DIM == 2 ? Nx : Nz
    center = naxis / 2
    best_n = 1
    best_c = 0
    best_φ = 0.0
    best_d = Inf
    for i in 0:naxis-1
        n = ac_index(i, Nx, Ny)
        φ = Float64(phi[n])
        d = abs(Float64(i) - center) + 2 * abs(φ - 0.5)
        if d < best_d
            best_d = d
            best_n = n
            best_c = i
            best_φ = φ
        end
    end
    if DIM == 2
        return best_n, best_c, 0, 0, best_φ
    else
        return best_n, 0, 0, best_c, best_φ
    end
end

# Stationary interface: a thick tanh sharpens to |∇φ| = 4φ(1-φ)/W and does not drift.
@testset "Allen–Cahn stationary tanh" begin
    backend = CPU()
    @static if DIM == 2
        Nx, Ny, Nz = 32, 1, 1
    else
        Nx, Ny, Nz = 1, 1, 32
    end
    N = Nx * Ny * Nz
    wg = min(32, N)
    model = Model(Nx, Ny, Nz, 0.05; backend=backend, workgroup=wg,
                  σ=0.01, W=4, Mphi=0.05, rho_a=1, rho_b=1)
    domain = model.domains[1]
    W = Float64(domain.W)
    phi = domain.phi.data
    for z in 0:Nz-1, y in 0:Ny-1, x in 0:Nx-1
        s = DIM == 2 ? (x - Nx / 2) : (z - Nz / 2)
        n = x + y * Nx + z * Nx * Ny + 1
        phi[n] = (1 + tanh(s / W)) / 2
    end
    x0 = ac_crossing(phi, Nx, Ny, Nz)

    initialize!(model)
    for _ in 1:400
        LatticeBoltzmann.step!(model)
    end

    phi1 = domain.phi.data
    x1 = ac_crossing(phi1, Nx, Ny, Nz)
    @test abs(x1 - x0) <= 0.5

    _, x, y, z, φ = ac_midpoint(phi1, Nx, Ny, Nz)
    gx, gy, gz = LatticeBoltzmann.grad_phi(phi1, x, y, z, model.weights, model.velocities, Nx, Ny, Nz)
    mag = sqrt(Float64(gx) * Float64(gx) + Float64(gy) * Float64(gy) + Float64(gz) * Float64(gz))
    target = 4 * φ * (1 - φ) / W
    @test abs(mag - target) <= 0.05 * abs(target)
    @test all(isfinite, domain.hi.data)
end
