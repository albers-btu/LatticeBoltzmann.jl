using Test
using LatticeBoltzmann
using KernelAbstractions

# Static cylinder: Δp = σ/R, not 2σ/R. p = Σ f_i / 3.
# 24×8×24 is 3D. DIM=2 rejects Nz != 1, so this testset is not compiled there.
@static if DIM == 3
@testset "Allen–Cahn static cylinder" begin
    backend = CPU()
    σ = 1e-3
    R = 6.0
    Nx, Ny, Nz = 24, 8, 24
    model = Model(Nx, Ny, Nz, 0.05; backend=backend, workgroup=64,
                  σ=σ, W=4, Mphi=0.05, rho_a=1, rho_b=1)
    @test model.n_hydro == 1
    domain = model.domains[1]
    @test domain.fx == 0 && domain.fy == 0 && domain.fz == 0
    @test domain.β == 0
    W = Float64(domain.W)
    cx = (Nx - 1) / 2
    cz = (Nz - 1) / 2
    phi = domain.phi.data
    for z in 0:Nz-1, y in 0:Ny-1, x in 0:Nx-1
        r = sqrt((x - cx)^2 + (z - cz)^2)
        n = x + y * Nx + z * Nx * Ny + 1
        phi[n] = (1 + tanh(2 * (R - r) / W)) / 2
    end
    initialize!(model)
    for _ in 1:800
        LatticeBoltzmann.step!(model)
    end
    moments!(model)

    ρ = Array(domain.ρ.data)
    u = Array(domain.u.data)
    pin = 0.0
    nin = 0
    pout = 0.0
    nout = 0
    for z in 0:Nz-1, y in 0:Ny-1, x in 0:Nx-1
        r = sqrt((x - cx)^2 + (z - cz)^2)
        n = x + y * Nx + z * Nx * Ny + 1
        p = Float64(ρ[n]) / 3
        if r <= 1
            pin += p
            nin += 1
        elseif r >= 11
            pout += p
            nout += 1
        end
    end
    @test nin > 0 && nout > 0
    pin /= nin
    pout /= nout
    Δp = pin - pout
    target = σ / R
    @test abs(Δp - target) <= 0.2 * abs(target)

    umax = 0.0
    for n in 1:size(u, 1)
        s = sqrt(Float64(u[n, 1])^2 + Float64(u[n, 2])^2 + Float64(u[n, 3])^2)
        umax = s > umax ? s : umax
    end
    @test umax < 5e-3
    @test all(isfinite, domain.fi.data)
    @test all(isfinite, domain.hi.data)
end
end # DIM == 3
