# Static cylinder, uniform in y, φ = 1 inside. Prints Δp and maximum(|u|).
# Set the Preferences key `interface` to `allen_cahn` and start a new Julia process.
# This script does not set that preference.

using LatticeBoltzmann
using KernelAbstractions

@assert ALLEN_CAHN

let
    Nx, Ny, Nz = 24, 8, 24
    σ = 1e-3
    R = 6.0
    model = Model(Nx, Ny, Nz, 0.05; backend=CPU(), workgroup=64,
                  σ=σ, W=4, Mphi=0.05, rho_a=1, rho_b=1)
    domain = model.domains[1]
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
    for _ in 1:200
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
    pin /= nin
    pout /= nout
    Δp = pin - pout

    umax = 0.0
    for n in 1:size(u, 1)
        s = sqrt(Float64(u[n, 1])^2 + Float64(u[n, 2])^2 + Float64(u[n, 3])^2)
        umax = s > umax ? s : umax
    end

    println("Δp = ", Δp)
    println("maximum(|u|) = ", umax)
end
