# Static cylinder, uniform in y, φ = 1 inside. Writes a ParaView time series.
# Set the Preferences key `interface` to `allen_cahn` and start a new Julia process.
# This script does not set that preference.
#
#   output_allen_cahn/lbm.pvd
#
# Open that file, color by phi, and contour at 0.5. p is the hydrodynamic
# pressure cs² Σ fᵢ. rho in the file is Σ fᵢ, which is 1 away from the jump
# when both densities are 1.

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

    dir = "output_allen_cahn"
    fields = (:phi, :u, :p, :rho)
    export!(model; dir=dir, fields=fields, sync=true)
    for _ in 1:5
        for _ in 1:40
            LatticeBoltzmann.step!(model)
        end
        export!(model; dir=dir, fields=fields, sync=true)
    end
    println("Wrote $dir/lbm.pvd (t = 0, 40, …, 200)")
    println("ParaView: color by phi and contour at 0.5.")
end
