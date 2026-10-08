# A few rays along +x, all inside the sphere's shadow, onto a metal sphere.
# Open both in ParaView:
#   output_laser_rays/lbm.pvd    sphere (color by phi or flags)
#   output_laser_rays/rays.pvd   one polyline per ray, plus the 1/e² cylinder (color by power)
#
# Rays that hit the sphere reflect. Rays that miss run on to the far wall.
# SURFACE and TEMPERATURE must be on (src/extensions.jl).

using LatticeBoltzmann

@assert SURFACE && TEMPERATURE

Nx = Ny = Nz = 48
R = 10.0
cx = (Nx + 1) / 2
cy = (Ny + 1) / 2
cz = (Nz + 1) / 2

# Grid spans ±2w, so corners sit at 2w√2. w = 3 keeps that inside R = 10.
# 3×3 rays, all aimed at the sphere.
laser = Laser{Float32}(;
    P = 100.0f0,
    w = 3.0f0,
    x = 2.0f0,
    y = cy,
    z = cz,
    dir = (1.0f0, 0.0f0, 0.0f0),
    nrays = 3,
    max_bounce = 3,
    skin = 1,
    enabled = true,
)

model = Model(Nx, Ny, Nz, 0.1f0; α=0.2f0, σ=0.0f0, fz=0.0f0,
              laser=laser, workgroup=64)

host = zeros(UInt8, Nx * Ny * Nz)
R2 = R * R
for z in 1:Nz, y in 1:Ny, x in 1:Nx
    n = x + (y - 1) * Nx + (z - 1) * Nx * Ny
    wall = x == 1 || x == Nx || y == 1 || y == Ny || z == 1 || z == Nz
    dx = x - cx
    dy = y - cy
    dz = z - cz
    host[n] = wall ? TYPE_S : (dx * dx + dy * dy + dz * dz <= R2 ? TYPE_F : TYPE_G)
end
copyto!(model.domains[1].flags.data, host)
initialize!(model)

dir = "output_laser_rays"
export!(model; dir=dir, fields=(:phi, :flags, :Q))
println("Wrote $dir/lbm.pvd and $dir/rays.pvd ($(length(laser.Pray)) rays)")
println("ParaView: open both, color the sphere by phi and the rays by power.")
