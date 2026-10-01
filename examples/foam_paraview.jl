# One bubble growing in a closed box, written for ParaView.
# Same dimensionless case as examples/foam_bubble.jl at Δc = 0.5, not a metal.
# The cooling aluminum foam is examples/foam_poisson.jl.
# Requires FOAM = true (and SURFACE). n_hydro stays 1.
#
# Open output_foam/lbm.pvd.
# Colour the volume by c (dissolved gas). It starts at c0 in the liquid
# and 0 in the bubble; a halo forms as the bubble grows.
# tag is the bubble id (0 in the liquid, 1 on the bubble).
# Contour phi = 0.5 for the interface.
# ParaView 6: T is flat. Type an isosurface value; do not drag the slider
# on a constant array (rho can crash it the same way).
#
# `nsteps ÷ 40` is the frame count over the run. Double 40 for twice the files.

using LatticeBoltzmann
using Printf

@assert FOAM && SURFACE

full = false
N = full ? 100 : 48
nsteps = full ? 400 : 200
R0 = 3.0
k_H = 0.001
D = 0.03
V_m = 3.0
ν = 0.25
Δc = 0.5
dir = "output_foam"
# c and tag are not in the default export list.
fields = (:phi, :c, :tag, :rho, :u, :flags, :T)

function closed_box(N)
    flags = fill(TYPE_F, N * N * N)
    for z in 0:(N - 1), y in 0:(N - 1), x in 0:(N - 1)
        if x == 0 || y == 0 || z == 0 || x == N - 1 || y == N - 1 || z == N - 1
            flags[1 + x + N * y + N * N * z] = TYPE_S
        end
    end
    return flags
end

function bubble_radius(model, id)
    return (3 * bubble_volume(model, id) / (4π))^(1 / 3)
end

analytic(t) = sqrt(2 * Δc * V_m * D * t + R0^2)

model = Model(N, N, N, ν; σ=0, n_hydro=1)
domain = model.domains[1]
copyto!(domain.flags.data, closed_box(N))
nucleate_bubbles!(model, [(N / 2, N / 2, N / 2)], [R0])
set_foam!(model; D=D, k_H=k_H, k_Π=0, q=0, V_m=V_m, γ_b=1,
          c0=k_H / 3 + Δc, ρ_liquid=1)
initialize!(model)
id = only(bubble_ids(model))

every = max(1, nsteps ÷ 40)
println("Δc=$Δc N=$N  frames every $every steps  →  $dir/lbm.pvd")

function save_frame!(model, id, t)
    export!(model; dir, fields, sync=true)
    R = bubble_radius(model, id)
    @printf("t=%d  R=%.4f  analytic=%.4f\n", t, R, analytic(t))
    return nothing
end

save_frame!(model, id, 0)
for t in 1:nsteps
    LatticeBoltzmann.step!(model)
    if t % every == 0 || t == nsteps
        save_frame!(model, id, t)
    end
end
println("wrote $dir/lbm.pvd")
