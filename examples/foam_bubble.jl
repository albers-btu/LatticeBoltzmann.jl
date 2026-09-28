# One bubble in a closed box. Requires FOAM = true.
# Prints R(t) next to sqrt(2 Δc V_m D t + R0^2). No pass/fail band.
# `full = true` runs the paper's 100³ box.

using LatticeBoltzmann

full = false
N = full ? 100 : 48
nsteps = full ? 400 : 200
R0 = 3.0
k_H = 0.001
D = 0.03
V_m = 3.0
ν = 0.25

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

analytic(t, Δc) = sqrt(2 * Δc * V_m * D * t + R0^2)

for Δc in (0.1, 0.3, 0.5)
    model = Model(N, N, N, ν; σ=0, n_hydro=1)
    domain = model.domains[1]
    copyto!(domain.flags.data, closed_box(N))
    nucleate_bubbles!(model, [(N / 2, N / 2, N / 2)], [R0])
    set_foam!(model; D=D, k_H=k_H, k_Π=0, q=0, V_m=V_m, γ_b=1,
              c0=k_H / 3 + Δc, ρ_liquid=1)
    initialize!(model)
    id = only(bubble_ids(model))
    println("Δc=$Δc N=$N")
    for t in 0:nsteps
        if t > 0
            LatticeBoltzmann.step!(model)
        end
        if t == 0 || t == nsteps || t % 50 == 0
            R = bubble_radius(model, id)
            println("t=$t R=$R analytic=$(analytic(t, Δc))")
        end
    end
end
