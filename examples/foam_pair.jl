# Two bubbles growing toward each other in a closed box.
# Same dimensionless liquid as examples/foam_bubble.jl (Δc = 0.5), not a metal.
# Requires FOAM = true.
# n_hydro stays 1. σ = 0, so the only force in the film is disjoining.
#
# The surfaces start 5 cells apart, outside the film range. Growth brings
# the liquid gap under 4 cells. Π = k_Π * (4 − d) then lowers the gas
# density on the near side and holds the film, so the gas regions do not
# touch face to face and the two ids do not merge on contact.
# k_Π = 0 lets that film rupture and the flood fill joins them.
#
# Open output_foam_pair/lbm.pvd.
# Colour by tag: 1 and 2 are the two bubbles, 0 is liquid.
# Contour phi = 0.5 for the two surfaces and the film between them.
# ParaView 6: T is flat. Type an isosurface value; do not drag the slider
# on a constant array.
#
# `nsteps ÷ 40` is the frame count over the run.

using LatticeBoltzmann
using Printf

@assert FOAM && SURFACE

N = 48
nsteps = 240
R0 = 4.0
gap0 = 5.0                              # liquid between the surfaces at t = 0
sep = 2 * R0 + gap0
k_H = 0.001
D = 0.03
V_m = 3.0
ν = 0.25
Δc = 0.5
k_Π = 0.02                              # disjoining strength; 0 merges on contact
dir = "output_foam_pair"
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

# Cells of liquid on the mid-line between the two gas cores.
function film_cells(domain)
    Nx = Int(domain.Nx)
    Ny = Int(domain.Ny)
    Nz = Int(domain.Nz)
    y = Ny ÷ 2
    z = Nz ÷ 2
    flags = Array(domain.flags.data)
    seen_gas = false
    film = 0
    best = 0
    for x in 0:(Nx - 1)
        n = 1 + x + y * Nx + z * Nx * Ny
        if (flags[n] & TYPE_SU) == TYPE_G
            # film == 0 is the interior of the bubble we are already in.
            if seen_gas && film > 0
                best = film
            end
            seen_gas = true
            film = 0
        elseif seen_gas
            film += 1
        end
    end
    return best
end

centers = [
    (N / 2 - sep / 2, N / 2, N / 2),
    (N / 2 + sep / 2, N / 2, N / 2),
]

model = Model(N, N, N, ν; σ=0, n_hydro=1)
domain = model.domains[1]
copyto!(domain.flags.data, closed_box(N))
nucleate_bubbles!(model, centers, [R0, R0])
set_foam!(model; D=D, k_H=k_H, k_Π=k_Π, q=0, V_m=V_m, γ_b=1,
          c0=k_H / 3 + Δc, ρ_liquid=1)
initialize!(model)

every = max(1, nsteps ÷ 40)
println("sep=$sep  gap0=$gap0  k_Π=$k_Π  →  $dir/lbm.pvd")

function save_frame!(model, domain, t)
    export!(model; dir, fields, sync=true)
    ids = bubble_ids(model)
    Rs = join((@sprintf("%.3f", bubble_radius(model, id)) for id in ids), ", ")
    @printf("t=%d  n=%d  ids=%s  R=[%s]  film=%d\n",
            t, length(ids), ids, Rs, film_cells(domain))
    return nothing
end

save_frame!(model, domain, 0)
for t in 1:nsteps
    LatticeBoltzmann.step!(model)
    if t % every == 0 || t == nsteps
        save_frame!(model, domain, t)
    end
end
println("wrote $dir/lbm.pvd")
