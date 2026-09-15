# Blowing-agent decomposition (TiH2-style Arrhenius) into dissolved gas.
# Uniform agent in a liquid box; a hot core releases H2-equivalent `c`.
# Nucleation is on, so bubbles appear where the agent burns off.
#
# ParaView: output_blowing_agent/lbm.pvd. Colour by a (agent) or c, Contour
# phi = 0.5. Hot core is T.
#
#   SURFACE = true, TEMPERATURE = true
using LatticeBoltzmann
using CUDA
using ProgressMeter
using Logging

@assert SURFACE && TEMPERATURE
start_run_log!("output_blowing_agent")

Nx = Ny = Nz = 36
k_H = 3.0f0
α_c = 0.2f0
a0 = 0.08f0
k_a = 0.04f0                 # 1/step at T=1 if E_a=0; here E_a>0 so only the hot core goes
E_a = 3.0f0                  # lattice; k(T) = k_a exp(-E_a/T)
T_cold = 0.7f0
T_hot = 1.4f0
R_hot = 8.0f0
nsteps = 300
every = 20

backend = CUDA.functional() ? CUDABackend() : CPU()
nuc = Nucleation{Float32}(; d_min=7, R=1, c_star=1.15f0, p_cell=1,
                          n_max=4, n_over=1.25f0, every=20, n_total_max=30)
model = Model(Nx, Ny, Nz, 0.1f0; α=0.2f0, α_c=α_c, k_H=k_H, fz=0, σ=0,
              k_a=k_a, E_a=E_a, Y_a=1, a_fs_max=1, T_avg=T_cold,
              backend=backend, workgroup=backend isa CUDABackend ? 256 : 64,
              bubbles=BubbleTracker{Float32}(; nucleation=nuc))

xc = yc = zc = Float32(Nx + 1) / 2
host = zeros(UInt8, Nx * Ny * Nz)
Th = fill(T_cold, Nx * Ny * Nz)
ah = zeros(Float32, Nx * Ny * Nz)
ch = fill(k_H * P_ATM_LAT, Nx * Ny * Nz)
for z in 1:Nz, y in 1:Ny, x in 1:Nx
    n = x + (y - 1) * Nx + (z - 1) * Nx * Ny
    if x == 1 || x == Nx || y == 1 || y == Ny || z == 1 || z == Nz
        host[n] = TYPE_S
        Th[n] = T_cold
    else
        host[n] = TYPE_F
        ah[n] = a0
        dx, dy, dz = Float32(x) - xc, Float32(y) - yc, Float32(z) - zc
        r2 = dx * dx + dy * dy + dz * dz
        Th[n] = T_cold + (T_hot - T_cold) * exp(-r2 / (2 * R_hot * R_hot))
    end
end
copyto!(model.domains[1].flags.data, host)
copyto!(model.domains[1].T.data, Th)
copyto!(model.domains[1].a.data, ah)
copyto!(model.domains[1].c.data, ch)

LatticeBoltzmann.initialize!(model)
export!(model; dir="output_blowing_agent")
inv0 = agent_inventory(model.domains[1])
@info "blowing agent" Nx k_a E_a a0 T_hot T_cold inv0 nsteps backend

function run_agent!(model, nsteps, every)
    prog = Progress(cld(nsteps, every); dt=0.3, desc="blowing agent ")
    t = 0
    while t < nsteps
        nrun = min(every, nsteps - t)
        with_logger(NullLogger()) do
            run!(model, nrun)
        end
        t += nrun
        export!(model; dir="output_blowing_agent")
        inv = agent_inventory(model.domains[1])
        recs = bubble_records(model)
        next!(prog; showvalues = [
            (:t, t),
            (:a, round(inv.a; digits=3)),
            (:res, round(inv.res; digits=3)),
            (:c_n, round(inv.dissolved; digits=3)),
            (:nb, length(recs)),
        ])
    end
    finish!(prog)
    return agent_inventory(model.domains[1])
end

inv1 = run_agent!(model, nsteps, every)
@info "blowing agent done" inv0 inv1 n_planted=model.bubbles.nucleation.n_planted
