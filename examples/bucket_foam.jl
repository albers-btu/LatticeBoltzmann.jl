# Physical bucket foam (LBfoam bucket analog at higher resolution).
# Gravity, surface tension, hot Dirichlet bottom (Arrhenius), then freeze by
# switching the crucible walls to a cold Dirichlet bath. No laser, no fs stamp.
#
# ParaView: output_bucket_foam/lbm.pvd. Colour by a, c, or T. Contour phi = 0.5
# after Threshold flags 8–32. fs shows the freeze front from the walls.
#
#   SURFACE = true, TEMPERATURE = true
using LatticeBoltzmann
using CUDA
using ProgressMeter
using Logging

@assert SURFACE && TEMPERATURE && VOLUME_FORCE
start_run_log!("output_bucket_foam")

Nx = Ny = 64
Nz = 80
Hfill = 48
k_H = 3.0f0
α_c = 0.2f0
a0 = 0.08f0
k_a = 1.2f0
E_a = 8.0f0                  # release is concentrated at the hot bottom
T_m = 1.0f0
T_hot = 1.40f0               # bottom plate during foam
T_bulk = 1.02f0              # initial melt, just above T_m
T_cold = 0.70f0              # wall bath during freeze
σ = 0.008f0
fz = -1.5f-5                 # downward
n_foam = 500
n_cool = 600
every = 25

backend = CUDA.functional() ? CUDABackend() : CPU()
nuc = Nucleation{Float32}(; d_min=10, R=1, c_star=1.10f0, p_cell=1,
                          n_max=8, n_over=1.35f0, every=25, n_total_max=80)
model = Model(Nx, Ny, Nz, 0.1f0; α=0.2f0, α_c=α_c, k_H=k_H, fz=fz, σ=σ,
              k_a=k_a, E_a=E_a, Y_a=1, a_fs_max=0.5f0, T_avg=T_bulk,
              Λ=0.6f0, Ts=T_m, Tl=T_m, K0=1.0f-3,
              backend=backend, workgroup=backend isa CUDABackend ? 256 : 64,
              bubbles=BubbleTracker{Float32}(; nucleation=nuc, k_Π=0.08f0, d_max=4))

function paint_bucket!(host, Th, ah, ch, Nx, Ny, Nz, Hfill;
                       T_bot, T_bulk, a0, k_H)
    patm_c = k_H * P_ATM_LAT
    for z in 1:Nz, y in 1:Ny, x in 1:Nx
        n = x + (y - 1) * Nx + (z - 1) * Nx * Ny
        ch[n] = patm_c
        if z == 1
            host[n] = TYPE_S | TYPE_T
            Th[n] = T_bot
        elseif x == 1 || x == Nx || y == 1 || y == Ny || z == Nz
            host[n] = TYPE_S
            Th[n] = T_bulk
        elseif z <= Hfill
            host[n] = TYPE_F
            Th[n] = T_bulk
            ah[n] = a0
        else
            host[n] = TYPE_G
            Th[n] = T_bulk
            ah[n] = 0
        end
    end
    return nothing
end

# Cold oven: bottom + side walls Dirichlet. Lid stays adiabatic.
function quench_crucible!(model, T_cold)
    d = model.domains[1]
    Nx, Ny, Nz = Int(d.Nx), Int(d.Ny), Int(d.Nz)
    host = Array(d.flags.data)
    Th = Array(d.T.data)
    for z in 1:Nz, y in 1:Ny, x in 1:Nx
        n = x + (y - 1) * Nx + (z - 1) * Nx * Ny
        if z == 1 || x == 1 || x == Nx || y == 1 || y == Ny
            host[n] = TYPE_S | TYPE_T
            Th[n] = T_cold
        end
    end
    copyto!(d.flags.data, host)
    copyto!(d.T.data, Th)
    fill!(d.Q.data, 0)
    return nothing
end

host = zeros(UInt8, Nx * Ny * Nz)
Th = fill(T_bulk, Nx * Ny * Nz)
ah = zeros(Float32, Nx * Ny * Nz)
ch = fill(k_H * P_ATM_LAT, Nx * Ny * Nz)
paint_bucket!(host, Th, ah, ch, Nx, Ny, Nz, Hfill;
              T_bot=T_hot, T_bulk=T_bulk, a0=a0, k_H=k_H)
copyto!(model.domains[1].flags.data, host)
copyto!(model.domains[1].T.data, Th)
copyto!(model.domains[1].a.data, ah)
copyto!(model.domains[1].c.data, ch)

LatticeBoltzmann.initialize!(model)
export!(model; dir="output_bucket_foam")
m0 = foam_metrics(model)
@info "bucket foam" Nx Ny Nz Hfill a0 k_a E_a T_hot T_bulk T_cold σ fz n_foam n_cool Ncell=(Nx*Ny*Nz) m0 backend

function run_bucket!(model, nsteps, every; tag="foam")
    prog = Progress(cld(nsteps, every); dt=0.3, desc="$tag ")
    t = 0
    local m = foam_metrics(model)
    while t < nsteps
        nrun = min(every, nsteps - t)
        with_logger(NullLogger()) do
            run!(model, nrun)
        end
        t += nrun
        export!(model; dir="output_bucket_foam")
        m = foam_metrics(model)
        next!(prog; showvalues = [
            (:t, t),
            (:nb, m.nb),
            (:porosity, round(m.porosity; digits=3)),
            (:fill_z, round(m.fill_z; digits=2)),
            (:a, round(m.a; digits=3)),
            (:frozen, m.n_frozen),
        ])
    end
    finish!(prog)
    return m
end

m1 = run_bucket!(model, n_foam, every; tag="foam")
@info "foam phase done" m1

quench_crucible!(model, T_cold)
m2 = run_bucket!(model, n_cool, every; tag="freeze")
@info "bucket foam done" m0 m1 m2
