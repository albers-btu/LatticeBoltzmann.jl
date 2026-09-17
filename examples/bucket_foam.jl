# Physical bucket foam. SI box, SI σ, SI g. Δt from capillary_s so CSF stays
# stable and Bond number matches  ρ g L²/σ. Hot Dirichlet bottom (Arrhenius),
# then freeze by switching the walls to a cold bath. No laser.
#
# ParaView: output_bucket_foam/lbm.pvd. Colour by a, c, or T. Contour phi = 0.5
# after Threshold flags 8–32. fs shows the freeze front from the walls.
#
#   SURFACE = true, TEMPERATURE = true, VOLUME_FORCE = true
using LatticeBoltzmann
using CUDA
using Unitful
using ProgressMeter
using Logging

@assert SURFACE && TEMPERATURE && VOLUME_FORCE
start_run_log!("output_bucket_foam")

si_Lx = 16.0e-3u"m"
si_H  = 12.0e-3u"m"
si_gas = 4.0e-3u"m"
si_dx = 0.25e-3u"m"

si_ρ  = 8000u"kg/m^3"
si_cp = 500.0u"J/kg/K"
si_k  = 30.0u"W/m/K"
si_Tm = 1673.0u"K"
si_T_hot  = 1.40 * si_Tm
si_T_bulk = 1.02 * si_Tm
si_T_cold = 0.70 * si_Tm
si_Lheat = 2.7e5u"J/kg"
si_σ = 1.5u"N/m"
si_g = 9.81u"m/s^2"
si_k_a = 2.0e4u"s^-1"
si_E_a = 1.34e4u"K"
si_t_foam = 0.08u"s"
si_t_cool = 0.10u"s"

dx = ustrip(u"m", si_dx)
L = max(8, round(Int, ustrip(u"m", si_H) / dx))
m = ustrip(u"m", si_H) / L
Nx = max(16, round(Int, ustrip(u"m", si_Lx) / m))
Ny = Nx
Hfill = L + 2
n_gas_top = max(6, ceil(Int, ustrip(u"m", si_gas) / m))
Nz = Hfill + n_gas_top + 1

α_si = si_k / (si_ρ * si_cp)
si_ν = ustrip(u"m^2/s", α_si) * u"m^2/s"
units = Units(si_H, si_ρ, si_σ; x=L, σ_lat=0.03, u=0.05, ρ=1, T=Float32,
              K=ustrip(u"K", si_Tm), cp=si_cp)
s = units.s
n_foam = max(40, round(Int, ustrip(u"s", si_t_foam) / s))
n_cool = max(40, round(Int, ustrip(u"s", si_t_cool) / s))
every = max(20, n_foam ÷ 20)

T_hot  = Float32(lbm_T(units, si_T_hot))
T_bulk = Float32(lbm_T(units, si_T_bulk))
T_cold = Float32(lbm_T(units, si_T_cold))
T_m    = Float32(lbm_T(units, si_Tm))
k_H = 3.0f0
a0 = 0.08f0

backend = CUDA.functional() ? CUDABackend() : CPU()
nuc = Nucleation{Float32}(; d_min=10, R=1, c_star=1.10f0, p_cell=1,
                          n_max=3, n_over=1.12f0, every=40, n_total_max=16)
model = Model(Nx, Ny, Nz, units;
              ν=si_ν, α=2 * α_si, α_c=α_si, k_H=k_H,
              gz=-si_g, σ=si_σ,
              k_a=si_k_a, E_a=si_E_a, Y_a=1, a_fs_max=0.5,
              T_avg=T_bulk, latent=si_Lheat, Ts=si_Tm, Tl=si_Tm,
              K0=1.0e-10u"m^2",
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
d = model.domains[1]
@info "bucket foam" Nx Ny Nz Hfill m_um=(1e6*m) s_us=(1e6*s) σ_lat=d.σ fz=d.fz Bo=(abs(d.fz) * Hfill^2 / max(d.σ, 1.0f-12)) n_foam n_cool Ncell=(Nx*Ny*Nz) m0 backend

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
