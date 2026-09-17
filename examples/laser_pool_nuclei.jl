# Step 4 of laser–foam coupling: nucleate in a thick laser-melted pool.
# Same melt as laser_pool.jl (solid → ≥12 liquid cells on the axis), then
# Laplace-equilibrium nuclei (n_over=1) with the beam still on. No Arrhenius
# yet — c is already supersaturated. Foam must stay in the melt, pad holds.
#
# Tests: test/test_laser_nucleate.jl
# ParaView: output_laser_pool_nuclei/lbm.pvd. Colour fs (pool) and phi=0.5
# (bubbles). Threshold flags 8–32.
#
#   SURFACE = true, TEMPERATURE = true
using LatticeBoltzmann
using CUDA
using Unitful
using ProgressMeter
using Logging

@assert SURFACE && TEMPERATURE
start_run_log!("output_laser_pool_nuclei")

si_Lx = 3.0e-3u"m"
si_Ly = 3.0e-3u"m"
si_H  = 1.2e-3u"m"
si_gas = 0.8e-3u"m"
si_dx = 50.0e-6u"m"

si_ρ  = 8000u"kg/m^3"
si_cp = 500.0u"J/kg/K"
si_k_s = 15.0u"W/m/K"
si_k_l = 30.0u"W/m/K"
si_Tm = 1673.0u"K"
si_T_init = 0.88 * si_Tm
si_Lheat = 2.7e5u"J/kg"
si_Lv = 7.45e6u"J/kg"
si_Tv = 3086.0u"K"
si_M = 0.0558u"kg/mol"
si_σ = 1.5u"N/m"
si_P = 280.0u"W"
si_d_spot = 0.5e-3u"m"
si_δ = 0.60e-3u"m"
si_t_heat = 4.0e-3u"s"

dx = ustrip(u"m", si_dx)
L = max(4, round(Int, ustrip(u"m", si_H) / dx))
m = ustrip(u"m", si_H) / L
Nx = max(16, round(Int, ustrip(u"m", si_Lx) / m))
Ny = max(16, round(Int, ustrip(u"m", si_Ly) / m))
Hfill = L + 2
n_gas_top = max(6, ceil(Int, ustrip(u"m", si_gas) / m))
Nz = Hfill + n_gas_top + 1

α_s_si = si_k_s / (si_ρ * si_cp)
α_l_si = si_k_l / (si_ρ * si_cp)
si_ν = ustrip(u"m^2/s", α_l_si) * u"m^2/s"
units = Units(si_H, si_ρ, si_σ; x=L, σ_lat=0.03, u=0.05, ρ=1, T=Float32,
              K=ustrip(u"K", si_Tm), cp=si_cp)
s = units.s
n_heat = max(40, round(Int, ustrip(u"s", si_t_heat) / s))
every = max(10, n_heat ÷ 8)

T_init = Float32(lbm_T(units, si_T_init))
nskin = max(3, round(Int, ustrip(u"m", si_δ) / m))
k_H = 3.0f0
c∞ = 1.40f0
backend = CUDA.functional() ? CUDABackend() : CPU()

nuc = Nucleation{Float32}(; enabled=false, d_min=8, R=1, c_star=1.10f0,
                          p_cell=1, n_max=3, n_over=1.0f0, every=10,
                          n_total_max=12)
model = Model(Nx, Ny, Nz, units;
              ν=si_ν, α=2 * α_l_si, α_s=2 * α_s_si, α_l=2 * α_l_si, α_c=α_l_si,
              ν_s=0.5 * si_ν, ν_l=si_ν, k_H=k_H,
              gz=0, σ=si_σ,
              T_avg=T_init, latent=si_Lheat, Ts=si_Tm, Tl=si_Tm,
              latent_v=si_Lv, T_v=si_Tv, M=si_M,
              emissivity=0.4, T_rad=si_T_init,
              K0=1.0e-10u"m^2",
              backend=backend, workgroup=backend isa CUDABackend ? 256 : 64,
              bubbles=BubbleTracker{Float32}(; nucleation=nuc, k_Π=0.08f0, d_max=4))

xc = (Nx + 1) / 2
yc = (Ny + 1) / 2
model.laser = Laser(units; P=si_P, w=0.5 * si_d_spot,
                    x=xc, y=yc, z=Float32(Nz) - 1.1f0,
                    nrays=11, max_bounce=4, every=1, skin=nskin)

host = zeros(UInt8, Nx * Ny * Nz)
Th = fill(T_init, Nx * Ny * Nz)
fsh = ones(Float32, Nx * Ny * Nz)
ch = fill(c∞, Nx * Ny * Nz)
for z in 1:Nz, y in 1:Ny, x in 1:Nx
    n = x + (y - 1) * Nx + (z - 1) * Nx * Ny
    if x == 1 || x == Nx || y == 1 || y == Ny || z == 1 || z == Nz
        host[n] = TYPE_S | TYPE_T
        fsh[n] = 1
        Th[n] = T_init
        ch[n] = k_H * P_ATM_LAT
    elseif z <= Hfill
        host[n] = TYPE_F
        fsh[n] = 1
        Th[n] = T_init
    else
        host[n] = TYPE_G
        fsh[n] = 0
        ch[n] = k_H * P_ATM_LAT
    end
end
copyto!(model.domains[1].flags.data, host)
copyto!(model.domains[1].T.data, Th)
copyto!(model.domains[1].fs.data, fsh)
copyto!(model.domains[1].c.data, ch)
LatticeBoltzmann.initialize!(model)

ix = round(Int, xc)
iy = round(Int, yc)

function bubble_z_mean(model)
    B = model.bubbles
    B isa BubbleTracker || return 0.0
    Nx, Ny = Int(model.domains[1].Nx), Int(model.domains[1].Ny)
    s = 0.0
    n = 0
    @inbounds for i in eachindex(B.label)
        B.label[i] > 0 || continue
        s += (i - 1) ÷ (Nx * Ny) + 1
        n += 1
    end
    return n == 0 ? 0.0 : s / n
end

function pool_probe(model)
    d = model.domains[1]
    TA = Array(d.T.data)
    fl = Array(d.flags.data)
    fsA = Array(d.fs.data)
    uA = Array(d.u.data)
    Tmn, Tmx = Inf32, -Inf32
    nF = nI = nliq = 0
    umax = 0.0f0
    @inbounds for n in eachindex(TA)
        su = fl[n] & TYPE_SU
        su == TYPE_F && (nF += 1)
        su == TYPE_I && (nI += 1)
        (su == TYPE_F || su == TYPE_I) || continue
        Tmn = min(Tmn, TA[n])
        Tmx = max(Tmx, TA[n])
        fsA[n] < 0.5f0 && (nliq += 1)
        umax = max(umax, hypot(uA[n, 1], uA[n, 2], uA[n, 3]))
    end
    depth = liquid_column_depth(model, ix, iy)
    fm = foam_metrics(model)
    return Tmn, Tmx, nF, nI, nliq, umax, depth, fm.nb, fm.n_planted, bubble_z_mean(model)
end

export!(model; dir="output_laser_pool_nuclei")
Tmn0, Tmx0, nF0, nI0, nliq0, umax0, depth0, nb0, np0, z0 = pool_probe(model)
@info "laser pool nuclei" Nx Ny Nz Hfill n_heat nskin depth0 nliq0 nF0 backend

function run_phase!(model, nsteps, every, tag)
    prog = Progress(cld(nsteps, every); dt=0.3, desc="$tag ")
    t = 0
    Tmn = Tmx = zero(Float32)
    nF = nI = nliq = depth = nb = np = 0
    umax = 0.0f0
    zbar = 0.0
    while t < nsteps
        nrun = min(every, nsteps - t)
        with_logger(NullLogger()) do
            run!(model, nrun)
        end
        t += nrun
        export!(model; dir="output_laser_pool_nuclei")
        Tmn, Tmx, nF, nI, nliq, umax, depth, nb, np, zbar = pool_probe(model)
        next!(prog; showvalues = [
            (:t, t),
            (:Tmax_K, round(si_T(model.units, Tmx); digits=0)),
            (:depth, depth),
            (:nliq, nliq),
            (:nb, nb),
            (:planted, np),
            (:z_bub, round(zbar; digits=1)),
            (:umax, round(umax; digits=3)),
        ])
    end
    finish!(prog)
    return Tmn, Tmx, nF, nI, nliq, umax, depth, nb, np, zbar
end

s1 = run_phase!(model, n_heat, every, "melt")
@info "melt done, nucleation on" depth=s1[7] nliq=s1[5] nF=s1[3]
model.bubbles.nucleation.enabled = true
model.laser = Laser(units; P=20.0u"W", w=0.5 * si_d_spot,
                    x=xc, y=yc, z=Float32(Nz) - 1.1f0,
                    nrays=11, max_bounce=4, every=1, skin=nskin)
s2 = run_phase!(model, 16, 8, "nucleate")
@info "laser pool nuclei done" Tmax_K=si_T(units, s2[2]) depth0 depth=s2[7] nliq=s2[5] nF0 nF=s2[3] nb=s2[8] planted=s2[9] z_bub=s2[10] umax=s2[6]
