# Laser melts a cold foaming pad, then the beam turns off and the foam develops
# in the pool. Stationary beam, no powder, no scan. Starts at room T (solid);
# agent decomposes in the hot liquid while the beam is on (Arrhenius + fs).
# Hertz–Knudsen evaporation + recoil hold a keyhole: T stays near Tv instead
# of running away in the subsurface liquid.
#
# ParaView: output_laser_foam/lbm.pvd. Colour by T (set the range 300–2500 K) —
# gas is written as 0 K so the pad should read ~room T with a hot spot under
# the beam. Contour phi = 0.5 (Threshold flags 8–32). Nuclei form in the melt,
# not against the floor.
#
#   SURFACE = true, TEMPERATURE = true
using LatticeBoltzmann
using CUDA
using Unitful
using ProgressMeter
using Logging

@assert SURFACE && TEMPERATURE
start_run_log!("output_laser_foam")

# --- SI box / 316L-like pad ---
si_Lx = 4.0e-3u"m"
si_Ly = 4.0e-3u"m"
si_H  = 1.6e-3u"m"
si_gas = 1.2e-3u"m"
si_dx = 25.0e-6u"m"

si_ρ  = 8000u"kg/m^3"
si_cp = 500.0u"J/kg/K"
si_k_s = 15.0u"W/m/K"
si_k_l = 30.0u"W/m/K"
si_Tm = 1673.0u"K"
si_T_init = 293.0u"K"             # room T; pad is solid until the laser melts it
si_Lheat = 2.7e5u"J/kg"
si_Lv = 7.45e6u"J/kg"             # HK evaporation: caps T near Tv, opens a keyhole
si_Tv = 3086.0u"K"
si_M = 0.0558u"kg/mol"
# Steel σ at this Δx/Δt is σ_lat~O(1) and Ma blows FSLBM. 0.015 N/m is the
# melt-pool value (σ_lat ~ 0.01) and still holds a keyhole against recoil.
si_σ = 0.015u"N/m"
si_P = 300.0u"W"
si_d_spot = 0.7e-3u"m"
si_δ = 0.20e-3u"m"
si_k_a = 2.0e5u"s^-1"
si_E_a = 6.7e3u"K"                # E/R ≈ 4 Tm; cold solid does not release

n_heat = 800
n_develop = 280
every = 40

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
lbm_α_true = 0.1
s = lbm_α_true * m^2 / ustrip(u"m^2/s", α_l_si)
lbm_u = 0.05
si_u = (lbm_u * m / s) * u"m/s"
units = Units(si_H, si_u, si_ρ; x=L, u=lbm_u, ρ=1, T=Float32,
              K=ustrip(u"K", si_Tm), cp=si_cp)

T_init = Float32(lbm_T(units, si_T_init))
nskin = max(3, round(Int, ustrip(u"m", si_δ) / m))
k_H = 3.0f0
A0 = fresnel_absorptance(1.0f0, 3.27f0, 4.48f0)
w_m = 0.5 * ustrip(u"m", si_d_spot)
I_peak = 2 * ustrip(u"W", si_P) / (π * w_m^2)
Q_si = A0 * I_peak / (nskin * m)
Q_full = Float32(lbm_Q(units, Q_si * u"W/m^3"))

backend = CUDA.functional() ? CUDABackend() : CPU()
nuc = Nucleation{Float32}(; d_min=10, R=1, c_star=1.02f0, p_cell=1,
                          n_max=3, n_over=1.12f0, every=20, n_total_max=16)
model = Model(Nx, Ny, Nz, units;
              ν=si_ν, α=2 * α_l_si, α_s=2 * α_s_si, α_l=2 * α_l_si, α_c=α_l_si,
              ν_s=0.5 * si_ν, ν_l=si_ν, k_H=k_H,
              gz=0, σ=si_σ,
              k_a=si_k_a, E_a=si_E_a, Y_a=1, a_fs_max=0.5,
              T_avg=T_init, latent=si_Lheat, Ts=si_Tm, Tl=si_Tm,
              latent_v=si_Lv, T_v=si_Tv, M=si_M,
              emissivity=0.4, T_rad=si_T_init,
              K0=1.0e-10u"m^2",
              backend=backend, workgroup=backend isa CUDABackend ? 256 : 64,
              bubbles=BubbleTracker{Float32}(; nucleation=nuc, k_Π=0.08f0, d_max=4))

x_las = Float32(Nx + 1) / 2
y_las = Float32(Ny + 1) / 2
z_las = Float32(Nz) - 1.1f0
model.laser = Laser(units; P=si_P, w=0.5 * si_d_spot,
                    x=x_las, y=y_las, z=z_las,
                    nrays=13, max_bounce=4, every=1, skin=nskin)

a0 = 0.08f0
host = zeros(UInt8, Nx * Ny * Nz)
Th = fill(T_init, Nx * Ny * Nz)
fsh = ones(Float32, Nx * Ny * Nz)
ah = zeros(Float32, Nx * Ny * Nz)
ch = fill(k_H * P_ATM_LAT, Nx * Ny * Nz)
for z in 1:Nz, y in 1:Ny, x in 1:Nx
    n = x + (y - 1) * Nx + (z - 1) * Nx * Ny
    if x == 1 || x == Nx || y == 1 || y == Ny || z == 1 || z == Nz
        # Dirichlet room-T walls. Plain TYPE_S is adiabatic and D3Q7 rings
        # at T_lat ≪ 1, which heats the corners and hides the laser spot.
        host[n] = TYPE_S | TYPE_T
        fsh[n] = 1
        Th[n] = T_init
    elseif z <= Hfill
        host[n] = TYPE_F
        ah[n] = a0
        fsh[n] = 1                  # solid at room T
        Th[n] = T_init
    else
        host[n] = TYPE_G
        fsh[n] = 0
        Th[n] = T_init
    end
end
copyto!(model.domains[1].flags.data, host)
copyto!(model.domains[1].T.data, Th)
copyto!(model.domains[1].fs.data, fsh)
copyto!(model.domains[1].a.data, ah)
copyto!(model.domains[1].c.data, ch)

LatticeBoltzmann.initialize!(model)

d = model.domains[1]
LatticeBoltzmann.deposit_laser!(model, d)
QA = Array(d.Q.data)
fl0 = Array(d.flags.data)
qfac = LatticeBoltzmann.laser_qfac(units)
Pabs, nQ, Qmx = let
    p = 0.0
    nq = 0
    qmx = 0.0f0
    for n in eachindex(QA)
        q = QA[n]
        q == 0 && continue
        su = fl0[n] & TYPE_SU
        (su == TYPE_F || su == TYPE_I) || continue
        p += Float64(q)
        nq += 1
        qmx = max(qmx, q)
    end
    (p / qfac, nq, qmx)
end
fill!(d.Q.data, zero(eltype(d.Q.data)))

function metal_T_minmax(model)
    d = model.domains[1]
    TA = Array(d.T.data)
    fl = Array(d.flags.data)
    fsA = Array(d.fs.data)
    Nx, Ny, Nz = Int(d.Nx), Int(d.Ny), Int(d.Nz)
    Tmn, Tmx = Inf32, -Inf32
    nliq = 0
    @inbounds for n in eachindex(TA)
        su = fl[n] & TYPE_SU
        (su == TYPE_F || su == TYPE_I) || continue
        Tmn = min(Tmn, TA[n])
        Tmx = max(Tmx, TA[n])
        fsA[n] < 0.5f0 && (nliq += 1)
    end
    nspot = round(Int, (Nx + 1) / 2) + (round(Int, (Ny + 1) / 2) - 1) * Nx + (Hfill - 1) * Nx * Ny
    xf, yf, zf = max(4, Nx ÷ 4), max(4, Ny ÷ 4), max(4, Hfill ÷ 2)
    nfar = xf + (yf - 1) * Nx + (zf - 1) * Nx * Ny
    uA = Array(d.u.data)
    umax = 0.0f0
    @inbounds for n in eachindex(TA)
        su = fl[n] & TYPE_SU
        (su == TYPE_F || su == TYPE_I) || continue
        umax = max(umax, hypot(uA[n, 1], uA[n, 2], uA[n, 3]))
    end
    return Tmn, Tmx, nliq, TA[nspot], TA[nfar], umax
end

function bubble_z_mean(model)
    B = model.bubbles
    B isa BubbleTracker || return 0.0
    Nx, Ny, Nz = Int(model.domains[1].Nx), Int(model.domains[1].Ny), Int(model.domains[1].Nz)
    s = 0.0
    n = 0
    @inbounds for i in eachindex(B.label)
        B.label[i] > 0 || continue
        s += (i - 1) ÷ (Nx * Ny) + 1
        n += 1
    end
    return n == 0 ? 0.0 : s / n
end

Tmn0, Tmx0, nliq0, Tspot0, Tfar0, umax0 = metal_T_minmax(model)
export!(model; dir="output_laser_foam")
m0 = foam_metrics(model)
@info "laser foam" Nx Ny Nz Hfill m_um=(1e6*m) s_us=(1e6*s) spot_mm=(1e3*ustrip(u"m", si_d_spot)) si_P n_heat n_develop nskin σ_lat=d.σ fz=d.fz k_a=d.k_a E_a=d.E_a T_init Tmin0_K=si_T(units, Tmn0) Tmax0_K=si_T(units, Tmx0) Tspot0_K=si_T(units, Tspot0) Tfar0_K=si_T(units, Tfar0) nliq0 umax0 A0 Q_full Pabs_W=Pabs nQ Qmax_lat=Qmx Ncell=(Nx*Ny*Nz) m0 backend

function run_phase!(model, nsteps, every, tag)
    prog = Progress(cld(nsteps, every); dt=0.3, desc="$tag ")
    t = 0
    local m = foam_metrics(model)
    while t < nsteps
        nrun = min(every, nsteps - t)
        with_logger(NullLogger()) do
            run!(model, nrun)
        end
        t += nrun
        export!(model; dir="output_laser_foam")
        m = foam_metrics(model)
        Tmn, Tmx, nliq, Tspot, Tfar, umax = metal_T_minmax(model)
        zbar = bubble_z_mean(model)
        next!(prog; showvalues = [
            (:t, t),
            (:laser, model.laser.enabled),
            (:Tmin_K, round(si_T(model.units, Tmn); digits=0)),
            (:Tmax_K, round(si_T(model.units, Tmx); digits=0)),
            (:Tspot_K, round(si_T(model.units, Tspot); digits=0)),
            (:Tfar_K, round(si_T(model.units, Tfar); digits=0)),
            (:nliq, nliq),
            (:nb, m.nb),
            (:z_bub, round(zbar; digits=1)),
            (:umax, round(umax; digits=3)),
            (:porosity, round(m.porosity; digits=3)),
            (:c_n, round(m.dissolved; digits=1)),
            (:a, round(m.a; digits=3)),
        ])
    end
    finish!(prog)
    return m
end

m1 = run_phase!(model, n_heat, every, "laser")
Tmn1, Tmx1, nliq1, Tspot1, Tfar1, umax1 = metal_T_minmax(model)
@info "laser off" m1 Tmin_K=si_T(units, Tmn1) Tmax_K=si_T(units, Tmx1) Tspot_K=si_T(units, Tspot1) Tfar_K=si_T(units, Tfar1) nliq=nliq1 umax=umax1 z_bub=bubble_z_mean(model)
model.laser.enabled = false
m2 = run_phase!(model, n_develop, every, "develop")
Tmn2, Tmx2, nliq2, Tspot2, Tfar2, umax2 = metal_T_minmax(model)
@info "laser foam done" m0 m1 m2 Tmin_K=si_T(units, Tmn2) Tmax_K=si_T(units, Tmx2) Tspot_K=si_T(units, Tspot2) Tfar_K=si_T(units, Tfar2) nliq=nliq2 umax=umax2 z_bub=bubble_z_mean(model)
