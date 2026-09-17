# Combined 4+5+6: mixed powder, Arrhenius in the melt, then nuclei, beam on.
# Laser power is unchanged. After the pool exists, numerical skin is cut to 4
# so Q stays in the lid instead of punching a 3³ hole (that was the Ma→1 crash).
#
# Tests: test/test_laser_foam_pad.jl
# ParaView: output_laser_foam_pad/lbm.pvd. Colour fs (pool), a (jet loading),
# c (dissolved), Contour phi=0.5 (nuclei). Threshold flags 8–32.
#
#   SURFACE = true, TEMPERATURE = true
using LatticeBoltzmann
using CUDA
using Unitful
using ProgressMeter
using Logging

@assert SURFACE && TEMPERATURE
start_run_log!("output_laser_foam_pad")

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
si_δ_melt = 0.60e-3u"m"
si_δ_nuc = 0.20e-3u"m"
si_t_heat = 4.0e-3u"s"
si_t_powder_delay = 1.5e-3u"s"
si_k_a = 2.0e5u"s^-1"
si_E_a = 6.7e3u"K"
si_mdot = 0.1u"g/s"
si_v_jet = 8.0u"m/s"
si_agent_frac = 0.15
k_H = 3.0f0

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
n_delay = max(0, round(Int, ustrip(u"s", si_t_powder_delay) / s))
every = max(10, n_heat ÷ 8)
nskin_melt = max(3, round(Int, ustrip(u"m", si_δ_melt) / m))
nskin_nuc = max(3, round(Int, ustrip(u"m", si_δ_nuc) / m))
T_init = Float32(lbm_T(units, si_T_init))
backend = CUDA.functional() ? CUDABackend() : CPU()

nuc = Nucleation{Float32}(; enabled=false, d_min=8, R=1, c_star=1.08f0,
                          p_cell=1, n_max=3, n_over=1.0f0, every=4,
                          n_total_max=8)
model = Model(Nx, Ny, Nz, units;
              ν=si_ν, α=2 * α_l_si, α_s=2 * α_s_si, α_l=2 * α_l_si, α_c=α_l_si,
              ν_s=0.5 * si_ν, ν_l=si_ν, k_H=k_H,
              gz=0, σ=si_σ,
              k_a=si_k_a, E_a=si_E_a, Y_a=1, a_fs_max=0.5,
              T_avg=T_init, latent=si_Lheat, Ts=si_Tm, Tl=si_Tm,
              latent_v=si_Lv, T_v=si_Tv, M=si_M,
              emissivity=0.4, T_rad=si_T_init,
              powder_τ=0, powder_T=si_T_init,
              K0=1.0e-10u"m^2",
              backend=backend, workgroup=backend isa CUDABackend ? 256 : 64,
              bubbles=BubbleTracker{Float32}(; nucleation=nuc, k_Π=0.08f0, d_max=4))

xc = (Nx + 1) / 2
yc = (Ny + 1) / 2
model.laser = Laser(units; P=si_P, w=0.5 * si_d_spot,
                    x=xc, y=yc, z=Float32(Nz) - 1.1f0,
                    nrays=11, max_bounce=4, every=1, skin=nskin_melt)
model.powder_jet = PowderJet(units; mdot=si_mdot, w=0.6 * si_d_spot, v=si_v_jet,
                             x=xc, y=yc, z=Float32(Nz) - 1.2f0,
                             dir=(0, 0, -1), nparcels=12,
                             enabled=false, agent_frac=si_agent_frac)

host = zeros(UInt8, Nx * Ny * Nz)
Th = fill(T_init, Nx * Ny * Nz)
fsh = ones(Float32, Nx * Ny * Nz)
for z in 1:Nz, y in 1:Ny, x in 1:Nx
    n = x + (y - 1) * Nx + (z - 1) * Nx * Ny
    if x == 1 || x == Nx || y == 1 || y == Ny || z == 1 || z == Nz
        host[n] = TYPE_S | TYPE_T
        fsh[n] = 1
        Th[n] = T_init
    elseif z <= Hfill
        host[n] = TYPE_F
        fsh[n] = 1
        Th[n] = T_init
    else
        host[n] = TYPE_G
        fsh[n] = 0
    end
end
copyto!(model.domains[1].flags.data, host)
copyto!(model.domains[1].T.data, Th)
copyto!(model.domains[1].fs.data, fsh)
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

function foam_probe(model)
    d = model.domains[1]
    TA = Array(d.T.data)
    fl = Array(d.flags.data)
    fsA = Array(d.fs.data)
    aA = Array(d.a.data)
    cA = Array(d.c.data)
    uA = Array(d.u.data)
    Tmx = -Inf32
    nliq = nF = 0
    a_tot = 0.0
    c_liq_max = 0.0f0
    umax = 0.0f0
    @inbounds for n in eachindex(TA)
        su = fl[n] & TYPE_SU
        su == TYPE_F && (nF += 1)
        (su == TYPE_F || su == TYPE_I) || continue
        Tmx = max(Tmx, TA[n])
        umax = max(umax, hypot(uA[n, 1], uA[n, 2], uA[n, 3]))
        a_tot += Float64(aA[n])
        if fsA[n] < 0.5f0
            nliq += 1
            c_liq_max = max(c_liq_max, cA[n])
        end
    end
    depth = liquid_column_depth(model, ix, iy)
    fm = foam_metrics(model)
    nfar = 4 + 3 * Int(d.Nx) + 3 * Int(d.Nx) * Int(d.Ny)
    return (Tmx, nliq, depth, umax, a_tot, aA[nfar], c_liq_max,
            fm.nb, fm.n_planted, fm.a_res, bubble_z_mean(model), nF)
end

export!(model; dir="output_laser_foam_pad")
p0 = foam_probe(model)
@info "laser foam pad" Nx Ny Nz Hfill n_heat n_delay nskin_melt nskin_nuc si_P si_agent_frac backend

function run_phase!(model, nsteps, every, n_delay, tag)
    prog = Progress(cld(nsteps, every); dt=0.3, desc="$tag ")
    t = 0
    Tmx = zero(Float32)
    nliq = depth = nb = np = nF = 0
    umax = a_tot = a_far = zbar = 0.0
    c_liq_max = a_res = zero(Float32)
    while t < nsteps
        if t >= n_delay && model.powder_jet !== nothing
            model.powder_jet.enabled = true
        end
        nrun = min(every, nsteps - t)
        with_logger(NullLogger()) do
            run!(model, nrun)
        end
        t += nrun
        export!(model; dir="output_laser_foam_pad")
        Tmx, nliq, depth, umax, a_tot, a_far, c_liq_max, nb, np, a_res, zbar, nF =
            foam_probe(model)
        next!(prog; showvalues = [
            (:t, t),
            (:P, model.laser.P),
            (:skin, model.laser.skin),
            (:Tmax_K, round(si_T(model.units, Tmx); digits=0)),
            (:depth, depth),
            (:nliq, nliq),
            (:a_tot, round(a_tot; digits=2)),
            (:c_liq, round(c_liq_max; digits=3)),
            (:nb, nb),
            (:z_bub, round(zbar; digits=1)),
            (:umax, round(umax; digits=3)),
        ])
    end
    finish!(prog)
    return Tmx, nliq, depth, umax, a_tot, a_far, c_liq_max, nb, np, a_res, zbar, nF
end

s1 = run_phase!(model, n_heat, every, n_delay, "melt+powder")
@info "melt+powder done" depth=s1[3] nliq=s1[2] a_tot=s1[5] c_liq=s1[7] P=model.laser.P
# Dissolved c from Arrhenius sits at the free surface; mix into liquid F so a
# 3³ cube can nucleate (same as the unit test).
let
    d = model.domains[1]
    cA = Array(d.c.data)
    fsA = Array(d.fs.data)
    fl = Array(d.flags.data)
    @inbounds for n in eachindex(cA)
        su = fl[n] & TYPE_SU
        (su == TYPE_F || su == TYPE_I) || continue
        fsA[n] < 0.5f0 || continue
        cA[n] = max(cA[n], 1.20f0)
    end
    copyto!(d.c.data, cA)
end
# Keep P; surface-local skin so the beam does not dump into a new 3³ cavity.
model.laser.skin = nskin_nuc
model.bubbles.nucleation.enabled = true
s2 = run_phase!(model, 16, 8, 0, "nucleate")
@info "laser foam pad done" Tmax_K=si_T(units, s2[1]) depth=s2[3] nliq=s2[2] a_tot=s2[5] a_far=s2[6] c_liq=s2[7] nb=s2[8] planted=s2[9] z_bub=s2[11] umax=s2[4] P=model.laser.P skin=model.laser.skin nF=s2[12]
