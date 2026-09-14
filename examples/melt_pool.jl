# DED-style 316L melt track. Coarse lattice, larger pad, 200 W — fast demo.
# Heat: PLIC + Fresnel laser. Powder: ballistic Gaussian jet (src/powder.jl)
# from behind or ahead of the spot along the scan (x), so the bead is
# symmetric in y. Set `n_layers` to retrace the bead.
# Evaporation cooling, mass loss, and recoil are on (latent_v, T_v).
#
# ParaView 6 + Qt6: Contour on a constant array (rho, Q, S) crashes the
# isosurface slider. Open lbm.pvd, colour by T (or phi), then Contour.
# Type the isosurface in the text box (e.g. T=1673, or phi=0.5 for the
# free surface). Do not drag the slider if the range looks empty.
#
#   SURFACE = true, TEMPERATURE = true, VOLUME_FORCE = true
using LatticeBoltzmann
using Printf
using CUDA
using Unitful
using ProgressMeter
using Logging

@assert SURFACE && TEMPERATURE
start_run_log!("output_melt_pool")

# --- user: scan ---
n_layers      = 1          # passes over the same bead (1 = single track)
bidirectional = true       # even layers scan x1 → x0; false = always x0 → x1
si_dwell      = 0.0u"s"    # laser + powder off between layers; 0 → none
jet_along     = :back      # :back = trailing (behind the travel), :front = leading

# --- user: grid / 316L ---
# ~6.4 × 5.1 mm in xy, pad 1.6 mm, Δx ≈ 80 µm. Nz is set from `n_layers` below.
Nx, Ny = 80, 64
Hfill = 22
L = Hfill - 2                           # pad thickness in cells

si_H      = 1.6e-3u"m"                  # → Δx ≈ 80 µm
si_ρ      = 8000u"kg/m^3"
si_cp     = 500u"J/kg/K"
# k(T) = k(Tm) + kT (T - Tm). k_s is at Tm so k(T_init) stays ~15 W/m/K.
si_Tm     = 1673.0u"K"
si_T_init = 300.0u"K"
si_k_sT   = 0.013u"W/m/K^2"
si_k_lT   = 0.005u"W/m/K^2"
si_k_s    = 15.0u"W/m/K" + si_k_sT * (si_Tm - si_T_init)
si_k_l    = 30.0u"W/m/K"
si_Lheat  = 2.8e5u"J/kg"
si_Lv     = 7.45e6u"J/kg"
si_Tv     = 3086.0u"K"
si_M      = 0.0558u"kg/mol"
si_K0     = 1.0e-10u"m^2"
si_P      = 200.0u"W"
si_d_spot = 0.5e-3u"m"
si_v      = 8.0e-3u"m/s"
si_δ      = 0.24e-3u"m"
si_mdot   = 2.0u"g/minute"
si_eta    = 0.7
si_powder_τ = 0.05u"s"                  # unmelted powder lifetime; 0 → instant metal
si_v_jet  = 8.0u"m/s"                   # parcel speed in the jet
si_σ      = 0.015u"N/m"
si_σT     = -1.5e-5u"N/m/K"
q_max     = 0.04f0

# sgn = +1 when the laser travels +x. Trailing nozzle sits behind that motion.
function place_powder_jet!(jet, x_las, y_las, z_noz, z_aim, sgn, dx_noz, jet_along, Nx)
    along = jet_along === :front ? sgn : -sgn
    x_noz = clamp(x_las + along * dx_noz, 2.5f0, Float32(Nx) - 1.5f0)
    set_powder_jet_position!(jet, x_noz, y_las, z_noz)
    aim_powder_jet!(jet, x_las, y_las, z_aim)
    return jet
end

function track_metrics(fsA, flags, TA, uA, Nx, Ny, Nz, Hfill, x_las, y_las, U)
    nliq = 0
    Tmax = -Inf32
    Tmin = Inf32
    umax = 0.0f0
    zI = 0
    nsolid = 4 + (4 - 1) * Nx + (4 - 1) * Nx * Ny
    xl = clamp(round(Int, x_las), 2, Nx - 1)
    yl = clamp(round(Int, y_las), 2, Ny - 1)
    for z in 1:Nz, y in 1:Ny, x in 1:Nx
        n = x + (y - 1) * Nx + (z - 1) * Nx * Ny
        su = flags[n] & TYPE_SU
        su == TYPE_I && (zI = max(zI, z))
        if su == TYPE_F || su == TYPE_I
            umax = max(umax, hypot(uA[n, 1], uA[n, 2], uA[n, 3]))
            Tmax = max(Tmax, TA[n])
            Tmin = min(Tmin, TA[n])
            fsA[n] < 0.5f0 && (nliq += 1)
        end
    end
    depth = 0
    for z in Hfill:-1:2
        n = xl + (yl - 1) * Nx + (z - 1) * Nx * Ny
        fsA[n] < 0.5f0 || break
        depth += 1
    end
    u_sol = hypot(uA[nsolid, 1], uA[nsolid, 2], uA[nsolid, 3])
    bead = max(0, zI - Hfill)
    return nliq, depth, bead, Float32(si_T(U, Tmax)), Float32(si_T(U, Tmin)), umax, u_sol, zI, xl
end

n_layers >= 1 || throw(ArgumentError("n_layers must be ≥ 1"))

α_l_si = si_k_l / (si_ρ * si_cp)
α_s_si = si_k_s / (si_ρ * si_cp)
si_ν_l = ustrip(u"m^2/s", α_l_si) * u"m^2/s"
si_ν_s = 0.5 * si_ν_l
si_ν_lT = -2.0e-9u"m^2/s/K"              # liquid thins with T; ν_sT = 0
lbm_α_true = 0.1
m = ustrip(u"m", si_H) / L
s = lbm_α_true * m^2 / ustrip(u"m^2/s", α_l_si)
lbm_u = 0.05
si_u = (lbm_u * m / s) * u"m/s"

units = Units(si_H, si_u, si_ρ; x=L, u=lbm_u, ρ=1, T=Float32,
              K=ustrip(u"K", si_Tm), cp=si_cp)

w_m    = 0.5 * ustrip(u"m", si_d_spot)
I_peak = 2 * ustrip(u"W", si_P) / (π * w_m^2)
δ_m    = ustrip(u"m", si_δ)
A0     = fresnel_absorptance(1.0f0, 3.27f0, 4.48f0)
nskin  = max(3, round(Int, δ_m / m))
Q_si   = A0 * I_peak / (nskin * m)
Q_full = Float32(lbm_Q(units, Q_si * u"W/m^3"))
mdot_kg_s = si_eta * ustrip(u"kg/s", uconvert(u"kg/s", si_mdot))

# Gas headroom: expected layer height from captured powder on a ~spot-wide bead,
# plus a fixed clearance so the launch plane stays in TYPE_G.
v_m      = ustrip(u"m/s", si_v)
ρ_m      = ustrip(u"kg/m^3", si_ρ)
w_bead   = 2 * w_m
h_layer  = mdot_kg_s / max(ρ_m * w_bead * v_m, 1e-30)
n_z_layer = max(2, ceil(Int, h_layer / m))
n_gas_top = 8
Nz = Hfill + n_layers * n_z_layer + n_gas_top + 1

model = Model(Nx, Ny, Nz, units;
              ν = si_ν_l,
              α = 2 * α_l_si,
              α_s = 2 * α_s_si,
              α_l = 2 * α_l_si,
              ν_s = si_ν_s,
              ν_l = si_ν_l,
              k_sT = si_k_sT, k_lT = si_k_lT, ν_lT = si_ν_lT,
              β = 0.0f0, gz = 0.0f0,
              σ = si_σ, σT = si_σT, Tσ = si_Tm,
              latent = si_Lheat,
              Ts = si_Tm, Tl = si_Tm, K0 = si_K0,
              latent_v = si_Lv, T_v = si_Tv, M = si_M,
              T_avg = Float32(lbm_T(units, si_Tm)),
              emissivity = 0.4, T_rad = si_T_init,
              powder_τ = si_powder_τ, powder_T = si_T_init,
              backend = CUDABackend())

Tm      = Float32(lbm_T(units, si_Tm))
T_init  = Float32(lbm_T(units, si_T_init))
w_cells = w_m / Float64(units.m)
v_lat   = Float32(ustrip(u"m/s", si_v) * units.s / units.m)
y_las   = Float32(Ny + 1) / 2
x0      = Float32(8 + 2 * w_cells)
x1      = Float32(Nx - 7 - 2 * w_cells)
nsteps_pass  = max(2, round(Int, abs(x1 - x0) / max(v_lat, Float32(1e-8))))
nsteps_dwell = si_dwell > 0u"s" ?
    max(0, round(Int, ustrip(u"s", si_dwell) / Float64(units.s))) : 0
nsteps_total = n_layers * nsteps_pass + max(0, n_layers - 1) * nsteps_dwell
qevery  = max(1, round(Int, 0.25f0 / max(v_lat, Float32(1e-8))))
# Frame spacing from one pass, not the whole job — otherwise n_layers=3
# writes 3× fewer VTK/progress samples and the beam looks 3× faster.
every   = max(qevery, max(1, nsteps_pass ÷ 40))
dx_noz  = Float32(max(6, 2 * w_cells))   # standoff along the track
z_noz   = Float32(Nz) - 1.4f0
z_aim   = Float32(Hfill)
σlat    = model.domains[1].σ
if Q_full > q_max
    @warn "surface peak Q_lat=$(Q_full) > q_max=$(q_max); raise skin/δ or lower P" Q_full nskin
end
if σlat > 0.05f0
    @warn "lattice σ=$(σlat) is large; CSF may blow up. Lower si_σ." σlat
end
@info "DED 316L multilayer track" n_layers bidirectional si_dwell Nx Ny Nz m_um=(1e6*m) box_mm=(1e3*m*Nx, 1e3*m*Ny, 1e3*m*Nz) pad_mm=(1e3*m*L) h_layer_mm=(1e3*h_layer) n_z_layer n_gas_top w_cells v_lat nsteps_pass nsteps_dwell nsteps_total qevery Ncell=(Nx*Ny*Nz) si_P si_v si_mdot si_v_jet A0 nskin Q_full τ_p=model.domains[1].τ_p T_p=model.domains[1].T_p Λ_v=model.domains[1].Λ_v T_v=model.domains[1].T_v T_init

host = zeros(UInt8, Nx * Ny * Nz)
Th   = fill(T_init, Nx * Ny * Nz)
fsh  = ones(Float32, Nx * Ny * Nz)
for z in 1:Nz, y in 1:Ny, x in 1:Nx
    n = x + (y - 1) * Nx + (z - 1) * Nx * Ny
    if z == 1 || z == Nz || x == 1 || x == Nx || y == 1 || y == Ny
        host[n] = TYPE_S
        Th[n] = T_init
    elseif z == 2 && z <= Hfill
        # cold Dirichlet plate: D3Q7 AA does not bounce the -z wall, so the
        # pad bottom would otherwise drift toward Tm.
        host[n] = TYPE_F | TYPE_T
        Th[n] = T_init
        fsh[n] = 1
    elseif z <= Hfill
        host[n] = TYPE_F
        Th[n] = T_init
        fsh[n] = 1
    end
end
copyto!(model.domains[1].flags.data, host)
copyto!(model.domains[1].T.data, Th)
copyto!(model.domains[1].fs.data, fsh)

d = model.domains[1]
model.laser = Laser(units; P = si_P, w = 0.5 * si_d_spot,
                    x = x0, y = y_las, z = Float32(Nz) - 1.1f0,
                    nrays = 9, max_bounce = 6, every = qevery, skin = nskin)
model.powder_jet = PowderJet(units; mdot = si_eta * si_mdot, w = 0.6 * si_d_spot,
                             v = si_v_jet, x = x0, y = y_las, z = z_noz,
                             nparcels = 16)
place_powder_jet!(model.powder_jet, x0, y_las, z_noz, z_aim, 1.0f0, dx_noz, jet_along, Nx)
LatticeBoltzmann.initialize!(model)
export!(model; dir="output_melt_pool")

function run_layers!(model, d, x0, x1, y_las, v_lat, n_layers, bidirectional,
                     nsteps_pass, nsteps_dwell, nsteps_total, qevery, every,
                     Nx, Ny, Nz, Hfill, z_noz, z_aim, dx_noz, jet_along)
    Ncell = Nx * Ny * Nz
    xmin, xmax = min(x0, x1), max(x0, x1)
    istep = 0
    x_now = x0
    mlups_ema = NaN
    αema = 0.2
    prog = Progress(cld(nsteps_total, every); dt=0.3, desc="DED layers ", showspeed=true)

    function report!(layer, x_las)
        export!(model; dir="output_melt_pool")
        nliq, depth, bead, Tmax_K, Tmin_K, umax, _, zI, xl = track_metrics(
            Array(d.fs.data), Array(d.flags.data), Array(d.T.data), Array(d.u.data),
            Nx, Ny, Nz, Hfill, x_las, y_las, model.units)
        Hlat = enthalpy(d)
        next!(prog; showvalues = [
            (:layer, layer),
            (:t, Int(d.t)),
            (:t_si, round(si_t(model.units, Int(d.t)); digits=4)),
            (:x_las, round(xl; digits=1)),
            (:nliq, nliq),
            (:depth, depth),
            (:bead, bead),
            (:Tmax_K, round(Tmax_K; digits=1)),
            (:Tmin_K, round(Tmin_K; digits=1)),
            (:H_J, round(si_enthalpy(model.units, Hlat); digits=3)),
            (:umax, round(umax; digits=3)),
            (:zI, zI),
            (:MLUPS, round(mlups_ema; digits=1)),
        ])
        return nothing
    end

    function do_step!(layer, x_las, powder, sgn)
        set_laser_position!(model.laser, x_las, y_las)
        jet = model.powder_jet
        jet.enabled = powder
        powder && place_powder_jet!(jet, x_las, y_las, z_noz, z_aim, sgn, dx_noz, jet_along, Nx)
        t0 = time_ns()
        with_logger(NullLogger()) do
            run!(model, 1)
        end
        dt = (time_ns() - t0) * 1e-9
        mlups = Ncell / max(dt, 1e-12) / 1e6
        mlups_ema = isfinite(mlups_ema) ? αema * mlups + (1 - αema) * mlups_ema : mlups
        istep += 1
        if istep % every == 0 || istep == nsteps_total
            report!(layer, x_las)
        end
        return nothing
    end

    for layer in 1:n_layers
        model.laser.enabled = true
        backward = bidirectional && iseven(layer)
        x_start = backward ? x1 : x0
        sgn = backward ? -one(Float32) : one(Float32)
        for k in 0:(nsteps_pass - 1)
            x_now = clamp(x_start + sgn * v_lat * Float32(k), xmin, xmax)
            do_step!(layer, x_now, true, sgn)
        end
        if layer < n_layers && nsteps_dwell > 0
            model.laser.enabled = false
            fill!(d.Q.data, 0)
            for _ in 1:nsteps_dwell
                do_step!(layer, x_now, false, sgn)
            end
        end
    end
    finish!(prog)
    return x_now
end

x_end = run_layers!(model, d, x0, x1, y_las, v_lat, n_layers, bidirectional,
                    nsteps_pass, nsteps_dwell, nsteps_total, qevery, every,
                    Nx, Ny, Nz, Hfill, z_noz, z_aim, dx_noz, jet_along)

LatticeBoltzmann.moments!(model)
nliq, depth, bead, Tmax_K, Tmin_K, umax, u_sol, zI, xl = track_metrics(
    Array(d.fs.data), Array(d.flags.data), Array(d.T.data), Array(d.u.data),
    Nx, Ny, Nz, Hfill, x_end, y_las, model.units)
Hlat = enthalpy(d)
@info "DED multilayer report" n_layers bidirectional nliq depth_cells=depth bead_cells=bead Tmax_K Tmin_K H_J=si_enthalpy(model.units, Hlat) umax u_sol zI Hfill x_las=xl t_end=si_t(model.units, Int(d.t)) P_W=ustrip(u"W", si_P) A0 nskin Q_full mdot_kg_s
