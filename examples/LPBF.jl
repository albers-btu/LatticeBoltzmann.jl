# 316L LPBF single track on input/powder_bed.h5.
# 200 W, 100 µm 1/e² spot, 1 m/s. Plate + powder come from the HDF5 bed.
# Process numbers are loaded from input/ (material_316L, LPBF_build, LPBF_laser, LPBF_powder).
# Heat: PLIC + Fresnel (multi-bounce). Evaporation, recoil, radiation, gravity.
# Bottom is TYPE_S|TYPE_H Robin into a cold backing (not a Dirichlet sink).
# σ stays the physical 1.6 N/m; n_hydro shortens the flow step so σ_lat stays small.
#
# ParaView 6 + Qt6: Contour on a constant array (rho, Q, S) crashes the
# isosurface slider. Open lbm.pvd and rays.pvd (colour rays by power).
# Colour the volume by T (or phi), then Contour.
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
start_run_log!("output_LPBF")

# Process settings. Edit the files in input/; names stay in this scope.
# LPBF_build.jl is loaded after the bed, because si_gas uses bed.dz.
const input_dir = joinpath(@__DIR__, "..", "input")
include(joinpath(input_dir, "material_316L.jl"))
include(joinpath(input_dir, "LPBF_laser.jl"))
include(joinpath(input_dir, "LPBF_powder.jl"))

# The grid is the DEM file (one domain cell per file cell). This bed is a
# short track, about 2.1 × 0.53 × 0.35 mm. Plate is file z = 41 (~0.20 mm);
# powder sits on it through the top layer.
const bed_path = joinpath(input_dir, "powder_bed.h5")
isfile(bed_path) || error("LPBF example needs $bed_path")
const bed = read_powder_bed(bed_path)
si_dx = bed.dx * u"m"                   # cell size (the file's dx)
si_Lx = size(bed.phi, 1) * bed.dx * u"m"  # domain length, from the file
si_Ly = size(bed.phi, 2) * bed.dy * u"m"  # domain width, from the file
zsub  = max(1, substrate_top(bed))
si_H  = zsub * bed.dz * u"m"            # substrate thickness, from the file
include(joinpath(input_dir, "LPBF_build.jl"))

# Lattice CSF cannot hold the physical σ at this Δx, Δt (σ_lat would be O(1)).
# Capped after Units.
σ_lat_cap = 0.03f0
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
lbm_α_true = 0.1
# Pad cells fix Δx through Units(si_H; x=L). Nx, Ny follow the same Δx.
# paint_powder_bed! strides with the file's Nx, Ny, so the domain must match.
L     = max(4, round(Int, ustrip(u"m", si_H) / ustrip(u"m", si_dx)))
m     = ustrip(u"m", si_H) / L
Nx    = max(16, round(Int, ustrip(u"m", si_Lx) / m))
Ny    = max(16, round(Int, ustrip(u"m", si_Ly) / m))
Hfill = L + 2
if bed !== nothing
    # File z = 1 sits on the Robin wall (domain z = 2).
    Hfill = 1 + substrate_top(bed)
    (Nx, Ny) == (size(bed.phi, 1), size(bed.phi, 2)) ||
        error("domain $(Nx)×$(Ny) does not match powder file $(size(bed.phi))")
end
s = lbm_α_true * m^2 / ustrip(u"m^2/s", α_l_si)
lbm_u = 0.05
si_u = (lbm_u * m / s) * u"m/s"       # reference speed that sets Δt; follows si_dx and si_k_l

units = Units(si_H, si_u, si_ρ; x=L, u=lbm_u, ρ=1, T=Float32,
              K=ustrip(u"K", si_Tm), cp=si_cp)

w_m    = 0.5 * ustrip(u"m", si_d_spot)
I_peak = 2 * ustrip(u"W", si_P) / (π * w_m^2)
δ_m    = ustrip(u"m", si_δ)
A0     = fresnel_absorptance(1.0f0, fresnel_n, fresnel_k)
nskin  = max(3, round(Int, δ_m / m))
Q_si   = A0 * I_peak / (nskin * m)
Q_full = Float32(lbm_Q(units, Q_si * u"W/m^3"))
mdot_kg_s = si_eta * ustrip(u"kg/s", uconvert(u"kg/s", si_mdot))

# No powder bead. Gas headroom is only si_gas (keyhole / launch plane).
v_m      = ustrip(u"m/s", si_v)
ρ_m      = ustrip(u"kg/m^3", si_ρ)
w_bead   = 2 * w_m
h_layer  = use_powder ? mdot_kg_s / max(ρ_m * w_bead * v_m, 1e-30) : 0.0
n_z_layer = use_powder ? max(2, ceil(Int, h_layer / m)) : 0
n_gas_top = max(6, ceil(Int, ustrip(u"m", si_gas) / m))
if bed === nothing
    Nz = Hfill + n_layers * n_z_layer + n_gas_top + 1
else
    # z = 1 Robin wall, then the file, then gas, then the top wall.
    Nz = 1 + size(bed.phi, 3) + n_gas_top + 1
end

# Physical σ on the outer step is σ_lat ∝ Δt². Hydro substeps use Δt/n_hydro,
# so σ_lat,sub = σ_lat / n_hydro². Pick n_hydro so that stays ≤ σ_lat_cap.
kg_cell = ρ_m * m^3
σ_lat_phys = ustrip(u"N/m", si_σ_phys) * s^2 / kg_cell
n_hydro = max(1, ceil(Int, sqrt(max(σ_lat_phys, 0.0) / Float64(σ_lat_cap))))
# An even count reuses one temperature slot, so the pool does not conduct.
iseven(n_hydro) && (n_hydro += 1)
si_σ  = si_σ_phys                      # surface tension passed to the model; edit si_σ_phys
si_σT = si_σT_phys                      # dσ/dT passed to the model; edit si_σT_phys

model = Model(Nx, Ny, Nz, units;
              ν = si_ν_l,
              α = 2 * α_l_si,
              α_s = 2 * α_s_si,
              α_l = 2 * α_l_si,
              ν_s = si_ν_s,
              ν_l = si_ν_l,
              k_sT = si_k_sT, k_lT = si_k_lT, cp_sT = si_cp_sT, cp_lT = si_cp_lT,
              ν_lT = si_ν_lT,
              β = Float32(ustrip(u"K^-1", si_β) * ustrip(u"K", si_Tm)),
              gz = -si_g,
              σ = si_σ, σT = si_σT, Tσ = si_Tm,
              latent = si_Lheat,
              Ts = si_Tm, Tl = si_Tm, K0 = si_K0,
              latent_v = si_Lv, T_v = si_Tv, M = si_M,
              T_avg = Float32(lbm_T(units, si_Tm)),
              emissivity = emissivity, T_rad = si_T_init,
              powder_τ = powder_τ, powder_T = si_T_init,
              n_hydro = n_hydro,
              backend = CUDABackend())

Tm      = Float32(lbm_T(units, si_Tm))
T_init  = Float32(lbm_T(units, si_T_init))
w_cells = w_m / Float64(units.m)
# Same ray pitch as DED: half a cell across the 4w bundle, not a fixed 11×11.
nrays = max(11, ceil(Int, 4 * w_cells / 0.5))
v_lat   = Float32(ustrip(u"m/s", si_v) * units.s / units.m)
y_las   = Float32(Ny + 1) / 2
# Scan ends in metres, not a fixed cell count — otherwise refining eats the
# run-in and the bead looks like it fills the box.
n_end = ustrip(u"m", si_end_margin) / m
x0    = Float32(clamp(n_end, 4.0, Nx / 4))
x1    = Float32(clamp(Nx + 1 - n_end, 3 * Nx / 4, Nx - 3))
nsteps_pass  = max(2, round(Int, abs(x1 - x0) / max(v_lat, Float32(1e-8))))
nsteps_dwell = si_dwell > 0u"s" ?
    max(0, round(Int, ustrip(u"s", si_dwell) / Float64(units.s))) : 0
nsteps_powder_delay = 0
nsteps_freeze = si_freeze > 0u"s" ?
    max(0, round(Int, ustrip(u"s", si_freeze) / Float64(units.s))) : 0
nsteps_total = n_layers * nsteps_pass + max(0, n_layers - 1) * nsteps_dwell + nsteps_freeze

qevery  = max(1, round(Int, 0.25f0 / max(v_lat, Float32(1e-8))))
# Frame spacing from one pass, not the whole job — otherwise n_layers=3
# writes 3× fewer VTK/progress samples and the beam looks 3× faster.
every   = max(qevery, max(1, nsteps_pass ÷ 80))
dx_noz  = Float32(max(0.5e-3 / m, 2 * w_cells))   # ≥0.5 mm along the track
z_noz   = Float32(Nz) - 1.4f0
z_aim   = Float32(Hfill)
σlat    = model.domains[1].σ
h_lat   = Float32(lbm_h(units, si_h_sub))
if Q_full > q_max
    @warn "surface peak Q_lat=$(Q_full) > q_max=$(q_max); raise skin/δ or lower P" Q_full nskin
end
if σlat > 0.05f0
    @warn "lattice σ=$(σlat) is large; CSF may blow up. Lower si_σ." σlat
end
@info "316L LPBF single track" bed_path n_hydro n_layers Nx Ny Nz Hfill m_um=(1e6*m) box_mm=(1e3*m*Nx, 1e3*m*Ny, 1e3*m*Nz) gas_mm=(1e3*m*n_gas_top) spot_um=(1e6*ustrip(u"m", si_d_spot)) w_cells nrays scan_mm=(1e3*m*abs(x1-x0)) v_mps=ustrip(u"m/s", si_v) v_lat line_J_per_mm=(ustrip(u"W", si_P) / ustrip(u"m/s", si_v) / 1e3) nsteps_pass nsteps_freeze nsteps_total Ncell=(Nx*Ny*Nz) si_P si_σ si_σT σlat=model.domains[1].σ σ_lat_phys A0 nskin Q_full h_lat gz=model.domains[1].fz β=model.domains[1].β ν=model.domains[1].ν T_init

host = zeros(UInt8, Nx * Ny * Nz)
Th   = fill(T_init, Nx * Ny * Nz)
fsh  = ones(Float32, Nx * Ny * Nz)
hh   = zeros(Float32, Nx * Ny * Nz)
for z in 1:Nz, y in 1:Ny, x in 1:Nx
    n = x + (y - 1) * Nx + (z - 1) * Nx * Ny
    if z == 1
        host[n] = TYPE_S | TYPE_H
        Th[n] = T_init
        hh[n] = h_lat
    elseif z == Nz || x == 1 || x == Nx || y == 1 || y == Ny
        host[n] = TYPE_S
        Th[n] = T_init
    elseif bed === nothing && z <= Hfill
        host[n] = TYPE_F
        Th[n] = T_init
        fsh[n] = 1
    end
end
if bed !== nothing
    Hfill = paint_powder_bed!(host, Th, fsh, bed, T_init; z0=2)
    Hfill == 0 && (Hfill = 2)
    for z in 1:Nz, y in 1:Ny, x in 1:Nx
        n = x + (y - 1) * Nx + (z - 1) * Nx * Ny
        if z == 1
            host[n] = TYPE_S | TYPE_H
            hh[n] = h_lat
        elseif z == Nz || x == 1 || x == Nx || y == 1 || y == Ny
            host[n] = TYPE_S
        end
    end
end
copyto!(model.domains[1].flags.data, host)
copyto!(model.domains[1].T.data, Th)
copyto!(model.domains[1].fs.data, fsh)
copyto!(model.domains[1].h.data, hh)

d = model.domains[1]
model.laser = Laser(units; P = si_P, w = 0.5 * si_d_spot,
                    x = x0, y = y_las, z = Float32(Nz) - 1.1f0,
                    nrays = nrays, max_bounce = 8, every = qevery, skin = nskin)
model.powder_jet = PowderJet(units; mdot = 0.0u"kg/s", w = 0.6 * si_d_spot,
                             v = 1.0u"m/s", x = x0, y = y_las, z = z_noz,
                             nparcels = 1, enabled = false)
LatticeBoltzmann.initialize!(model)
export!(model; dir="output_LPBF")

function run_layers!(model, d, x0, x1, y_las, v_lat, n_layers, bidirectional,
                     nsteps_pass, nsteps_dwell, nsteps_freeze, nsteps_powder_delay, nsteps_total,
                     qevery, every, Nx, Ny, Nz, Hfill, z_noz, z_aim, dx_noz)
    Ncell = Nx * Ny * Nz
    xmin, xmax = min(x0, x1), max(x0, x1)
    istep = 0
    x_now = x0
    mlups_ema = NaN
    αema = 0.2
    prog = Progress(cld(nsteps_total, every); dt=0.3, desc="LPBF ", showspeed=true)

    function report!(layer, x_las)
        export!(model; dir="output_LPBF")
        nliq, depth, bead, Tmax_K, Tmin_K, umax, _, zI, xl = track_metrics(
            Array(d.fs.data), Array(d.flags.data), Array(d.T.data), Array(d.u.data),
            Nx, Ny, Nz, Hfill, x_las, y_las, model.units)
        b = energy_budget(d)
        m = mass_budget(d)
        U = model.units
        next!(prog; showvalues = [
            (:layer, layer),
            (:t, Int(d.t)),
            (:t_si, round(si_t(U, Int(d.t)); digits=4)),
            (:x_las, round(xl; digits=1)),
            (:nliq, nliq),
            (:depth, depth),
            (:bead, bead),
            (:Tmax_K, round(Tmax_K; digits=1)),
            (:Tmin_K, round(Tmin_K; digits=1)),
            (:H_J, round(si_enthalpy(U, b.H); digits=3)),
            (:Qin_J, round(si_enthalpy(U, b.Q); digits=3)),
            (:Qout_J, round(si_enthalpy(U, b.rad + b.evap + b.wall); digits=3)),
            (:pow_J, round(si_enthalpy(U, b.powder); digits=3)),
            (:res_J, round(si_enthalpy(U, b.residual); digits=3)),
            (:M_g, round(1e3 * si_mass(U, m.M); digits=3)),
            (:Mpow_g, round(1e3 * si_mass(U, m.powder); digits=3)),
            (:Mevap_g, round(1e3 * si_mass(U, m.evap); digits=3)),
            (:Mres_g, round(1e3 * si_mass(U, m.residual); digits=3)),
            (:umax, round(umax; digits=3)),
            (:zI, zI),
            (:powder, model.powder_jet.enabled),
            (:MLUPS, round(mlups_ema; digits=1)),
        ])
        return nothing
    end

    function do_step!(layer, x_las, powder, sgn, force=false)
        set_laser_position!(model.laser, x_las, y_las)
        jet = model.powder_jet
        jet.enabled = powder && use_powder
        jet.enabled && place_powder_jet!(jet, x_las, y_las, z_noz, z_aim, sgn, dx_noz, :back, Nx)
        t0 = time_ns()
        with_logger(NullLogger()) do
            run!(model, 1)
        end
        dt = (time_ns() - t0) * 1e-9
        mlups = Ncell / max(dt, 1e-12) / 1e6
        mlups_ema = isfinite(mlups_ema) ? αema * mlups + (1 - αema) * mlups_ema : mlups
        istep += 1
        if force || istep % every == 0 || istep == nsteps_total
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
            do_step!(layer, x_now, false, sgn, k == nsteps_pass - 1)
        end
        if layer < n_layers && nsteps_dwell > 0
            model.laser.enabled = false
            fill!(d.Q.data, 0)
            for _ in 1:nsteps_dwell
                do_step!(layer, x_now, false, sgn)
            end
        end
    end
    if nsteps_freeze > 0
        model.laser.enabled = false
        fill!(d.Q.data, 0)
        for _ in 1:nsteps_freeze
            do_step!(n_layers, x_now, false, one(Float32))
        end
    end
    finish!(prog)
    return x_now
end

x_end = run_layers!(model, d, x0, x1, y_las, v_lat, n_layers, bidirectional,
                    nsteps_pass, nsteps_dwell, nsteps_freeze, nsteps_powder_delay, nsteps_total,
                    qevery, every, Nx, Ny, Nz, Hfill, z_noz, z_aim, dx_noz)

LatticeBoltzmann.moments!(model)
nliq, depth, bead, Tmax_K, Tmin_K, umax, u_sol, zI, xl = track_metrics(
    Array(d.fs.data), Array(d.flags.data), Array(d.T.data), Array(d.u.data),
    Nx, Ny, Nz, Hfill, x_end, y_las, model.units)
b = energy_budget(d)
m = mass_budget(d)
U = model.units
dx = Float64(U.m)
@info "LPBF single-track report" nliq depth_cells=depth depth_mm=(1e3*dx*depth) bead_cells=bead bead_mm=(1e3*dx*bead) Tmax_K Tmin_K H_J=si_enthalpy(U, b.H) Q_J=si_enthalpy(U, b.Q) rad_J=si_enthalpy(U, b.rad) evap_J=si_enthalpy(U, b.evap) wall_J=si_enthalpy(U, b.wall) res_J=si_enthalpy(U, b.residual) M_kg=si_mass(U, m.M) Mevap_kg=si_mass(U, m.evap) Mres_kg=si_mass(U, m.residual) umax u_sol zI Hfill x_las=xl box_mm=(1e3*dx*Nx, 1e3*dx*Ny, 1e3*dx*Nz) spot_mm=(1e3*ustrip(u"m", si_d_spot)) t_end=si_t(U, Int(d.t)) P_W=ustrip(u"W", si_P) A0 nskin Q_full
