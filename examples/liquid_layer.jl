# Steady conduction in a still liquid layer with volumetric generation and an
# adiabatic free surface (Incropera / Bergman, plane wall with generation,
# one face isothermal, the other insulated).
#
#   k T'' + Q_vol = 0
#   z* = 0 (bottom TYPE_T):  T = T_L
#   z* = H (free surface):   T' = 0     (gas bounce-back of g)
#   T(z*) = T_L + (Q_vol/k) (H z* - z*^2 / 2)
#
# SURFACE + TEMPERATURE. Edit the SI block. Gas is skipped; T lives in F/I.
#
#   SURFACE = true, TEMPERATURE = true, VOLUME_FORCE = true
using LatticeBoltzmann
using Printf
using CUDA
using Unitful
using ProgressMeter
using Logging

@assert SURFACE && TEMPERATURE
start_run_log!("output_liquid_layer")

# --- SI inputs ---
Nx, Ny, Nz = 16, 8, 32
zB = 2                                  # Dirichlet fluid layer
Hfill = 2 * Nz ÷ 3                      # last TYPE_F cell (interface forms above)
L = Hfill - zB                          # conducting liquid cells

si_H   = 0.02u"m"                       # liquid depth (bottom plate → free surface)
si_ρ   = 2700u"kg/m^3"
si_cp  = 900u"J/kg/K"
si_k   = 20.0u"W/m/K"
si_ν   = 1.0e-5u"m^2/s"
si_T_L = 300.0u"K"
si_Q   = 2.0e6u"W/m^3"                  # uniform generation in the liquid

α_si = si_k / (si_ρ * si_cp)
lbm_α_true = 0.1
m = ustrip(u"m", si_H) / L
s = lbm_α_true * m^2 / ustrip(u"m^2/s", α_si)
lbm_u = 0.05
si_u = (lbm_u * m / s) * u"m/s"

units = Units(si_H, si_u, si_ρ; x=L, u=lbm_u, ρ=1, T=Float32,
              K=ustrip(u"K", si_T_L), cp=si_cp)

model = Model(Nx, Ny, Nz, units;
              ν = si_ν,
              α = 2 * α_si,
              σ = 0,
              β = 0.0f0, gz = 0.0f0,
              backend = CUDABackend())

T_L   = Float32(lbm_T(units, si_T_L))
Q_vol = Float32(lbm_Q(units, si_Q))
k     = thermal_k(model.domains[1])
@info "liquid layer" si_H si_k si_Q si_T_L α_si k Q_vol T_L Hfill L

host = zeros(UInt8, Nx * Ny * Nz)
Th   = fill(T_L, Nx * Ny * Nz)
Qh   = zeros(Float32, Nx * Ny * Nz)
for z in 1:Nz, y in 1:Ny, x in 1:Nx
    n = x + (y - 1) * Nx + (z - 1) * Nx * Ny
    if z == 1 || z == Nz || x == 1 || x == Nx || y == 1 || y == Ny
        host[n] = TYPE_S
    elseif z == zB
        host[n] = TYPE_T
        Th[n] = T_L
        Qh[n] = Q_vol
    elseif z <= Hfill
        host[n] = TYPE_F
        Qh[n] = Q_vol
    end
end
copyto!(model.domains[1].flags.data, host)
copyto!(model.domains[1].T.data, Th)
copyto!(model.domains[1].Q.data, Qh)

k_si = ustrip(u"W/m/K", si_k)
Q_si = ustrip(u"W/m^3", si_Q)
T0   = ustrip(u"K", si_T_L)
Hsi  = ustrip(u"m", si_H)
analytic(z_m) = T0 + (Q_si / k_si) * (Hsi * z_m - z_m^2 / 2)

d = model.domains[1]
LatticeBoltzmann.initialize!(model)
export!(model; dir="output_liquid_layer")

nsteps = 30000
every  = 500
nchunks = nsteps ÷ every
Ncell = Nx * Ny * Nz
mlups_ema = NaN
αema = 0.2
prog = Progress(nchunks; dt=0.2, desc="liquid layer ", showspeed=true)
for i in 1:nchunks
    t0 = time_ns()
    with_logger(NullLogger()) do
        run!(model, every)
    end
    dt = (time_ns() - t0) * 1e-9
    mlups = Ncell * every / dt / 1e6
    global mlups_ema = isfinite(mlups_ema) ? αema * mlups + (1 - αema) * mlups_ema : mlups
    export!(model; dir="output_liquid_layer")
    next!(prog; showvalues = [
        (:t, Int(d.t)),
        (:t_si, si_t(model.units, Int(d.t))),
        (:MLUPS, round(mlups_ema; digits=1)),
    ])
end
finish!(prog)

LatticeBoltzmann.moments!(model)
T = Array(d.T.data)
flags = Array(d.flags.data)
x0, y0 = Nx ÷ 2, Ny ÷ 2
let err2 = 0.0, nerr = 0, max_err = 0.0, Tmid_num = 0.0, Tmid_an = 0.0
    zmid = (zB + Hfill) ÷ 2
    for z in zB:Hfill
        n = x0 + (y0 - 1) * Nx + (z - 1) * Nx * Ny
        (flags[n] & TYPE_SU) == TYPE_G && continue
        z_m = Float64(z - zB) * Float64(model.units.m)
        Tan = analytic(z_m)
        Tnum = Float64(si_T(model.units, T[n]))
        e = abs(Tnum - Tan)
        err2 += e^2
        nerr += 1
        max_err = max(max_err, e)
        if z == zmid
            Tmid_num = Tnum
            Tmid_an = Tan
        end
    end
    L2 = sqrt(err2 / max(nerr, 1))
    @info "liquid layer vs analytic (K)" Tmid_num Tmid_an max_err L2 nerr Fo=(k * nsteps / Float32(L)^2)
end
