# Steady 1-D conduction in a plane wall with uniform volumetric generation
# and a specified heat flux (Incropera / Bergman, Fundamentals of Heat and
# Mass Transfer, Ch. 3). Validates Q(x,t) and TYPE_H Neumann/Robin.
#
#   k T'' + Q_vol = 0
#   z* = 0 (TYPE_H):  -k T' = q_in     (heat into the wall)
#   z* = L (TYPE_T):  T = T_L
#   T(z*) = T_L + (q_in/k)(L - z*) + (Q_vol/(2k))(L^2 - z*^2)
#
# D3Q7 Fourier k_lbm = α/2, so Model `α` is twice the physical diffusivity.
#
#   TEMPERATURE = true
using LatticeBoltzmann
using Printf
using CUDA
using Unitful
using ProgressMeter
using Logging

@assert TEMPERATURE
start_run_log!("output_plane_wall")

# --- SI inputs ---
Nx, Ny, Nz = 16, 8, 32
zH, zC = 2, Nz - 1
L = zC - zH                             # conducting cells (TYPE_H → TYPE_T)

si_H   = 0.02u"m"                       # plate thickness
si_ρ   = 2700u"kg/m^3"
si_cp  = 900u"J/kg/K"
si_k   = 20.0u"W/m/K"                   # conductivity (generic metal; Al is ~237)
si_ν   = 1.0e-5u"m^2/s"                 # dummy; no flow
si_T_L = 300.0u"K"                      # Dirichlet face
si_q   = 2.0e5u"W/m^2"                  # flux into the plate
si_Q   = 2.0e6u"W/m^3"                  # volumetric generation

α_si = si_k / (si_ρ * si_cp)            # thermal diffusivity
lbm_α_true = 0.1                        # target lattice diffusivity (stable)
m = ustrip(u"m", si_H) / L
s = lbm_α_true * m^2 / ustrip(u"m^2/s", α_si)
lbm_u = 0.05
si_u = (lbm_u * m / s) * u"m/s"

units = Units(si_H, si_u, si_ρ; x=L, u=lbm_u, ρ=1, T=Float32,
              K=ustrip(u"K", si_T_L), cp=si_cp)

# α-parameter is twice CE diffusivity: thermal_k = α/2.
model = Model(Nx, Ny, Nz, units;
              ν = si_ν,
              α = 2 * α_si,
              β = 0.0f0, gz = 0.0f0,
              backend = CUDABackend())

T_L   = Float32(lbm_T(units, si_T_L))
q_in  = Float32(lbm_q(units, si_q))
Q_vol = Float32(lbm_Q(units, si_Q))
T_avg = T_L
k     = thermal_k(model.domains[1])
@info "plane wall" si_H si_k si_q si_Q si_T_L α_si k q_in Q_vol T_L ν=model.domains[1].ν

host = zeros(UInt8, Nx * Ny * Nz)
Th   = fill(T_avg, Nx * Ny * Nz)
Qh   = zeros(Float32, Nx * Ny * Nz)
for z in 1:Nz, y in 1:Ny, x in 1:Nx
    n = x + (y - 1) * Nx + (z - 1) * Nx * Ny
    if z == 1 || z == Nz
        host[n] = TYPE_S
    elseif z == 2
        host[n] = TYPE_H
        Qh[n] = q_in
        Th[n] = T_L
    elseif z == Nz - 1
        host[n] = TYPE_T
        Th[n] = T_L
    else
        Qh[n] = Q_vol
    end
end
copyto!(model.domains[1].flags.data, host)
copyto!(model.domains[1].T.data, Th)
copyto!(model.domains[1].Q.data, Qh)

k_si = ustrip(u"W/m/K", si_k)
q_si = ustrip(u"W/m^2", si_q)
Q_si = ustrip(u"W/m^3", si_Q)
T0   = ustrip(u"K", si_T_L)
Hsi  = ustrip(u"m", si_H)
analytic(z_m) = T0 + (q_si / k_si) * (Hsi - z_m) + (Q_si / (2 * k_si)) * (Hsi^2 - z_m^2)

d = model.domains[1]
LatticeBoltzmann.initialize!(model)
export!(model; dir="output_plane_wall")

nsteps = 30000
every  = 500
nchunks = nsteps ÷ every
Ncell = Nx * Ny * Nz
mlups_ema = NaN
αema = 0.2
prog = Progress(nchunks; dt=0.2, desc="plane wall ", showspeed=true)
for i in 1:nchunks
    t0 = time_ns()
    with_logger(NullLogger()) do
        run!(model, every)
    end
    dt = (time_ns() - t0) * 1e-9
    mlups = Ncell * every / dt / 1e6
    global mlups_ema = isfinite(mlups_ema) ? αema * mlups + (1 - αema) * mlups_ema : mlups
    export!(model; dir="output_plane_wall")
    next!(prog; showvalues = [
        (:t, Int(d.t)),
        (:t_si, si_t(model.units, Int(d.t))),
        (:MLUPS, round(mlups_ema; digits=1)),
    ])
end
finish!(prog)

LatticeBoltzmann.moments!(model)
T = Array(d.T.data)
x0, y0 = Nx ÷ 2, Ny ÷ 2
let err2 = 0.0, nerr = 0, max_err = 0.0, Tmid_num = 0.0, Tmid_an = 0.0
    for z in zH:zC
        n = x0 + (y0 - 1) * Nx + (z - 1) * Nx * Ny
        z_m = Float64(z - zH) * Float64(model.units.m)
        Tan = analytic(z_m)
        Tnum = Float64(si_T(model.units, T[n]))
        e = abs(Tnum - Tan)
        err2 += e^2
        nerr += 1
        max_err = max(max_err, e)
        if z == (zH + zC) ÷ 2
            Tmid_num = Tnum
            Tmid_an = Tan
        end
    end
    L2 = sqrt(err2 / nerr)
    @info "plane wall vs analytic (K)" Tmid_num Tmid_an max_err L2 Fo=(k * nsteps / Float32(L)^2)
end
