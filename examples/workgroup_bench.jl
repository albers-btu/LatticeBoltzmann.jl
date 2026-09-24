# Sweep GPU (or CPU) workgroup size on one melt-pool-like step:
# free surface + D3Q19 KBC + D3Q7 heat, n_hydro = 1.
#
#   julia --threads 8 --project=. examples/workgroup_bench.jl
#   julia --threads 8 --project=. examples/workgroup_bench.jl cpu
#
# MLUPS counts lattice updates per second for that one step, not the
# n_hydro substeps of examples/melt_pool.jl. The printed default is what
# Model uses if you do not pass workgroup=.

using LatticeBoltzmann
using CUDA
using KernelAbstractions
using Printf

use_cpu = any(==("cpu"), ARGS)
backend = (!use_cpu && CUDA.functional()) ? CUDABackend() : CPU()
const N = backend isa CUDABackend ? 128 : 64
const NWARM = 2
const NSTEP = 8

if backend isa CUDABackend
    dev = CUDA.device()
    maxthr = CUDA.attribute(dev, CUDA.DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK)
    println("device: ", CUDA.name(dev), "   max threads/block: ", maxthr)
    sizes = (32, 64, 128, 256, 512, 1024)
else
    println(use_cpu ? "CPU backend (requested)" : "CUDA not available, timing the CPU backend")
    sizes = (16, 32, 64, 128, 256)
end
println("default workgroup: ", LatticeBoltzmann.default_workgroup(backend))
println("grid: ", N, "³   warm-up ", NWARM, "   timed ", NSTEP)

function bench_model(wg)
    model = Model(N, N, N, 0.1f0;
                  α=0.05f0, σ=0.01f0, σT=-1.0f-4, fz=-1.0f-5,
                  Λ=0.2f0, Ts=1.0f0, Tl=1.0f0,
                  backend=backend, workgroup=wg, n_hydro=1)
    Nx = Ny = Nz = N
    host = fill(TYPE_G, Nx * Ny * Nz)
    for z in 1:Nz, y in 1:Ny, x in 1:Nx
        n = x + (y - 1) * Nx + (z - 1) * Nx * Ny
        wall = x == 1 || x == Nx || y == 1 || y == Ny || z == 1
        host[n] = wall ? TYPE_S : (z <= (2 * Nz) ÷ 3 ? TYPE_F : TYPE_G)
    end
    copyto!(model.domains[1].flags.data, host)
    initialize!(model)
    return model
end

function sweep(sizes)
    println()
    @printf("%10s %12s %10s\n", "workgroup", "MLUPS", "ms/step")
    best_wg, best_mlups = 0, -1.0
    for wg in sizes
        local mlups, ms
        try
            model = bench_model(wg)
            run!(model, NWARM)
            KernelAbstractions.synchronize(backend)
            t = @elapsed begin
                run!(model, NSTEP)
                KernelAbstractions.synchronize(backend)
            end
            mlups = (N * N * N) * NSTEP / t / 1e6
            ms = 1000 * t / NSTEP
        catch err
            @printf("%10d %12s %10s\n", wg, "fail", "")
            println("    ", sprint(showerror, err)[1:min(end, 180)])
            continue
        end
        @printf("%10d %12.1f %10.2f\n", wg, mlups, ms)
        if mlups > best_mlups
            best_wg = wg
            best_mlups = mlups
        end
    end
    println()
    println("best workgroup: ", best_wg, "   (", round(best_mlups; digits=1), " MLUPS)")
    println("pass it as Model(..., workgroup=", best_wg, ")")
end

sweep(sizes)
