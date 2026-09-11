using LatticeBoltzmann
using Test, StaticArrays, CUDA

@testset "LatticeBoltzmann.jl" begin
	include("test_weights.jl")
	include("test_velocities.jl")
	include("test_kernel.jl")
	include("test_memory.jl")
	include("test_model.jl")
	include("test_plic.jl")
	include("test_equilibrium.jl")
	include("test_moving.jl")
	include("test_force_field.jl")

	model = Model(64, 64, 64, 1.0; backend=CUDABackend())
	#model = Model(64, 64, 64, 1.0)

	# set boundaries for box with all walls solid
	Nx, Ny, Nz = Int(model.Nx), Int(model.Ny), Int(model.Nz)
	host = zeros(UInt8, Nx * Ny * Nz)
	for z in 1:Nz, y in 1:Ny, x in 1:Nx
		if x == 1 || x == Nx || y == 1 || y == Ny || z == 1 || z == Nz
			n = x + (y-1)*Nx + (z-1)*Nx*Ny
			host[n] = TYPE_S
		end
	end
	copyto!(model.domains[1].flags.data, host)

	for i in 1:1 # 1000
		run!(model, 10)
	end
end