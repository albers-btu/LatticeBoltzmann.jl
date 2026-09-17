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
	include("test_temperature.jl")
	include("test_bubble.jl")
	include("test_dissolved.jl")
	include("test_nucleation.jl")
	include("test_blowing_agent.jl")
	include("test_crucible_foam.jl")
	include("test_disjoining.jl")
	include("test_closed_foam.jl")
	include("test_capillary.jl")
	include("test_laser_opaque.jl")
	include("test_laser_cavity.jl")
	include("test_laser_pool.jl")
	include("test_laser_nucleate.jl")
	include("test_laser_agent.jl")
	include("test_laser_powder.jl")
	include("test_laser_foam_pad.jl")
	include("test_ded_track.jl")

	model = Model(64, 64, 64, 1.0; backend=CUDABackend())
	#model = Model(64, 64, 64, 1.0)

	# set boundaries for box with all walls solid
	Nx, Ny, Nz = Int(model.Nx), Int(model.Ny), Int(model.Nz)
	host = zeros(UInt8, Nx * Ny * Nz)
	for z in 1:Nz, y in 1:Ny, x in 1:Nx
		n = x + (y-1)*Nx + (z-1)*Nx*Ny
		if x == 1 || x == Nx || y == 1 || y == Ny || z == 1 || z == Nz
			host[n] = TYPE_S
		else
			host[n] = TYPE_F
		end
	end
	copyto!(model.domains[1].flags.data, host)

	for i in 1:1 # 1000
		run!(model, 10)
	end
end