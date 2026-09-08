using LatticeBoltzmann: run
using Test, StaticArrays

@testset "LatticeBoltzmann.jl" begin
	include("test_weights.jl")
	include("test_velocities.jl")
	include("test_kernel.jl")
	include("test_memory.jl")
	include("test_model.jl")

	model = Model(256, 256, 256, 1.0)

	mlups::UInt = 0
	for i in 1:1 # 1000
		run(model, 10)
		mlups = max(mlups, 0)
		@info "$mlups MLUPS"
	end
end