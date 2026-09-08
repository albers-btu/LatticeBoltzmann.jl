using LatticeBoltzmann
using Test
using StaticArrays

@testset "LatticeBoltzmann.jl" begin
	w = WEIGHTS[:d2q9]

	@test w[1] == 4//9
	@test w[6] == 1//36

	v = VELOCITIES[:d2q9]

	@test v[1] == SVector( 0,  0,  0)
	@test v[2][1] == 1

	include("test_kernel.jl")
	include("test_domain.jl")
	include("test_memory.jl")
end