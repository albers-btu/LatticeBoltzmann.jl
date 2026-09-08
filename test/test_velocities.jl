using Test, LatticeBoltzmann

@testset "DdQq velocities" begin
	v = VELOCITIES[:d2q9]

	@test v[1] == SVector( 0,  0,  0)
	@test v[2][1] == 1
end