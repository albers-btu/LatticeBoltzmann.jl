using Test, LatticeBoltzmann

@testset "DdQq velocities" begin
	v = VELOCITIES[:D2Q9]

	@test v[1] == SVector( 0,  0,  0)
	@test v[2][1] == 1

	v5 = VELOCITIES[:D2Q5]
	@test length(v5) == 5
	@test all(c -> c[3] == 0, v5)
	@test v5[2] == -v5[3]
	@test v5[4] == -v5[5]
	@test v5[2] == SVector( 1,  0,  0)
	@test v5[3] == SVector(-1,  0,  0)
	@test v5[4] == SVector( 0,  1,  0)
	@test v5[5] == SVector( 0, -1,  0)
end