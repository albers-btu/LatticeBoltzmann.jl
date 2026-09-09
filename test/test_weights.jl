using Test, LatticeBoltzmann

@testset "DdQq weights" begin
	w = WEIGHTS[:D2Q9]

	@test w[1] == 4//9
	@test w[6] == 1//36
end