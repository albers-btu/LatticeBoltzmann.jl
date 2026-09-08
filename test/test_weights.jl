using Test, LatticeBoltzmann

@testset "DdQq weights" begin
	w = WEIGHTS[:d2q9]

	@test w[1] == 4//9
	@test w[6] == 1//36
end