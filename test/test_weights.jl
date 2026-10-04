using Test, LatticeBoltzmann

@testset "DdQq weights" begin
	w = WEIGHTS[:D2Q9]

	@test w[1] == 4//9
	@test w[6] == 1//36

	@test sum(w * (0 - 1//3) for w in WEIGHTS[:D2Q9]) == -1//3

	w5 = WEIGHTS[:D2Q5]
	@test sum(w5) == 1
	@test w5[1] == 1//3
	@test w5[2] == 1//6
	@test w5[3] == 1//6
	@test w5[4] == 1//6
	@test w5[5] == 1//6

	@static if DIM == 3
		@test SCHEME == :D3Q19
		@test SCHEME_T == :D3Q7
	end
	@static if DIM == 2
		@test SCHEME == :D2Q9
		@test SCHEME_T == :D2Q5
	end
end