using Test
using LatticeBoltzmann

# Rational so Σ s_i == 0 and Σ s_i == -(9/2) Πzz / 3 are exact.
const RT = Rational{Int}

@testset "2D KBC moment basis" begin
    w = WEIGHTS[:D2Q9]
    c = VELOCITIES[:D2Q9]
    z0 = zero(RT)

    # Non-equilibrium D2Q9 populations. Every cz is 0.
    f = collect(w)
    f[2] += 1//50
    f[3] += 1//20
    f[4] -= 1//40
    f[6] += 1//80

    ρn = f[1]
    ux = z0
    uy = z0
    uz = z0
    NP = 4
    pairs = ntuple(NP) do k
        i = 2k
        (f[i], f[i + 1])
    end
    for k in 1:NP
        i = 2k
        fp, fm = pairs[k]
        ρn += fp + fm
        ux += RT(c[i][1]) * fp + RT(c[i + 1][1]) * fm
        uy += RT(c[i][2]) * fp + RT(c[i + 1][2]) * fm
        uz += RT(c[i][3]) * fp + RT(c[i + 1][3]) * fm
    end
    invρ = one(RT) / ρn
    ux *= invρ
    uy *= invρ
    uz *= invρ
    uu = RT(3) // 2 * (ux * ux + uy * uy + uz * uz)

    # Same accumulators as the 3D store: dp*cα*cβ + dm*cα*cβ, including cz².
    Πxx = Πyy = Πzz = Πxy = Πxz = Πyz = z0
    for k in 1:NP
        i = 2k
        fp, fm = pairs[k]
        feqp = LatticeBoltzmann.feq(w[i], ρn, ux, uy, uz, uu, c[i], RT)
        feqm = LatticeBoltzmann.feq(w[i + 1], ρn, ux, uy, uz, uu, c[i + 1], RT)
        dp = fp - feqp
        dm = fm - feqm
        cxp = RT(c[i][1]); cyp = RT(c[i][2]); czp = RT(c[i][3])
        cxm = RT(c[i + 1][1]); cym = RT(c[i + 1][2]); czm = RT(c[i + 1][3])
        Πxx += dp * cxp * cxp + dm * cxm * cxm
        Πyy += dp * cyp * cyp + dm * cym * cym
        Πzz += dp * czp * czp + dm * czm * czm
        Πxy += dp * cxp * cyp + dm * cxm * cym
        Πxz += dp * cxp * czp + dm * cxm * czm
        Πyz += dp * cyp * czp + dm * cym * czm
    end

    @test Πzz == 0
    @test Πxz == 0
    @test Πyz == 0
    @test !(Πxx == 0 && Πyy == 0 && Πxy == 0)

    Σs2 = Σcx2 = Σcy2 = Σcz2 = z0
    Σs3 = Σcx3 = Σcy3 = Σcz3 = z0
    for i in eachindex(w)
        cx = RT(c[i][1])
        cy = RT(c[i][2])
        cz = z0
        s2 = LatticeBoltzmann.kbc_shear_2d(w[i], cx, cy, Πxx, Πyy, Πxy)
        s3 = LatticeBoltzmann.kbc_shear(w[i], cx, cy, cz, Πxx, Πyy, Πzz, Πxy, Πxz, Πyz)
        Σs2 += s2
        Σcx2 += cx * s2
        Σcy2 += cy * s2
        Σcz2 += cz * s2
        Σs3 += s3
        Σcx3 += cx * s3
        Σcy3 += cy * s3
        Σcz3 += cz * s3
    end
    @test Σs2 == 0
    @test Σcx2 == 0 && Σcy2 == 0 && Σcz2 == 0
    @test Σs3 == 0
    @test Σcx3 == 0 && Σcy3 == 0 && Σcz3 == 0

    # Injected Πzz, other deviatoric moments 0, cz = 0.
    # Σ_i s_i = (9/2) * Πzz * Σ w_i (0 - 1/3) = -(9/2) * Πzz / 3.
    Πzz_inj = 2//7
    Σs = z0
    for i in eachindex(w)
        si = LatticeBoltzmann.kbc_shear(w[i], RT(c[i][1]), RT(c[i][2]), z0,
                                         z0, z0, Πzz_inj, z0, z0, z0)
        Σs += si
    end
    @test Σs == -(9//2) * Πzz_inj / 3

    # Compile the 2D store. It does not return moments.
    C = Float64
    N = 1
    fi = zeros(C, length(w))
    ff = C.(f)
    for i in eachindex(ff)
        fi[LatticeBoltzmann.f_index(1, i, N)] = ff[i]
    end
    wf = ntuple(i -> C(w[i]), length(w))
    cf = ntuple(i -> c[i], length(c))
    pairsf = ntuple(k -> (ff[2k], ff[2k + 1]), NP)
    LatticeBoltzmann.kbc_store!(
        Val(false), fi, ff[1], pairsf, wf, cf,
        C(ρn), C(ux), C(uy), C(uz), C(uu),
        zero(C), zero(C), zero(C), one(C),
        N, 1, 1, 1, 1, 0, 0, 0, C,
    )
    @test all(isfinite, fi)
end
