# Prathmesh LPBF laser and travel. Line energy is P/v (here 100 W / 0.45 m/s ≈ 0.222 J/mm).
# Loaded by examples/Prathmesh_LPBF.jl after input/material_316L.jl.
# input/prathmesh.h5 is 1.5 mm in x. si_scan is that length minus the 0.16 mm
# clearance at each end (Prathmesh_LPBF_build.jl), plus one cell.
# The solver has no wavelength field. si_λ is the process wavelength; n and k below
# are what Fresnel uses. Included after material_316L.jl so they replace that file's
# n = 3.27, k = 4.48 (normal-incidence absorptance ≈ 0.341). k/n stays 4.48/3.27.
# fresnel_absorptance(1, n, k) is then exactly 0.6 in Float32.

si_P      = 100.0u"W"                   # laser power
si_d_spot = 80.0e-6u"m"                 # laser 1/e² spot diameter
si_v      = 0.45u"m/s"                  # scan speed
si_scan   = 1.1849342105263157e-3u"m"   # laser travel along x on prathmesh.h5
si_λ      = 1070.0e-9u"m"               # wavelength (documented; Fresnel uses n, k)
si_δ      = 30.0e-6u"m"                 # laser absorption depth (about one powder diameter)
fresnel_n = 1.3679824f0                 # real refractive index at 1070 nm
fresnel_k = 1.8741778f0                 # extinction coefficient at 1070 nm
