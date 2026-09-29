# LPBF laser and travel speed. Line energy is P/v (here 0.2 J/mm).
# Loaded by examples/LPBF.jl after `using Unitful`.

si_P      = 200.0u"W"                   # laser power (0.2 J/mm at 1 m/s)
si_d_spot = 100.0e-6u"m"                # laser 1/e² spot diameter
si_v      = 1.0u"m/s"                   # scan speed
si_δ      = 30.0e-6u"m"                 # laser absorption depth (about one d50)
