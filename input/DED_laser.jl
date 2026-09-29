# DED laser and travel speed. Line energy is P/v (here 100 J/mm).
# Loaded by examples/DED.jl after `using Unitful`.

si_P      = 1000.0u"W"                  # laser power (100 J/mm at 10 mm/s)
si_d_spot = 2.0e-3u"m"                  # laser 1/e² spot diameter
si_v      = 10.0e-3u"m/s"               # scan speed
si_δ      = 0.15e-3u"m"                 # laser absorption depth
