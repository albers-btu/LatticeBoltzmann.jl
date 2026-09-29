# DED plate, mesh, layers, and the cold backing.
# Loaded by examples/DED.jl after `using Unitful`.
# 12.8 × 6.4 mm, 3.2 mm plate, 2.4 mm of gas for the bead and the coaxial
# flight. Refine with `si_dx` only. This example does not read a powder bed.

n_layers      = 1                       # one weld line
bidirectional = false                   # always x0 → x1
si_dwell      = 0.0u"s"                 # pause between layers, laser off
si_freeze     = 0.2u"s"                 # time after the pass with laser and powder off
si_Lx         = 12.8e-3u"m"             # domain length (scan direction)
si_Ly         = 6.4e-3u"m"              # domain width
si_H          = 3.2e-3u"m"              # substrate thickness (z = 2 … Hfill)
si_gas        = 2.4e-3u"m"              # gas headroom above the plate (bead and nozzle flight)
si_dx         = 80.0e-6u"m"             # cell size
si_end_margin = 2.0e-3u"m"              # scan start/stop inset from the x walls (spot radius is 1 mm)
si_h_sub      = 2.0e4u"W/m^2/K"         # Robin heat-transfer coefficient of the backing
