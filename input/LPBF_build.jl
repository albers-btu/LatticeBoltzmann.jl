# LPBF layers, scan inset, backing, and gas headroom.
# Loaded by examples/LPBF.jl after the powder bed is read (`si_gas` uses bed.dz).
# Cell size and the plate box come from input/powder_bed.h5, not from this file.

n_layers      = 1                       # one LPBF track
bidirectional = false                   # always x0 → x1
si_dwell      = 0.0u"s"                 # pause between layers, laser off
si_freeze     = 2.0e-3u"s"              # time after the pass with the laser off
si_end_margin = 0.15e-3u"m"             # scan start/stop inset from the x walls (spot radius is 50 µm)
si_h_sub      = 2.0e4u"W/m^2/K"         # Robin heat-transfer coefficient of the backing
si_gas        = 40 * bed.dz * u"m"      # gas headroom above the powder (~0.20 mm)
