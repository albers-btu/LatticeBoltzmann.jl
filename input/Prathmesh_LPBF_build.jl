# Prathmesh LPBF layers, scan inset, backing, and gas headroom.
# Loaded by examples/Prathmesh_LPBF.jl after the powder bed is read (`si_gas` uses bed.dz).
# Cell size and the plate box come from input/prathmesh.h5, not from this file.
# Travel is si_scan in Prathmesh_LPBF_laser.jl. This inset keeps that track off
# the x walls: the ray bundle is 4w wide, and w is half of the 80 µm spot.

n_layers      = 1                       # one LPBF track
bidirectional = false                   # always x0 → x1
si_dwell      = 0.0u"s"                 # pause between layers, laser off
si_freeze     = 2.0e-3u"s"              # time after the pass with the laser off
si_end_margin = 0.16e-3u"m"             # clearance from each x wall (bundle half-width, 2 × 80 µm)
si_h_sub      = 2.0e4u"W/m^2/K"         # Robin heat-transfer coefficient of the backing
si_gas        = 40 * bed.dz * u"m"      # gas headroom above the powder
