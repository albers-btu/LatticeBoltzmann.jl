# LPBF powder feed. The track's powder is already in input/powder_bed.h5.
# Loaded by examples/LPBF.jl after `using Unitful`.

use_powder = false                      # powder is already in the HDF5 bed, not a jet
si_mdot    = 0.0u"g/minute"             # powder feed rate (off: the bed is already there)
si_eta     = 1.0                        # fraction of the feed that sticks (unused while mdot = 0)
powder_τ   = 0.0u"s"                    # shedding time (jet is off)
