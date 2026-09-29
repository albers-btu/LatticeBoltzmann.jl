# DED coaxial powder. The pool keeps what melts; cold cells shed it on powder_τ.
# Loaded by examples/DED.jl after `using Unitful`.

use_powder  = true                      # coaxial jet on during the pass
si_mdot     = 8.0u"g/minute"            # powder feed rate; the pool keeps what melts
si_d_powder = 2.4e-3u"m"                # powder stream 1/e² diameter
si_v_powder = 2.0u"m/s"                 # powder particle speed
si_eta      = 1.0                       # fraction of the feed that sticks (1 = all of it)
powder_τ    = 2.0e-3u"s"                # time for a cold cell to shed unmelted powder
