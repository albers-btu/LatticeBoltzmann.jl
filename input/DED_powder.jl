# DED powder. The pool keeps what melts; cold solid cells shed it on powder_τ.
# Loaded by examples/DED.jl and examples/Tim_DED.jl after `using Unitful`.
# si_mdot is the whole feed. It is split evenly across powder_jet_azimuths.
# tilt is the angle from the vertical. azimuth is around the beam, measured
# from the scan direction: 0° is ahead, 90° is to the left of travel.
# One jet at tilt 0 is coaxial. examples/Tim_DED.jl loads Tim_DED_powder.jl
# after this file and replaces the ring.

use_powder    = true                    # jets on during the pass
si_mdot       = 8.0u"g/minute"          # powder feed rate; the pool keeps what melts
si_d_powder   = 2.4e-3u"m"              # powder stream 1/e² diameter
si_powder_tilt = 0u"°"                  # from the vertical; 0 is straight down
powder_jet_azimuths = (0u"°",)          # one coaxial jet
# Lognormal sphere diameters, same rule as a powder specification:
# median d50, width from d10 and d90, draws rejected outside [dmin, dmax].
# The median stays 75 µm. The window is a coarse DED cut, not a fine LPBF cut.
# si_d_particle is that median and the single-diameter fallback.
si_d10        = 45.0e-6u"m"
si_d50        = 75.0e-6u"m"
si_d90        = 125.0e-6u"m"
si_dmin       = 20.0e-6u"m"
si_dmax       = 150.0e-6u"m"
si_d_particle = si_d50                  # optical diameter; parcel mass is still mdot/nparcels
si_v_powder   = 2.0u"m/s"               # powder particle speed
si_eta        = 1.0                     # reserved bead height uses eta * mdot; the jet feeds mdot
powder_τ      = 2.0e-3u"s"              # time for a cold solid cell to shed unmelted powder
