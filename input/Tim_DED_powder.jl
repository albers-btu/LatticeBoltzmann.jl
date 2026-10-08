# Tim DED powder ring. Included after input/DED_powder.jl.
# Three nozzles, 30° from the vertical, on a circle about the laser axis.
# 120° apart, starting at 90° so none sits on the scan axis (0° or 180°).
# Every jet is aimed at the laser focus. si_mdot from the example is split
# evenly across these three.

si_powder_tilt = 30u"°"
powder_jet_azimuths = (90u"°", 210u"°", 330u"°")
