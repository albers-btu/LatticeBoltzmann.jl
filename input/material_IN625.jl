# IN625
# Thermophysical properties based on literature/JMatPro data.
# Solidus = 1563 K, Liquidus = 1623 K.

si_ρ       = 8440u"kg/m^3"              # density
si_cp      = 677u"J/kg/K"               # specific heat capacity at Tm
si_cp_sT   = 0.13u"J/kg/K^2"            # d(cp)/dT of the solid; cp(T) = cp + cpT (T − Tm)
si_cp_lT   = 0.00u"J/kg/K^2"            # d(cp)/dT of the liquid
si_Tm      = 1623.0u"K"                 # melting temperature (solidus = liquidus)
si_T_init  = 300.0u"K"                  # initial temperature, and the cold backing
si_k_sT    = 0.00628u"W/m/K^2"          # dk/dT of the solid
si_k_lT    = 0.00u"W/m/K^2"             # dk/dT of the liquid
si_k_s     = 28.8u"W/m/K" + si_k_sT * (si_Tm - si_T_init)  # solid thermal conductivity at Tm
si_k_l     = 30.1u"W/m/K"               # liquid thermal conductivity at Tm
si_Lheat   = 2.72e5u"J/kg"              # latent heat of fusion
si_Lv      = 6.40e6u"J/kg"              # Ni vaporization ≈ 6.4 MJ/kg (≈ 377 kJ/mol)
si_Tv      = 3000.0u"K"                 # boiling temperature
si_M       = 0.05869u"kg/mol"           # molar mass (evaporation pressure)
si_K0      = 1.0e-10u"m^2"              # Kozeny–Carman permeability of the mush
si_g       = 9.81u"m/s^2"               # gravity
si_β       = 5.0e-5u"K^-1"              # volumetric thermal expansion coefficient
si_σ_phys  = 1.8u"N/m"                  # surface tension at Tm
si_σT_phys = -2.0e-5u"N/m/K"            # dσ/dT (Marangoni)
si_ν_l     = 8.3e-7u"m^2/s"             # liquid kinematic viscosity
si_ν_s     = 1.0e-4u"m^2/s"             # solid kinematic viscosity (large: the solid does not flow)
si_ν_lT    = 0.0u"m^2/s/K"              # dν/dT of the liquid
emissivity = 0.4                        # surface emissivity (radiation to the cold backing)
fresnel_n  = 3.97f0                     # real refractive index (laser absorptance)
fresnel_k  = 5.40f0                     # extinction coefficient (laser absorptance)