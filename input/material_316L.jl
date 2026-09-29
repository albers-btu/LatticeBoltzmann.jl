# 316L, shared by examples/DED.jl and examples/LPBF.jl.
# Loaded after `using Unitful`.
# k(T) = k(Tm) + kT (T − Tm). k_s is at Tm so k(T_init) stays ~15 W/m/K.

si_ρ       = 8000u"kg/m^3"              # density
si_cp      = 500u"J/kg/K"               # specific heat capacity at Tm
si_cp_sT   = 0.20u"J/kg/K^2"            # d(cp)/dT of the solid; cp(T) = cp + cpT (T − Tm)
si_cp_lT   = 0.08u"J/kg/K^2"            # d(cp)/dT of the liquid
si_Tm      = 1673.0u"K"                 # melting temperature (solidus = liquidus)
si_T_init  = 300.0u"K"                  # initial temperature, and the cold backing
si_k_sT    = 0.013u"W/m/K^2"            # dk/dT of the solid
si_k_lT    = 0.005u"W/m/K^2"            # dk/dT of the liquid
si_k_s     = 15.0u"W/m/K" + si_k_sT * (si_Tm - si_T_init)  # solid thermal conductivity at Tm
si_k_l     = 30.0u"W/m/K"               # liquid thermal conductivity at Tm
si_Lheat   = 2.8e5u"J/kg"               # latent heat of fusion
si_Lv      = 7.45e6u"J/kg"              # latent heat of vaporization
si_Tv      = 3086.0u"K"                 # boiling temperature
si_M       = 0.0558u"kg/mol"            # molar mass (evaporation pressure)
si_K0      = 1.0e-10u"m^2"              # Kozeny–Carman permeability of the mush
si_g       = 9.81u"m/s^2"               # gravity
si_β       = 1.2e-4u"K^-1"              # volumetric thermal expansion coefficient
# 316L near Tm is ~1.6 N/m and dσ/dT ~ −4.3e-4 N/m/K.
si_σ_phys  = 1.6u"N/m"                  # surface tension at Tm
si_σT_phys = -4.3e-4u"N/m/K"            # dσ/dT (Marangoni)
si_ν_l     = 6.0e-7u"m^2/s"             # liquid kinematic viscosity
si_ν_s     = 1.0e-4u"m^2/s"             # solid kinematic viscosity (large: the solid does not flow)
si_ν_lT    = -2.0e-10u"m^2/s/K"         # dν/dT of the liquid
emissivity = 0.4                        # surface emissivity (radiation to the cold backing)
fresnel_n  = 3.27f0                     # real refractive index (laser absorptance)
fresnel_k  = 4.48f0                     # extinction coefficient (laser absorptance)
