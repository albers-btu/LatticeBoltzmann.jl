# LBM_Saclay and this solver

Overview of what [LBM_Saclay 1.0](https://cea-lbm-saclay.github.io/LBM_Saclay_Documentation/index.html) (CEA/STMF, documentation dated 24 June 2026) offers, and what this Julia solver still lacks if the aim is to compete with that code on multiphase CFD.

The Saclay C++ sources sit on an access-controlled CEA git server (`codev-tuleap.cea.fr`). This note uses the public Sphinx manual, which quotes the kernels it says are compiled. Statements about LatticeBoltzmann.jl are from `src/` on `main` (`5d9b886`).

The two codes do not discretize the same interface. Saclay captures a diffuse interface with a phase field and solves both fluids. This solver captures a sharp free surface (Körner): gas is a boundary condition, and the product is a melt pool with a laser and a powder jet. Matching Saclay means adding a different model, not filling a few missing flags.

## What LBM_Saclay is

LBM_Saclay is a C++ CFD code for multiphase and multicomponent flow. The interface comes from phase-field theory (Cahn–Hilliard and Allen–Cahn), coupled either to incompressible Navier–Stokes or to a low-Mach Navier–Stokes/Korteweg model. The same source builds for OpenMP on CPU or for CUDA on GPU through Kokkos. The manual describes multi-GPU jobs on ORCUS (V100, A100, H100) and Topaze. One published H100 multi-GPU configure transcript prints “MPI not enabled”, so a multi-GPU binary is not the same thing as an MPI domain decomposition. One executable is one problem: `configure_build.sh` asks which kernels to compile.

The training menu in the [quick start](https://cea-lbm-saclay.github.io/LBM_Saclay_Documentation/src_doc/01_USER_GUIDE/QUICKSTART/quickstart.html) is:

| # | Kernel |
|---|---|
| 0 | `AC` |
| 1 | `Advection-Diffusion` |
| 2 | `Crystal_growth_Younsi` |
| 3 | `GPMixt` |
| 4 | `GPMixtNS` |
| 5 | `GPMixtTernary` |
| 6 | `GPMuTernary` |
| 7 | `MPwSLphC` |
| 8 | `NS` |
| 9 | `NS_3phases_1comp_phase_change` |
| 10 | `NSAC_Comp` |
| 11 | `NSAC_Comp_3phases` |
| 12 | `NSAC_Comp_3phases3D` |
| 13 | `NSAC_coupling` |
| 14 | `NSAC_Fakhari` |
| 15 | `NSAC_Surfactant` |

`NSAC_Comp` is the default training kernel. Cases live in `run_training_lbm` and are driven by a `.ini` file (`problem=NSAC_Comp`). Output is `.vti` and HDF5/XDMF for ParaView. A documented capillary-wave case is compared with Prosperetti’s solution at density ratio 2, and the manual points to a separate comparison near density ratio 830.

## Models they document as implemented

[PART II](https://cea-lbm-saclay.github.io/LBM_Saclay_Documentation/src_doc/02_MODELS/models.html) is the list of models in the code. PART V is a course and is not counted here as a kernel.

### Two incompressible fluids and a composition

[`NSAC_Comp`](https://cea-lbm-saclay.github.io/LBM_Saclay_Documentation/src_doc/02_MODELS/01_Fluid_Fluid/Model_NSAC_Comp.html) solves

\[
\nabla\cdot\mathbf{u}=0,
\]

a momentum equation with a diffuse-interface force, the conservative Allen–Cahn equation for the phase field \(\phi\in[0,1]\), and a composition equation for \(c\). The capillary force is the chemical-potential form

\[
\mathbf{F}_c=\mu_\phi\nabla\phi,\qquad
\mu_\phi=\frac{3}{2}\sigma\left[\frac{16}{W}\phi(1-\phi)(1-2\phi)-W\nabla^2\phi\right],
\]

with surface tension \(\sigma\) and interface width \(W\). Gravity is \(\mathbf{F}_g=\varrho(\phi,c)\,\mathbf{g}\). Marangoni comes from composition, not from temperature:

\[
\mathbf{F}_M=\frac{3W}{2}\Big[\nabla\sigma\,|\nabla\phi|^2-\nabla\phi\,(\nabla\phi\cdot\nabla\sigma)\Big],\qquad
\sigma(c)=\sigma_{\mathrm{ref}}+\sigma_c(c-c_{\mathrm{ref}}).
\]

Density and viscosity are interpolations of the two bulk values. The same input file can switch the interface PDE toward Cahn–Hilliard (`cahn_hilliard`) or the conservative counter-term (`counter_term`). The manual ties Cahn–Hilliard to a spinodal test and Allen–Cahn with phase change to a Stefan test.

### Liquid–gas phase change on that interface

The [phase-change kernel](https://cea-lbm-saclay.github.io/LBM_Saclay_Documentation/src_doc/02_MODELS/01_Fluid_Fluid/Model_NSAC_Comp_PhaseChange.html) keeps incompressible momentum, replaces composition by temperature, and puts a volumetric mass source in the divergence:

\[
\nabla\cdot\mathbf{u}=\dot{m}'''\left(\frac{1}{\rho_g}-\frac{1}{\rho_l}\right),\qquad
\frac{\dot{m}'''}{\rho_g}=-\frac{4\alpha}{\mathscr{A}W^2}(\theta_I-\theta)\,\phi(1-\phi).
\]

The temperature equation carries the latent heat of the moving phase field:

\[
\partial_t T+\nabla\cdot(\mathbf{u}T)=\alpha\nabla^2 T-\frac{\mathcal{L}}{\mathcal{C}_p}\big[\partial_t\phi+\nabla\cdot(\mathbf{u}\phi)\big].
\]

The force is capillary plus gravity. The page names Marangoni in the force list and then defines \(\mathbf{F}_{\mathrm{tot}}=\mathbf{F}_c+\mathbf{F}_g\).

### Surfactant, three fluids, fluid against a solid

- [`NSAC_Surfactant`](https://cea-lbm-saclay.github.io/LBM_Saclay_Documentation/src_doc/02_MODELS/models.html) is a compiled kernel and a documented model. This note did not re-derive its PDE page.
- [Three immiscible fluids](https://cea-lbm-saclay.github.io/LBM_Saclay_Documentation/src_doc/02_MODELS/04_Fluid_Fluid_Fluid/NS_2AC_Comp.html) use two Allen–Cahn fields and \(\phi_0=1-\phi_1-\phi_2\). Each pair of phases has its own surface tension. Viscosity is a harmonic mean. Published pictures include a double serpentine, spreading lenses, three-fluid spinodal decomposition, and a three-fluid Rayleigh–Taylor splash. The compile menu has both `NSAC_Comp_3phases` and `NSAC_Comp_3phases3D`.
- A further kernel couples Navier–Stokes / Allen–Cahn / composition to a solid phase.

### Fluid–solid phase change without flow

[PART II](https://cea-lbm-saclay.github.io/LBM_Saclay_Documentation/src_doc/02_MODELS/models.html) states that crystal growth, dissolution, and gel maturation move the interface by thermodynamic imbalance only. Navier–Stokes is not solved in the liquid for those models. Crystal growth and dissolution are binary. Gel maturation is ternary and is the one of these pages with an input keyword, `problem=ternary_GP_mixt`. The crystal-growth and dissolution page states the PDEs and then leaves the `.ini` parameter section empty, with no `problem=` keyword. The compile menu’s `Crystal_growth_Younsi`, `GPMixt`, `GPMixtNS`, `GPMixtTernary`, and `GPMuTernary` are the matching program names. `GPMixtNS` is the one name that implies a flow coupling.

### Navier–Stokes/Korteweg

The [NSK model](https://cea-lbm-saclay.github.io/LBM_Saclay_Documentation/src_doc/02_MODELS/01_Fluid_Fluid/Model_NSK.html) is the other two-phase route: low-Mach Navier–Stokes, a Korteweg pressure tensor, and density itself as the phase index. No separate \(\phi\) is required. The page says van der Waals and Carnahan–Starling have been tested. Redlich–Kwong and Peng–Robinson are described as not yet tested. The NSK-with-surfactant page is a title and a navigation bar. PART III also describes a pseudo-potential force, a potential form of the pressure tensor, and a well-balanced scheme. The model page does not say those last two are compiled kernels.

## Schemes

From [lattices and streaming](https://cea-lbm-saclay.github.io/LBM_Saclay_Documentation/src_doc/03_LBM_Schemes/A_Basic-LBM/01_Lattices-Streaming_LBMSaclay.html) and [collision operators](https://cea-lbm-saclay.github.io/LBM_Saclay_Documentation/src_doc/03_LBM_Schemes/A_Basic-LBM/02_Collision-Operators_LBMSaclay.html), all in `src/LBM_Base_Functor.cpp` and `src/kernels/Collision_operators.h`:

- Lattices: D2Q5, D2Q9, D3Q7, D3Q15, D3Q19, D3Q27. Weights, velocities, and the MRT matrix \(M\) exist for each. \(M^{-1}\) is documented only for D2Q9 and D3Q27.
- Streaming is a two-array push. `stream` writes post-collision populations into `f_tmp` at the neighbour. A link that would leave the box is dropped (`stream_alldir` returns false).
- Collisions: BGK, TRT (magic parameter \(\Lambda=(\tau^+-1/2)(\tau^--1/2)\), viscosity on \(\tau^-\)), Gram–Schmidt MRT, and central moments. Central moments are what `NSAC_Comp` actually runs. The shear rate is \(s_\nu=1/(1/2+\nu/(c_s^2\delta t))\), with a separate bulk rate \(s_b\). Equilibrium and force are written in the central-moment basis (Gruszczyński et al., 2020, for high density ratio).
- Equilibria are documented for incompressible single-phase Navier–Stokes, incompressible two-phase flow, advection–diffusion with a source, Cahn–Hilliard, and conservative Allen–Cahn.
- The [force page](https://cea-lbm-saclay.github.io/LBM_Saclay_Documentation/src_doc/03_LBM_Schemes/A_Basic-LBM/05_Forces-LBMSaclay.html) is only a heading, so it does not specify \(\mathcal{F}_i\). In the bulk, the gradient is the weighted directional stencil \(\nabla\phi=(1/e^2)\sum_i w_i\mathbf{e}_i(\mathbf{e}_i\cdot\nabla\phi)\) with \(e^2=1/3\) except on D2Q5, where the exception is stated and the value is not. The bulk Laplacian on that page is \(3\sum_i w_i(\mathbf{e}_i\cdot\nabla)^2\phi\). The boundary-gradient sentence stops mid-line.
- The [boundary-condition page](https://cea-lbm-saclay.github.io/LBM_Saclay_Documentation/src_doc/03_LBM_Schemes/C_Other/Boundary_Conditions.html) is only a heading. The capillary-wave setup that is written down uses periodicity in one direction and bounce-back in the other. Contact angle is a course chapter (PART V.B), not a kernel section read for this note.

### Equilibria

Their single-phase incompressible equilibrium stores a pressure, not a density:

\[
f_i^{\mathrm{eq}}=w_i\left[p_h+\rho_0 c_s^2\left(\frac{\mathbf{c}_i\cdot\mathbf{u}}{c_s^2}+\frac{(\mathbf{c}_i\cdot\mathbf{u})^2}{2c_s^4}-\frac{|\mathbf{u}|^2}{2c_s^2}\right)\right],
\]

with \(p_h=\sum_i f_i\). The two-phase form used by `NSAC_Comp` replaces the constant \(\rho_0\) by the interpolated density inside a dimensionless pressure \(p_h^\star=p_h/(\varrho(\phi)c_s^2)\). Advection–diffusion uses \(g_i^{\mathrm{eq}}=w_i\phi(1+\mathbf{u}\cdot\mathbf{c}_i/c_s^2)\) and a half-source shift. Cahn–Hilliard puts the chemical potential in the rest and moving weights. Conservative Allen–Cahn is either that advection–diffusion equilibrium plus a counter-term flux, or the same equilibrium with a source \(G_i\propto\phi(1-\phi)\,\mathbf{c}_i\cdot\mathbf{n}\).

`src/kernels/hydro.jl` uses the compressible polynomial \(f_i=w_i\rho(1+3\mathbf{c}_i\cdot\mathbf{u}+4.5(\mathbf{c}_i\cdot\mathbf{u})^2-uu)\) with \(uu=1.5|\mathbf{u}|^2\), and `src/kernels/moments.jl` recovers \(\rho=\sum_i f_i\). The heat lattice is linear in \(T\) and \(\mathbf{u}\), and temperature is \(T=1+\sum_i g_i\) with no \(\delta t/2\) shift (`src/kernels/temperature.jl`).

## What this solver is

`src/extensions.jl` fixes one image per Julia process. The default is 3D: D3Q19 hydro and D3Q7 heat. A Preferences key `dim => 2` rebuilds D2Q9 and D2Q5. Compile-time switches that are on in this tree: TRT (SRT if that flag is false), KBC, free surface, volume force, equilibrium boundaries, moving walls, force field, temperature.

The hydro equation is the weakly compressible lattice Boltzmann equation, \(\rho=\sum_i f_i\), not \(\nabla\cdot\mathbf{u}=0\). Streaming is EsotericPull: one population array, even and odd steps swap the read and write sides. There is no second distribution buffer.

The interface is a sharp free surface. Cells are fluid, interface, or gas. Gas does not collide. Missing populations are reconstructed from a gas density. In 3D that density is \(\mathrm{clamp}(1-6\sigma\kappa,\,0.2,\,2)\); in 2D it is \(\mathrm{clamp}(1-3\sigma\kappa,\,0.2,\,2)\), with \(\kappa\) from a PLIC fit (`src/plic.jl`). Mass crosses the interface by the Körner exchange. That is a different closure from \(\mathbf{F}_c=\mu_\phi\nabla\phi\).

Heat is a second lattice (D3Q7 or D2Q5) with an enthalpy solid fraction, evaporation, and radiation (`src/kernels/temperature.jl`). Marangoni and recoil are evaluated on the reconstructed interface from \(\nabla T\) and \(\nabla\phi\) (`src/kernels/forces.jl`). A ray-traced laser deposits heat along the interface normal (`src/laser.jl`). One `PowderJet` advects parcels of a single mass rate into unmelted powder (`src/powder.jl`). `paint_powder_bed!` reads a 3D HDF5 volume and refuses to run in the 2D image (`src/powderbed.jl`).

Examples on `main` cover a channel, a lid-driven cavity, a dam break, Rayleigh–Bénard, a Stefan problem, a freeze layer, thermocapillary flow, evaporation, a cylinder, vortex shedding, a melt pool, a DED track, and an LPBF track, plus the three 2D scripts `channel_2d.jl`, `lid_driven_cavity_2d.jl`, and `dam_break_2d.jl`. Execution is one CPU or one CUDA device through KernelAbstractions. There is no MPI and no Kokkos.

Weight tables also exist for D3Q27 (`src/weights.jl`). The compiled hydro scheme is still D3Q19 or D2Q9. There is no D3Q15, no MRT matrix, and no central-moment collision.

## Side by side

| | LBM_Saclay | This solver |
|---|---|---|
| Interface | Diffuse \(\phi\), width \(W\), Allen–Cahn or Cahn–Hilliard | Sharp free surface, PLIC curvature |
| Hydro | Incompressible, \(\nabla\cdot\mathbf{u}=0\), both fluids solved | Weakly compressible, gas is a boundary |
| Density ratio | Both densities and viscosities interpolated; published wave test near 830 | Gas density clamped to \([0.2,2]\) around the liquid |
| Second scalar | Composition \(c\), or temperature in the phase-change kernel | Temperature and solid fraction only |
| Capillary force | \(\mu_\phi\nabla\phi\) | Laplace jump inside the gas reconstruction |
| Marangoni | \(\sigma(c)\) along the diffuse interface | \(\sigma_T\) from \(\nabla T\) on the sharp interface |
| Phase change | Volumetric \(\dot{m}'''\) from \((\theta_I-\theta)\phi(1-\phi)\), latent heat in the \(\phi\) equation | Evaporation and recoil on the free surface; enthalpy freezing |
| Extra phases | Surfactant kernel; two extra Allen–Cahn fields for three fluids | One liquid, one gas boundary |
| Solidification / dissolution | Crystal growth, dissolution, ternary gel, grand-potential kernels, no flow in the liquid for the pure thermodynamic models | Solid fraction in the melt; no dissolution, no crystal anisotropy |
| NSK / pseudo-potential | Low-Mach Korteweg; van der Waals and Carnahan–Starling tested. Well-balanced scheme described, not confirmed compiled | Absent |
| Lattices | D2Q5, D2Q9, D3Q7, D3Q15, D3Q19, D3Q27 | D2Q9+D2Q5 or D3Q19+D3Q7 |
| Collision | BGK, TRT, MRT, central moments (CM used by `NSAC_Comp`) | BGK/TRT and KBC |
| Streaming | Two-array push into `f_tmp` | One-array EsotericPull |
| Boundaries | Bounce-back and periodic in the written test; BC manual page is empty | Bounce-back, equilibrium, moving walls |
| Laser, powder, powder bed | Not in the model list read here | Ray tracer, one powder jet, 3D HDF5 bed |
| Parallelism | Kokkos, OpenMP, CUDA. Multi-GPU pages exist; one H100 configure transcript has MPI off | One CPU or one CUDA device |
| Case setup | `.ini` plus a compiled kernel | Julia scripts |

## What is missing if the target is their capability

Ranked by what their training code actually runs, not by the course notes.

1. **A diffuse-interface two-fluid solver.** `NSAC_Comp` is the core: incompressible variable-density momentum, conservative Allen–Cahn, a composition field, chemical-potential capillarity, and composition-driven Marangoni. Nothing in `src/` advances a \(\phi\) equation or a species \(c\). The free-surface gas reconstruction does not substitute for a second fluid at density ratio hundreds.

2. **The collision they use for that regime.** High density ratio in their manual is a central-moment scheme with a separate bulk viscosity. TRT and KBC on D3Q19 are a different stability tool. MRT is also absent. Copying BGK onto a phase field will not reproduce their capillary-wave results.

3. **Their lattice set as a runtime choice.** They keep D2Q5 through D3Q27, with MRT matrices, in one C++ lattice object. This code picks D2Q9 or D3Q19 at compile time. D3Q15 is missing. D3Q27 is only a weight table.

4. **Three immiscible fluids and a surfactant.** Both are compiled kernels (`NSAC_Comp_3phases`, `NSAC_Comp_3phases3D`, `NSAC_Surfactant`) with published pictures. A second Allen–Cahn field and a surfactant transport equation are new equations, not a flag on the free surface.

5. **Phase change in their sense.** Their Stefan problem is a source in Allen–Cahn proportional to \((\theta_I-\theta)\phi(1-\phi)\), with the divergence constraint carrying \(1/\rho_g-1/\rho_l\). Our Stefan and evaporation examples move a sharp interface and an enthalpy. The numbers are not interchangeable.

6. **Thermodynamic solid–liquid models.** Crystal growth, dissolution, and ternary gel maturation are separate programs, several of them without Navier–Stokes. Grand-potential formulations (`GPMixt*`) have no counterpart here.

7. **Navier–Stokes/Korteweg.** A low-Mach Korteweg pressure tensor with a tested van der Waals or Carnahan–Starling equation of state is a third interface method. This solver has neither. The well-balanced and potential-form schemes are described and not confirmed as compiled kernels. Redlich–Kwong, Peng–Robinson, and NSK with surfactant are not tested or not written up.

8. **A pressure equilibrium, if the hydro is to match theirs.** Their incompressible \(f_i^{\mathrm{eq}}\) carries \(p_h\), and the two-phase form carries \(p_h/(\varrho c_s^2)\). This solver’s equilibrium carries \(\rho\). That change belongs with item 1. It is useless on the free surface by itself.

9. **Multi-GPU domain decomposition.** Kokkos GPU builds are documented. One H100 multi-GPU configure transcript still says MPI is off, and the Topaze sample script does not show an `[mpi]` block. This solver has one device and no domain decomposition. Treat multi-node MPI as documented intent, not as a script that was shown to launch.

10. **Their case harness.** `.ini` files, one binary per kernel, XDMF/HDF5, and a training set with an analytical capillary wave. We have Julia examples and VTK. The public force page and boundary-condition chapter are stubs, so this item is about the training suite, not about a richer BC theory.

## What competing does not require

A melt-pool track is outside the model list above. Their manual, as read here, does not describe a laser ray, a powder jet, a powder-bed volume, recoil pressure, or a Körner free surface. Those pieces are the reason this solver exists, and they are already in `src/laser.jl`, `src/powder.jl`, `src/powderbed.jl`, and `src/kernels/surface.jl`.

KBC, EsotericPull, moving walls, and the 2D/3D compile switch are also already here. They are not steps toward `NSAC_Comp`.

Building Saclay’s training set on top of the free surface will not pass a Prosperetti test at density ratio 830, because the gas is not a fluid. The first new kernel that would make the two codes comparable is incompressible Navier–Stokes plus conservative Allen–Cahn, with central-moment collision and an interpolated second density. Surfactant, three phases, dissolution, and NSK come after that kernel exists.

## Sources

- [LBM_Saclay documentation index](https://cea-lbm-saclay.github.io/LBM_Saclay_Documentation/index.html), 24 June 2026.
- [PART II, mathematical models](https://cea-lbm-saclay.github.io/LBM_Saclay_Documentation/src_doc/02_MODELS/models.html).
- [NSAC_Comp](https://cea-lbm-saclay.github.io/LBM_Saclay_Documentation/src_doc/02_MODELS/01_Fluid_Fluid/Model_NSAC_Comp.html).
- [Liquid–gas phase change](https://cea-lbm-saclay.github.io/LBM_Saclay_Documentation/src_doc/02_MODELS/01_Fluid_Fluid/Model_NSAC_Comp_PhaseChange.html).
- [Three immiscible phases](https://cea-lbm-saclay.github.io/LBM_Saclay_Documentation/src_doc/02_MODELS/04_Fluid_Fluid_Fluid/NS_2AC_Comp.html).
- [PART III, schemes](https://cea-lbm-saclay.github.io/LBM_Saclay_Documentation/src_doc/03_LBM_Schemes/TOC_LBM_Schemes.html).
- [Lattices and streaming](https://cea-lbm-saclay.github.io/LBM_Saclay_Documentation/src_doc/03_LBM_Schemes/A_Basic-LBM/01_Lattices-Streaming_LBMSaclay.html).
- [Collision operators](https://cea-lbm-saclay.github.io/LBM_Saclay_Documentation/src_doc/03_LBM_Schemes/A_Basic-LBM/02_Collision-Operators_LBMSaclay.html).
- [Quick start, kernel menu and GPU/MPI](https://cea-lbm-saclay.github.io/LBM_Saclay_Documentation/src_doc/01_USER_GUIDE/QUICKSTART/quickstart.html).
- [Navier–Stokes/Korteweg](https://cea-lbm-saclay.github.io/LBM_Saclay_Documentation/src_doc/02_MODELS/01_Fluid_Fluid/Model_NSK.html).
- [Incompressible equilibria](https://cea-lbm-saclay.github.io/LBM_Saclay_Documentation/src_doc/03_LBM_Schemes/A_Basic-LBM/03_Equilibrium-Functions_Navier-Stokes.html) and [transport equilibria](https://cea-lbm-saclay.github.io/LBM_Saclay_Documentation/src_doc/03_LBM_Schemes/A_Basic-LBM/04_Equilibrium-Functions_Transport-Equations.html).
- [Gradients](https://cea-lbm-saclay.github.io/LBM_Saclay_Documentation/src_doc/03_LBM_Schemes/C_Other/Additional_Gradients.html). The force page and the boundary-condition page are headings only.
- This tree: `src/extensions.jl`, `src/weights.jl`, `src/plic.jl`, `src/kernels/forces.jl`, `src/kernels/temperature.jl`, `src/laser.jl`, `src/powder.jl`, `src/powderbed.jl`.
