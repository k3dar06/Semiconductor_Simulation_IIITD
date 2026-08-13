import matplotlib
matplotlib.use("Agg")          # headless: save figures to file instead of showing
import NanoTCAD_ViDES
from NanoTCAD_ViDES import *
import numpy as np
import matplotlib.pyplot as plt
import module_TMD                 # imported by NAME too (not just star) so NEmax can be
from module_TMD import *          # overridden on it below -- see the NEmax block in main()
import os

# =====================================================================================
#  Energy band diagram + energy-resolved current for the MoS2 nanoscale FET
#  (same device as 2d_NanoScale_FETS.py), for a single gate+drain bias point.
#
#  Band edges use the model's NATIVE frame (actual tight-binding on-site energies):
#     Ec = BC_MX2 - Phi  = 0.20 - Phi
#     Ev = BV_MX2 - Phi  = -1.58 - Phi
#  The midgap is NOT at E=0; it sits at (BC_MX2+BV_MX2)/2 - Phi = -0.69 - Phi.
#  The device geometry / grid / doping are UNCHANGED from the original file.
#
#  STRUCTURE. The script is split into one function per stage, each taking what it
#  needs and returning a plain dict, so a stage can be re-run or reused without
#  re-running the expensive NEGF solve:
#     build_device()                -- geometry, grid, oxides, gates, doping (NO bias/solve)
#     apply_bias()                  -- set (Vg,Vd), solve_init (optional) + self-consistent
#     compute_bands()               -- Ec, Ev, E_i, and the NEGF/LDOS quasi-Fermi level
#     compute_current_density()     -- dJ/dE spectrum
#     compute_barrier()             -- barrier top (absolute energy) and barrier height
#     split_thermionic_tunnelling() -- decomposition of the current
#     plot_all()                    -- both figures
#     main()                        -- wires them together
# =====================================================================================

# Bias magnitudes for the "on" case (matching the sweep maxima in 2d_NanoScale_FETS.py)
VG_ON = 0.1    # gate voltage  [V] #for n type make it positive
VD_ON = 0.6    # drain voltage [V] #for n type make it positive


# =====================================================================================
def build_device(L_device=17.0, dE=0.005, Nky=32.0,
                 source_ymax=5.0, drain_ymin=12.0,
                 fraction_source=0.01, fraction_drain=0.01,
                 eta=1e-5, compute_ldos="yes", Temperature=300, normpois=1e-3,
                 normd_=5e-3, underel=0.5):
    """Build the MoS2 FET structure. Does NOT apply a bias and does NOT solve.

    Geometry, grid, oxides, gates, doping and material parameters only -- i.e.
    everything that is the SAME at every bias point. Call apply_bias() afterwards
    to set (Vg, Vd) and run the solver.

    WHY THE SPLIT: a bias sweep should build this once and reuse it, for two
    reasons.
      MEMORY -- rebuilding grid2D/interface2D/TMD for all 65 bias points
        accumulates (the C core allocates per charge_T call as well), which is
        what exhausted an 8 GB WSL2 VM partway through a sweep.
      SPEED -- solve_self_consistent starts from whatever interface.Phi already
        holds (see NanoTCAD_ViDES.py: `channel.Phi = interface.Phi[grid.swap]`).
        Reusing the interface therefore WARM-STARTS each solve from the previous
        bias point's converged potential, which is an excellent initial guess for
        a nearby bias and cuts the iteration count sharply. Rebuilding throws that
        away and restarts cold every time.

    NOTE ON THE SIGNATURE: the length parameter is L_device, not `len` -- `len` is
    a Python builtin, and shadowing it inside this function would break every
    len(...) call in it. Also, every parameter here has a default, because a
    parameter WITHOUT a default may not follow one WITH a default: that is a
    SyntaxError, so it kills the whole file at import rather than at call time.

    Returns a dict holding everything downstream stages need. grid, source_ymax
    and drain_ymin come back as well as MoS2/interface, because the band
    extraction needs grid.swap and the plots need the reservoir edges; the two
    gate objects come back because apply_bias() has to set their Ef.
    """
    # -------------------------------------------------------------------------
    # Geometry / grid / doping are IDENTICAL to 2d_NanoScale_FETS.py.
    # -------------------------------------------------------------------------
    # Creating the grid and also the MoS2 region  (xmin, dx1, x1, dx2, x2, dx3)
    

    x_g = nonuniformgrid(array([-2, 1, 0, 0.1, 2, 1]))

    # the MoS2 material, as an nMOS
    MoS2 = TMD(L_device, "n")

    kymax = np.pi / MoS2.delta            # maximum value of k in the y-direction
    MoS2.kmax = np.pi / MoS2.delta
    MoS2.kmin = 0
    MoS2.dk = kymax / Nky                 # step in the k-space
    MoS2.dE = dE
    MoS2.Temp = Temperature                        # energy step for the NEGF transport method
    # Opt in to the local density of states. This is what makes the C core return
    # MoS2.ldos / MoS2.ne, from which the genuine NEGF quasi-Fermi level (EFn_ldos)
    # is obtained below. It defaults to "no" so that pure I-V scripts
    # (2d_NanoScale_FETS.py) do not pay for a (Nc, NEmax) array per k-point.
    MoS2.compute_ldos = compute_ldos
    MoS2.eta = eta                        # Green's-function broadening (default 1e-5).
                                          # Larger eta smooths the razor-sharp band-edge
                                          # DOS that causes the gate-on channel ripple.

    # creating the 2D grid for the simulation
    grid = grid2D(x_g, MoS2.y, MoS2.x, MoS2.y)

    # Creating oxide and gate regions
    Oxide1 = region("hex", grid.xmin, 0, grid.ymin, grid.ymax)
    Oxide1.eps = 10                       # Al2O3 dielectric constant
    Oxide2 = region("hex", 0, grid.xmax, grid.ymin, grid.ymax)
    Oxide2.eps = 10                       # Al2O3 dielectric constant

    # Creating the top and bottom gates. The gate spans exactly the gap between the
    # two reservoirs, so it is tied to source_ymax / drain_ymin rather than
    # repeating those numbers -- that way changing the channel length moves the
    # gate and the reservoir edges together instead of letting them drift apart.
    gate_ymin = source_ymax
    gate_ymax = drain_ymin
    top_gate = gate("hex", grid.xmax, grid.xmax, gate_ymin, gate_ymax)
    bottom_gate = gate("hex", grid.xmin, grid.xmin, gate_ymin, gate_ymax)

    interface = interface2D(grid, Oxide1, Oxide2, top_gate, bottom_gate)

    # Creating the source and drain reservoirs.
    # volume density: N_D = fraction * n(MoS2), with the MoS2 formula-unit density
    # n(MoS2) = rho*N_A/M = 5.06 g/cm^3 * 6.022e23 / 160 g/mol ~= 1.9e22 cm^-3.
    # fraction 0.05 -> N_D ~= 9.5e20 cm^-3   (0.001 -> ~1.9e19 cm^-3)
    # 0.01 keeps the reservoirs degenerate (n+), so Ec-E_F stays constant under bias;
    # lower values (e.g. 0.001) are non-degenerate and let the drain Ec-E_F drift.
    dope_reservoir(grid, interface, MoS2, fraction_source,
                   array([-1, 1, grid.ymin, source_ymax]))
    dope_reservoir(grid, interface, MoS2, fraction_drain,
                   array([-1, 1, drain_ymin, grid.ymax]))

    # Material band offsets. These define the tight-binding on-site energies, so
    # they must be set before any solve -- but they are bias-INDEPENDENT, hence
    # here rather than in apply_bias.
    MoS2.BC_MX2 = 0.3 # Work function = electron affinity(4 eV) + BC_MX2 = 4.3 eV
    MoS2.BV_MX2 = 0.3 - MoS2.Egap

    # Source contact (low-y) is the grounded reference at every bias point; only
    # mu2 moves with Vd, so mu1 is set once here.
    MoS2.mu1 = 0.0

    # Solver settings.
    # underel is the UNDER-RELAXATION weight in the Poisson/NEGF update
    #     Phi <- (1-underel)*Phi_new + underel*Phi_old
    # so it is the weight on the OLD potential: HIGHER = more damping.
    #   0.0  undamped -- fastest when it works, oscillates when the loop is stiff
    #   0.5  the long-standing default
    #   0.7  heavier damping for hard points
    #   ~1.0 almost no progress per iteration
    # It is a U-shaped tradeoff, not "higher is safer": too LOW and the loop
    # overshoots and rings (at 50 K a 1 meV potential error swings the charge
    # ~26%, so high Vd at low T oscillates); too HIGH and every point crawls,
    # including the easy ones, and can still miss normd insde the iteration
    # cap -- the same failure by a different route. Exposed as a parameter so the
    # damping can be raised only for the bias points that ineed it.
    interface.underel = underel
    # normd / normpoisson in eV. kboltz here is 1.3807e-23 J/K (SI), so kT in eV is
    # kboltz*Temperature/q -- the /q is what converts to eV and MUST be present
    # (every other kT in this file uses kboltz*Temp/q; without it the formula gives
    # ~1e-22 and floors to 1e-5 at every temperature, ~100x too tight at 300 K).
    vt = kboltz * Temperature / q                 # kT [eV]
    if Temperature < 30:
        interface.normd = max(vt / 8.0, 1e-5)     # looser cold: drain rings at kT/25
    else:
        interface.normd = max(vt / 25.0, 1e-5)    # barrier resolved to ~2%
    interface.normpoisson = interface.normd / 10.0

    return {
        "MoS2": MoS2,
        "interface": interface,
        "grid": grid,
        # apply_bias() needs these to set gate.Ef
        "top_gate": top_gate,
        "bottom_gate": bottom_gate,
        "source_ymax": source_ymax,
        "drain_ymin": drain_ymin,
        # Filled in by apply_bias(); None until then.
        "Vg": None,
        "Vd": None,
    }


# =====================================================================================
def apply_bias(dev, Vg, Vd, init=False, underel=None, normd=None, warn=True, Phi_init=None):
    """Set (Vg, Vd) on an already-built device and solve it self-consistently.

    Separated from build_device so a sweep can call this repeatedly on ONE device
    instead of rebuilding the structure per bias point.

    init:
      False (default) -- WARM START. solve_self_consistent begins from the
        interface.Phi left behind by the previous call, which for a nearby bias is
        a very good initial guess and converges in far fewer iterations. Use this
        while ramping Vg in small steps.
      True -- COLD START: run solve_init first, which re-anchors the reservoirs to
        their charge-neutral, mu-offset levels and overwrites the potential. Use it
        for the FIRST point, and at the start of each new Vd (where the stale Phi
        from the previous Vd's last Vg is a poor guess and can slow, or mislead,
        convergence).

    (solve_init works here only because its log(exp(argo)-1) underflow was fixed
    with expm1 -- see the TEMPORARY CHANGE in NanoTCAD_ViDES.py; previously it
    produced a NaN potential on this wide-gap material. It also restores the
    reservoir boundary conditions on return, so the reservoirs are free rather
    than hard-clamped during the SC loop and the junctions come out smoother.)

    underel: per-point override of the under-relaxation weight (see build_device
      for the full tradeoff). None keeps whatever the device was built with. Pass
      a higher value (e.g. 0.7) only for bias points that oscillate -- typically
      high Vd at low temperature -- since heavier damping slows every point.

    normd: per-point override of the SC convergence threshold on max|dPhi| [eV].
      None keeps whatever the device was built with. The point of overriding it is
      the high-Vd/low-T ringing: there the Poisson update limit-cycles in the DRAIN
      (measured: max|dPhi| swings ~0.6-2.7 meV at Vd=0.5, T=30 K, node at y~15 nm),
      so a tight kT/25 (~1 meV) threshold can never be held and the loop burns all
      the iteration cap. Loosening normd for those points is SAFE for the current: the
      residual is a drain artifact, not the channel barrier (y in [5,12]) that sets
      I ~ exp(-barrier/kT) -- the barrier converges far below max|dPhi|, which is
      dominated by that one stiff drain node. Keep it tight at low Vd, where the
      norm is not drain-dominated and the barrier could actually carry it.

    Phi_init: optional potential (numpy array on the grid) used as the initial
      guess instead of solve_init's cold start. Only consulted when init=True. Used
      for temperature continuation: pass the previous (warmer) temperature's
      converged interface.Phi so a colder temperature does not cold-start again.

    warn: print a warning if the point did not converge.

    Sets dev["converged"] and dev["normad_final"] from the solver. The underlying
    solve_self_consistent gives up at the iteration cap and RETURNS AN UNCONVERGED
    POTENTIAL; checking this flag is the only way to tell a failed point from a
    good one, because the currents it produces look perfectly ordinary.

    Mutates dev in place and returns it, so it can be chained or ignored.
    """
    MoS2      = dev["MoS2"]
    interface = dev["interface"]
    grid      = dev["grid"]

    if underel is not None:
        interface.underel = underel

    if normd is not None:
        interface.normd = normd

    # Drain bias. solve_init assigns its reservoir potentials as
    #   low-y reservoir (source) <- mu1 ,   high-y reservoir (drain) <- mu2
    # so to bias the high-y drain we move mu2 and leave the source on mu1 (set in
    # build_device). A positive Vd lowers the drain's electron energy, so its
    # quasi-Fermi level sits BELOW the source's.
    MoS2.mu2 = -Vd

    # Gate bias: same convention as 2d_NanoScale_FETS.py  (gate.Ef = -Vg)
    dev["bottom_gate"].Ef = -Vg; set_gate(interface, dev["bottom_gate"])
    dev["top_gate"].Ef    = -Vg; set_gate(interface, dev["top_gate"])

    if init:
        # Phi_init carries a warm-start potential across temperature steps; None
        # falls back to solve_init's from-scratch cold start.
        solve_init(grid, interface, MoS2, Phi_init=Phi_init)

    solve_self_consistent(grid, interface, MoS2)

    # getattr defaults keep this working against an unpatched NanoTCAD_ViDES.py
    # (the converged/normad_final attributes come from the fork patch).
    dev["converged"]     = getattr(interface, "converged", True)
    dev["normad_final"]  = getattr(interface, "normad_final", float("nan"))
    dev["Vg"] = Vg
    dev["Vd"] = Vd

    if warn and not dev["converged"]:
        print("  *** NOT CONVERGED at Vg=%+.2f Vd=%.2f : ||dPhi||=%.3e (target %.1e). "
              "Results for this point are NOT trustworthy. ***"
              % (Vg, Vd, dev["normad_final"], interface.normd), flush=True)

    return dev


# =====================================================================================
def compute_bands(dev, with_ldos=True):
    """Conduction / valence edges, intrinsic level, and the NEGF quasi-Fermi level.

    Returns the position-resolved band structure, sorted by y so the profiles are
    monotonic for plotting.

    with_ldos=False SKIPS the quasi-Fermi level inversion and returns
    EFn_ldos = dEF_i_channel = None. Everything else (Ec, Ev, Efi, Ec_unsorted,
    mu_source, mu_drain, Egap) is unaffected, because those come from the
    self-consistent potential alone and cost nothing.

    Use it for I-V sweeps. The thermionic/tunnelling chain needs only Ec_unsorted
    from this function: compute_current_density reads MoS2.E / MoS2.T and
    compute_barrier reads Ec_unsorted, neither of which involves the LDOS. The
    inversion below, by contrast, is a 60-iteration bisection over EVERY site, run
    once per bias point, so across a 65-point sweep it dominates the analysis cost
    while producing a quantity the sweep never looks at.

    Turning it off also lets the sweep call build_device(compute_ldos="no"), which
    is the bigger saving -- the C core then skips filling a (Nc, NEmax) array per
    k-point. Those two switches MUST agree: with compute_ldos="no" the TMD class
    deletes MoS2.ldos / MoS2.ne outright, so with_ldos=True would fail. That
    mismatch is checked explicitly below instead of surfacing as a bare
    AttributeError several lines later.

    plot_all() draws the EFn_ldos curve, so any run whose figures you intend to
    keep needs with_ldos=True.
    """
    MoS2        = dev["MoS2"]
    interface   = dev["interface"]
    grid        = dev["grid"]
    source_ymax = dev["source_ymax"]
    drain_ymin  = dev["drain_ymin"]

    # interface.Phi lives on the full 2D grid; grid.swap gathers it onto the atom
    # chain, giving a vector aligned one-to-one with MoS2.y.
    Phi = interface.Phi[grid.swap]
    Eg = MoS2.Egap

    # Band edges in the model's NATIVE frame: the actual tight-binding on-site
    # energies shifted by the potential. The midgap is NOT at 0 -- it sits at
    # (BC_MX2 + BV_MX2)/2 - Phi = -0.69 - Phi eV.
    Ec = MoS2.BC_MX2 - Phi                # conduction band edge
    Ev = MoS2.BV_MX2 - Phi                # valence band edge

    # Intrinsic Fermi level E_i over the whole structure: midgap + a small
    # effective-mass correction (~1 meV, so E_i ~ midgap). Use 0.5 as a FLOAT.
    E_mgap = (Ec + Ev) / 2.0
    kT_eV  = MoS2.Temp * kboltz / q
    Efi    = E_mgap + 0.5 * kT_eV * np.log(MoS2.mh/MoS2.me)

    # Channel mask + mean intrinsic level. The actual (E_F - E_i) difference is
    # computed below, once the genuine NEGF quasi-Fermi level exists.
    y_all       = np.array(MoS2.y)
    ch_all      = (y_all >= source_ymax) & (y_all <= drain_ymin)
    Efi_ch_mean = np.array(Efi)[ch_all].mean()

    # Everything above is potential-only and always runs. Everything below is the
    # LDOS inversion, which is what with_ldos switches off.
    EFn_ldos = None
    dEF_i_channel = None

    if with_ldos:
        # Fail loudly and specifically if the device was built without the LDOS.
        # module_TMD deletes MoS2.ldos / MoS2.ne when compute_ldos != "yes", so the
        # symptom would otherwise be an AttributeError on a line that looks
        # unrelated to the flag that actually caused it.
        if not hasattr(MoS2, "ldos") or not hasattr(MoS2, "ne"):
            raise AttributeError(
                "compute_bands(with_ldos=True) needs MoS2.ldos / MoS2.ne, but the "
                "device was built with compute_ldos != 'yes'. Either rebuild with "
                "build_device(..., compute_ldos='yes') or call "
                "compute_bands(dev, with_ldos=False).")

        # -------------------------------------------------------------------------
        # GENUINE NEGF quasi-Fermi level from the local density of states.
        # The C core (H_charge_T LDOS patch) now returns MoS2.ldos[site, energy] =
        # the electron spectral function 2*(A1+A2), scaled identically to MoS2.charge
        # so that   n(site) = sum_E ldos(site,E) * Fermi((E-Ef)/vt) * dE   holds
        # (verified to ratio 1.000 at the contacts). We invert that per site for the
        # single Ef = E_Fn(site) that reproduces the actual local NEGF density -- the
        # rigorous local quasi-Fermi level. Unlike a parabolic-DOS reconstruction,
        # this uses the TRUE spectral function, so it is exact wherever the NEGF
        # density and DOS are trustworthy (contacts, on-state channel).
        # -------------------------------------------------------------------------
        ldos_arr = np.array(MoS2.ldos)                 # (Nc, NEmax), native energy frame
        NE_v     = int(MoS2.NE)                        # number of valid energy points
        E_axis   = np.array(MoS2.E)[:NE_v]             # native energy grid (== Ec frame)
        A_site   = ldos_arr[:, :NE_v]                  # local spectral function per site
        # Use the TRUE electron density MoS2.ne -- NOT abs(MoS2.charge). MoS2.charge
        # is the NET density (holes - electrons); in the depleted channel the hole
        # term is comparable to the tiny electron term, so |charge| != n_e there and
        # the inversion produced spurious modulation/early bending in the channel
        # (while still looking correct in the n+ contacts, where p ~ 0).
        n_site   = np.array(MoS2.ne)                   # actual local NEGF electron density
        vt_v     = MoS2.Temp * kboltz / q
        EFn_ldos = np.zeros(len(MoS2.y))
        lo0, hi0 = E_axis.min() - 0.5, E_axis.max() + 0.5
        for s in range(len(MoS2.y)):
            A = A_site[s]; target = n_site[s]
            lo, hi = lo0, hi0
            for _ in range(60):                        # bisection: integral is monotonic in Ef
                mid = 0.5 * (lo + hi)
                occ = 1.0 / (1.0 + np.exp(np.clip((E_axis - mid) / vt_v, -500, 500)))
                val = (A * occ).sum() * MoS2.dE
                if val < target:
                    lo = mid
                else:
                    hi = mid
            EFn_ldos[s] = 0.5 * (lo + hi)

        # (E_F - E_i) averaged over the channel: how far the actual local Fermi level
        # sits above the local intrinsic level.  > 0 => n-type, = 0 => intrinsic,
        # < 0 => p-type.
        # We use the GENUINE NEGF quasi-Fermi level (EFn_ldos), not the fixed contact
        # level mu1. mu1 is only the Fermi level *of the source reservoir*; it equals
        # the channel Fermi level at equilibrium (Vd=0), but under drain bias the
        # channel's actual E_F drops away from mu1 toward mu2, so referencing mu1
        # would overstate how n-type the channel really is. EFn_ldos is the true
        # local E_F at every site, so this difference now means what it claims.
        dEF_i_channel = EFn_ldos[ch_all].mean() - Efi_ch_mean

    # Sort by position so the profile is monotonic in y for plotting
    order = np.argsort(MoS2.y)
    return {
        "y": np.array(MoS2.y)[order],
        "Ec": np.array(Ec)[order],
        "Ev": np.array(Ev)[order],
        "Efi": np.array(Efi)[order],           # intrinsic Fermi level E_i(y)
        # genuine NEGF quasi-Fermi level (LDOS inversion), or None with_ldos=False.
        # The sort has to be guarded: np.array(None)[order] raises.
        "EFn_ldos": (np.array(EFn_ldos)[order] if EFn_ldos is not None else None),
        "dEF_i_channel": dEF_i_channel,        # (E_F - E_i) averaged over the channel
        "Egap": Eg,
        "mu_source": MoS2.mu1,                 # source Fermi level (mu1, low-y)
        "mu_drain": MoS2.mu2,                  # drain Fermi level (mu2, high-y)
        # Ec in ATOM order (unsorted) as well: compute_barrier masks it against
        # MoS2.y, which is also unsorted, so the two must share an ordering.
        # Masking the y-sorted Ec with an unsorted y mask would silently pick the
        # wrong atoms and give a wrong barrier top.
        "Ec_unsorted": np.array(Ec),
    }


# =====================================================================================
def compute_current_density(dev):
    """Energy-resolved sheet current density dJ/dE.

      dI/dE = (2q^2/h) T(E) [f(E-mu1) - f(E-mu2)]

    the integrand of MoS2.current() before summing over energy. Overlaid on the
    band-diagram energy axis it shows WHERE in energy current flows: weight below
    the channel-barrier top => tunnelling; only above it => thermionic emission.
    """
    MoS2 = dev["MoS2"]

    vt = kboltz * MoS2.Temp / q
    E_native = np.array(MoS2.E)
    Tr = np.array(MoS2.T)
    # MoS2.E / MoS2.T are preallocated to length NEmax (=5000); only the first ~600
    # entries hold the real energy grid, the rest are (E=0, T=0) padding. Drop that
    # padding and sort by energy -- otherwise the dI/dE curve draws spurious lines
    # jumping back to E=0. The sort is also what lets np.interp be used later:
    # np.interp requires its xp to be increasing and gives silent garbage otherwise.
    valid = ~((E_native == 0) & (Tr == 0))
    E_native = E_native[valid]; Tr = Tr[valid]
    order_E = np.argsort(E_native)
    E_native = E_native[order_E]; Tr = Tr[order_E]

    # Energy-resolved SHEET current density dJ/dE, per unit device WIDTH [A/(m*eV)].
    # It integrates to MoS2.current() = A/m (the NEGF integrates the transverse
    # direction out as a k-sum, so its result is inherently per unit width).
    #
    # This is still a current DENSITY -- a linear/sheet one (current per unit width,
    # the surface current density K), as opposed to a VOLUME density in A/cm^2
    # (current per unit width*thickness). Only the normalisation differs.
    #
    # Since  1 A/m == 1 uA/um  EXACTLY (1 A/m = 1e-6 A per um = 1 uA/um), this array
    # is ALREADY the spectrum in uA/(um*eV): no conversion, no thickness needed.
    # Dividing by a 0.65 nm monolayer thickness -- the interlayer spacing of BULK
    # MoS2, a convention for pretending a 2D sheet has a volume, not a real
    # cross-section the current flows through -- would inflate every value by
    # 1e-4/0.65e-9 = 1.54e5, giving ~1e9 A/cm^2 (~1000x copper's electromigration
    # limit). uA/um is the standard figure of merit for 2D FETs.
    dJdE_sheet = 2 * q * q / (2 * np.pi * hbar) * Tr * (
        Fermi((E_native - MoS2.mu1) / vt) - Fermi((E_native - MoS2.mu2) / vt))

    # dJdE_sheet is in uA/(um*eV) == A/(m*eV) as it stands (see the A/m == uA/um
    # identity above). Convert ONCE here to mA/(um*eV) so that every downstream
    # consumer -- the plot and the thermionic/tunnelling integrals -- is already in
    # the reporting unit and no per-use *1e-3 rescale is needed.
    # NOTE: this means the energy integral of jE is now in mA/um, so comparisons
    # against MoS2.current() (which stays in uA/um) must scale by 1e-3.
    jE = dJdE_sheet * 1e-3                  # dJ/dE in mA/(um * eV)

    # SINGLE SOURCE OF TRUTH FOR THE TOTAL CURRENT.
    # MoS2.current() is the C core's own Landauer sum and comes back in uA/um;
    # everything else in this file is mA/um. Rather than leave callers to remember
    # that one exception, convert it here and hand it out already in mA/um. A sweep
    # should store res["J_total"], NOT MoS2.current() -- mixing the two puts columns
    # that differ by 1000x into the same output file.
    # Signed, like MoS2.current(): the sign is the transport direction (mu2 = -Vd).
    J_total = MoS2.current() * 1e-3          # mA/um

    return {
        "E_plot": E_native,     # NEGF energy axis, ascending
        "jE": np.abs(jE),       # |dJ/dE| sheet current density spectrum [mA/(um*eV)]
        "J_total": J_total,     # total sheet current density [mA/um], signed
    }


# =====================================================================================
def compute_barrier(dev, bands, verbose=True):
    """Channel barrier top (an absolute energy) and barrier height (a difference).

    barrier_top    -- the peak conduction-band edge in the channel, on the same
                      energy axis as E_plot. This is the quantity to compare
                      energies against (e.g. the thermionic/tunnelling split).
    barrier_height -- barrier_top measured FROM the source Fermi level. Positive
                      => a real barrier (off-ish); negative => the channel is
                      pulled below the source Fermi level (on). This is a depth,
                      NOT a point on the energy axis, so it must never be used as
                      an energy threshold.
    """
    MoS2 = dev["MoS2"]

    # Mask MoS2.y against the UNSORTED Ec: both are in atom order, so they line up.
    y_atoms = np.array(MoS2.y)
    ych = (y_atoms >= dev["source_ymax"]) & (y_atoms <= dev["drain_ymin"])
    barrier_top = bands["Ec_unsorted"][ych].max()   # change Ev to Ec for nMOS
    barrier_height = barrier_top - MoS2.mu1         # change the signs for nMOS

    if verbose:
        # MoS2.current() is A/m == uA/um numerically (the sheet current density, no
        # fictitious thickness); x1e-3 puts it in mA/um like everything else here.
        print("   barrier height (channel Ec_max - E_F,source) = %+.3f eV ;  J = %.4f mA/um"
              % (barrier_height, MoS2.current() * 1e-3))

    return {"barrier_top": barrier_top, "barrier_height": barrier_height}


# =====================================================================================
def split_thermionic_tunnelling(res, dev, verbose=True):
    """Split the current into thermionic (over the barrier) and tunnelling (through).

    Operational definition: current carried at energies ABOVE the channel barrier
    top passes OVER the barrier (thermionic emission); current at energies BELOW
    the top has to pass THROUGH it (tunnelling).

    The split is made on ENERGY (E vs barrier_top), NOT on the current-density
    value. barrier_top is an absolute energy in the same frame as E_plot, so
    "E_plot vs barrier_top" is a valid comparison. The related barrier_height is a
    DIFFERENCE (barrier_top - mu1), i.e. a depth rather than a point on the energy
    axis, so it must not be used as the split -- it would only coincide with
    barrier_top while mu1 happens to be 0.
    """
    MoS2 = dev["MoS2"]
    E_sp = res["E_plot"]              # energy axis [eV], sorted ascending
    j_sp = res["jE"]                  # |dJ/dE| [mA/(um*eV)]
    Etop = res["barrier_top"]         # channel barrier top [eV]

    # np.interp does not extrapolate -- it silently clamps to the endpoint value. A
    # barrier top outside the transport grid would therefore be masked rather than
    # raised, so check it explicitly.
    assert E_sp[0] <= Etop <= E_sp[-1], "barrier top lies outside the NEGF energy grid"

    # Splice an exact sample at Etop so the two halves share a boundary. Without it
    # the trapezoid spanning the crossing belongs to NEITHER masked sub-array and the
    # two pieces do not sum to the total.
    j_at_top = np.interp(Etop, E_sp, j_sp)

    # Strict < and > : a grid point landing exactly on Etop is excluded from both
    # masks, and Etop is re-inserted by hand, so it is counted exactly once per half.
    lo = E_sp < Etop
    hi = E_sp > Etop

    E_tun = np.append(E_sp[lo], Etop);    j_tun = np.append(j_sp[lo], j_at_top)
    E_the = np.insert(E_sp[hi], 0, Etop); j_the = np.insert(j_sp[hi], 0, j_at_top)

    # np.trapezoid(y, x) -- values FIRST, coordinates second. (np.trapz is the
    # deprecated NumPy 1.x spelling; it still runs on NumPy 2 but warns.)
    tunnelJ = np.trapezoid(j_tun, E_tun)          # mA/um
    thermJ  = np.trapezoid(j_the, E_the)          # mA/um
    totalJ  = np.trapezoid(j_sp,  E_sp)           # mA/um

    # The two pieces must tile the whole integral exactly (this is what the splice
    # above buys). A failure here means the masks or the insert ends are wrong.
    assert np.isclose(tunnelJ + thermJ, totalJ, rtol=1e-9), "splice lost area"

    pct_the = 100.0 * thermJ  / totalJ if totalJ else 0.0
    pct_tun = 100.0 * tunnelJ / totalJ if totalJ else 0.0

    if verbose:
        # Independent cross-checks. (1) The rectangular sum on the uniform dE grid
        # should agree with the trapezoid to well under 1%; a large gap means the
        # energy grid is not uniform/contiguous after the padding filter. (2) The
        # NumPy integral must reproduce the C core's own Landauer sum -- if it does
        # not, the decomposition is meaningless no matter where the boundary is put.
        rectJ = j_sp.sum() * MoS2.dE                  # mA/um
        # The one place MoS2.current() is read raw: it is uA/um, so x1e-3 -> mA/um.
        # (Callers should use res["J_total"], which is already converted.)
        coreJ = abs(MoS2.current()) * 1e-3            # mA/um

        print("\n=== Thermionic vs tunnelling decomposition (%s) ===" % res["title_flat"])
        print("   barrier top  = %+.4f eV   (E_F,source = %+.3f eV, barrier height = %+.3f eV)"
              % (Etop, res["mu_source"], res["barrier_height"]))
        print("   thermionic (E > barrier top) = %10.4f mA/um   %6.2f %%" % (thermJ,  pct_the))
        print("   tunnelling (E < barrier top) = %10.4f mA/um   %6.2f %%" % (tunnelJ, pct_tun))
        print("   ---------------------------------------------------------------")
        print("   total                        = %10.4f mA/um   %6.2f %%"
              % (totalJ, pct_the + pct_tun))
        print("   cross-check  rectangular sum = %10.4f mA/um   (rel. diff %.2e)"
              % (rectJ, abs(rectJ - totalJ) / totalJ if totalJ else 0.0))
        print("   cross-check  MoS2.current()  = %10.4f mA/um   (rel. diff %.2e)"
              % (coreJ, abs(coreJ - totalJ) / coreJ if coreJ else 0.0))

    return {
        "tunnelJ": tunnelJ, "thermJ": thermJ, "totalJ": totalJ,
        "pct_tun": pct_tun, "pct_the": pct_the,
        # The spliced arrays are returned so the plot shades EXACTLY the regions
        # that were integrated -- the picture cannot disagree with the numbers.
        "E_tun": E_tun, "j_tun": j_tun,
        "E_the": E_the, "j_the": j_the,
    }


# =====================================================================================
def plot_all(res, split, dev,
             band_png="band_diagrams_MoS2.png",
             current_png="band_and_current_MoS2.png"):
    """Both figures: the band diagram, and the band diagram + current density."""
    source_ymax = dev["source_ymax"]
    drain_ymin  = dev["drain_ymin"]

    # Source / drain Fermi levels are drawn ONLY over their own reservoir, not
    # spanning the device. mu1/mu2 are the Fermi levels OF THE CONTACTS; they are
    # only defined where that contact physically is. Across the channel the actual
    # local Fermi level is EFn_ldos (teal), which is precisely what joins mu1 to mu2.
    msrc = res["y"] <= source_ymax
    mdrn = res["y"] >= drain_ymin

    def draw_bands(a):
        """The band lines shared by both figures."""
        a.plot(res["y"], res["Ec"], lw=2, label="$E_c$ (conduction)")
        a.plot(res["y"], res["Ev"], label="$E_v$ (valence)")
        a.plot(res["y"], res["Efi"], color="brown", ls="-.", lw=1.2,
               label="$E_i$ (intrinsic)")
        # Absent when compute_bands ran with with_ldos=False (an I-V sweep). Skip
        # the curve rather than crash, so the bands still plot for such a run.
        if res.get("EFn_ldos") is not None:
            a.plot(res["y"], res["EFn_ldos"], color="teal", ls="-", lw=1.7,
                   label="$E_{Fn}$ (NEGF/LDOS)")
        a.plot(res["y"][msrc], np.full(int(msrc.sum()), res["mu_source"]),
               color="green", ls="--", lw=2.0, label="$E_{F,source}$")
        a.plot(res["y"][mdrn], np.full(int(mdrn.sum()), res["mu_drain"]),
               color="purple", ls=":", lw=2.2, label="$E_{F,drain}$")
        a.set_ylim(-2.5, 1.0)   # switch the bound to (-2.5, 1.0) for nMOS
        a.set_xlabel("Position along channel, y (nm)")
        a.set_ylabel("Energy (eV)")

    # -----------------------------------------------------------------------
    # FIGURE 1 -- band diagram alone
    # -----------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 6))
    draw_bands(ax)
    ax.fill_between(res["y"], res["Ev"], res["Ec"], color="0.85", zorder=0)  # band gap
    ax.set_title(res["title"], fontsize=11)
    ax.legend(loc="upper center", ncol=3, fontsize=9)
    fig.suptitle("MoS$_2$ nanoscale FET  -  Energy band diagram (E$_g$ = %.2f eV)"
                 % res["Egap"], fontsize=13)
    fig.tight_layout()
    fig.savefig(band_png, dpi=150, bbox_inches="tight")
    print("\nSaved figure to %s" % band_png)

    # -----------------------------------------------------------------------
    # FIGURE 2 -- band diagram + energy-resolved current density
    # -----------------------------------------------------------------------
    fig2, axb = plt.subplots(figsize=(8, 6))
    draw_bands(axb)
    axb.set_title("%s   (barrier %+.3f eV)"
                  % (res["title_flat"], res["barrier_height"]), fontsize=10, pad=30)

    # |dJ/dE| overlaid IN THE SAME AXES via a twin x-axis (twiny): the ENERGY (y)
    # axis is shared, so the current spectrum lines up horizontally with the bands
    # it flows through -- read straight across to see which energies carry current.
    #
    # NOTE ON THE UNITS. The solver returns A/m, and A/m == uA/um EXACTLY: going
    # A->uA multiplies by 1e6 while going per-m -> per-um divides by 1e6, so the two
    # cancel. That is why the Landauer expression yields uA/(um*eV) directly, and
    # the only applied factor is the x1e-3 to mA -- done once, inside
    # compute_current_density, so the curve, the fills and the printed integrals all
    # share one unit and cannot drift apart. Nothing is rescaled at draw time.
    #
    # Note also that uA -> mA is x1e-3 and per-eV -> per-meV is ALSO x1e-3, so these
    # same numbers would equally well be labelled uA/(um*meV). We label them
    # mA/(um*eV) because keeping the energy denominator in eV matches the eV y-axis
    # of the band diagram -- you can read straight across in one energy unit.
    axt = axb.twiny()
    ln_j, = axt.plot(res["jE"], res["E_plot"], color="darkorange", lw=1.4,
                     label=r"$|dJ/dE|$")
    axt.set_xlim(0, 30)   # mA/(um*eV); this was 30000 back when jE was in uA/(um*eV)
    axt.set_xlabel(r"Current Density  (mA/($\mu$m$\cdot$eV))",
                   color="darkorange", fontsize=9)
    axt.tick_params(axis="x", colors="darkorange", labelsize=8)

    # Shade the two contributions under the dJ/dE curve, split at the barrier top.
    # fill_betweenx (not fill_between) because energy is the y-axis on these axes:
    # it fills horizontally from x=0 out to the curve, over a span of y.
    # The SPLICED arrays are reused, so the two patches meet exactly at the barrier
    # top with no gap -- the shaded areas are literally the integrals printed above.
    f_the = axt.fill_betweenx(split["E_the"], 0, split["j_the"],
                              color="red",  alpha=0.5, lw=0)
    f_tun = axt.fill_betweenx(split["E_tun"], 0, split["j_tun"],
                              color="blue", alpha=0.5, lw=0)

    h2, l2 = axb.get_legend_handles_labels()
    h2.append(ln_j);  l2.append(r"$|dJ/dE|$ (current density)")
    # The patches live on axt, so they are not picked up by axb's handle collection
    # and have to be appended by hand.
    h2.append(f_the); l2.append("thermionic (%.1f%%)" % split["pct_the"])
    h2.append(f_tun); l2.append("tunnelling (%.1f%%)" % split["pct_tun"])
    axb.legend(h2, l2, loc="upper center", ncol=3, fontsize=8)

    fig2.suptitle("Band diagram + energy-resolved sheet current density (shared energy axis)",
                  fontsize=12)
    fig2.tight_layout()
    fig2.savefig(current_png, dpi=150, bbox_inches="tight")
    print("Saved %s" % current_png)


# =====================================================================================
def main(Vg=VG_ON, Vd=VD_ON, Temp=300):
    # creating the data folder if it does not exist
    if not os.path.isdir('datiout'):
        os.makedirs('datiout')

    print("\n" + "=" * 70)
    print("Solving case: Vg = %.2f V , Vd = %.2f V , T = %g K" % (Vg, Vd, Temp))
    print("=" * 70)

    # -------------------------------------------------------------------------
    # TEMPERATURE-DRIVEN RESOLUTION AND TOLERANCES -- identical policy to
    # 2d_NanoScale_FETS.py, so a band diagram matches what the sweep actually
    # solves. The build_device() defaults (dE=0.005, normd_=5e-3) are 300 K
    # values and are far too coarse/loose cold: at 30 K they under-resolve the
    # ~kT=2.6 meV Fermi edge (dE=5 meV steps OVER it) and stop the solve ~5 meV
    # short, so the drain band comes out under-bent (mu2 lands in the gap).
    # -------------------------------------------------------------------------
    kB = 8.617e-5                                   # eV/K
    dE = max(kB * Temp / 5.0, 2e-4)                 # >= kT/5 to resolve the Fermi edge

    # NEmax MUST scale with 1/dE: the C core writes NE=(Eupper-Elower)/dE points
    # into MoS2.E / MoS2.T, and if NE > NEmax it writes past the arrays and the
    # heap is smashed ("free(): invalid next size"). SPAN_MAX covers Egap + Vd +
    # band-bending + Fermi tails with headroom; +1000 is margin. max(5000, .) keeps
    # the small default when dE is coarse. Set on BOTH modules (star import aliases
    # do not propagate a rebinding, so the name has to be set where each reads it).
    SPAN_MAX = 4.0
    NEmax_new = max(5000, int(SPAN_MAX / dE) + 1000)
    NanoTCAD_ViDES.NEmax = NEmax_new
    module_TMD.NEmax = NEmax_new

    # Structure first, then bias. init=True because this is a single cold run with
    # no previous potential to warm-start from. build_device() sets normd/normpoisson
    # itself from Temperature (see there), so we pass Temperature + dE and let it own
    # the tolerances -- dE and NEmax are the two things build_device does NOT handle.
    # underel=0.7 for the hard high-Vd points (same gate as the sweep's VD_HARD=0.4).
    dev = build_device(Temperature=Temp, dE=dE)
    print("  dE=%.5f eV  normd=%.2e  normpoisson=%.2e  NEmax=%d"
          % (dE, dev["interface"].normd, dev["interface"].normpoisson, NEmax_new))
    apply_bias(dev, Vg, Vd, init=True, underel=(0.7 if Vd > 0.4 else 0.5))

    # Did the self-consistent loop actually converge? A band diagram from a solve
    # that hit the iteration cap is NOT a solution -- the drain band in
    # particular will look wrong (under-bent) -- so say so loudly before plotting.
    if not dev["converged"]:
        print("\n  *** WARNING: this point did NOT converge "
              "(||dPhi||=%.3e, target %.1e). The band diagram below is NOT a valid "
              "solution -- do not trust the drain band or the Fermi levels. ***\n"
              % (dev["normad_final"], dev["interface"].normd))
    else:
        print("  converged: ||dPhi||=%.3e (target %.1e)"
              % (dev["normad_final"], dev["interface"].normd))

    # res accumulates the results of every analysis stage into one flat dict, so
    # the plotter takes a single object instead of five positional arguments.
    res = {
        "title": "Gate + Drain\n$V_g=%.2f$ V, $V_d=%.2f$ V" % (Vg, Vd),
        "title_flat": "Gate + Drain  $V_g=%.2f$ V, $V_d=%.2f$ V" % (Vg, Vd),
    }
    res.update(compute_bands(dev))
    res.update(compute_current_density(dev))
    res.update(compute_barrier(dev, res))

    split = split_thermionic_tunnelling(res, dev)

    plot_all(res, split, dev)

    # Only the difference between the intrinsic level and the genuine NEGF/LDOS
    # quasi-Fermi level, averaged over the channel. (dEF_i_channel is built from
    # EFn_ldos, not from the fixed contact level mu1.)
    d = res["dEF_i_channel"]
    if d is not None:
        print("\n=== Channel: NEGF quasi-Fermi level vs intrinsic level  (E_Fn - E_i, channel average) ===")
        print("   (channel = %.1f < y < %.1f nm)" % (dev["source_ymax"], dev["drain_ymin"]))
        typ = "n-type" if d > 0 else ("p-type" if d < 0 else "intrinsic")
        print("   %-26s : E_Fn - E_i = %+.3f eV   (%s)" % (res["title_flat"], d, typ))

    print("\nDone.")
    return res, split, dev


# Guard so the module can be imported (to reuse build_device / compute_bands from
# another script, or to drive a Vg sweep) without immediately running a full NEGF
# solve on import.
if __name__ == "__main__":
    main()
