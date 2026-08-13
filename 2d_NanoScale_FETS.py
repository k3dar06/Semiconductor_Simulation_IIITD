import NanoTCAD_ViDES
from NanoTCAD_ViDES import *
import numpy as np
import matplotlib.pyplot as plt
matplotlib.use("Agg")
import module_TMD                 # imported by NAME (not just star) so NEmax can be
from module_TMD import *          # overridden on it below -- see the NEmax block
import E_band
from E_band import *
import os

kB = 8.617e-5    # Boltzmann constant [eV/K]

# MPI. Run under `mpirun -np N python3 2d_NanoScale_FETS.py` to split the energy
# integral across N ranks (NanoTCAD_ViDES.MPIze). Only rank 0 writes files / prints;
# every rank computes its energy slice and stays consistent via the bcast inside
# MPIze. Plain `python3 ...` runs as a single rank -> exact serial path.
try:
    from mpi4py import MPI
    rank = MPI.COMM_WORLD.Get_rank()
    nranks = MPI.COMM_WORLD.Get_size()
except Exception:
    rank, nranks = 0, 1

# Device characteristics
length = 17
sourceYmax = 5.0
drainYmin = 12.0
eta_ = 1e-5
ldos = "no"

# ======================= RESUME EDIT 1/4 (revert to restore) =======================
# TO REVERT: put back the original line below and DELETE the SEED-block that follows.
#   ORIGINAL:  Temperatures = [300, 100, 77, 60, 50, 40, 30, 28, 25, 20, 15]
Temperatures = [13, 10, 8, 5]

# RESUME SEED (whole block is NEW -- delete it to revert). Continue the ladder below
# 15 K from the potential the previous run left at 15 K, Vd=0.5. Create it once the
# 15 K run finishes with:   cp Phi.ini Phi_15K_seed.out
# The grid is temperature-independent (1232 nodes), so this Phi transplants directly
# into the colder devices. It seeds ONLY the first point of the first temperature
# (13 K, Vd=0.5) via solve_init(Phi_init=...); every later point warm-starts as usual.
SEED_FILE = "Phi_15K_seed.out"
if not os.path.isfile(SEED_FILE):
    raise SystemExit("Seed file %s not found -- after the 15 K run finishes, run "
                     "`cp Phi.ini %s` before launching this resume." % (SEED_FILE, SEED_FILE))
seed_Phi = np.loadtxt(SEED_FILE)
# =================================== END EDIT 1/4 ===================================

# Potential carried across temperatures (within THIS run).
saved_Phi = None

for ti, Temp in enumerate(Temperatures):

    if rank == 0 and not os.path.isdir(f"datiout{Temp}"):
        os.makedirs(f"datiout{Temp}")

    # ENERGY RESOLUTION, set from temperature.
    # The Fermi edge is ~kT wide, and the transport integral needs several grid points
    # across it to be clean; the rule of thumb is dE <= kT/5. Deriving dE from Temp
    # keeps it resolved automatically as the ladder cools. The floor (2e-4 eV) stops
    # it collapsing to an absurd value; it only bites below ~12 K.
    dEnergy = max(kB * Temp / 5.0, 2e-4)
    if dEnergy > kB * Temp / 5.0:
        print("WARNING: dE floored at %.1e eV > kT/5 = %.1e eV (T=%g K) -- Fermi edge "
              "slightly under-resolved." % (dEnergy, kB * Temp / 5.0, Temp))

    # CONVERGENCE TOLERANCES (eV), from temperature.
    # Outer SC tolerance: loop stops at max|dPhi| < normd, which sets barrier accuracy.
    normd_T = max(kB * Temp / 8.0, 2e-3)

    # Looser tolerance for the hard Vds > VD_HARD points, where the drain node rings
    # above normd_T; the barrier there still converges far below this drain-dominated
    # norm. (Most needed cold, where the drain genuinely rings.)
    normd_hard_T = max(kB * Temp / 4.0, 2e-3)

    # Inner Poisson tolerance -- kept tighter than the outer normd. Applied to the
    # interface below so solve_init and solve_self_consistent both use it (solve_init
    # no longer hard-codes 1e-3 -- see NanoTCAD_ViDES.solve_init).
    normpoisson_T = 1e-4

    print("Tolerances at T=%g K:  normd=%.2e  normpoisson=%.2e  (dI/I ~ %.0f%% per meV)"
          % (Temp, normd_T, normpoisson_T, 100.0 * 1e-3 / (kB * Temp)))

    # NEmax pre-allocates MoS2.E / MoS2.T. The C core writes NE = (Eupper-Elower)/dE
    # points into them; if NE > NEmax it overruns the arrays and smashes the heap
    # ("free(): invalid next size"). Size NEmax from dE with headroom.
    SPAN_MAX = 4.0                                       # eV, worst-case Eupper-Elower
    NE_needed = int(SPAN_MAX / dEnergy) + 1000
    NEmax_new = max(5000, NE_needed)
    NanoTCAD_ViDES.NEmax = NEmax_new
    module_TMD.NEmax = NEmax_new
    print("NEmax = %d  (dE=%.5f eV, worst-case NE~%d for span<=%.1f eV) at T=%g K"
          % (NEmax_new, dEnergy, NE_needed, SPAN_MAX, Temp))

    # Sweeping the gate voltage
    Vgmin = -0.1
    Vgmax = 0.5
    Vgstep = 0.05
    Np = int(round(abs(Vgmax - Vgmin) / Vgstep)) + 1
    # Fine Vd ramp: each step is a small perturbation, so every point warm-starts
    # from the previous converged solution -- easier initial guess, faster convergence.
    Vds_vals = [0.02, 0.04, 0.03, 0.1, 0.2, 0.3, 0.4, 0.42, 0.44, 0.46, 0.49, 0.5] 
    #0.3, 0.4, 0.42, 0.44, 0.48, 0.5, 0.52, 0.53, 0.55, 0.57, 0.59, 0.6]
    # Under-relaxation weight on the OLD potential: higher = more damping. The loop
    # gets stiff at high Vd and low T, so those points ring at the default 0.5. The
    # gentler value for the easy low-Vd points, heavier damping only where needed.
    UNDEREL_EASY = 0.5
    UNDEREL_HARD = 0.8
    VD_HARD      = 0.4      # above this Vd (and only below T_RING) use the hard regime

    # Rebuilt per temperature because dE / NEmax change with T. interface.Phi comes
    # back zeroed; it is then seeded below -- RESUME EDIT 4/4: the 15 K seed_Phi at
    # ti==0 (was: cold start) or the saved_Phi transplant at ti>0.
    device = build_device(compute_ldos=ldos, L_device=length, eta=eta_,
                          source_ymax=sourceYmax, drain_ymin=drainYmin,
                          Temperature=Temp, dE=dEnergy)
    Mos = device["MoS2"]
    # Apply the inner-Poisson tolerance we actually want (build_device sets its own
    # default; solve_init now respects this instead of its old hard-coded 1e-3).
    device["interface"].normpoisson = normpoisson_T
    # Enable the energy-split MPI charge (MPIze) only when actually under >1 rank;
    # a single process keeps the exact serial charge_T path.
    if nranks > 1:
        device["interface"].MPI = "yes"

    all_vds = []
    all_currents = []
    current_thermionic = []
    current_tunnel = []

    # BOUSTROPHEDON IN Vd (option "b"). Each temperature STARTS at the Vd where the
    # previous one ENDED, so the cross-temperature seed matches the drain bending.
    # ===================== RESUME EDIT 2/4 (revert to restore) =====================
    # Flipped so the FIRST temperature (13 K, ti=0) DESCENDS -- it starts at Vd=0.5,
    # matching the 15 K seed (Vd=0.5) for a clean matched-bias continuation. The main
    # run ascended from a cold start at Vd=0.02, hence the original below.
    #   ORIGINAL:  vd_order = range(len(Vds_vals)) if (ti % 2 == 0) else range(len(Vds_vals) - 1, -1, -1)
    vd_order = range(len(Vds_vals) - 1, -1, -1) if (ti % 2 == 0) else range(len(Vds_vals))
    # ================================= END EDIT 2/4 ================================

    # Drain ringing is a cold, degenerate-reservoir effect. Only below T_RING do we
    # loosen normd for the high-Vd points; at warmer T the drain does not ring, so a
    # loose normd_hard would only inject the hysteretic sawtooth into saturation.
    T_RING = 50

    for pos, val in enumerate(vd_order):
        Vds = Vds_vals[val]
        vg = zeros(Np)
        total_current = zeros(Np)
        thermion_current = zeros(Np)
        tunnel_current = zeros(Np)
        barrier_tops = zeros(Np)
        converged_flag = zeros(Np)
        done = zeros(Np, dtype=bool) # convergence checker

        # SERPENTINE Vg, keyed to the POSITION in this temperature's Vd traversal (not
        # the Vd index), so consecutive batches still alternate direction and warm-start
        # from each other whichever way Vd is running. The results arrays stay indexed
        # by counter_vg, so the saved file is in ascending Vg order either way.
        sweep_order = range(Np) if (pos % 2 == 0) else range(Np - 1, -1, -1)

        for vg_pos, counter_vg in enumerate(sweep_order):
            Vgs = Vgmin + counter_vg * Vgstep
            # WARM-START POLICY (resume run):
            #   - first point of the FIRST temperature (13 K, Vd=0.5) -> seed from the
            #     saved 15 K potential (seed_Phi) via solve_init -- NO cold start.
            #   - first point of every colder temperature (10/8/5 K) -> warm start from
            #     saved_Phi (the temperature above's converged field) via solve_init.
            #   - every other point -> warm start from the live interface.Phi (init=False).
            first_point = (pos == 0 and vg_pos == 0)
            # =================== RESUME EDIT 3/4 (revert to restore) ===================
            # ONLY the ti==0 branch changed: seed_Phi (15 K restart) instead of None
            # (cold start).  ORIGINAL:  init_flag, phi_seed = True, None
            if first_point and ti == 0:
                init_flag, phi_seed = True, seed_Phi
            # =============================== END EDIT 3/4 ==============================
            elif first_point:
                init_flag, phi_seed = True, saved_Phi
            else:
                init_flag, phi_seed = False, None

            # Hard regime only in the cold, high-Vd corner where the drain rings.
            hard_regime = (Temp < T_RING) and (Vds > VD_HARD)
            apply_bias(device, Vgs, Vds,
                       init=init_flag, Phi_init=phi_seed,
                       underel=(UNDEREL_HARD if hard_regime else UNDEREL_EASY),
                       normd=(normd_hard_T if hard_regime else normd_T))
            if rank == 0:
                print("  T=%g  Vd=%.2f  Vg=%+.2f" % (Temp, Vds, Vgs), end="", flush=True)
            results = compute_bands(device, with_ldos=False)
            results.update(compute_current_density(device))
            results.update(compute_barrier(device, results, verbose=False))
            results.update(split_thermionic_tunnelling(results, device, verbose=False))
            vg[counter_vg] = Vgs
            # All four columns are in mA/um. Use results["J_total"], NOT Mos.current()
            # (uA/um), so therm+tunn=total holds.
            total_current[counter_vg] = results["J_total"]
            thermion_current[counter_vg] = results["thermJ"]
            tunnel_current[counter_vg] = results["tunnelJ"]
            barrier_tops[counter_vg] = results["barrier_top"]
            converged_flag[counter_vg] = 1.0 if device["converged"] else 0.0
            done[counter_vg] = True

            # Consistency check: the two components are integrals of the SAME dJ/dE
            # spectrum split at the barrier top, so they must add back to the C core's
            # Landauer sum. Catching a mismatch HERE names the (Vds, Vgs) that failed.
            assert np.isclose(thermion_current[counter_vg] + tunnel_current[counter_vg],
                              abs(total_current[counter_vg]), rtol=1e-6), \
                "component split does not match total at Vds=%.2f Vgs=%.2f" % (Vds, Vgs)

            if rank == 0:
                print("   J=%.4e mA/um   barrier=%+.4f eV   tun=%5.1f%%"
                      % (total_current[counter_vg], barrier_tops[counter_vg],
                         100 * tunnel_current[counter_vg] / abs(total_current[counter_vg])
                         if total_current[counter_vg] else 0.0), flush=True)

            # SAVE EVERY POINT, not once per drain bias, so an OOM/crash partway does
            # not discard the batch. Rewrites the whole file each time (cheap).
            # [done] keeps only the rows COMPUTED SO FAR -- a boolean mask, not a
            # slice, because the serpentine runs some batches in DESCENDING Vg.
            # 6 columns: Vg | total | thermionic | tunnelling | barrier_top | converged
            # (cols 1-3 mA/um, col 4 eV). plots_idVd.py reads cols 0-3 by index.
            if rank == 0:
                savetxt("./datiout%d/idvgs_vd%.2f.out" % (Temp, Vds),
                        transpose([vg, total_current, thermion_current,
                                   tunnel_current, barrier_tops,
                                   converged_flag])[done])
        all_vds.append(Vds)
        all_currents.append(total_current.copy())
        current_thermionic.append(thermion_current.copy())
        current_tunnel.append(tunnel_current.copy())

    # Carry this temperature's last converged potential to seed the next (colder)
    # one. .copy() detaches it from the interface that build_device will discard.
    saved_Phi = device["interface"].Phi.copy()
    print(f"------DONE With {Temp} K------")

print("----SOLVED ALL TEMPERATURE CASES")

#I save the output files

# if (rank==0):
# string="./datiout/Phi%s.out" %Vgs;
# savetxt(string,interface.Phi);
# string="./datiout/ncar%s.out" %Vgs;
# savetxt(string,interface.free_charge);
# a=[MoS2.E,MoS2.T];
# string="./datiout/T%s.out" %Vgs;
# savetxt(string,transpose(a));
# string="./datiout/jayn%s.out" %Vgs;
# fp=open(string,"w");
# string2="%s" %current[counter];
# fp.write(string2);
# fp.close();
