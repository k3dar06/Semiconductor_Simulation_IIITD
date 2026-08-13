import matplotlib.pyplot as plt
import numpy as np
import os

W_device = 1  # [um]

# Fixed drain bias for this transfer-characteristic plot. The saved files are named
# idvgs_vd%.2f.out, so 0.1 -> "idvgs_vd0.10.out".
Vd = 0.1

# One curve per temperature. Only those with a datiout{T}/idvgs_vd0.10.out are drawn.
Temperatures = [300, 100, 77, 60, 50, 40, 30, 28, 25, 20, 15, 13, 10, 8, 5]

# Column layout of the .out files:
#   0: Vg [V]   1: total J [mA/um]   2: thermionic   3: tunnelling
#   4: barrier_top [eV]   5: converged flag (1.0 = converged)
COL_VG, COL_JTOT, COL_CONV = 0, 1, 5


MARK_UNCONVERGED = True   # overlay an 'x' on any point whose SC loop did not converge

fig, ax = plt.subplots(figsize=(7.5, 5.5))

# Colour by INDEX, not by temperature value: the temperatures are bunched at the
# cold end (25..50 K) with a big gap up to 300 K, so a value-normalised colormap
# maps all the cold curves to nearly the same blue. Sampling the colormap at evenly
# spaced positions (one slot per curve) gives every temperature a distinct hue, and
# cycling markers disambiguates any that still look close.
temps_hot_to_cold = sorted(Temperatures, reverse=True)   # 300 K first -> top of legend
cmap = plt.get_cmap("turbo")
colours = cmap(np.linspace(0.05, 0.95, len(temps_hot_to_cold)))
markers = ["o", "s", "^", "v", "D", "P", "X", "*", "<", ">", "h"]

# Peak transconductance gm,max = max(dId/dVgs) per temperature, collected in the loop
# below and plotted vs T afterwards.
gm_by_T = {}

any_unconverged = False
for i, T in enumerate(temps_hot_to_cold):
    path = "datiout%d/idvgs_vd%.2f.out" % (T, Vd)
    if not os.path.isfile(path):
        print("skip %d K: %s not found" % (T, path))
        continue

    data = np.atleast_2d(np.loadtxt(path))
    if data.size == 0:
        print("skip %d K: %s is empty" % (T, path))
        continue

    order = np.argsort(data[:, COL_VG])               # ensure ascending Vg for the line
    data = data[order]
    Vg = data[:, COL_VG]
    Id = np.abs(data[:, COL_JTOT]) * W_device         # mA/um * um = mA
    conv = data[:, COL_CONV] if data.shape[1] > COL_CONV else np.ones_like(Vg)

    # Transconductance gm = dId/dVgs on the LINEAR current (np.gradient handles the
    # Vg spacing). Stored as a full curve vs Vg. Id [mA] / Vg [V] -> gm in mA/V = mS.
    if Vg.size >= 2:
        gm = np.gradient(Id, Vg)
        gm_by_T[T] = (Vg, gm)

    colour = colours[i]
    marker = markers[i % len(markers)]
    ax.semilogy(Vg, Id, ls="-", marker=marker, ms=5, lw=1.4, color=colour, label="%d K" % T)

    if MARK_UNCONVERGED:
        bad = conv < 0.5
        if np.any(bad):
            any_unconverged = True
            ax.semilogy(Vg[bad], Id[bad], "x", ms=9, mew=2, color="k", zorder=5)

ax.set_xlabel("Gate voltage  $V_{GS}$  [V]")
ax.set_ylabel(r"Drain current  $I_D$  [mA]")
ax.set_title(r"Transfer characteristics at $V_{DS}$ = %.2f V" % Vd)
ax.grid(True, which="both", ls=":", alpha=0.5)

handles, labels = ax.get_legend_handles_labels()
if MARK_UNCONVERGED and any_unconverged:
    from matplotlib.lines import Line2D
    handles.append(Line2D([0], [0], marker="x", color="k", ls="none",
                          ms=9, mew=2, label="not converged"))
ax.legend(handles=handles, title="Temperature", fontsize=9, ncol=2)

fig.tight_layout()
out = "transfer_vd%.2f_vs_T.png" % Vd
fig.savefig(out, dpi=150)
print("wrote", out)

# ---------------------------------------------------------------------------
# Transconductance gm(Vgs) = dId/dVgs, one curve per temperature.
# x = gate voltage Vgs [V] (gm is a function of the operating point); y = gm [mS];
# temperature is the legend. Same colours/markers as the transfer plot above.
# ---------------------------------------------------------------------------
fig2, ax2 = plt.subplots(figsize=(7.5, 5.5))
for i, T in enumerate(temps_hot_to_cold):
    if T not in gm_by_T:
        continue
    Vg, gm = gm_by_T[T]
    ax2.plot(Vg, gm, ls="-", marker=markers[i % len(markers)], ms=5, lw=1.4,
             color=colours[i], label="%d K" % T)
ax2.set_xlabel("Gate voltage  $V_{GS}$  [V]")
ax2.set_ylabel(r"Transconductance  $g_m = dI_D/dV_{GS}$  [mS]")
ax2.set_title(r"Transconductance at $V_{DS}$ = %.2f V" % Vd)
ax2.grid(True, ls=":", alpha=0.5)
ax2.legend(title="Temperature", fontsize=9, ncol=2)
fig2.tight_layout()
out2 = "gm_vd%.2f_vs_T.png" % Vd
fig2.savefig(out2, dpi=150)
print("wrote", out2)
