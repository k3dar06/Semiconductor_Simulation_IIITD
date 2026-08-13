"""
Plot the I-V data written by 2d_NanoScale_FETS.py.

That script writes ONE FILE PER DRAIN VOLTAGE:
    datiout/idvgs_vd0.01.out ... datiout/idvgs_vd0.60.out
and each file is written with
    savetxt(filename, transpose([vg, total, thermionic, tunnelling]))
so every file is a 4-COLUMN table:
    column 0 = Vg                                    [V]
    column 1 = current density, total                [mA/um]
    column 2 = current density, thermionic (over the barrier)    [mA/um]
    column 3 = current density, tunnelling (through the barrier) [mA/um]

UNITS. The files hold a current DENSITY in mA/um. E_band.py does the uA -> mA
conversion once, where the current density is built, and 2d_NanoScale_FETS.py
stores res["J_total"] (NOT MoS2.current(), which is uA/um). So no unit rescaling
happens in this script.

The one factor still applied here is the DEVICE WIDTH, which is not a unit
conversion but the step from a density to a current:

    I [mA] = J [mA/um] * W [um]

The NEGF treats the transverse direction as infinite and translationally
invariant, integrating it out as a k-sum over the Brillouin zone, so its result
is inherently per unit width; there is no width anywhere inside the simulation.
W is therefore a free parameter of the model. The files are deliberately left as
a density (width-independent, the actual physical result) so the width can be
changed here without re-running the sweep.

Caveat: I = J*W is linear only because the model assumes an infinitely wide
device. For a genuinely narrow device (a few nm) the transverse direction would
be quantized and this scaling would break down -- a different calculation, not a
rescaling.
"""

import matplotlib.pyplot as plt
import numpy as np
import glob
import re

# ---------------------------------------------------------------------------
# DEVICE WIDTH -- the only factor applied in this script (see the module
# docstring). It converts the stored current DENSITY into a current:
#
#     I [mA] = J [mA/um] * W [um]
#
# In um, so it pairs directly with the mA/um on disk. W = 1 um is the standard
# reporting convention for 2D FETs, which makes the factor numerically 1 -- the
# number in the file is then already the current in mA.
# ---------------------------------------------------------------------------
W_DEVICE_UM = 1.0      # device width [um]

# ---------------------------------------------------------------------------
# Find the files. glob is used instead of a hard-coded range() so this keeps
# working if you change Vdsmin/Vdsmax/Vdsstep in 2d_NanoScale_FETS.py.
# (A range(0,7) would also ask for idvgs_vd0.00.out, which is never produced --
# the drain sweep starts at 0.1 -- so it would crash on the first iteration.)
# ---------------------------------------------------------------------------
files = sorted(glob.glob("datiout5/idvgs_vd*.out"))
if not files:
    raise SystemExit("No datiout5/idvgs_vd*.out files found -- run 2d_NanoScale_FETS.py first.")

# Pull the Vd value straight out of each filename, so the legend is always right.
vds_list = [float(re.search(r"vd([0-9]+\.[0-9]+)\.out", f).group(1)) for f in files]

print("Found %d files:" % len(files))
for f, vd in zip(files, vds_list):
    print("   %-30s -> Vd = %.2f V" % (f, vd))

# ---------------------------------------------------------------------------
# FIGURE 1 -- Transfer characteristics: Id vs Vg, one curve per Vd
# ---------------------------------------------------------------------------
# Read every file once, up front, so the log and linear figures below plot
# identical data instead of each re-loading and re-scaling it.
curves = []
has_components = True
for f, vd in zip(files, vds_list):
    data = np.loadtxt(f)            # shape (n_vg, 4)
    Vg = data[:, 0]                 # column 0 = gate voltage [V]
    Jd = data[:, 1]                 # column 1 = total current DENSITY [mA/um]
    # Density -> current: multiply by the width in um. No unit conversion; the
    # file is already in mA/um and the plots are in mA.
    # abs() because a log scale cannot show negative values; the sign of Jd is
    # just the transport direction (mu2 = -Vd), not physically interesting here.
    # Files written before the thermionic/tunnelling split have only 2 columns;
    # detect that rather than IndexError, so old data still plots figures 1-2.
    if data.shape[1] >= 4:
        Jthe = np.abs(data[:, 2]) * W_DEVICE_UM
        Jtun = np.abs(data[:, 3]) * W_DEVICE_UM
    else:
        has_components = False
        Jthe = Jtun = None
    curves.append((vd, Vg, np.abs(Jd) * W_DEVICE_UM, Jthe, Jtun))

if not has_components:
    print("\nNOTE: some files have only 2 columns (pre-split format) -- "
          "figures 3 and 4 skipped. Re-run 2d_NanoScale_FETS.py to get "
          "thermionic/tunnelling columns.")

ylabel = r"Drain current $|I_d|$ (mA)"
title  = r"Transfer characteristics  $I_d$ vs $V_{gs}$   ($W$ = %g $\mu$m)" % W_DEVICE_UM

# Log (left y-axis) and linear (right y-axis) in ONE axes, via twinx. Each Vd
# curve is therefore drawn TWICE -- solid against the log scale, dashed against
# the linear scale -- in the same colour, so a colour is one Vd on both scales.
#
# Why both: on the log axis the high-Vd curves look identical at large Vg, but
# they are not. At Vg = 0.5 the currents run ~1.43 / 1.56 / 1.63 mA for
# Vd = 0.2 / 0.3 / 0.6 -- a real ~14% spread (DIBL) that a 4-decade log axis
# compresses to under a pixel, and that the linear scale shows plainly.
fig1, ax1 = plt.subplots(figsize=(8, 5.5))
ax1r = ax1.twinx()

for vd, Vg, Id_p, _the, _tun in curves:
    # Draw the log curve first and reuse ITS colour for the linear twin, so the
    # two axes do not each start the colour cycle over and mismatch.
    ln, = ax1.plot(Vg, Id_p, marker="o", ms=3, label=r"$V_{ds}$ = %.2f V" % vd)
    ax1r.plot(Vg, Id_p, ls="--", lw=1.2, alpha=0.75, color=ln.get_color())

ax1.set_yscale("log")
ax1r.set_ylim(bottom=0)          # a linear current axis must start at 0 to be read honestly

ax1.set_xlabel(r"Gate voltage $V_{gs}$ (V)")
ax1.set_ylabel(ylabel + "   [log, solid]")
ax1r.set_ylabel(ylabel + "   [linear, dashed]")
ax1.set_title(title)

# The twin is created second, so it is drawn on top and would cover ax1's lines
# and legend. Lift ax1 above it and make its background transparent so both
# sets of curves stay visible.
ax1.set_zorder(ax1r.get_zorder() + 1)
ax1.patch.set_visible(False)
ax1.legend(fontsize=8, loc="upper left")

fig1.tight_layout()
fig1.savefig("idvgs_transfer.png", dpi=150)


# ---------------------------------------------------------------------------
# FIGURE 2 -- Output characteristics: Id vs Vd, one curve per Vg
#
# The data on disk is stored the "wrong way round" for this plot (each file is a
# fixed Vd), so stack it into a matrix and slice the other axis:
#     id_matrix[i, j] = Id at (Vd = vds_list[i], Vg = Vg[j])
# Reading DOWN a column j then gives Id vs Vd at fixed Vg.
# ---------------------------------------------------------------------------
Vg = np.loadtxt(files[0])[:, 0]                              # same Vg grid in every file
id_matrix = np.array([np.loadtxt(f)[:, 1] for f in files])   # shape (n_vd, n_vg)
vds = np.array(vds_list)

fig2, ax2 = plt.subplots(figsize=(7, 5))
# Plot only every 4th Vg, otherwise 21 curves is unreadable.
for j in range(0, len(Vg)):
    ax2.plot(vds, np.abs(id_matrix[:, j]) * W_DEVICE_UM, marker="s", ms=3,
             label=r"$V_{gs}$ = %.2f V" % Vg[j])

ax2.set_xlabel(r"Drain voltage $V_{ds}$ (V)")
ax2.set_ylabel(ylabel)
ax2.set_title(r"Output characteristics  $I_d$ vs $V_{ds}$   ($W$ = %g $\mu$m)" % W_DEVICE_UM)
ax2.legend()
fig2.tight_layout()
fig2.savefig("idvds_output.png", dpi=150)

saved = ["idvgs_transfer.png", "idvds_output.png"]

if has_components:
    # -----------------------------------------------------------------------
    # FIGURE 3 -- current COMPONENTS vs Vg: total / thermionic / tunnelling.
    #
    # ONE PANEL PER DRAIN VOLTAGE. Overlaying all five Vd values put 15 lines in
    # one axes, where the colour said "which Vd" and the linestyle said "which
    # component" -- two encodings competing for attention. Splitting by Vd means
    # each panel carries only the 3 curves that are actually being compared, and
    # the colour is then free to mean ONE thing: the component.
    #     blue  solid  = total       (over + through the barrier)
    #     red   dashed = thermionic  (E > barrier top, over it)
    #     green dotted = tunnelling  (E < barrier top, through it)
    #
    # Log scale + shared axes: in the on-state the components differ by orders of
    # magnitude, so a linear axis would flatten tunnelling onto zero exactly where
    # it matters. sharey lets you compare panel to panel by eye -- without it each
    # panel would autoscale separately and the Vd trend would be invisible.
    # -----------------------------------------------------------------------
    n = len(curves)
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))
    fig3, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.6 * nrows),
                              sharex=True, sharey=True, squeeze=False)
    flat = axes.ravel()

    for k, (vd, Vg_c, Id_p, Jthe, Jtun) in enumerate(curves):
        a = flat[k]
        a.plot(Vg_c, Id_p,  color="tab:blue",  ls="-",  lw=1.7, label="total")
        a.plot(Vg_c, Jthe,  color="tab:red",   ls="--", lw=1.4, label="thermionic")
        a.plot(Vg_c, Jtun,  color="tab:green", ls=":",  lw=1.6, label="tunnelling")
        a.set_yscale("log")
        a.set_title(r"$V_{ds}$ = %.2f V" % vd, fontsize=10)
        a.grid(alpha=0.25, which="both")
        # Axis labels only on the outer edge, so they are not repeated 5 times.
        if k % ncols == 0:
            a.set_ylabel(r"$|I_d|$ (mA)")
        if k >= n - ncols:
            a.set_xlabel(r"$V_{gs}$ (V)")

    # Blank out any unused cell in the grid (5 curves in a 2x3 grid leaves one).
    for k in range(n, len(flat)):
        flat[k].axis("off")

    # sharex hides tick labels on every row but the literal last one. With a
    # ragged grid (5 panels in 2x3) the panel above a blanked cell is the bottom
    # of ITS column, so it gets an x-label from the loop above but would show no
    # tick numbers -- an axis labelled in volts with nothing to read them against.
    # Re-enable the numbers on any panel that has nothing beneath it.
    for k in range(n):
        if k + ncols >= n:
            flat[k].tick_params(labelbottom=True)

    # One legend for the whole figure: the three labels are identical in every
    # panel, so repeating it per panel would be pure clutter.
    flat[0].legend(fontsize=8, loc="upper left")
    fig3.suptitle("Current components vs gate voltage, per drain bias", fontsize=12)
    fig3.tight_layout()
    fig3.savefig("idvgs_components.png", dpi=150)
    saved.append("idvgs_components.png")

    # -----------------------------------------------------------------------
    # FIGURE 4 -- tunnelling FRACTION vs Vg, one curve per Vd.
    #
    # The ratio is the physically meaningful figure: it separates "how much
    # current" from "how it gets across". Expect it to fall as Vg rises -- the
    # gate pulls the barrier down, the thermionic window widens, and tunnelling
    # is swamped. The subthreshold end is where tunnelling matters, since that
    # is the leakage floor a thicker/longer barrier would suppress.
    #
    # CAVEAT: the split point is the barrier top, which MOVES with bias. A
    # changing fraction is therefore partly a changing partition of the energy
    # axis, not only changing physics -- compare figure 3's absolute curves too.
    # -----------------------------------------------------------------------
    fig4, ax4 = plt.subplots(figsize=(7, 5))
    for vd, Vg_c, Id_p, Jthe, Jtun in curves:
        tot = Jthe + Jtun
        # Guard the deep-off points where both components underflow to 0.
        frac = np.where(tot > 0, 100.0 * Jtun / np.where(tot > 0, tot, 1.0), np.nan)
        ax4.plot(Vg_c, frac, marker="o", ms=3, label=r"$V_{ds}$ = %.2f V" % vd)

    ax4.set_xlabel(r"Gate voltage $V_{gs}$ (V)")
    ax4.set_ylabel("Tunnelling fraction of $I_d$ (%)")
    ax4.set_title("Tunnelling share of the total current")
    ax4.legend(fontsize=8)
    fig4.tight_layout()
    fig4.savefig("idvgs_tunnel_fraction.png", dpi=150)
    saved.append("idvgs_tunnel_fraction.png")

print("\nSaved: " + " , ".join(saved))
plt.show()
