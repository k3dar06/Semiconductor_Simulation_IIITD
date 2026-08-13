import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from E_band import *          # build_device, apply_bias, compute_bands (also sets the Agg backend)

kb = 8.617e-5   # [eV/K]

# ---------------- bias point / conditions ----------------
Temperature = 300
Vg  = 0.4
Vds = 0.4
normd = max(kb * Temperature / 8.0, 2e-3)     # builtin max, not np.max (np.max's 2nd arg is axis)

# ---------------- build (LDOS ON), bias, solve ----------------
# compute_ldos="yes" makes the C core fill MoS2.ldos during the solve. Temperature is
# passed so dE / NEmax / the internal tolerances match this bias.
device = build_device(compute_ldos="yes", Temperature=Temperature)
# init=True -> cold start via solve_init (fresh device, interface.Phi is zeros).
# NOTE the exact kwargs: Vd (not Vds), normd (not normd_); apply_bias has no Temp kwarg.
apply_bias(device, Vg=Vg, Vd=Vds, init=True, normd=normd)

# compute_bands gives Ec(y) and the reservoir Fermi levels; the solve above already
# left the raw spectral function in MoS2.ldos, which we read directly below.
rest = compute_bands(device, with_ldos=True)

# ---------------- pull the raw LDOS = spectral function A(site, E) ----------------
MoS2   = device["MoS2"]
ldos   = np.array(MoS2.ldos)          # (Nsites, NEmax)
NE     = int(MoS2.NE)                 # number of valid energy points (rest is padding)
E_axis = np.array(MoS2.E)[:NE]        # energy grid -- SAME frame as Ec
A      = ldos[:, :NE]                 # (Nsites, NE) local spectral weight

# sort sites by position so the map is monotonic in y (MoS2.y is in atom order)
order = np.argsort(np.array(MoS2.y))
y_s   = np.array(MoS2.y)[order]
A_s   = A[order]                      # (Nsites, NE), rows now in ascending y

# ---------------- 2D colormap: x = y, y = E, colour = LDOS ----------------
fig, ax = plt.subplots(figsize=(8.5, 5.5))

vmax = A_s.max()
vmin = vmax * 1e-4                    # ~4 decades; LogNorm masks the empty gap (<=0)
pcm = ax.pcolormesh(y_s, E_axis, A_s.T,
                    norm=LogNorm(vmin=vmin, vmax=vmax),
                    shading="auto", cmap="inferno")
cb = fig.colorbar(pcm, ax=ax)
cb.set_label("LDOS  (electron spectral function)  [a.u.]")

# Conduction band edge Ec(y) from the self-consistent potential -- it should trace the
# LOWER onset of the bright conduction weight. Overlaying it is the visual check that
# the LDOS band edge and the potential-derived Ec agree.
ax.plot(rest["y"], rest["Ec"], color="w", lw=1.8, label=r"$E_c$ (from $\Phi$)")

# Reservoir Fermi levels (constants): mu_source = mu1 = 0, mu_drain = mu2 = -Vd.
ax.axhline(rest["mu_source"], color="cyan",    ls="--", lw=1.3, label=r"$\mu_S$")
ax.axhline(rest["mu_drain"],  color="magenta", ls="--", lw=1.3, label=r"$\mu_D$")

# Focus on JUST the conduction band: from a touch below the lowest Ec up past the
# barrier top, so the deep valence band (~Egap below) is excluded.
ec = np.array(rest["Ec"])
ax.set_ylim(ec.min() - 0.1, ec.max() + 0.4)

ax.set_xlabel("y [nm]")
ax.set_ylabel("Energy [eV]")
ax.set_title("Conduction-band LDOS   (Vg=%.2f V, Vd=%.2f V, T=%g K)" % (Vg, Vds, Temperature))
ax.legend(loc="upper right", fontsize=9, framealpha=0.85)

fig.tight_layout()
out = "LDOS_conduction_band.png"
fig.savefig(out, dpi=150)
print("wrote", out)
