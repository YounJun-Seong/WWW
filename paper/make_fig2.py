"""Figure 2: (A) W4 overlap response, (B) E2 gold placement. Numbers are read from the sealed result files."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

W4 = json.load(open("runs/overlap/RESULTS_overlap.json"))
E2 = json.load(open("runs/e2/RESULTS_e2.json"))
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6})
DARK, LIGHT = "#1f4e8c", "#8fb3de"
f1 = lambda v: f"{v:+.1f}".replace("-", "\u2212")
f2 = lambda v: f"{v:+.2f}".replace("-", "\u2212")
fig, (a, b) = plt.subplots(1, 2, figsize=(7.0, 2.6), gridspec_kw=dict(width_ratios=[1.15, 1], wspace=0.32))

o = list(range(6)); P = [W4["secondary"]["P_int"][str(k)] for k in o]
y = [p[0] * 100 for p in P]; lo = [(p[0] - p[1]) * 100 for p in P]; hi = [(p[2] - p[0]) * 100 for p in P]
a.axhline(0, color="0.6", lw=0.6, ls="--", zorder=0)
a.errorbar(o, y, yerr=[lo, hi], color=DARK, marker="o", ms=4, lw=1.4, capsize=2.5, elinewidth=0.8)
s = W4["primary"]["slope_P_int"]
a.text(0.97, 0.95, f"Slope: {f2(s[0]*100)} pp per shared passage\n(95% CI [{f2(s[1]*100)}, {f2(s[2]*100)}])",
       transform=a.transAxes, ha="right", va="top", fontsize=7)
a.set_xticks(o); a.set_xlabel("Shared passages (5 per agent)")
a.set_ylabel("Communication gain\nMAS − No-comm (pp)"); a.set_ylim(-1.5, 15.5)
a.set_title("A. Overlap response (FRAMES, n = 624)", loc="left", fontsize=8.5, fontweight="bold")

cells = ["EXCL_LOW", "EXCL_HIGH", "SHAR_LOW", "SHAR_HIGH"]
labels = ["Low", "High", "Low", "High"]
C = [E2["cells"][c]["P_int"] for c in cells]
yb = [c[0] * 100 for c in C]; lb = [(c[0] - c[1]) * 100 for c in C]; hb = [(c[2] - c[0]) * 100 for c in C]
xs = [0, 1, 2.4, 3.4]
b.axhline(0, color="0.6", lw=0.6, ls="--", zorder=0)
b.bar(xs, yb, width=0.8, color=[DARK, DARK, LIGHT, LIGHT], hatch=["", "///", "", "///"], edgecolor="white", lw=0.4)
b.errorbar(xs, yb, yerr=[lb, hb], fmt="none", ecolor="0.15", capsize=2.5, elinewidth=0.8)
b.set_xticks(xs); b.set_xticklabels(labels, fontsize=7.5)
for xc, g in ((0.5, "Exclusive gold"), (2.9, "Shared gold")):
    b.text(xc, -0.13, g, transform=b.get_xaxis_transform(), ha="center", va="top", fontsize=7.5, fontweight="bold")
    b.plot([xc - 0.75, xc + 0.75], [-0.115, -0.115], transform=b.get_xaxis_transform(), color="0.4", lw=0.6, clip_on=False)
b.set_xlabel("Non-gold passage overlap\n(Low: 0; High: 3 shared non-gold passages)", labelpad=17, fontsize=7.5)
b.set_ylabel("Communication gain\nMAS − No-comm (pp)"); b.set_ylim(-1.5, 15.5)
pl, ov = E2["primary"]["placement"], E2["primary"]["overlap"]
g1 = lambda v: (f"{v:.1f}" if abs(v) >= 0.05 else "0.0").replace("-", "\u2212")
b.text(0.98, 0.97, f"Placement effect: {f1(pl[0]*100)} pp\n(95% CI [{g1(pl[1]*100)}, {g1(pl[2]*100)}])\n"
       f"Overlap effect: {g1(ov[0]*100)} pp\n(95% CI [{g1(ov[1]*100)}, {g1(ov[2]*100)}])", linespacing=1.25,
       transform=b.transAxes, ha="right", va="top", fontsize=7)
b.set_title(f"B. Gold placement (FRAMES, {E2['n']} eligible)", loc="left", fontsize=8.5, fontweight="bold")
for f in ("pdf", "png"):
    fig.savefig(f"paper/figures/fig2_overlap_placement.{f}", bbox_inches="tight", dpi=300)
print("saved")
