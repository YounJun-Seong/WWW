"""Figures for the revised WWW paper. Numbers are read from the committed result files (runs/bx, runs/bx2).
  Figure 1  evaluation framework (concept; no results)          -> paper/figures/fig1_framework.{pdf,png}
  Figure 2  evidence placement and integration (B3, BX2-A)       -> paper/figures/fig_placement_integration.{pdf,png}
  Figure 3  information access vs system utility (B1)            -> paper/figures/fig_access_utility.{pdf,png}
Palette: reference categorical slots 1-3 (blue / orange / aqua), documented as passing all-pairs CVD checks; anything
beyond three identities is neutral gray + marker shape + direct labels.   python paper/make_figs_bx.py   (from repo root)"""
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

sys.path.insert(0, "scripts")
from common import normalize  # noqa: E402

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
                     "axes.edgecolor": "#52514e", "xtick.color": "#52514e", "ytick.color": "#52514e"})
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
GRAY, INK, INK2, GRID = "#898781", "#0b0b0b", "#52514e", "#e1e0d9"
LIGHTGRAY = "#c3c2b7"
OUT = "paper/figures"


def save(fig, name):
    for f in ("pdf", "png"):
        fig.savefig(f"{OUT}/{name}.{f}", bbox_inches="tight", dpi=300)
    plt.close(fig)


# ----------------------------------------------------------------------------------------------------------- Figure 1
def fig1():
    fig = plt.figure(figsize=(7.0, 2.75)); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 100); ax.set_ylim(0, 40); ax.axis("off")
    stages = [("1  Evidence access", "Local · Peer · Remote"), ("2  Communication", "Answer · Evidence"),
              ("3  Evidence use", "Revision · Synthesis"), ("4  Utility", "Accuracy · Cost")]
    xs = [1, 35, 60, 81]; ws = [32, 23, 19, 18]
    for (t, s), x, w in zip(stages, xs, ws):
        ax.add_patch(FancyBboxPatch((x, 33.2), w, 5.6, boxstyle="round,pad=0,rounding_size=0.8", fc="#f0efec", ec="none"))
        ax.text(x + 1.2, 37.2, t, fontsize=8.2, fontweight="bold", color=INK, va="center")
        ax.text(x + 1.2, 34.7, s, fontsize=7, color=INK2, va="center")
    for x0, x1 in ((33.2, 34.8), (58.2, 59.8), (79.2, 80.8)):
        ax.add_patch(FancyArrowPatch((x0, 36), (x1, 36), arrowstyle="-|>", mutation_scale=8, color=GRAY, lw=0.8))

    def box(x, y, w, h, label, fc="white", ec=INK2, fs=7.2, bold=False, color=INK):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.7", fc=fc, ec=ec, lw=0.7))
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=fs, fontweight="bold" if bold else "normal", color=color)

    def arrow(p, q, color, ls="-", rad=0.0, lw=1.3, ms=8, style="-|>"):
        ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=ms, color=color, lw=lw, linestyle=ls,
                                     connectionstyle=f"arc3,rad={rad}", shrinkA=1, shrinkB=1))

    def docs(x, y, label):
        for k in range(3):
            ax.add_patch(Rectangle((x + 0.6 * k, y - 0.6 * k), 5.2, 4.4, fc="white", ec=INK2, lw=0.6))
        for k in range(3):
            ax.plot([x + 2.0, x + 6.0], [y - 0.4 - 0.9 * k, y - 0.4 - 0.9 * k], color=LIGHTGRAY, lw=0.6)
        ax.text(x + 3.2, y - 3.0, label, ha="center", va="top", fontsize=6.6, color=INK2)

    # panel 1: two agents, two collections
    docs(2.0, 24.5, "Collection A"); docs(2.0, 9.5, "Collection B")
    box(14.5, 21.0, 7.5, 5.0, "Agent A", bold=True); box(14.5, 6.0, 7.5, 5.0, "Agent B", bold=True)
    arrow((8.6, 23.4), (14.4, 23.4), GRAY); arrow((8.6, 8.4), (14.4, 8.4), GRAY)
    arrow((18.2, 11.1), (18.2, 20.9), BLUE, lw=1.6, style="<|-|>")                    # peer transfer A <-> B (both directions)
    ax.text(19.0, 16.0, "A ↔ B", fontsize=6.4, color=BLUE, va="center")
    arrow((15.0, 20.9), (6.8, 10.4), ORANGE, ls=(0, (3, 2)), rad=0.12)               # remote query A -> collection B
    arrow((8.7, 8.0), (16.6, 20.9), ORANGE, rad=0.12)                                 # remote results collection B -> A
    ax.text(9.0, 16.2, "query", fontsize=6.2, color=ORANGE, rotation=55, ha="center", va="center")
    ax.text(14.6, 13.4, "results", fontsize=6.2, color=ORANGE, rotation=58, ha="center", va="center")
    for c, ls, t, lx, ly in ((GRAY, "-", "Local access", 1.5, 2.6), (BLUE, "-", "Peer evidence transfer", 15.5, 2.6),
                             (ORANGE, (0, (3, 2)), "Remote query", 1.5, 0.6), (ORANGE, "-", "Remote results", 15.5, 0.6)):
        ax.plot([lx, lx + 2.4], [ly, ly], color=c, lw=1.4, ls=ls); ax.text(lx + 2.9, ly, t, va="center", fontsize=6.4, color=INK)
    # panel 2: message content (one type per condition)
    ax.text(46.0, 29.3, "Alternative message conditions", ha="center", va="center", fontsize=6.7, color=INK, style="italic")
    box(36.0, 22.5, 20.0, 5.0, "Answer (+ reasoning)", ec=BLUE)
    box(36.0, 15.0, 20.0, 5.0, "Evidence passages", ec=BLUE, fc="#eaf2fc")
    box(36.0, 7.5, 20.0, 5.0, "Answer + evidence", ec=BLUE)
    ax.plot([56.8, 57.6, 57.6, 56.8], [25.0, 25.0, 10.0, 10.0], color=BLUE, lw=0.9)
    arrow((57.6, 17.5), (60.3, 17.5), BLUE, lw=1.1, ms=7)
    ax.text(46.0, 5.4, "one message type per condition", ha="center", va="top", fontsize=6.3, color=INK2)
    # panel 3: evidence use
    box(60.5, 19.5, 8.3, 5.5, "Revision\nagent A", fs=6.8); box(69.7, 19.5, 8.3, 5.5, "Revision\nagent B", fs=6.8)
    box(60.5, 8.0, 17.5, 5.0, "Synthesis / combiner", fc="#f0efec")
    arrow((64.65, 19.4), (64.65, 13.1), GRAY, lw=0.9, ms=6); arrow((73.85, 19.4), (73.85, 13.1), GRAY, lw=0.9, ms=6)
    ax.text(69.25, 26.6, "① candidate generation", ha="center", va="center", fontsize=6.5, color=INK, style="italic")
    ax.text(69.25, 6.6, "② candidate preservation", ha="center", va="center", fontsize=6.5, color=INK, style="italic")
    ax.text(69.25, 4.3, "evidence received ≠ evidence\nintegrated ≠ answer preserved", ha="center", va="top", fontsize=6.3,
            color=INK, fontweight="bold")
    # panel 4: utility
    box(82.0, 22.5, 16.0, 5.0, "Accuracy", bold=True)
    ax.text(90.0, 19.6, "Cost, reported separately:", ha="center", va="top", fontsize=6.6, color=INK)
    ax.text(90.0, 17.3, "LLM calls · input/output tokens\nretrieval calls · measured latency",
            ha="center", va="top", fontsize=6.3, color=INK2, linespacing=1.3)
    arrow((78.2, 10.5), (81.8, 24.0), GRAY, lw=0.9, ms=6)
    save(fig, "fig1_framework")


# ----------------------------------------------------------------------------------------------------------- Figure 2
def fig2():
    b3 = json.load(open("runs/bx/RESULTS_b3_hotpot.json"))
    cells = ["NONE", "P_EXCL", "P_SHAR", "F_SPLIT", "F_CONC", "F_SHAR"]
    lab = json.load(open("data/hotpot_final_labels.json"))
    em = lambda i, a: bool(a) and normalize(a) == normalize(lab[str(i)])
    parts = {}
    for c in ("F_SPLIT", "F_CONC"):
        G = {x["idx"]: x for x in map(json.loads, open(f"runs/bx/hotpot_B3_{c}_gen.jsonl"))}
        k = dict(kept=0, lost=0, created=0, none=0)
        for i, g in G.items():
            cand = em(i, g["A1m"]["answer"]) or em(i, g["B1m"]["answer"]); fin = em(i, g["MAS"]["answer"])
            k["kept" if cand and fin else "lost" if cand else "created" if fin else "none"] += 1
        parts[c] = k
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.0, 2.55), gridspec_kw=dict(width_ratios=[1.55, 1], wspace=0.32))
    # panel A: grouped bars MAS vs NoComm per cell
    xs = []; x = 0
    for k, c in enumerate(cells):
        xs.append(x); x += 1.12 + (0.5 if k in (0, 2) else 0)
    w = 0.38
    for j, (key, col, name) in enumerate((("MAS", BLUE, "MAS (communication)"), ("NOCOMM", GRAY, "No-comm"))):
        ys = [b3["cells"][c][key][0] * 100 for c in cells]
        lo = [(b3["cells"][c][key][0] - b3["cells"][c][key][1]) * 100 for c in cells]
        hi = [(b3["cells"][c][key][2] - b3["cells"][c][key][0]) * 100 for c in cells]
        xx = [v + (j - 0.5) * (w + 0.04) for v in xs]
        a.bar(xx, ys, width=w, color=col, label=name, zorder=2)
        a.errorbar(xx, ys, yerr=[lo, hi], fmt="none", ecolor=INK, elinewidth=0.6, capsize=1.6, zorder=3)
    for c, v in zip(cells, xs):
        p = b3["cells"][c]["P_int"][0] * 100
        top = max(b3["cells"][c]["MAS"][2], b3["cells"][c]["NOCOMM"][2]) * 100
        a.text(v, top + 2.0, f"{p:+.1f}".replace("-", "−"), ha="center", fontsize=6.8, color=INK)
    a.text(0.99, 0.99, "bars: accuracy, 95% CI\nlabels: MAS − No-comm (pp)", transform=a.transAxes, ha="right", va="top",
           fontsize=6.3, color=INK2, linespacing=1.2)
    a.set_xticks(xs); a.set_xticklabels(["none", "exclusive", "shared", "split", "concen-\ntrated", "shared"], fontsize=6.6)
    for (lo_, hi_), t in (((xs[0], xs[0]), "0 of 2"), ((xs[1], xs[2]), "1 of 2"), ((xs[3], xs[5]), "2 of 2")):
        xc = (lo_ + hi_) / 2
        a.text(xc, -0.215, t, transform=a.get_xaxis_transform(), ha="center", va="top", fontsize=7, fontweight="bold")
        a.plot([lo_ - 0.45, hi_ + 0.45], [-0.2, -0.2], transform=a.get_xaxis_transform(), color=INK2, lw=0.6, clip_on=False)
    a.set_xlabel("Supporting paragraphs in the team (0, 1 or 2 of 2) and their placement", labelpad=24, fontsize=7.5)
    a.set_ylabel("Team accuracy (%)"); a.set_ylim(0, 80); a.yaxis.grid(True, color=GRID, lw=0.5); a.set_axisbelow(True)
    a.legend(frameon=False, fontsize=6.8, loc="upper left")
    a.set_title(f"A. Controlled support availability and placement\n(HotpotQA, n = {b3['n']})", loc="left", fontsize=8.5, fontweight="bold")
    # panel B: 100% stacked decomposition
    segs = (("kept", BLUE, "Correct, preserved"), ("lost", ORANGE, "Lost in synthesis"),
            ("created", AQUA, "Created in synthesis"), ("none", LIGHTGRAY, "No correct revised/final answer"))
    for r, c in enumerate(("F_SPLIT", "F_CONC")):
        left = 0
        for key, col, name in segs:
            v = parts[c][key] / 10
            b.barh(r, v, left=left, height=0.5, color=col, edgecolor="white", lw=1.0, label=name if r == 0 else None)
            if v >= 8:
                b.text(left + v / 2, r, f"{v:.1f}", ha="center", va="center", fontsize=6.8,
                       color="white" if col in (BLUE,) else INK)
            left += v
    for r, c in enumerate(("F_SPLIT", "F_CONC")):
        x0 = parts[c]["kept"] / 10; v = parts[c]["lost"] / 10
        b.annotate(f"Lost {v:.1f}%", (x0 + v / 2, r - 0.28), xytext=(x0 + v / 2 + 4, r - 0.48), fontsize=6.5, color=INK,
                   arrowprops=dict(arrowstyle="-", color=INK2, lw=0.6), va="center")
    b.set_yticks([0, 1]); b.set_yticklabels(["Split\n(1 + 1)", "Concentrated\n(2 + 0)"], fontsize=7); b.invert_yaxis()
    b.set_ylim(1.5, -0.75)
    b.set_xlim(0, 100); b.set_xlabel("Questions (%)")
    gain = {c: b3["cells"][c]["P_int"][0] * 100 for c in ("F_SPLIT", "F_CONC")}
    b.text(103, -0.6, "Comm. gain\n(Panel A)", ha="left", va="center", fontsize=6.3, color=INK2, clip_on=False)
    b.text(133, -0.6, "Final\naccuracy", ha="left", va="center", fontsize=6.3, color=INK2, clip_on=False)
    for r, c in enumerate(("F_SPLIT", "F_CONC")):
        k, cr = parts[c]["kept"] / 10, parts[c]["created"] / 10
        b.text(103, r, f"{gain[c]:+.1f} pp", va="center", fontsize=6.6, color=INK, clip_on=False)
        b.text(133, r, f"{k + cr:.1f}%" + (f"\n= {k:.1f} + {cr:.1f}" if cr else ""), va="center", fontsize=6.6, color=INK,
               clip_on=False, linespacing=1.2)
    b.legend(frameon=False, fontsize=6.4, ncol=2, loc="upper left", bbox_to_anchor=(-0.02, -0.28), columnspacing=0.8, handlelength=1.0)
    b.set_title("B. Where the split-evidence loss occurs\n(outcome of each question, %)", loc="left", fontsize=8.5, fontweight="bold")
    save(fig, "fig_placement_integration")
    return parts


# ----------------------------------------------------------------------------------------------------------- Figure 3
FAM = {"LOCAL_ONE": ("local", None), "LOCAL_SC": ("local_sc", None), "NOCOMM": ("nocomm", None),
       "FULL(MAS)": ("peer", "MAS"), "PEER_EV_5": ("peer", "Peer-EV"), "CQ_Q_5": ("remote", "Direct-Q"),
       "CQ_GEN_5": ("remote", "Direct-Gen"), "UNI_ONE": ("central", "UNI-ONE"), "UNI_SC": ("central_sc", None)}
# POOL / SC are not drawn separately: on HotpotQA they are the identical calls (A5 ∪ B5 = pooled list), on FRAMES within
# 0.2 / 0.8 pp and 2 / 31 tokens of UNI_ONE / UNI_SC (stated in the caption).
STYLE = {"local": dict(c=GRAY, m="o", fill=True), "local_sc": dict(c=GRAY, m="o", fill=False),
         "nocomm": dict(c=GRAY, m="s", fill=True), "peer": dict(c=BLUE, m="s", fill=True),
         "remote": dict(c=ORANGE, m="^", fill=True), "central": dict(c=AQUA, m="D", fill=True),
         "central_sc": dict(c=AQUA, m="D", fill=False)}
OFF = {"frames": {"FULL(MAS)": (3, -11), "PEER_EV_5": (-33, 3), "CQ_Q_5": (5, -11), "CQ_GEN_5": (-20, 7), "UNI_ONE": (5, -11)},
       "hotpot": {"FULL(MAS)": (5, -11), "PEER_EV_5": (-12, 8), "CQ_Q_5": (-18, -13), "CQ_GEN_5": (-2, -13), "UNI_ONE": (-14, 8)}}


def fig3():
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.85), gridspec_kw=dict(wspace=0.24))
    for ax, ds, title in zip(axes, ("frames", "hotpot"), ("A. FRAMES (n = 624, judge)", "B. HotpotQA (n = 1000, EM)")):
        S = json.load(open(f"runs/bx/RESULTS_b14_{ds}.json"))["systems"]
        for k, (fam, lab) in FAM.items():
            v = S[k]; st = STYLE[fam]; x = v["tokens"] / 1000; y = v["accuracy"][0] * 100
            ax.plot([x, x], [v["accuracy"][1] * 100, v["accuracy"][2] * 100], color=st["c"], lw=0.6, alpha=0.4, zorder=2)
            ax.scatter([x], [y], s=36, marker=st["m"], color=st["c"] if st["fill"] else "white",
                       edgecolors="white" if st["fill"] else st["c"], linewidths=0.8 if st["fill"] else 1.4, zorder=3)
            if lab:
                dx, dy = OFF[ds][k]
                ax.annotate(lab, (x, y), xytext=(dx, dy), textcoords="offset points", fontsize=6.5, color=INK, zorder=4)
        ax.set_xlabel("Tokens per question (thousands, input + output)", fontsize=7.3)
        ax.set_ylabel("Accuracy (%)", fontsize=7.5)
        ax.yaxis.grid(True, color=GRID, lw=0.5); ax.xaxis.grid(True, color=GRID, lw=0.5); ax.set_axisbelow(True)
        ax.set_title(title, loc="left", fontsize=8.5, fontweight="bold"); ax.set_xlim(0, 10)
    axes[0].set_ylim(22, 44); axes[1].set_ylim(30, 64)
    fig.text(0.99, 0.985, "y-axis ranges differ between panels", ha="right", va="top", fontsize=6.3, color=INK2)
    L2 = lambda m, c, fill, name: plt.Line2D([], [], ls="", marker=m, ms=5.5, color=c, markerfacecolor=c if fill else "white",
                                             markeredgecolor=c, label=name)
    hs = [L2("o", GRAY, True, "Local single agent"), L2("s", GRAY, True, "Two agents, no communication"),
          L2("s", BLUE, True, "Peer evidence transfer (MAS, Peer-EV)"), L2("^", ORANGE, True, "Direct remote retrieval (Direct-Q, Direct-Gen)"),
          L2("D", AQUA, True, "Federated / pooled retrieval (UNI-ONE)"), L2("o", INK2, False, "Hollow markers: self-consistency (k = 4)")]
    fig.legend(handles=hs, loc="upper center", ncol=3, frameon=False, fontsize=6.5, bbox_to_anchor=(0.5, -0.04),
               columnspacing=1.2, handletextpad=0.3)
    save(fig, "fig_access_utility")


if __name__ == "__main__":
    fig1(); p = fig2(); fig3(); print("saved; B3 decomposition", p)
