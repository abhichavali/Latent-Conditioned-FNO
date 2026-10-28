# Mean error vs number of simulations, v3 vs scratch (summary.json -> fresh_eqs.png, used in the main README)
import json
import matplotlib.pyplot as plt

m = json.load(open("summary.json"))["metrics"]["rel_slot0"]["mean"]
x = lambda d: [max(int(k), 0.5) for k in d]  # put k = 0 at 0.5 on the log axis
fig, ax = plt.subplots(figsize=(5.5, 3.5), dpi=150)
ax.plot(x(m["ctx"]), list(m["ctx"].values()), "o-", label="v3, no gradient steps")
ax.plot(x(m["scratch"]), list(m["scratch"].values()), "s-", label="FNO from scratch")
ax.axhline(m["persistence"], ls="--", c="gray", label="persistence")
ax.set_xscale("log", base=2)
ax.set_xticks([0.5, 1, 2, 4, 8, 16, 32, 64, 128], ["0", 1, 2, 4, 8, 16, 32, 64, 128])
ax.set_xlabel("simulations of the new equation")
ax.set_ylabel("test rel L2")
ax.set_title("10 unseen equations from the training families")
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig("fresh_eqs.png")
