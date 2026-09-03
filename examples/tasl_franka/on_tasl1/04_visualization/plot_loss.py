import csv, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rows=list(csv.DictReader(open("/tmp/loss_10task.csv")))
step=[int(r["step"]) for r in rows]
loss=[float(r["loss"]) for r in rows]
gn=[float(r["grad_norm"]) for r in rows]

fig,ax=plt.subplots(1,2,figsize=(13,4.6))
ax[0].plot(step,loss,lw=1.6,color="#1f77b4")
ax[0].set_yscale("log")
ax[0].set_xlabel("training step"); ax[0].set_ylabel("training loss (log scale)")
ax[0].set_title("pi05_droid_franka_lora_10task — training loss\n250 demos / 10 tasks / 66,463 frames, batch 32")
ax[0].grid(alpha=.3, which="both")
ax[0].annotate(f"step 0\n{loss[0]:.3f}", xy=(step[0],loss[0]), xytext=(1200,0.9),
               arrowprops=dict(arrowstyle="->",lw=.8), fontsize=8)
ax[0].annotate(f"step {step[-1]}\n{loss[-1]:.4f}", xy=(step[-1],loss[-1]), xytext=(13000,0.03),
               arrowprops=dict(arrowstyle="->",lw=.8), fontsize=8)
for s in range(2000,20001,2000):
    ax[0].axvline(s, color="gray", ls=":", lw=.6)
ax[0].text(0.98,0.96,"dotted lines = checkpoints (every 2000 steps)",transform=ax[0].transAxes,
           ha="right",va="top",fontsize=7.5,color="gray")

ax[1].plot(step,gn,lw=1.4,color="#d62728")
ax[1].set_yscale("log")
ax[1].set_xlabel("training step"); ax[1].set_ylabel("grad norm (log scale)")
ax[1].set_title("gradient norm")
ax[1].grid(alpha=.3, which="both")

plt.tight_layout()
plt.savefig("/tmp/loss_curve_10task.png", dpi=160)
print("saved. 首末:", step[0], loss[0], "->", step[-1], loss[-1])
