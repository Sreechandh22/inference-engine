"""Regenerate the plot from hardcoded sweep results without re-running Modal."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CONCURRENCIES = [1, 4, 16, 32, 64]

our = {
    1:  {"throughput": 56.8,   "peak_mem_gb": 2.96, "oom": False},
    4:  {"throughput": 178.0,  "peak_mem_gb": 2.96, "oom": False},
    16: {"throughput": 672.8,  "peak_mem_gb": 2.97, "oom": False},
    32: {"throughput": 1187.8, "peak_mem_gb": 2.99, "oom": False},
    64: {"throughput": 1340.1, "peak_mem_gb": 3.02, "oom": False},
}
hf = {
    1:  {"throughput": 49.6,  "peak_mem_gb": 2.22, "oom": False},
    4:  {"throughput": 57.3,  "peak_mem_gb": 2.22, "oom": False},
    16: {"throughput": 49.3,  "peak_mem_gb": 2.22, "oom": False},
    32: {"throughput": 64.7,  "peak_mem_gb": 2.22, "oom": False},
    64: {"throughput": 63.1,  "peak_mem_gb": 2.22, "oom": False},
}

plot_concurrencies = [c for c in CONCURRENCIES if c > 1]
our_x = [c for c in plot_concurrencies if not our[c]["oom"]]
hf_x  = [c for c in plot_concurrencies if not hf[c]["oom"]]

def _vals(results, key, xs):
    return [results[c][key] for c in xs]

fig, axes = plt.subplots(1, 2, figsize=(11, 5))
fig.suptitle(
    "Inference Engine vs HuggingFace — Concurrency Sweep (A10G, TinyLlama-1.1B, 150 tok/req)",
    fontsize=12,
)

ax = axes[0]
ax.plot(our_x, _vals(our, "throughput", our_x), "o-", label="Ours (PagedAttn + batched decode)", color="steelblue")
ax.plot(hf_x,  _vals(hf,  "throughput", hf_x),  "s--", label="HuggingFace (sequential)", color="coral")
ax.set_xlabel("Concurrent requests")
ax.set_ylabel("Throughput (tok/sec)")
ax.set_title("Throughput")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

ax = axes[1]
ax.plot(our_x, _vals(our, "peak_mem_gb", our_x), "o-", label="Ours", color="steelblue")
ax.plot(hf_x,  _vals(hf,  "peak_mem_gb", hf_x),  "s--", label="HuggingFace", color="coral")
ax.set_xlabel("Concurrent requests")
ax.set_ylabel("Peak GPU memory (GB)")
ax.set_title("GPU Memory Usage")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
os.makedirs("benchmarks/plots", exist_ok=True)
out = "benchmarks/plots/concurrency_sweep.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved → {out}")
