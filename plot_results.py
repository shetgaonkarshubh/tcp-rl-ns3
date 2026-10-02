import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("rl_tcp_traces.csv")

step = df["step"].to_numpy()

# 1. Extract cWnd and Dynamic BDP / ssThresh in MSS units (1448B scale)
if "cwnd_mss" in df.columns:
    cwnd_mss = df["cwnd_mss"].to_numpy(dtype=float)
else:
    cwnd_mss = (df["cwnd"] / 1448.0).to_numpy(dtype=float)

raw_ssthresh = (df["ssThresh"] / 1448.0).to_numpy(dtype=float)
ssthresh_mss = np.where(raw_ssthresh > 500.0, np.nan, raw_ssthresh)

# 2. Extract RTT (ms) and Auto-Detect Unloaded Physical Base RTT
rtt_ms = (df["rtt_us"] / 1000.0).to_numpy(dtype=float)
if "base_rtt_ms" in df.columns and df["base_rtt_ms"].iloc[-1] > 1.0:
    base_rtt_ms = float(df["base_rtt_ms"].iloc[-1])
else:
    # Fallback: estimate unloaded base RTT from the 5th percentile of observed RTTs
    base_rtt_ms = float(np.nanpercentile(rtt_ms, 5))

reward = df["reward"].to_numpy(dtype=float)
reward_smooth = pd.Series(reward).rolling(window=15, min_periods=1).mean().to_numpy()

# 3. Detect Random Bottleneck Shift Steps (where dynamic BDP / ssThresh jumps > 1.0 MSS)
valid_ssthresh = np.nan_to_num(ssthresh_mss, nan=0.0)
shift_indices = np.where(np.abs(np.diff(valid_ssthresh)) > 1.0)[0] + 1
shift_steps = step[shift_indices]

# 4. Compute Summary KPIs for the Figure Header
valid_mask = ~np.isnan(ssthresh_mss)
tracking_mae = (
    float(np.mean(np.abs(cwnd_mss[valid_mask] - ssthresh_mss[valid_mask])))
    if np.any(valid_mask)
    else 0.0
)
mean_rtt = float(np.mean(rtt_ms))
p95_rtt = float(np.percentile(rtt_ms, 95))
mean_reward = float(np.mean(reward))

# ==============================================================================
# Plotting Setup
# ==============================================================================
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(11, 8.5), sharex=True)

# --- Subplot 1: Congestion Window (cWnd) vs. Dynamic BDP / ssThresh ---
ax1.plot(step, cwnd_mss, label="Agent cWnd (MSS)", color="#1f77b4", linewidth=2.2, zorder=4)
if np.any(valid_mask):
    ax1.step(
        step,
        ssthresh_mss,
        where="post",
        label="Dynamic BDP / ssThresh (MSS)",
        color="#ff7f0e",
        linestyle="--",
        linewidth=1.8,
        zorder=3
    )
    ax1.fill_between(
        step,
        ssthresh_mss - 1.5,
        ssthresh_mss + 1.5,
        step="post",
        color="#ff7f0e",
        alpha=0.15,
        label="BBR Sweet-Spot Band (±1.5 MSS)"
    )

y1_max = max(
    np.nanmax(cwnd_mss) if len(cwnd_mss) else 40.0,
    np.nanmax(ssthresh_mss) if np.any(valid_mask) else 40.0,
    40.0
)
ax1.set_ylabel("Segments (MSS)")
ax1.set_ylim(0, y1_max * 1.18)
ax1.set_title(
    f"PPO Agent TCP Evaluation Dynamics (Stochastic Bottleneck)\n"
    f"BDP Tracking MAE: {tracking_mae:.2f} MSS  |  Mean RTT: {mean_rtt:.1f} ms "
    f"(P95: {p95_rtt:.1f} ms)  |  Mean Reward: {mean_reward:+.2f}",
    fontsize=11,
    fontweight="bold"
)
ax1.grid(True, alpha=0.3)
ax1.legend(loc="upper right", framealpha=0.9)

# --- Subplot 2: Round Trip Time (RTT in ms) & Queue Envelope ---
ax2.plot(step, rtt_ms, label="Observed RTT (ms)", color="#d62728", linewidth=1.8, zorder=4)
ax2.axhline(
    y=base_rtt_ms,
    color="#4d4d4d",
    linestyle=":",
    linewidth=1.6,
    label=f"Physical Base RTT (~{base_rtt_ms:.1f} ms)"
)
ax2.axhline(
    y=base_rtt_ms * 1.25,
    color="#ff7f0e",
    linestyle="-.",
    linewidth=1.2,
    alpha=0.8,
    label=f"BBR +25% Queue Ceiling ({base_rtt_ms * 1.25:.1f} ms)"
)
ax2.set_ylabel("RTT (ms)")
ax2.set_ylim(base_rtt_ms * 0.85, max(np.nanmax(rtt_ms) * 1.10, base_rtt_ms * 1.45))
ax2.grid(True, alpha=0.3)
ax2.legend(loc="upper right", framealpha=0.9)

# --- Subplot 3: Instant & Smoothed Step Reward ---
ax3.plot(
    step,
    reward,
    label="Instant Step Reward",
    color="#2ca02c",
    alpha=0.40,
    linewidth=1.2,
    zorder=3
)
ax3.plot(
    step,
    reward_smooth,
    label="15-Step Rolling Mean Reward",
    color="#156b15",
    linewidth=2.0,
    zorder=4
)
ax3.axhline(y=0.0, color="gray", linestyle=":", linewidth=1.0, alpha=0.7)
ax3.set_xlabel("Step Count (100 ms / step)")
ax3.set_ylabel("Reward")
ax3.grid(True, alpha=0.3)
ax3.legend(loc="lower right", framealpha=0.9)

# Mark random bottleneck shift events across all three subplots
for idx, s_step in enumerate(shift_steps):
    for ax in (ax1, ax2, ax3):
        ax.axvline(
            x=s_step,
            color="#7f7f7f",
            linestyle="--",
            linewidth=1.0,
            alpha=0.55,
            label="Bottleneck Shift" if (idx == 0 and ax is ax1) else None
        )
if len(shift_steps) > 0:
    ax1.legend(loc="upper right", framealpha=0.9)

plt.tight_layout()
plt.savefig("rl_tcp_zoomed_performance.png", dpi=300)
print(
    f"[*] Zoomed plot saved: rl_tcp_zoomed_performance.png "
    f"(Detected {len(shift_steps)} random bottleneck shifts | Base RTT: {base_rtt_ms:.1f} ms)"
)