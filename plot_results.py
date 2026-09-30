import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("rl_tcp_traces.csv")

step = df["step"].to_numpy()
# Convert raw bytes to segments (MSS = 1448)
cwnd_mss = (df["cwnd"] / 1448.0).to_numpy()
# Filter out 2^32-1 initialization sentinel values for display
raw_ssthresh = (df["ssThresh"] / 1448.0).to_numpy()
ssthresh_mss = np.where(raw_ssthresh > 500, np.nan, raw_ssthresh)

rtt_ms = (df["rtt_us"] / 1000.0).to_numpy()
reward = df["reward"].to_numpy()

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

# 1. Congestion Window (cWnd) & ssThresh
ax1.plot(step, cwnd_mss, label="cWnd (MSS)", color="#1f77b4", linewidth=2.0)
ax1.plot(step, ssthresh_mss, label="ssThresh (MSS)", color="#ff7f0e", linestyle="--", linewidth=1.5)
ax1.set_ylabel("Segments (MSS)")
ax1.set_ylim(0, max(np.nanmax(cwnd_mss) * 1.25, 40))
ax1.set_title("PPO Agent TCP Evaluation Dynamics (Zoomed Scale)")
ax1.grid(True, alpha=0.3)
ax1.legend()

# 2. Round Trip Time (RTT in ms)
ax2.plot(step, rtt_ms, label="RTT (ms)", color="#d62728", linewidth=1.8)
ax2.axhline(y=42.0, color='gray', linestyle=':', label="Physical Base RTT (~42ms)")
ax2.set_ylabel("RTT (ms)")
ax2.grid(True, alpha=0.3)
ax2.legend()

# 3. Model Action & Instant Reward
ax3.plot(step, reward, label="Step Reward", color="#2ca02c", linewidth=1.5)
ax3.set_xlabel("Step Count")
ax3.set_ylabel("Reward")
ax3.grid(True, alpha=0.3)
ax3.legend()

plt.tight_layout()
plt.savefig("rl_tcp_zoomed_performance.png", dpi=300)
print("[*] Zoomed plot saved: rl_tcp_zoomed_performance.png")