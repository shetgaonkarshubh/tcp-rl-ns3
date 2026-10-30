"""
Interactive Multi-Model Real-Time TCP Congestion Control Dashboard.
Supports live switching between:
  [1] Oracle Baseline (Privileged PPO, MAE: 0.65 MSS, Mean RTT: 84.0 ms)
  [2] Naive Blind Agent (No Oracle / Basic Feedback, MAE: 3.75 MSS, Mean RTT: 90.1 ms)
  [3] Tuned Little's Law Agent (Physics-Driven Real-World, MAE: 1.41 MSS, Mean RTT: 85.1 ms)

Hotkeys:
  [1, 2, 3] : Switch models on the fly
  [Space]   : Pause / Resume playback
  [Right/N] : Single step forward
  [R]       : Restart from Step 0
  [Q / Esc] : Quit
"""

import sys
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

MODELS_METADATA = {
    1: {
        "title": "Model 1: Oracle Baseline (Privileged PPO)",
        "tag": "ORACLE (SUPERVISED)",
        "file": "/home/shubh/tcpai/rl_tcp_traces.csv",
        "description": "Agent directly observes true BDP in 8D observation vector. Theoretical upper bound.",
        "expected_mae": "0.65 MSS",
        "expected_rtt": "84.0 ms",
        "color": "#1f77b4"
    },
    2: {
        "title": "Model 2: Naive Blind Agent (Zero Oracle)",
        "tag": "NAIVE BLIND (UNPRIVILEGED)",
        "file": "/home/shubh/tcpai/oracle_free_tcp/blind_agent_benchmark.csv",
        "description": "Zero BDP knowledge. Relies on raw latency inflation and throughput ratios without physical delay scaling.",
        "expected_mae": "3.75 MSS",
        "expected_rtt": "90.1 ms",
        "color": "#ff7f0e"
    },
    3: {
        "title": "Model 3: Little's Law Tuned Agent (Real-World Physics)",
        "tag": "LITTLE'S LAW TUNED (STATE-OF-THE-ART)",
        "file": "/home/shubh/tcpai/oracle_free_tcp/perfect_blind_results.csv",
        "description": "Zero BDP knowledge. Uses Little's Law ratio BDP = cWnd * (BaseRTT / RTT) to infer exact capacity upon queuing.",
        "expected_mae": "1.41 MSS",
        "expected_rtt": "85.1 ms",
        "color": "#2ca02c"
    }
}


def load_dataset(model_id):
    meta = MODELS_METADATA[model_id]
    filepath = meta["file"]
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Missing dataset for Model {model_id}: {filepath}")

    df = pd.read_csv(filepath)
    steps = np.arange(len(df))

    if model_id == 1:
        cwnd_mss = df["cwnd_mss"].to_numpy(dtype=float)
        raw_ssthresh = (df["ssThresh"] / 1448.0).to_numpy(dtype=float)
        target_mss = np.where(raw_ssthresh > 500.0, np.nan, raw_ssthresh)
        rtt_ms = (df["rtt_us"] / 1000.0).to_numpy(dtype=float)
        actions = df["action"].to_numpy(dtype=int)
        rewards = df["reward"].to_numpy(dtype=float)
        base_rtt_ms = 84.0

    elif model_id == 2:
        cwnd_mss = df["cwnd"].to_numpy(dtype=float)
        target_mss = df["capacity"].to_numpy(dtype=float)
        rtt_ms = df["rtt_ms"].to_numpy(dtype=float)
        # Approximate discrete actions based on delta cWnd
        diffs = np.diff(cwnd_mss, prepend=cwnd_mss[0])
        actions = np.where(diffs < -2.0, 0, np.where(diffs < 0, 1, np.where(diffs == 0, 3, np.where(diffs < 2.0, 4, 6))))
        rewards = ((df["delivered"] / 10.0) - (df["inflation"] * 4.0)).to_numpy(dtype=float)
        base_rtt_ms = 84.0

    elif model_id == 3:
        cwnd_mss = df["cwnd_mss"].to_numpy(dtype=float)
        target_mss = df["true_capacity_mss"].to_numpy(dtype=float)
        rtt_ms = df["rtt_ms"].to_numpy(dtype=float)
        rewards = df["reward"].to_numpy(dtype=float)
        # Map continuous delta action to nearest discrete ID
        deltas = df["action_offset"].to_numpy(dtype=float)
        actions = np.where(deltas < -2.0, 0, np.where(deltas < -0.2, 1, np.where(abs(deltas) <= 0.2, 3, np.where(deltas < 2.0, 4, 6))))
        base_rtt_ms = float(df["base_rtt_ms"].iloc[0]) if "base_rtt_ms" in df.columns else 84.0

    valid_target = np.nan_to_num(target_mss, nan=0.0)
    shift_indices = set(np.where(np.abs(np.diff(valid_target)) > 1.0)[0] + 1)

    return {
        "df": df,
        "steps": steps,
        "total_steps": len(df),
        "cwnd_mss": cwnd_mss,
        "target_mss": target_mss,
        "rtt_ms": rtt_ms,
        "actions": actions,
        "rewards": rewards,
        "base_rtt_ms": base_rtt_ms,
        "shift_indices": shift_indices
    }


class RealtimeTcpDashboard:
    def __init__(self, initial_model=3, step_delay=0.05, loop=False):
        self.current_model_id = initial_model
        self.step_delay = step_delay
        self.loop = loop
        self.paused = False
        self.step_once = False
        self.reload_model = False
        self.running = True

        self.data = load_dataset(self.current_model_id)

    def on_key(self, event):
        if event.key == " ":
            self.paused = not self.paused
        elif event.key in ["right", "n"]:
            self.step_once = True
            self.paused = True
        elif event.key == "r":
            self.reload_model = True
        elif event.key in ["1", "2", "3"]:
            new_id = int(event.key)
            if new_id != self.current_model_id:
                print(f"\n[*] Switching live to Model {new_id}: {MODELS_METADATA[new_id]['title']}")
                self.current_model_id = new_id
                self.data = load_dataset(self.current_model_id)
                self.reload_model = True
        elif event.key in ["q", "escape"]:
            self.running = False
            plt.close("all")

    def run(self):
        plt.ion()
        fig = plt.figure(figsize=(14.5, 9.0), facecolor="#12141a")
        fig.canvas.mpl_connect("key_press_event", self.on_key)
        gs = gridspec.GridSpec(3, 3, width_ratios=[2.4, 2.4, 1.6], hspace=0.35, wspace=0.25, left=0.06, right=0.96, top=0.91, bottom=0.07)

        ax_cwnd = fig.add_subplot(gs[0, :2], facecolor="#1a1c23")
        ax_rtt = fig.add_subplot(gs[1, :2], facecolor="#1a1c23", sharex=ax_cwnd)
        ax_reward = fig.add_subplot(gs[2, :2], facecolor="#1a1c23", sharex=ax_cwnd)
        ax_hud = fig.add_subplot(gs[:, 2], facecolor="#16181f")
        ax_hud.axis("off")

        step_idx = 0

        while self.running:
            if self.reload_model:
                step_idx = 0
                self.reload_model = False
                ax_cwnd.cla()
                ax_rtt.cla()
                ax_reward.cla()

            if step_idx >= self.data["total_steps"]:
                if self.loop:
                    step_idx = 0
                    continue
                else:
                    self.paused = True

            while self.paused and self.running:
                fig.canvas.flush_events()
                plt.pause(0.08)
                if self.step_once:
                    self.step_once = False
                    break
                if self.reload_model:
                    break

            if not self.running:
                break

            curr_step = step_idx
            curr_cwnd = self.data["cwnd_mss"][curr_step]
            cur_target = self.data["target_mss"][curr_step]
            cur_rtt = self.data["rtt_ms"][curr_step]
            cur_rew = self.data["rewards"][curr_step]
            base_rtt = self.data["base_rtt_ms"]

            err = curr_cwnd - cur_target if not np.isnan(cur_target) else 0.0
            infl = max(0.0, (cur_rtt - base_rtt) / base_rtt) * 100.0

            # Header title
            meta = MODELS_METADATA[self.current_model_id]
            fig.suptitle(
                f"TCP CONGESTION CONTROL MULTI-MODEL LIVE DASHBOARD  |  [Active: {meta['title']}]\n"
                f"Press [1] Oracle (0.65 MSS) | [2] Naive Blind (3.75 MSS) | [3] Little's Law Tuned (1.41 MSS) | [Space] Pause/Play | [Q] Quit",
                color="#f8fafc", fontsize=11, fontweight="bold", y=0.97
            )

            # Subplot 1: Congestion Window
            ax_cwnd.cla()
            ax_cwnd.set_facecolor("#1a1c23")
            ax_cwnd.grid(True, color="#2e323e", linestyle="--", alpha=0.6)
            ax_cwnd.plot(self.data["steps"][:curr_step+1], self.data["cwnd_mss"][:curr_step+1], color=meta["color"], linewidth=2.2, label=f"cWnd ({meta['tag']})")
            ax_cwnd.step(self.data["steps"][:curr_step+1], self.data["target_mss"][:curr_step+1], where="post", color="#ff4b4b", linestyle="--", linewidth=1.8, label="Physical Bottleneck BDP")
            ax_cwnd.fill_between(self.data["steps"][:curr_step+1], self.data["target_mss"][:curr_step+1] - 1.5, self.data["target_mss"][:curr_step+1] + 1.5, step="post", color="#ffa116", alpha=0.15, label="BBR Sweet-Spot (±1.5 MSS)")
            ax_cwnd.scatter([curr_step], [curr_cwnd], color="#ffffff", s=55, zorder=5)
            ax_cwnd.set_ylabel("cWnd (MSS)", color="#cbd5e1", fontsize=9.5, fontweight="bold")
            ax_cwnd.set_ylim(0, 52)
            ax_cwnd.set_xlim(0, self.data["total_steps"])
            ax_cwnd.tick_params(colors="#94a3b8", labelsize=8.5)
            ax_cwnd.legend(loc="upper right", facecolor="#22242a", edgecolor="#444751", labelcolor="#cbd5e1", fontsize=8.5)

            # Subplot 2: RTT
            ax_rtt.cla()
            ax_rtt.set_facecolor("#1a1c23")
            ax_rtt.grid(True, color="#2e323e", linestyle="--", alpha=0.6)
            ax_rtt.plot(self.data["steps"][:curr_step+1], self.data["rtt_ms"][:curr_step+1], color="#ffa116", linewidth=1.8, label="Observed RTT (ms)")
            ax_rtt.axhline(y=base_rtt, color="#818cf8", linestyle=":", linewidth=1.4, label=f"Physical Base RTT ({base_rtt:.1f} ms)")
            ax_rtt.axhline(y=base_rtt * 1.25, color="#f43f5e", linestyle="-.", linewidth=1.2, label=f"Bufferbloat Ceiling ({base_rtt*1.25:.1f} ms)")
            ax_rtt.scatter([curr_step], [cur_rtt], color="#ffffff", s=45, zorder=5)
            ax_rtt.set_ylabel("RTT (ms)", color="#cbd5e1", fontsize=9.5, fontweight="bold")
            ax_rtt.set_ylim(base_rtt * 0.85, base_rtt * 1.45)
            ax_rtt.tick_params(colors="#94a3b8", labelsize=8.5)
            ax_rtt.legend(loc="upper right", facecolor="#22242a", edgecolor="#444751", labelcolor="#cbd5e1", fontsize=8.5)

            # Subplot 3: Cumulative / Step Reward
            ax_reward.cla()
            ax_reward.set_facecolor("#1a1c23")
            ax_reward.grid(True, color="#2e323e", linestyle="--", alpha=0.6)
            ax_reward.plot(self.data["steps"][:curr_step+1], self.data["rewards"][:curr_step+1], color="#34d399", alpha=0.4, label="Step Reward")
            smooth_rew = pd.Series(self.data["rewards"][:curr_step+1]).rolling(10, min_periods=1).mean().to_numpy()
            ax_reward.plot(self.data["steps"][:curr_step+1], smooth_rew, color="#10b981", linewidth=2.0, label="Rolling Mean Reward")
            ax_reward.axhline(y=0.0, color="#64748b", linestyle=":", linewidth=1.0)
            ax_reward.set_ylabel("Reward", color="#cbd5e1", fontsize=9.5, fontweight="bold")
            ax_reward.set_xlabel("Evaluation Step (100 ms / step)", color="#cbd5e1", fontsize=9.5, fontweight="bold")
            ax_reward.tick_params(colors="#94a3b8", labelsize=8.5)
            ax_reward.legend(loc="lower right", facecolor="#22242a", edgecolor="#444751", labelcolor="#cbd5e1", fontsize=8.5)

            # Mark shifts
            for s_idx in self.data["shift_indices"]:
                if s_idx <= curr_step:
                    for ax in (ax_cwnd, ax_rtt, ax_reward):
                        ax.axvline(x=s_idx, color="#64748b", linestyle="--", linewidth=1.0, alpha=0.6)

            # Subplot 4: Heads-Up Display (HUD)
            ax_hud.cla()
            ax_hud.axis("off")

            status_color = "#34d399" if abs(err) <= 1.5 and infl < 10.0 else ("#f43f5e" if infl >= 25.0 or err > 3.0 else "#fbbf24")
            status_text = "OPTIMAL (Kleinrock)" if abs(err) <= 1.5 and infl < 10.0 else ("BUFFERBLOAT RISK" if infl >= 25.0 else "TRACKING SHIFT")

            cum_mae = np.nanmean(np.abs(self.data["cwnd_mss"][:curr_step+1] - self.data["target_mss"][:curr_step+1]))
            mean_lat = np.mean(self.data["rtt_ms"][:curr_step+1])

            hud_text = (
                f"═══════════════════════════════\n"
                f"     TELEMETRY HUD [{meta['tag'][:11]}]\n"
                f"═══════════════════════════════\n\n"
                f"• Step: {curr_step:03d} / {self.data['total_steps']}  ({curr_step*0.1:4.1f}s)\n"
                f"• System State: {status_text}\n\n"
                f"▶ OBSERVATION READINGS:\n"
                f"  ├ Bottleneck BDP : {cur_target:5.1f} MSS\n"
                f"  ├ Current cWnd   : {curr_cwnd:5.1f} MSS\n"
                f"  ├ Window Error   : {err:+5.1f} MSS\n"
                f"  ├ Observed RTT   : {cur_rtt:5.1f} ms\n"
                f"  ├ Base RTT (min) : {base_rtt:5.1f} ms\n"
                f"  └ Queue Inflation: {infl:+5.1f} %\n\n"
                f"▶ BENCHMARK SCORECARD:\n"
                f"  ├ Cumulative MAE : {cum_mae:5.2f} MSS\n"
                f"  ├ Target MAE     : {meta['expected_mae']}\n"
                f"  ├ Mean Latency   : {mean_lat:5.1f} ms\n"
                f"  └ Target Latency : {meta['expected_rtt']}\n\n"
                f"▶ LIVE CONTROLS:\n"
                f"  [1] Oracle Baseline\n"
                f"  [2] Naive Blind Agent\n"
                f"  [3] Little's Law Tuned (Best)\n"
                f"  [Space] Pause/Play  [R] Reset\n"
                f"═══════════════════════════════"
            )

            ax_hud.text(
                0.05, 0.95,
                hud_text,
                transform=ax_hud.transAxes,
                fontsize=9.0,
                family="monospace",
                verticalalignment="top",
                color="#e2e8f0",
                bbox=dict(boxstyle="round,pad=0.7", facecolor="#1e222d", edgecolor=status_color, alpha=0.95, linewidth=1.5)
            )

            plt.pause(self.step_delay)
            step_idx += 1

        plt.ioff()
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Multi-Model Real-Time TCP Congestion Control Dashboard")
    parser.add_argument("--model", type=int, choices=[1, 2, 3], default=None, help="1=Oracle Baseline, 2=Naive Blind, 3=Little's Law Tuned")
    parser.add_argument("--delay", type=float, default=0.045, help="Step playback delay in seconds (speed)")
    parser.add_argument("--loop", action="store_true", help="Loop playback indefinitely")
    args = parser.parse_args()

    chosen_model = args.model
    if chosen_model is None:
        print("\n" + "=" * 70)
        print("          SELECT A MODEL TO LAUNCH THE REAL-TIME DEMO          ")
        print("=" * 70)
        print("  [1] Model 1: Oracle Baseline (Privileged PPO, MAE: 0.65 MSS, 84.0 ms)")
        print("  [2] Model 2: Naive Blind Agent (No Oracle, MAE: 3.75 MSS, 90.1 ms)")
        print("  [3] Model 3: Little's Law Tuned Agent (State-of-the-Art, MAE: 1.41 MSS, 85.1 ms) [RECOMMENDED]")
        print("=" * 70)
        user_choice = input("Enter model number [1, 2, or 3] (default: 3): ").strip()
        if user_choice in ["1", "2", "3"]:
            chosen_model = int(user_choice)
        else:
            chosen_model = 3

    print(f"\n[*] Launching Real-Time Dashboard with {MODELS_METADATA[chosen_model]['title']}...")
    print("    Tip: You can switch models dynamically anytime by pressing [1], [2], or [3] on your keyboard!")

    dashboard = RealtimeTcpDashboard(initial_model=chosen_model, step_delay=args.delay, loop=args.loop)
    dashboard.run()


if __name__ == "__main__":
    main()
