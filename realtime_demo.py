import sys
import os
import time
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle

ACTION_DESCRIPTIONS = {
    0: ("-4.0 MSS", "Fast Queue Drain (Contraction)", "#d62728"),
    1: ("-2.0 MSS", "Moderate Backoff", "#e377c2"),
    2: ("-0.5 MSS", "Fine Downward Trim", "#ff7f0e"),
    3: (" 0.0 MSS", "BDP Steady-State Lock", "#2ca02c"),
    4: ("+0.5 MSS", "Fine Upward Trim", "#17becf"),
    5: ("+2.0 MSS", "Moderate Probe", "#9467bd"),
    6: ("+4.0 MSS", "Fast Ramp-Up (Expansion)", "#1f77b4"),
}

CONG_STATES = {
    0: "CA_OPEN (Normal)",
    1: "CA_DISORDER",
    2: "CA_CWR",
    3: "CA_RECOVERY (Loss)",
    4: "CA_LOSS (Timeout)"
}

class RealtimeTcpDashboard:
    def __init__(self, trace_file="rl_tcp_traces.csv", step_delay=0.06, loop=False):
        self.trace_file = trace_file
        self.step_delay = step_delay
        self.loop = loop
        self.paused = False
        self.step_once = False

        if not os.path.exists(self.trace_file):
            raise FileNotFoundError(f"Trace file not found: {self.trace_file}")

        self.df = pd.read_csv(self.trace_file)
        self.total_steps = len(self.df)

        # Parse data
        if "cwnd_mss" in self.df.columns:
            self.cwnd_mss = self.df["cwnd_mss"].to_numpy(dtype=float)
        else:
            self.cwnd_mss = (self.df["cwnd"] / 1448.0).to_numpy(dtype=float)

        raw_ssthresh = (self.df["ssThresh"] / 1448.0).to_numpy(dtype=float)
        self.ssthresh_mss = np.where(raw_ssthresh > 500.0, np.nan, raw_ssthresh)
        self.rtt_ms = (self.df["rtt_us"] / 1000.0).to_numpy(dtype=float)
        self.actions = self.df["action"].to_numpy(dtype=int)
        self.rewards = self.df["reward"].to_numpy(dtype=float)

        self.base_rtt_ms = float(np.nanpercentile(self.rtt_ms, 5))
        if self.base_rtt_ms < 1.0:
            self.base_rtt_ms = 84.0

        # Precompute shifts
        valid_target = np.nan_to_num(self.ssthresh_mss, nan=0.0)
        self.shift_indices = set(np.where(np.abs(np.diff(valid_target)) > 1.0)[0] + 1)

    def on_key(self, event):
        if event.key == " ":
            self.paused = not self.paused
        elif event.key in ["right", "n"]:
            self.step_once = True
            self.paused = True
        elif event.key in ["q", "escape"]:
            plt.close("all")
            sys.exit(0)

    def run(self):
        plt.ion()
        fig = plt.figure(figsize=(15, 8.5))
        fig.canvas.manager.set_window_title("Real-Time PPO TCP Congestion Control Visualizer")
        fig.canvas.mpl_connect("key_press_event", self.on_key)

        # 2 columns: Plots on Left, Telemetry HUD Card on Right
        gs = gridspec.GridSpec(2, 2, width_ratios=[2.5, 1.2], height_ratios=[1.2, 1.0], figure=fig)
        ax_cwnd = fig.add_subplot(gs[0, 0])
        ax_rtt = fig.add_subplot(gs[1, 0], sharex=ax_cwnd)
        ax_hud = fig.add_subplot(gs[:, 1])

        ax_hud.axis("off")
        fig.patch.set_facecolor("#18191c")

        # Configure Left Plots styling
        for ax in [ax_cwnd, ax_rtt]:
            ax.set_facecolor("#22242a")
            ax.grid(True, linestyle="--", alpha=0.3, color="#555863")
            ax.tick_params(colors="#dcdcdc", labelsize=9)
            for spine in ax.spines.values():
                spine.set_color("#444751")

        ax_cwnd.set_ylabel("cWnd / Dynamic BDP (MSS)", color="#f0f0f0", fontweight="bold", fontsize=10)
        ax_cwnd.set_ylim(0, max(np.nanmax(self.cwnd_mss), np.nanmax(self.ssthresh_mss)) * 1.25)
        ax_rtt.set_ylabel("Round Trip Time (ms)", color="#f0f0f0", fontweight="bold", fontsize=10)
        ax_rtt.set_xlabel("Simulation Step (100 ms / step)", color="#f0f0f0", fontweight="bold", fontsize=10)
        ax_rtt.set_ylim(self.base_rtt_ms * 0.85, max(np.nanmax(self.rtt_ms) * 1.12, self.base_rtt_ms * 1.5))

        # Static reference lines
        ax_rtt.axhline(self.base_rtt_ms, color="#38ef7d", linestyle=":", linewidth=1.5, label=f"Base RTT (~{self.base_rtt_ms:.1f}ms)")
        ax_rtt.axhline(self.base_rtt_ms * 1.25, color="#ff4b4b", linestyle="-.", linewidth=1.2, label="+25% Queue Ceiling")
        ax_rtt.legend(loc="upper right", facecolor="#2b2d35", edgecolor="#444751", labelcolor="#eaeaea", fontsize=8)

        # Dynamic plot lines
        (line_target,) = ax_cwnd.step([], [], where="post", color="#ffa116", linestyle="--", linewidth=1.8, label="Dynamic BDP Target")
        (line_cwnd,) = ax_cwnd.plot([], [], color="#00b4d8", linewidth=2.3, label="Agent cWnd")
        (line_rtt,) = ax_rtt.plot([], [], color="#ff6b6b", linewidth=1.8, label="Observed RTT")

        ax_cwnd.legend(loc="upper right", facecolor="#2b2d35", edgecolor="#444751", labelcolor="#eaeaea", fontsize=8)

        print("\n" + "=" * 95)
        print(" [★] REAL-TIME PPO TCP CONGESTION CONTROL TELEMETRY STREAM ")
        print(" Controls: [Space] Pause/Resume | [Right Arrow] Step 1 frame | [Q] Quit")
        print("=" * 95)
        print(f"{'Step':>5} | {'Target BDP':>10} | {'Agent cWnd':>10} | {'Error':>9} | {'RTT (ms)':>9} | {'Action':<32} | {'Reward':>7}")
        print("-" * 95)

        step_idx = 0
        while True:
            if step_idx >= self.total_steps:
                if self.loop:
                    step_idx = 0
                    time.sleep(1.0)
                else:
                    break

            if self.paused and not self.step_once:
                plt.pause(0.05)
                continue
            self.step_once = False

            curr_step = step_idx
            cur_cwnd = self.cwnd_mss[curr_step]
            cur_target = self.ssthresh_mss[curr_step]
            cur_rtt = self.rtt_ms[curr_step]
            cur_act = self.actions[curr_step]
            cur_rew = self.rewards[curr_step]
            err = cur_cwnd - cur_target
            infl = max(0.0, (cur_rtt - self.base_rtt_ms) / self.base_rtt_ms) * 100.0

            act_delta, act_name, act_color = ACTION_DESCRIPTIONS.get(cur_act, (f"Act {cur_act}", "Unknown", "#ffffff"))

            # Update Plot Lines
            x_range = np.arange(curr_step + 1)
            line_target.set_data(x_range, self.ssthresh_mss[: curr_step + 1])
            line_cwnd.set_data(x_range, self.cwnd_mss[: curr_step + 1])
            line_rtt.set_data(x_range, self.rtt_ms[: curr_step + 1])

            # Adjust x-axis window dynamically
            min_x = max(0, curr_step - 80)
            max_x = max(80, curr_step + 5)
            ax_cwnd.set_xlim(min_x, max_x)

            # Check if this step is a dynamic bottleneck shift
            is_shift = curr_step in self.shift_indices
            if is_shift:
                ax_cwnd.axvline(curr_step, color="#ff7f0e", linestyle=":", alpha=0.7)
                ax_rtt.axvline(curr_step, color="#ff7f0e", linestyle=":", alpha=0.7)

            # Render Telemetry HUD
            ax_hud.clear()
            ax_hud.axis("off")
            ax_hud.set_facecolor("#18191c")

            status_color = "#38ef7d" if abs(err) <= 1.5 and infl < 25.0 else ("#ff4b4b" if infl >= 25.0 or err > 3.0 else "#ffa116")
            status_text = "OPTIMAL (Kleinrock)" if abs(err) <= 1.5 and infl < 25.0 else ("BUFFERBLOAT RISK" if infl >= 25.0 else "ADAPTING")

            hud_text = (
                f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                f"   LIVE AGENT TELEMETRY\n"
                f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                f"• Step: {curr_step:03d} / {self.total_steps}  ({curr_step*0.1:.1f}s)\n"
                f"• System State: {status_text}\n\n"
                f"▶ OBSERVATION RECEIVED:\n"
                f"  ├ Bottleneck BDP : {cur_target:5.1f} MSS\n"
                f"  ├ Current cWnd   : {cur_cwnd:5.1f} MSS\n"
                f"  ├ Window Error   : {err:+5.1f} MSS\n"
                f"  ├ Observed RTT   : {cur_rtt:5.1f} ms\n"
                f"  ├ Base RTT (min) : {self.base_rtt_ms:5.1f} ms\n"
                f"  └ Queue Inflation: {infl:+5.1f} %\n\n"
                f"▶ DECISION (PPO POLICY):\n"
                f"  ├ Action ID      : [{cur_act}]\n"
                f"  ├ Window Delta   : {act_delta}\n"
                f"  └ Strategy       : {act_name}\n\n"
                f"▶ PERFORMANCE FEEDBACK:\n"
                f"  ├ Step Reward    : {cur_rew:+.3f}\n"
                f"  ├ Cumulative Rew : {np.sum(self.rewards[:curr_step+1]):+.1f}\n"
                f"  └ Mean Abs Error : {np.nanmean(np.abs(self.cwnd_mss[:curr_step+1] - self.ssthresh_mss[:curr_step+1])):4.2f} MSS\n"
                f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                f" [Space] Pause/Resume | [Q] Quit"
            )

            ax_hud.text(
                0.05, 0.95,
                hud_text,
                transform=ax_hud.transAxes,
                fontsize=9.5,
                family="monospace",
                verticalalignment="top",
                color="#e6e6e6",
                bbox=dict(boxstyle="round,pad=0.6", facecolor="#22242a", edgecolor="#444751", alpha=0.95)
            )

            # Terminal log row
            shift_indicator = "⚡ SHIFT" if is_shift else "       "
            print(f"{curr_step:>5} | {cur_target:8.1f} MSS | {cur_cwnd:8.1f} MSS | {err:+7.1f} MSS | {cur_rtt:7.1f} ms | [{cur_act}] {act_delta:<8} {act_name:<18} | {cur_rew:>+7.3f} {shift_indicator}")

            plt.pause(self.step_delay)
            step_idx += 1

        print("\n[*] Demonstration finished successfully! Keeping window open.")
        print("    Press 'Q' or close window to exit.")
        plt.ioff()
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real-Time TCP PPO Dashboard")
    parser.add_argument("--trace", default="rl_tcp_traces.csv", help="Trace CSV file to visualize")
    parser.add_argument("--delay", type=float, default=0.05, help="Delay in seconds between steps (speed)")
    parser.add_argument("--loop", action="store_true", help="Loop playback indefinitely")
    args = parser.parse_args()

    dashboard = RealtimeTcpDashboard(trace_file=args.trace, step_delay=args.delay, loop=args.loop)
    dashboard.run()
