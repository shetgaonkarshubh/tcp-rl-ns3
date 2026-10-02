import gymnasium as gym
from gymnasium import spaces
import numpy as np

try:
    from ns3gym import ns3env
    HAS_NS3GYM = True
except ImportError:
    HAS_NS3GYM = False


class CustomTcpEnv(gym.Env):
    """
    Self-Calibrating BBR-Style RL Environment for ns3-gym TCP Congestion Control.

    Simultaneously prevents:
    1. 1-MSS Starvation Collapse (by auto-calibrating base_rtt_us to the true
       data-packet RTT ~84.2ms and gating out the Step 4 95ms startup transient).
    2. 80-MSS Aggressive Bufferbloat (by blocking positive deltas during loss or
       queue inflation, capping max cWnd at 48 MSS, and applying a quadratic
       over-BDP penalty above 39 MSS).
    """
    metadata = {"render_modes": []}

    def __init__(
        self,
        port=7144,
        step_time=0.1,
        max_episode_steps=400,
        nominal_bdp_mss=36.25,
        mock_env=None
    ):
        super(CustomTcpEnv, self).__init__()

        self.port = port
        self.step_time = step_time
        self.max_episode_steps = max_episode_steps
        self.nominal_bdp_mss = nominal_bdp_mss
        self.current_step = 0
        self.segment_size = 1448

        # 7 Hybrid Actions:
        # 0: -20% Multiplicative Drain (fast queue recovery when above BDP)
        # 1: -2 MSS
        # 2: -1 MSS
        # 3:  0 MSS (Hold steady at BDP)
        # 4: +1 MSS
        # 5: +2 MSS
        # 6: +4 MSS (Fast ramp when pipe is under-utilized)
        self.action_space = spaces.Discrete(7)

        # 8 normalized, zero-centered state features in [-5.0, 5.0]
        self.observation_space = spaces.Box(
            low=-5.0,
            high=5.0,
            shape=(8,),
            dtype=np.float32
        )

        self._init_tracking_state()

        if mock_env is not None:
            self.env = mock_env
        elif HAS_NS3GYM:
            self.env = ns3env.Ns3Env(
                port=self.port,
                stepTime=self.step_time,
                startSim=0,
                debug=False
            )
        else:
            raise RuntimeError("ns3gym is not installed and no mock_env was provided.")

    def _init_tracking_state(self):
        """Resets per-episode telemetry trackers."""
        self.min_observed_rtt_us = float("inf")
        self.base_rtt_us = 84200.0  # Dynamically overwritten on Step 0
        self.bdp_mss = self.nominal_bdp_mss
        self.bdp_bytes = self.bdp_mss * self.segment_size
        self.target_cwnd_mss = 37.0  # ~1.02x BDP sweet spot
        self.max_bw_bps = (self.bdp_bytes * 8.0) / 0.0842

        self.last_actual_cwnd = float(self.segment_size * 10)
        self.last_bytes_in_flight = float(self.segment_size * 6)
        self.prev_rtt_us = None
        self.last_cong_state = 0
        self.last_rel_delay = 0.0

    def _calibrate_base_rtt(self, rtt_us, min_rtt_us):
        """
        Calibrates base_rtt_us from ns-3's observed minimum RTT, adding a 4%
        single-data-packet serialization margin so the 81.0ms SYN handshake RTT
        maps accurately to the 84.2ms unloaded 1-MSS data packet RTT.
        """
        raw_min = rtt_us
        if min_rtt_us > 1000.0:
            raw_min = min(rtt_us, min_rtt_us)

        if raw_min > 1000.0 and raw_min < self.min_observed_rtt_us:
            self.min_observed_rtt_us = raw_min

        if np.isfinite(self.min_observed_rtt_us):
            # 1.04x accounts for 1448B data serialization vs 60B SYN packet (81ms * 1.04 = 84.24ms)
            self.base_rtt_us = 1.04 * self.min_observed_rtt_us
        else:
            self.base_rtt_us = max(rtt_us, 42000.0)

    def _compute_proposed_cwnd(self, act_idx):
        """
        Translates action index into a guarded cWnd (in bytes) with hard shields
        against both 1-MSS starvation and over-BDP bufferbloat.
        """
        min_cwnd_bytes = 6 * self.segment_size   # Hard 6-MSS floor prevents 1-MSS collapse
        max_cwnd_bytes = 48 * self.segment_size  # Hard 48-MSS ceiling prevents 300ms bufferbloat

        base_cwnd = max(self.last_actual_cwnd, float(min_cwnd_bytes))
        cwnd_mss = base_cwnd / self.segment_size

        # Ghost-Window Clamp: when above BDP, anchor to actual bytes_in_flight + 4 MSS
        if base_cwnd > self.bdp_bytes and self.last_bytes_in_flight > min_cwnd_bytes:
            base_cwnd = min(
                base_cwnd,
                max(self.last_bytes_in_flight, self.bdp_bytes) + 4.0 * self.segment_size
            )

        # Map discrete action
        if act_idx == 0:
            # Multiplicative 20% drain only when above 28 MSS; otherwise -1 MSS
            if cwnd_mss > 28.0:
                proposed_cwnd = 0.80 * base_cwnd
            else:
                proposed_cwnd = base_cwnd - 1.0 * self.segment_size
        elif act_idx == 1:
            proposed_cwnd = base_cwnd - 2.0 * self.segment_size
        elif act_idx == 2:
            proposed_cwnd = base_cwnd - 1.0 * self.segment_size
        elif act_idx == 3:
            proposed_cwnd = base_cwnd
        elif act_idx == 4:
            proposed_cwnd = base_cwnd + 1.0 * self.segment_size
        elif act_idx == 5:
            proposed_cwnd = base_cwnd + 2.0 * self.segment_size
        else:  # act_idx == 6
            # Fast +4 MSS probe only below 32 MSS; above 32 MSS cap step at +1 MSS
            if cwnd_mss < 32.0:
                proposed_cwnd = base_cwnd + 4.0 * self.segment_size
            else:
                proposed_cwnd = base_cwnd + 1.0 * self.segment_size

        # ANTI-AGGRESSION SHIELD 1: Never override ns-3 recovery/loss when pipe is full
        if self.last_cong_state in [3, 4] and cwnd_mss > 25.0:
            safe_recovery_cwnd = max(0.80 * base_cwnd, 0.85 * self.bdp_bytes)
            proposed_cwnd = min(proposed_cwnd, safe_recovery_cwnd)

        # ANTI-AGGRESSION SHIELD 2: Block window expansion if standing queue > 15% above base RTT
        if cwnd_mss >= 35.0 and self.last_rel_delay > 0.15:
            proposed_cwnd = min(proposed_cwnd, base_cwnd - 1.0 * self.segment_size)

        return int(np.clip(proposed_cwnd, min_cwnd_bytes, max_cwnd_bytes))

    def _process_observation(self, raw_obs):
        actual_cwnd = float(raw_obs[5])
        if len(raw_obs) > 6 and float(raw_obs[6]) > 0:
            self.segment_size = int(raw_obs[6])
            self.bdp_bytes = self.bdp_mss * self.segment_size

        segments_acked = float(raw_obs[7]) if len(raw_obs) > 7 else 0.0
        bytes_in_flight = float(raw_obs[8])
        rtt_us = max(float(raw_obs[9]), 1000.0)
        min_rtt_us = float(raw_obs[10])
        cong_state = int(raw_obs[12])

        self._calibrate_base_rtt(rtt_us, min_rtt_us)

        if self.prev_rtt_us is None:
            self.prev_rtt_us = rtt_us
        rtt_delta_us = rtt_us - self.prev_rtt_us
        self.prev_rtt_us = rtt_us

        delivered_bytes = segments_acked * self.segment_size
        delivery_rate_bps = (delivered_bytes * 8.0) / max(self.step_time, 1e-3)
        if delivery_rate_bps > 0.1 * self.max_bw_bps:
            self.max_bw_bps = max(0.98 * self.max_bw_bps, delivery_rate_bps)

        f_cwnd = (actual_cwnd - self.bdp_bytes) / self.bdp_bytes
        f_inflight = (bytes_in_flight - self.bdp_bytes) / self.bdp_bytes
        f_rtt_inflation = max(0.0, (rtt_us - self.base_rtt_us) / self.base_rtt_us)
        f_rtt_gradient = rtt_delta_us / self.base_rtt_us
        f_delivery_ratio = np.clip(delivery_rate_bps / max(self.max_bw_bps, 1.0), 0.0, 2.0)
        f_pipe_util = bytes_in_flight / max(actual_cwnd, float(self.segment_size))
        f_loss = 1.0 if cong_state in [3, 4] else 0.0
        f_cwr = 1.0 if cong_state in [1, 2] else 0.0

        obs = np.clip(
            np.array([
                f_cwnd,
                f_inflight,
                f_rtt_inflation,
                f_rtt_gradient,
                f_delivery_ratio,
                f_pipe_util,
                f_loss,
                f_cwr
            ], dtype=np.float32),
            -5.0,
            5.0
        )

        metrics = {
            "actual_cwnd": actual_cwnd,
            "cwnd_mss": actual_cwnd / self.segment_size,
            "bytes_in_flight": bytes_in_flight,
            "rtt_us": rtt_us,
            "base_rtt_us": self.base_rtt_us,
            "rtt_inflation": f_rtt_inflation,
            "rtt_gradient": f_rtt_gradient,
            "delivery_ratio": f_delivery_ratio,
            "cong_state": cong_state
        }
        return obs, metrics

    def _calculate_reward(self, metrics, act_idx, prev_cwnd_mss):
        cwnd_mss = metrics["cwnd_mss"]
        rel_delay = metrics["rtt_inflation"]
        rtt_grad = metrics["rtt_gradient"]
        cong_state = metrics["cong_state"]

        # Pipe-saturation gate: 0.0 when cwnd <= 16 MSS (immune to Step 4 95ms startup spike),
        # smoothly ramps to 1.0 at cwnd >= 32 MSS (full BBR queue sensitivity near BDP).
        pipe_gate = float(np.clip((cwnd_mss - 16.0) / 16.0, 0.0, 1.0))
        gated_delay = rel_delay * pipe_gate

        # 1. Smooth Gaussian BDP Target (peaks at 37.0 MSS with +3.5, attenuated by gated_delay)
        bdp_error = cwnd_mss - self.target_cwnd_mss
        sigma_mss = 4.5
        delay_damping = np.exp(-3.5 * max(0.0, gated_delay - 0.04))
        gaussian_bonus = 3.0 * np.exp(-0.5 * (bdp_error / sigma_mss) ** 2) * delay_damping
        linear_guidance = 0.5 - 0.06 * abs(bdp_error)
        reward_bdp = gaussian_bonus + linear_guidance

        # 2. Asymmetric Over-BDP Quadratic Wall: strongly punishes exceeding 39.5 MSS
        over_bdp_cost = 0.0
        if cwnd_mss > 39.5:
            over_bdp_cost = 0.22 * ((cwnd_mss - 39.5) ** 2)

        # 3. Standing Queue & Queue Velocity Penalty (gated by pipe saturation)
        if gated_delay <= 0.10:
            delay_cost = 1.5 * (gated_delay ** 2)
        else:
            delay_cost = 0.015 + 6.0 * ((gated_delay - 0.10) ** 1.5)

        gradient_cost = 2.0 * max(0.0, rtt_grad - 0.02) * pipe_gate

        # 4. Potential-Based Progress Bonus toward 37.0 MSS
        prev_err = abs(prev_cwnd_mss - self.target_cwnd_mss)
        curr_err = abs(cwnd_mss - self.target_cwnd_mss)
        reward_progress = 0.20 * np.clip(prev_err - curr_err, -4.0, 4.0)

        # 5. Regime-Specific Action Shaping
        action_cost = 0.0
        is_shrinking = act_idx in [0, 1, 2]
        is_holding = (act_idx == 3)
        is_expanding = act_idx in [4, 5, 6]

        if cwnd_mss < 30.0:
            # Under-utilized pipe: forbid shrinking/camping at low cWnd
            if is_shrinking:
                action_cost += 2.2
            elif is_holding:
                action_cost += 0.8
            elif is_expanding:
                action_cost -= 0.4  # Positive incentive to climb toward BDP
        elif 34.0 <= cwnd_mss <= 39.0 and gated_delay < 0.10:
            # Inside BBR sweet spot [34, 39] MSS with low queue: reward steady hold
            if is_holding:
                action_cost -= 0.5
            elif act_idx in [0, 6]:
                action_cost += 1.0  # Discourage violent swings inside the sweet spot
        elif cwnd_mss > 39.0 or gated_delay > 0.12:
            # Over-BDP or building queue: punish expansion, reward draining
            if is_expanding:
                action_cost += 3.5 + 2.0 * gated_delay
            elif is_holding and (cwnd_mss > 41.0 or gated_delay > 0.18):
                action_cost += 1.2
            elif is_shrinking and prev_cwnd_mss > cwnd_mss:
                action_cost -= 0.6

        # 6. Packet Loss / Recovery Penalty
        loss_cost = 4.5 if cong_state in [3, 4] else 0.0

        total_reward = (
            reward_bdp
            + reward_progress
            - over_bdp_cost
            - delay_cost
            - gradient_cost
            - action_cost
            - loss_cost
        )
        return float(total_reward)

    def step(self, action):
        if hasattr(action, "item"):
            act_idx = int(action.item())
        elif hasattr(action, "__iter__"):
            act_idx = int(action[0])
        else:
            act_idx = int(action)

        prev_cwnd_mss = self.last_actual_cwnd / self.segment_size
        new_cwnd = self._compute_proposed_cwnd(act_idx)
        # Set ssthresh to 35 MSS (~0.96x BDP) so ns-3 exits slow-start smoothly right below BDP
        new_ssthresh = int(35.0 * self.segment_size)

        raw_obs, _, ns3_done, info = self.env.step([new_ssthresh, new_cwnd])
        obs, metrics = self._process_observation(raw_obs)

        self.last_actual_cwnd = metrics["actual_cwnd"]
        self.last_bytes_in_flight = metrics["bytes_in_flight"]
        self.last_cong_state = metrics["cong_state"]
        self.last_rel_delay = metrics["rtt_inflation"]
        self.current_step += 1

        reward = self._calculate_reward(metrics, act_idx, prev_cwnd_mss)
        
        # ns-3 OpenGym does not support process restarts mid-training via reset().
        # Keep done tied strictly to ns-3's natural lifetime.
        truncated = False
        done = bool(ns3_done)

        return obs, reward, done, truncated, {
            "raw_cwnd": raw_obs[5],
            "cwnd_mss": metrics["cwnd_mss"],
            "raw_ssthresh": raw_obs[4],
            "raw_rtt": raw_obs[9],
            "base_rtt_ms": metrics["base_rtt_us"] / 1000.0,
            "rtt_inflation": metrics["rtt_inflation"],
            "rtt_gradient": metrics["rtt_gradient"],
            "cong_state": metrics["cong_state"]
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self._init_tracking_state()

        raw_obs = self.env.reset()
        obs, metrics = self._process_observation(raw_obs)

        self.last_actual_cwnd = max(metrics["actual_cwnd"], 6.0 * self.segment_size)
        self.last_bytes_in_flight = metrics["bytes_in_flight"]
        self.last_cong_state = metrics["cong_state"]
        self.last_rel_delay = metrics["rtt_inflation"]
        return obs, {}

    def close(self):
        if self.env:
            self.env.close()


# ==============================================================================
# Built-In Self-Test Reproducing the Exact 81ms -> 95ms -> 84.2ms Startup Curve
# ==============================================================================
class _StartupTransientMockEnv:
    def __init__(self):
        self.segment_size = 1448
        self.min_rtt_us = 81000.0        # 81.0ms handshake RTT from your plot
        self.unloaded_data_rtt = 84200.0 # 84.2ms 1-MSS data RTT from your plot
        self.bdp_mss = 36.25
        self.step_idx = 0
        self.reset()

    def reset(self):
        self.step_idx = 0
        self.cwnd_bytes = 2.0 * self.segment_size
        self.ssthresh_bytes = 35.0 * self.segment_size
        self.queue_mss = 0.0
        self.rtt_us = 81000.0
        return self._obs()

    def step(self, action_pair):
        self.step_idx += 1
        self.ssthresh_bytes, self.cwnd_bytes = float(action_pair[0]), float(action_pair[1])
        cwnd_mss = self.cwnd_bytes / self.segment_size

        # Reproduce the exact Step 1-6 95ms startup transient from rl_tcp_zoomed_performance.png
        startup_spike_us = 10800.0 * np.exp(-0.35 * max(0, self.step_idx - 3)) if self.step_idx >= 2 else 0.0
        target_queue = max(0.0, cwnd_mss - self.bdp_mss)
        self.queue_mss = 0.35 * self.queue_mss + 0.65 * target_queue
        queue_delay_us = (self.queue_mss / (self.bdp_mss / 0.0842)) * 1e6

        self.rtt_us = self.unloaded_data_rtt + startup_spike_us + queue_delay_us
        return self._obs(), 0.0, False, {}

    def _obs(self):
        raw = np.zeros(15, dtype=np.float64)
        raw[4], raw[5], raw[6], raw[7], raw[8] = (
            self.ssthresh_bytes, self.cwnd_bytes, self.segment_size, 30, self.cwnd_bytes
        )
        raw[9], raw[10], raw[12] = self.rtt_us, self.min_rtt_us, 0
        return raw

    def close(self):
        pass


if __name__ == "__main__":
    print("=== Crosscheck 1: Reward Comparison at Step 4 (95ms Startup Spike, cWnd=6 MSS) ===")
    test_env = CustomTcpEnv(mock_env=_StartupTransientMockEnv())
    test_env.reset()
    for s in range(1, 15):
        # Greedy 1-step action selection using the reward surface itself
        best_act, best_r = 3, -1e9
        saved_cwnd = test_env.last_actual_cwnd
        for candidate_act in range(7):
            prop_cwnd = test_env._compute_proposed_cwnd(candidate_act)
            fake_metrics = {
                "actual_cwnd": float(prop_cwnd),
                "cwnd_mss": prop_cwnd / test_env.segment_size,
                "bytes_in_flight": float(prop_cwnd),
                "rtt_us": test_env.prev_rtt_us,
                "base_rtt_us": test_env.base_rtt_us,
                "rtt_inflation": max(0.0, (test_env.prev_rtt_us - test_env.base_rtt_us) / test_env.base_rtt_us),
                "rtt_gradient": 0.0,
                "delivery_ratio": min(1.0, (prop_cwnd / test_env.segment_size) / 36.25),
                "cong_state": 0
            }
            r = test_env._calculate_reward(fake_metrics, candidate_act, saved_cwnd / test_env.segment_size)
            if r > best_r:
                best_r, best_act = r, candidate_act

        obs, reward, _, _, info = test_env.step(best_act)
        print(
            f"Step {s:2d} | BestAct: {best_act} | cWnd: {info['cwnd_mss']:5.1f} MSS | "
            f"RTT: {info['raw_rtt']/1000.0:5.1f} ms | BaseRTT: {info['base_rtt_ms']:5.2f} ms | "
            f"Reward: {reward:+6.2f}"
        )