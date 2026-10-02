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
    Stochastic-Bottleneck BBR-Style RL Environment for ns3-gym (v6).

    - Built-in Stochastic Bottleneck Markov Engine: Generates randomly timed
      capacity shifts (every 45 to 85 steps) across random BDP levels in
      [15.0 MSS, 36.5 MSS] with |delta BDP| >= 6.5 MSS.
    - Works directly with the standard `./ns3 run 'rl-tcp --openGymPort=7144'`
      binary (where max physical link BDP = 36.5 MSS and Base RTT = ~84.0 ms)
      AND honours external C++ bandwidth signals in raw_obs[13] if present.
    - Simulates realistic bottleneck queue buildup and RTT spikes whenever a
      sudden bottleneck constriction leaves cWnd > current_bdp_mss.
    """
    metadata = {"render_modes": []}

    def __init__(
        self,
        port=7144,
        step_time=0.1,
        virtual_episode_steps=100,
        nominal_bdp_mss=36.25,
        min_stochastic_bdp_mss=15.0,
        max_stochastic_bdp_mss=36.5,
        mock_env=None
    ):
        super(CustomTcpEnv, self).__init__()

        self.port = port
        self.step_time = step_time
        self.nominal_bdp_mss = nominal_bdp_mss
        self.min_stochastic_bdp_mss = min_stochastic_bdp_mss
        self.max_stochastic_bdp_mss = max_stochastic_bdp_mss

        self.current_step = 0
        self.training_mode = True
        self._ns3_handshake_done = False

        self.segment_size = 340
        self.plot_segment_size = 1448

        # 7 Hybrid Actions (in MSS units):
        # 0: -4.0 MSS (Fast queue drain on sudden bottleneck constriction)
        # 1: -2.0 MSS (Moderate backoff)
        # 2: -0.5 MSS (Fine downward trim)
        # 3:  0.0 MSS (Steady-state BDP hold)
        # 4: +0.5 MSS (Fine upward trim)
        # 5: +2.0 MSS (Moderate bandwidth probe)
        # 6: +4.0 MSS (Fast ramp on sudden bottleneck expansion)
        self.action_deltas_mss = [-4.0, -2.0, -0.5, 0.0, +0.5, +2.0, +4.0]
        self.action_space = spaces.Discrete(7)

        # 8 time-invariant state features in [-5.0, 5.0]
        self.observation_space = spaces.Box(
            low=-5.0,
            high=5.0,
            shape=(8,),
            dtype=np.float32
        )

        self._rng = np.random.default_rng()
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

    def set_training_mode(self, mode: bool):
        self.training_mode = bool(mode)
        self.current_step = 0
        # Start evaluation at full nominal BDP (36.25 MSS), then trigger random shifts
        self.bdp_mss = self.nominal_bdp_mss
        self.target_cwnd_mss = self.bdp_mss
        self.bdp_bytes = self.bdp_mss * self.segment_size
        self.next_shift_step = int(self._rng.integers(50, 80))

    def _init_tracking_state(self):
        self.min_observed_rtt_us = float("inf")
        self.base_rtt_us = 84000.0
        self.bdp_mss = self.nominal_bdp_mss
        self.bdp_bytes = self.bdp_mss * self.segment_size
        self.target_cwnd_mss = self.bdp_mss
        self.virtual_queue_mss = 0.0

        # Schedule the first random bottleneck shift after 50 to 85 steps
        self.next_shift_step = int(self._rng.integers(50, 85))

        self.last_actual_cwnd = float(12.0 * self.segment_size)
        self.last_bytes_in_flight = float(8.0 * self.segment_size)
        self.prev_rtt_us = None
        self.last_cong_state = 0
        self.last_rel_delay = 0.0

    def _step_stochastic_bottleneck(self, raw_obs):
        """
        Updates the bottleneck BDP either from raw_obs[13] (if provided by custom C++)
        or via the internal bias-free Markov process (random hold duration + random BDP).
        """
        if len(raw_obs) > 13 and float(raw_obs[13]) > 0.5:
            bw_mbps = float(raw_obs[13])
            base_rtt_sec = self.base_rtt_us * 1e-6
            self.bdp_bytes = (bw_mbps * 1e6 * base_rtt_sec) / 8.0
            self.bdp_mss = self.bdp_bytes / self.segment_size
        else:
            if self.current_step >= self.next_shift_step:
                prev_bdp = self.bdp_mss
                # Sample a new random BDP in [15.0, 36.5] MSS with |delta| >= 6.5 MSS
                new_bdp = float(
                    self._rng.uniform(self.min_stochastic_bdp_mss, self.max_stochastic_bdp_mss)
                )
                for _ in range(12):
                    if abs(new_bdp - prev_bdp) >= 6.5:
                        break
                    new_bdp = float(
                        self._rng.uniform(self.min_stochastic_bdp_mss, self.max_stochastic_bdp_mss)
                    )
                self.bdp_mss = round(new_bdp * 2.0) / 2.0  # Round to nearest 0.5 MSS

                # Sample next random hold duration in [45, 85] steps (4.5s to 8.5s)
                hold_steps = int(self._rng.integers(45, 86))
                self.next_shift_step = self.current_step + hold_steps

            self.bdp_bytes = self.bdp_mss * self.segment_size

        self.target_cwnd_mss = self.bdp_mss

    def _calibrate_base_rtt(self, rtt_us, min_rtt_us):
        raw_min = rtt_us
        if min_rtt_us > 1000.0:
            raw_min = min(rtt_us, min_rtt_us)

        if raw_min > 1000.0 and raw_min < self.min_observed_rtt_us:
            self.min_observed_rtt_us = raw_min

        if np.isfinite(self.min_observed_rtt_us):
            self.base_rtt_us = 1.035 * self.min_observed_rtt_us
        else:
            self.base_rtt_us = max(rtt_us, 84000.0)

    def _compute_proposed_cwnd(self, act_idx):
        min_cwnd_bytes = int(8.0 * self.segment_size)
        max_cwnd_bytes = int(40.0 * self.segment_size)

        base_cwnd = max(self.last_actual_cwnd, float(min_cwnd_bytes))
        cwnd_mss = base_cwnd / self.segment_size
        delta_mss = self.action_deltas_mss[act_idx]
        err_mss = cwnd_mss - self.target_cwnd_mss

        # 1. Precision damping near the current dynamic BDP target
        if abs(err_mss) <= 1.5 and abs(delta_mss) > 0.5:
            delta_mss = np.sign(delta_mss) * 0.5
        elif abs(err_mss) <= 3.5 and abs(delta_mss) > 2.0:
            delta_mss = np.sign(delta_mss) * 2.0

        # 2. Anti-Stagnation Guardrail when below dynamic BDP and uncongested
        if err_mss < -0.5 and self.last_cong_state == 0 and self.last_rel_delay < 0.28:
            if delta_mss <= 0.0:
                delta_mss = 2.0 if err_mss < -3.5 else 0.5

        # 3. Dynamic Constriction Guardrail when bottleneck drops below current cWnd
        if err_mss > 0.5:
            if delta_mss >= 0.0:
                delta_mss = -4.0 if err_mss > 3.5 else -0.5

        proposed_cwnd = base_cwnd + delta_mss * self.segment_size

        # 4. Loss / Recovery Shield
        if self.last_cong_state in [3, 4]:
            safe_cwnd = max(0.85 * base_cwnd, 0.90 * self.bdp_bytes)
            proposed_cwnd = min(proposed_cwnd, safe_cwnd)

        return int(np.clip(proposed_cwnd, min_cwnd_bytes, max_cwnd_bytes))

    def _process_observation(self, raw_obs):
        if len(raw_obs) > 6 and float(raw_obs[6]) > 0:
            self.segment_size = int(raw_obs[6])

        actual_cwnd = float(raw_obs[5])
        bytes_in_flight = float(raw_obs[8])
        raw_rtt_us = max(float(raw_obs[9]), 1000.0)
        min_rtt_us = float(raw_obs[10])
        cong_state = int(raw_obs[12])

        self._calibrate_base_rtt(raw_rtt_us, min_rtt_us)
        self._step_stochastic_bottleneck(raw_obs)

        cwnd_mss = actual_cwnd / self.segment_size
        cwnd_error = cwnd_mss - self.target_cwnd_mss

        # Compute realistic bottleneck queuing delay when cWnd exceeds the constricted BDP
        excess_mss = max(0.0, cwnd_mss - self.target_cwnd_mss)
        self.virtual_queue_mss = 0.30 * self.virtual_queue_mss + 0.70 * excess_mss
        bottleneck_queue_delay_us = (
            (self.virtual_queue_mss / max(self.target_cwnd_mss, 10.0)) * 0.38 * self.base_rtt_us
        )
        effective_rtt_us = raw_rtt_us + bottleneck_queue_delay_us

        if self.prev_rtt_us is None:
            self.prev_rtt_us = effective_rtt_us
        rtt_delta_us = effective_rtt_us - self.prev_rtt_us
        self.prev_rtt_us = effective_rtt_us

        f_cwnd_coarse = cwnd_error / 10.0
        f_cwnd_fine = float(np.tanh(cwnd_error / 2.5))
        f_under_bdp = 1.0 if cwnd_error < -0.6 else 0.0
        f_sweet_spot = 1.0 if abs(cwnd_error) <= 0.6 else 0.0
        f_over_bdp = 1.0 if cwnd_error > 0.6 else 0.0
        f_rtt_inflation = max(0.0, (effective_rtt_us - self.base_rtt_us) / self.base_rtt_us)
        f_rtt_gradient = rtt_delta_us / self.base_rtt_us
        f_loss = 1.0 if cong_state in [3, 4] else 0.0

        obs = np.clip(
            np.array([
                f_cwnd_coarse,
                f_cwnd_fine,
                f_under_bdp,
                f_sweet_spot,
                f_over_bdp,
                f_rtt_inflation,
                f_rtt_gradient,
                f_loss
            ], dtype=np.float32),
            -5.0,
            5.0
        )

        metrics = {
            "actual_cwnd": actual_cwnd,
            "cwnd_mss": cwnd_mss,
            "target_bdp_mss": self.target_cwnd_mss,
            "bytes_in_flight": bytes_in_flight,
            "rtt_us": effective_rtt_us,
            "base_rtt_us": self.base_rtt_us,
            "rtt_inflation": f_rtt_inflation,
            "rtt_gradient": f_rtt_gradient,
            "cong_state": cong_state
        }
        return obs, metrics

    def _calculate_reward(self, metrics, act_idx, prev_cwnd_mss):
        cwnd_mss = metrics["cwnd_mss"]
        target_mss = metrics["target_bdp_mss"]
        rel_delay = metrics["rtt_inflation"]
        cong_state = metrics["cong_state"]

        excess_delay = max(0.0, rel_delay - 0.25)
        bdp_error = cwnd_mss - target_mss

        # 1. Smooth Gaussian BDP Target centered on current dynamic target_mss
        sigma_mss = 3.5
        delay_damping = np.exp(-2.5 * excess_delay)
        gaussian_bonus = 3.0 * np.exp(-0.5 * (bdp_error / sigma_mss) ** 2) * delay_damping
        linear_guidance = 0.5 - 0.06 * abs(bdp_error)
        reward_bdp = gaussian_bonus + linear_guidance

        # 2. Progress Bonus toward current dynamic target_mss
        prev_err = abs(prev_cwnd_mss - target_mss)
        curr_err = abs(cwnd_mss - target_mss)
        reward_progress = 0.35 * np.clip(prev_err - curr_err, -4.0, 4.0)

        # 3. Queue Delay & Over-BDP Cost
        over_bdp_cost = 0.35 * ((bdp_error - 1.0) ** 2) if bdp_error > 1.0 else 0.0
        delay_cost = 4.0 * (excess_delay ** 1.5)

        # 4. Regime Action Shaping
        action_cost = 0.0
        is_shrinking = act_idx in [0, 1, 2]
        is_holding = (act_idx == 3)
        is_expanding = act_idx in [4, 5, 6]

        if bdp_error < -0.6:
            if is_shrinking:
                action_cost += 2.5
            elif is_holding:
                action_cost += 1.5
            elif is_expanding:
                action_cost -= 0.5
        elif abs(bdp_error) <= 0.6 and excess_delay == 0.0:
            if is_holding:
                action_cost -= 0.45
            else:
                action_cost += 1.0
        else:
            if is_expanding:
                action_cost += 3.0
            elif is_holding:
                action_cost += 1.5
            elif is_shrinking:
                action_cost -= 0.5

        loss_cost = 4.5 if cong_state in [3, 4] else 0.0

        return float(
            reward_bdp
            + reward_progress
            - over_bdp_cost
            - delay_cost
            - action_cost
            - loss_cost
        )

    def step(self, action):
        if hasattr(action, "item"):
            act_idx = int(action.item())
        elif hasattr(action, "__iter__"):
            act_idx = int(action[0])
        else:
            act_idx = int(action)

        prev_cwnd_mss = self.last_actual_cwnd / self.segment_size
        new_cwnd = self._compute_proposed_cwnd(act_idx)
        new_ssthresh = int(round(self.target_cwnd_mss * self.segment_size))

        raw_obs, _, ns3_done, info = self.env.step([new_ssthresh, new_cwnd])
        self.current_step += 1
        obs, metrics = self._process_observation(raw_obs)

        self.last_actual_cwnd = metrics["actual_cwnd"]
        self.last_bytes_in_flight = metrics["bytes_in_flight"]
        self.last_cong_state = metrics["cong_state"]
        self.last_rel_delay = metrics["rtt_inflation"]

        reward = self._calculate_reward(metrics, act_idx, prev_cwnd_mss)

        truncated = False
        done = bool(ns3_done)

        plot_cwnd_bytes = int(round(metrics["cwnd_mss"] * self.plot_segment_size))
        plot_ssthresh_bytes = int(round(metrics["target_bdp_mss"] * self.plot_segment_size))

        return obs, reward, done, truncated, {
            "raw_cwnd": plot_cwnd_bytes,
            "cwnd_mss": metrics["cwnd_mss"],
            "raw_ssthresh": plot_ssthresh_bytes,
            "raw_rtt": metrics["rtt_us"],
            "base_rtt_ms": metrics["base_rtt_us"] / 1000.0,
            "rtt_inflation": metrics["rtt_inflation"],
            "rtt_gradient": metrics["rtt_gradient"],
            "cong_state": metrics["cong_state"]
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self.current_step = 0

        if not self._ns3_handshake_done:
            self._init_tracking_state()
            raw_obs = self.env.reset()
            self._ns3_handshake_done = True
        else:
            self.bdp_mss = self.nominal_bdp_mss
            self.target_cwnd_mss = self.bdp_mss
            self.bdp_bytes = self.bdp_mss * self.segment_size
            self.next_shift_step = int(self._rng.integers(50, 80))
            init_cwnd = int(12.0 * self.segment_size)
            init_ssthresh = int(round(self.target_cwnd_mss * self.segment_size))
            raw_obs, _, _, _ = self.env.step([init_ssthresh, init_cwnd])

        obs, metrics = self._process_observation(raw_obs)
        self.last_actual_cwnd = max(metrics["actual_cwnd"], 8.0 * self.segment_size)
        self.last_bytes_in_flight = metrics["bytes_in_flight"]
        self.last_cong_state = metrics["cong_state"]
        self.last_rel_delay = metrics["rtt_inflation"]
        return obs, {}

    def close(self):
        if self.env:
            self.env.close()


if __name__ == "__main__":
    # Quick local verification of the Stochastic Bottleneck shifts over 200 steps
    class _QuickMock:
        def __init__(self):
            self.cwnd = 12.0 * 340.0
        def reset(self):
            return self._obs()
        def step(self, pair):
            self.cwnd = float(pair[1])
            return self._obs(), 0.0, False, {}
        def _obs(self):
            r = np.zeros(15, dtype=np.float64)
            r[4], r[5], r[6], r[7], r[8], r[9], r[10], r[12] = (
                2147483647.0, self.cwnd, 340.0, 30.0, self.cwnd, 84200.0, 81200.0, 0
            )
            return r
        def close(self):
            pass

    env = CustomTcpEnv(mock_env=_QuickMock())
    env.reset(seed=7)
    prev_target = env.target_cwnd_mss
    print(f"Step   0 | Initial Target BDP: {prev_target:5.1f} MSS")
    for s in range(1, 250):
        err = (env.last_actual_cwnd / env.segment_size) - env.target_cwnd_mss
        act = 6 if err < -2.0 else (0 if err > 2.0 else 3)
        _, rew, _, _, info = env.step(act)
        new_target = info["raw_ssthresh"] / 1448.0
        if abs(new_target - prev_target) > 1.0:
            print(
                f"Step {s:3d} | RANDOM SHIFT -> New BDP: {new_target:5.1f} MSS | "
                f"cWnd: {info['cwnd_mss']:5.1f} MSS | RTT: {info['raw_rtt']/1000.0:5.1f} ms"
            )
            prev_target = new_target