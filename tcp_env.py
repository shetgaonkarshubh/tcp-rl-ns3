import gymnasium as gym
from gymnasium import spaces
import numpy as np
from ns3gym import ns3env

class CustomTcpEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, port=7144, step_time=0.1):
        super(CustomTcpEnv, self).__init__()

        self.port = port
        self.step_time = step_time
        self.segment_size = 1448

        # 5 actions: -3 MSS, -1 MSS, 0, +1 MSS, +3 MSS
        self.action_deltas = [
            -3 * self.segment_size,
            -1 * self.segment_size,
            0,
            1 * self.segment_size,
            5 * self.segment_size
        ]
        self.action_space = spaces.Discrete(5)

        # 6 continuous state features
        self.observation_space = spaces.Box(
            low=-5.0,
            high=5.0,
            shape=(6,),
            dtype=np.float32
        )

        self.base_rtt_us = 42000.0
        self.bdp_bytes = 36.25 * self.segment_size

        self.env = ns3env.Ns3Env(
            port=self.port,
            stepTime=self.step_time,
            startSim=0,
            debug=False
        )

    def _process_observation(self, raw_obs):
        actual_cwnd = float(raw_obs[5])
        bytes_in_flight = float(raw_obs[8])
        rtt_us = max(float(raw_obs[9]), 1000.0)
        min_rtt_us = float(raw_obs[10])
        if min_rtt_us > 1000.0:
            self.base_rtt_us = min(self.base_rtt_us, min_rtt_us)

        cong_state = int(raw_obs[12])

        f_cwnd = (actual_cwnd - self.bdp_bytes) / self.bdp_bytes
        f_inflight = (bytes_in_flight - self.bdp_bytes) / self.bdp_bytes
        f_rtt_ratio = (rtt_us - self.base_rtt_us) / self.base_rtt_us
        f_loss = 1.0 if cong_state in [3, 4] else 0.0
        f_cwr = 1.0 if cong_state in [1, 2] else 0.0
        f_queue = max(0.0, (rtt_us - self.base_rtt_us) / 50000.0)

        obs = np.clip(np.array([f_cwnd, f_inflight, f_rtt_ratio, f_loss, f_cwr, f_queue], dtype=np.float32), -5.0, 5.0)
        return obs, actual_cwnd, rtt_us, cong_state

    def _calculate_reward(self, cwnd_bytes, rtt_us, cong_state, action_idx):
        # 1. Optimal BDP proximity incentive (Gaussian-like curve centered at 36.25 MSS)
        cwnd_mss = cwnd_bytes / self.segment_size
        bdp_error = abs(cwnd_mss - 36.25) / 36.25
        reward_bdp = np.exp(-2.0 * (bdp_error ** 2))

        # 2. Asymmetric queuing delay penalty
        rel_delay = max(0.0, (rtt_us - self.base_rtt_us) / self.base_rtt_us)
        queue_penalty = 3.0 * rel_delay

        # 3. Penalize selecting expansion actions when queueing delay is already elevated
        action_penalty = 0.0
        if rel_delay > 0.1 and action_idx in [3, 4]:
            action_penalty = 1.5

        # 4. Drop penalty
        loss_penalty = 4.0 if cong_state in [3, 4] else 0.0

        reward = reward_bdp - queue_penalty - action_penalty - loss_penalty
        return float(reward)

    def step(self, action):
        if hasattr(action, "item"):
            act_idx = int(action.item())
        elif hasattr(action, "__iter__"):
            act_idx = int(action[0])
        else:
            act_idx = int(action)

        delta = self.action_deltas[act_idx]
        new_cwnd = int(np.clip(self.last_actual_cwnd + delta, self.segment_size * 2, self.segment_size * 80))
        new_ssthresh = int(max(new_cwnd // 2, self.segment_size * 2))

        raw_obs, _, done, info = self.env.step([new_ssthresh, new_cwnd])
        obs, actual_cwnd, rtt_us, cong_state = self._process_observation(raw_obs)
        self.last_actual_cwnd = actual_cwnd

        reward = self._calculate_reward(actual_cwnd, rtt_us, cong_state, act_idx)
        truncated = False

        return obs, reward, done, truncated, {
            "raw_cwnd": raw_obs[5],
            "raw_ssthresh": raw_obs[4],
            "raw_rtt": raw_obs[9]
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        raw_obs = self.env.reset()
        obs, actual_cwnd, _, _ = self._process_observation(raw_obs)
        self.last_actual_cwnd = actual_cwnd
        return obs, {}

    def close(self):
        if self.env:
            self.env.close()