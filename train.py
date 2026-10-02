import os
from typing import Callable
import numpy as np
import pandas as pd
import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from tcp_env import CustomTcpEnv


def linear_schedule(initial_value: float, final_value: float = 2e-5) -> Callable[[float], float]:
    def func(progress_remaining: float) -> float:
        return final_value + progress_remaining * (initial_value - final_value)
    return func


class TcpTelemetryCallback(BaseCallback):
    """Logs mean cWnd (MSS), RTT (ms), and calibrated Base RTT (ms) every rollout."""
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.cwnd_buf = []
        self.rtt_buf = []
        self.base_rtt_buf = []
        self.infl_buf = []

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "cwnd_mss" in info:
                self.cwnd_buf.append(info["cwnd_mss"])
            if "raw_rtt" in info:
                self.rtt_buf.append(info["raw_rtt"] / 1000.0)
            if "base_rtt_ms" in info:
                self.base_rtt_buf.append(info["base_rtt_ms"])
            if "rtt_inflation" in info:
                self.infl_buf.append(info["rtt_inflation"])
        return True

    def _on_rollout_end(self) -> None:
        if self.cwnd_buf:
            self.logger.record("tcp/mean_cwnd_mss", float(np.mean(self.cwnd_buf)))
            self.logger.record("tcp/mean_rtt_ms", float(np.mean(self.rtt_buf)))
            self.logger.record("tcp/calibrated_base_rtt_ms", float(np.mean(self.base_rtt_buf)))
            self.logger.record("tcp/mean_rtt_inflation", float(np.mean(self.infl_buf)))
            self.cwnd_buf.clear()
            self.rtt_buf.clear()
            self.base_rtt_buf.clear()
            self.infl_buf.clear()


def main():
    print("--- Phase 1: Environment Setup & Version Verification ---")
    port = 7144
    total_timesteps = 500_000
    episode_steps = 400
    checkpoint_dir = "./checkpoints_ppo_tcp"
    os.makedirs(checkpoint_dir, exist_ok=True)

    raw_env = DummyVecEnv([
        lambda: CustomTcpEnv(
            port=port,
            step_time=0.1,
            max_episode_steps=episode_steps
        )
    ])

    # Sanity check to confirm the new 8-feature, 7-action CustomTcpEnv is active
    assert raw_env.observation_space.shape == (8,), (
        f"Expected 8-D observation space from updated tcp_env.py, got {raw_env.observation_space.shape}"
    )
    assert raw_env.action_space.n == 7, (
        f"Expected 7 hybrid actions from updated tcp_env.py, got {raw_env.action_space.n}"
    )
    print("[*] Verified updated CustomTcpEnv: 8-D State Space, 7 Hybrid Actions, [6, 48] MSS Guardrails.")

    # Keep norm_obs=False so BDP-centered coordinates are never shifted by running mean
    env = VecNormalize(
        raw_env,
        norm_obs=False,
        norm_reward=True,
        clip_reward=10.0,
        gamma=0.98
    )

    policy_kwargs = dict(
        activation_fn=th.nn.SiLU,
        net_arch=dict(pi=[128, 128], vf=[128, 128])
    )

    print(f"\n--- Phase 2: Training PPO Agent ({total_timesteps:,} Steps) ---")
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=linear_schedule(3e-4, 2e-5),
        n_steps=2048,
        batch_size=128,
        n_epochs=10,
        gamma=0.98,
        gae_lambda=0.95,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        clip_range=0.2,
        target_kl=0.03,
        policy_kwargs=policy_kwargs,
        seed=42
    )

    checkpoint_cb = CheckpointCallback(
        save_freq=50_000,
        save_path=checkpoint_dir,
        name_prefix="ppo_tcp_500k",
        save_vecnormalize=True
    )
    telemetry_cb = TcpTelemetryCallback()

    model.learn(
        total_timesteps=total_timesteps,
        callback=[checkpoint_cb, telemetry_cb]
    )

    model.save("ppo_tcp_normalized")
    env.save("vec_normalize.pkl")
    print("[*] Training finished. Model and VecNormalize saved.")

    print("\n--- Phase 3: Deterministic Evaluation & Trace Logging (400 Steps) ---")
    env.training = False
    env.norm_reward = False

    trace_data = []
    obs = env.reset()

    for step_count in range(episode_steps):
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = env.step(action)
        step_info = infos[0]

        trace_data.append({
            "step": step_count,
            "cwnd": step_info["raw_cwnd"],
            "cwnd_mss": step_info["cwnd_mss"],
            "ssThresh": step_info["raw_ssthresh"],
            "rtt_us": step_info["raw_rtt"],
            "rtt_ms": step_info["raw_rtt"] / 1000.0,
            "base_rtt_ms": step_info["base_rtt_ms"],
            "rtt_inflation": step_info["rtt_inflation"],
            "action": int(action[0]),
            "reward": float(rewards[0])
        })

        if dones[0]:
            break

    env.close()

    print("\n--- Phase 4: Saving Traces ---")
    df = pd.DataFrame(trace_data)
    csv_filename = "rl_tcp_traces.csv"
    df.to_csv(csv_filename, index=False)
    print(
        f"[*] Saved {csv_filename} | Mean cWnd: {df['cwnd_mss'].mean():.2f} MSS | "
        f"Mean RTT: {df['rtt_ms'].mean():.2f} ms | Calibrated Base RTT: {df['base_rtt_ms'].iloc[-1]:.2f} ms | "
        f"Mean Step Reward: {df['reward'].mean():+.2f}"
    )


if __name__ == "__main__":
    main()