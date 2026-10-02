import os
from typing import Callable
import numpy as np
import pandas as pd
import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from tcp_env import CustomTcpEnv


def linear_schedule(initial_value: float, final_value: float = 3e-5) -> Callable[[float], float]:
    def func(progress_remaining: float) -> float:
        return final_value + progress_remaining * (initial_value - final_value)
    return func


class TcpTelemetryCallback(BaseCallback):
    """Logs BBR telemetry across each 1,024-step rollout."""
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.cwnd_buf = []
        self.rtt_buf = []
        self.base_rtt_buf = []
        self.infl_buf = []

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "cwnd_mss" in info:
                self.cwnd_buf.append(float(info["cwnd_mss"]))
            if "raw_rtt" in info:
                self.rtt_buf.append(float(info["raw_rtt"]) / 1000.0)
            if "base_rtt_ms" in info:
                self.base_rtt_buf.append(float(info["base_rtt_ms"]))
            if "rtt_inflation" in info:
                self.infl_buf.append(float(info["rtt_inflation"]))
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
    print("--- Phase 1: Environment Setup ---")
    port = 7144
    total_timesteps = 100_000
    eval_steps = 400
    checkpoint_dir = "./checkpoints_ppo_tcp"
    os.makedirs(checkpoint_dir, exist_ok=True)

    raw_env = DummyVecEnv([
        lambda: CustomTcpEnv(
            port=port,
            step_time=0.1,
            virtual_episode_steps=100
        )
    ])

    env = VecNormalize(
        raw_env,
        norm_obs=False,
        norm_reward=True,
        clip_reward=10.0,
        gamma=0.98
    )

    print("[*] Performing initial handshake with ns-3...")
    obs = env.reset()
    print(f"[*] Handshake successful! Initial Observation: {obs}")

    policy_kwargs = dict(
        activation_fn=th.nn.SiLU,
        net_arch=dict(pi=[128, 128], vf=[128, 128])
    )

    print(f"\n--- Phase 2: Training Agent ({total_timesteps:,} Steps) ---")
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=linear_schedule(3e-4, 3e-5),
        n_steps=1024,
        batch_size=128,
        n_epochs=5,
        gamma=0.98,
        gae_lambda=0.95,
        ent_coef=0.015,
        vf_coef=0.5,
        max_grad_norm=0.5,
        clip_range=0.2,
        target_kl=0.05,
        policy_kwargs=policy_kwargs,
        seed=42
    )

    checkpoint_cb = CheckpointCallback(
        save_freq=25_000,
        save_path=checkpoint_dir,
        name_prefix="ppo_tcp_100k",
        save_vecnormalize=True
    )
    telemetry_cb = TcpTelemetryCallback()

    model.learn(
        total_timesteps=total_timesteps,
        callback=[checkpoint_cb, telemetry_cb]
    )

    model.save("ppo_tcp_normalized")
    env.save("vec_normalize.pkl")
    print("[*] Training finished. Model saved.")

    print("\n--- Phase 3: Evaluation & Trace Logging (400 Steps) ---")
    # Disable training mode on both VecNormalize and CustomTcpEnv
    env.training = False
    env.norm_reward = False
    raw_env.env_method("set_training_mode", False)

    trace_data = []
    obs = env.reset()

    for step_count in range(eval_steps):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, infos = env.step(action)
        step_info = infos[0]

        trace_data.append({
            "step": step_count,
            "cwnd": step_info["raw_cwnd"],
            "cwnd_mss": step_info["cwnd_mss"],
            "ssThresh": step_info["raw_ssthresh"],
            "rtt_us": step_info["raw_rtt"],
            "action": int(action[0]),
            "reward": float(reward[0])
        })

        if done[0]:
            break

    env.close()

    print("\n--- Phase 4: Saving Traces ---")
    df = pd.DataFrame(trace_data)
    csv_filename = "rl_tcp_traces.csv"
    df.to_csv(csv_filename, index=False)
    print(
        f"[*] Evaluation traces saved to {csv_filename}! | "
        f"Mean cWnd: {df['cwnd_mss'].mean():.2f} MSS | "
        f"Mean RTT: {df['rtt_us'].mean() / 1000.0:.2f} ms | "
        f"Mean Reward: {df['reward'].mean():+.2f}"
    )


if __name__ == "__main__":
    main()