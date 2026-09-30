import os
import gymnasium as gym
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from tcp_env import CustomTcpEnv

def main():
    print("--- Phase 1: Environment Setup ---")
    port = 7144

    # Wrap in VecNormalize to stabilize value estimation and policy gradients
    raw_env = DummyVecEnv([lambda: CustomTcpEnv(port=port)])
    env = VecNormalize(raw_env, norm_obs=True, norm_reward=True, clip_reward=5.0)

    print("[*] Performing initial handshake with ns-3...")
    obs = env.reset()
    print(f"[*] Handshake successful! Initial Observation: {obs}")

    print("\n--- Phase 2: Training Agent (50,000 Steps) ---")
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=1e-4,
        n_steps=512,
        batch_size=64,
        n_epochs=10,
        gamma=0.95,
        ent_coef=0.08,
        clip_range=0.2
    )

    model.learn(total_timesteps=50000)

    model.save("ppo_tcp_normalized")
    env.save("vec_normalize.pkl")
    print("[*] Training finished. Model saved.")

    print("\n--- Phase 3: Evaluation & Trace Logging ---")
    # Disable training mode on normalization wrapper for deterministic evaluation
    env.training = False
    env.norm_reward = False

    trace_data = []
    obs = env.reset()
    done = False
    step_count = 0

    while not done and step_count < 400:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, infos = env.step(action)
        step_info = infos[0]

        trace_data.append({
            "step": step_count,
            "cwnd": step_info["raw_cwnd"],
            "ssThresh": step_info["raw_ssthresh"],
            "rtt_us": step_info["raw_rtt"],
            "action": int(action[0]),
            "reward": float(reward[0])
        })

        step_count += 1

    env.close()

    print("\n--- Phase 4: Saving Traces ---")
    df = pd.DataFrame(trace_data)
    csv_filename = "rl_tcp_traces.csv"
    df.to_csv(csv_filename, index=False)
    print(f"[*] Evaluation traces saved to {csv_filename}!")

if __name__ == '__main__':
    main()