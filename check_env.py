import numpy as np
from tcp_env import CustomTcpEnv

def run_test():
    print("[*] Connecting to running ns-3 simulation on port 7144...")
    env = CustomTcpEnv(port=7144)

    obs, info = env.reset()
    print(f"[*] Reset successful. Initial Obs:\n{obs}")
    print(f"[*] Obs shape: {obs.shape}, Dtype: {obs.dtype}")

    total_reward = 0.0
    for step_idx in range(20):
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward

        print(f"Step {step_idx:02d} | Action: {action} | "
              f"cWnd: {obs[0]:.0f} | RTT(us): {obs[5]:.0f} | "
              f"Reward: {reward:.3f}")

        if done or truncated:
            print("[*] Episode completed.")
            break

    print(f"[*] Verification complete. Cumulative Reward: {total_reward:.3f}")
    env.close()

if __name__ == "__main__":
    run_test()