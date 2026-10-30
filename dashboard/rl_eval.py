"""Run a trained PPO policy against an ns-3 rl-tcp simulation until the simulation ends.

The ns-3 side (started separately with --variant=rl) writes the trace CSV; this process
only drives the agent. Python binds the ZeroMQ port, so start this before the simulator.
"""
import argparse
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from stable_baselines3 import PPO  # noqa: E402
from tcp_env import CustomTcpEnv  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--model", default=os.path.join(REPO, "ppo_tcp_normalized.zip"))
    args = parser.parse_args()

    model = PPO.load(args.model, device="cpu")
    env = CustomTcpEnv(port=args.port)
    env.set_training_mode(False)

    obs, _ = env.reset()
    steps = 0
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, truncated, _ = env.step(action)
        steps += 1
        if done or truncated:
            break
    env.close()
    print(f"RL_DONE steps={steps}", flush=True)


if __name__ == "__main__":
    main()
