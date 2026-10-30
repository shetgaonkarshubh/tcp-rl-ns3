# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

PPO (Stable-Baselines3) agent that controls TCP cWnd inside an ns-3.40 simulation via ns3-gym (ZeroMQ, port 7144). Two processes: the C++ sim (`rl-tcp.cc`) is the ZMQ server, the Python trainer (`train.py` → `tcp_env.py`) connects to it. Target platform is Linux; there is no test suite, linter, or build config in this repo.

## Commands

`rl-tcp.cc` is not built here — it must be copied into an ns-3.40 tree at `contrib/opengym/examples/rl-tcp/rl-tcp.cc` and built with `./ns3 build` from the ns-3 root.

```bash
# Terminal 1 (ns-3 root): start the sim server
./ns3 run 'rl-tcp --openGymPort=7144 --duration=15000'   # also: --seed=N (0 = random_device)

# Terminal 2 (this repo, venv active)
python train.py          # 100k-step PPO training, then 400-step deterministic eval → rl_tcp_traces.csv
python plot_results.py   # rl_tcp_traces.csv → rl_tcp_zoomed_performance.png
python check_env.py      # 20 random-action steps against a running sim
python tcp_env.py        # offline: runs CustomTcpEnv against a built-in mock (no ns-3 needed)
python realtime_demo.py --delay 0.04 --loop   # animated matplotlib replay of rl_tcp_traces.csv (Space pause, Q quit)
python generate_deck.py  # rebuilds tcp_rl_presentation.pdf (needs reportlab)
```

`realtime_demo.py` is a replay of the eval trace CSV, not a live connection to ns-3 or the policy. `run_realtime_demo.sh` looks for a venv at `../venv`, then `~/ns-allinone-3.40/ns-3.40/venv`, then `./venv`.

Python deps: `torch gymnasium stable-baselines3 pandas numpy matplotlib`, plus `ns3gym` installed from `ns-3.40/contrib/opengym/model/ns3gym`. `CustomTcpEnv(mock_env=...)` accepts any object with `reset/step/close` returning the 15-element raw observation, which is the way to test env logic without the simulator.

### Comparison dashboard (`dashboard/`)
```bash
dashboard/build_sim.sh      # symlinks rl-tcp.cc into ns-3 and rebuilds (make -C $NS3_DIR/cmake-cache rl-tcp)
dashboard/run_dashboard.sh  # FastAPI + static UI at http://localhost:8000 (needs fastapi, uvicorn in the venv)
```
- `runner.py` runs seeds × algorithms as parallel ns-3 subprocesses (`--variant=rl|newreno|cubic|bbr --trace=…`). An RL run also starts `rl_eval.py` on a free port (Python binds, ns-3 connects; order doesn't matter). Progress comes from the simulator's `PROGRESS t total` stderr lines.
- Results live in `experiments/<id>/` (gitignored). Unsaved runs sit in `experiments/_scratch/` and are wiped by the next run; Save moves them up a level.
- `metrics.py` derives throughput from `rx_bytes` deltas and the ideal window (BDP) from capacity × base RTT, where base RTT = 2 × (bottleneckDelay + 1 ms).
- Frontend is plain `static/{index.html,app.css,app.js}` (Chart.js and Lucide from a CDN, no build step). Algorithm colours are CSS variables (`--algo-*`), validated as a set for colourblind separation on the warm dark canvas; keep that order if adding series. The UI is a light dashboard (grey canvas, white cards, orange primary #d04a0a; PPO is the orange chart series) with a full-height sidebar.

### Pipeline scripts — read before running

- Both scripts hardcode `PROJECT_DIR=$HOME/ns-allinone-3.40/ns-3.40/tcp_rl_project` and `VENV_DIR=$NS3_DIR/venv`, and run `rm -f *.zip *.csv *.png *.pkl` in the project dir (this deletes the committed `rl_tcp_zoomed_performance.png`).
- `run_pipeline.sh` is not a plain launcher: it runs `tmux kill-server`, `pkill -9`s the sim/trainer, **overwrites** `run_pipeline_tmux.sh` with an embedded version (`--duration=6500`), then runs it.
- `run_pipeline_tmux.sh` (as committed) uses `--duration=15000` and waits for the port with `nc -z`.
- Sim duration matters: training is 100k steps × 0.1 s = 10,000 s of sim time plus eval, so `--duration=6500` ends the simulation before training finishes.

## Architecture

### C++ ↔ Python contract
- **Raw observation** (`TcpGymEnv::GetObservation`, 15 doubles): `[4]` ssThresh B, `[5]` cWnd B, `[6]` segment size, `[8]` bytes in flight, `[9]` last RTT µs, `[10]` min RTT µs, `[12]` cong state (forced to 3 if a bottleneck-queue drop occurred since last step), `[13]` current bottleneck Mbps. Python indexes these positions directly — changing the order breaks `_process_observation`.
- **Action** sent to ns-3: `[ssThresh_bytes, cWnd_bytes]` (uint32). C++ clamps to [2, 200] MSS and the custom `TcpRlDynamicCongestion` (subclass of `TcpNewReno`) pins `tcb->m_cWnd`/`m_ssThresh` to these targets on every ACK, so NewReno's own window growth is fully overridden.
- C++ reward is always 0; reward is computed entirely in Python. ns-3 steps every 0.1 s sim time (first notify at 0.15 s).

### Bottleneck dynamics
- `rl-tcp.cc`: sender —100 Mbps/0.5 ms— R1 —bottleneck 20 ms, 60-pkt DropTail— R2 —100 Mbps/0.5 ms— receiver (≈42 ms unloaded RTT). `RandomizeBottleneckBandwidth` resamples the bottleneck rate in [3.5, 11.5] Mbps every 3.5–9 s. All of these are command-line options (`--minRate --maxRate --minHold --maxHold --queueSize --bottleneckDelay`).
- The schedule draws only from `g_rng` and tracks the rate in `g_currentMbps`, so the same `--seed` gives every `--variant` an identical capacity schedule. Keep it that way: comparisons rely on it.
- The default traffic-control queue disc is uninstalled on the bottleneck egress so the DropTail queue really is the buffer. Overflow drops then appear on the TrafficControlLayer's `TcDrop` trace, not the queue's `Drop` trace; both feed `DropTrace`.
- Non-RL variants never create the OpenGym interface; `g_gymEnv` is null and the gym hooks are no-ops.
- `tcp_env.py` `_step_stochastic_bottleneck`: if `raw_obs[13] > 0.5` (always true with this `rl-tcp.cc`) the target BDP is derived from the sim's true bottleneck rate × calibrated base RTT. Otherwise it falls back to its own internal Markov BDP generator. So the agent's "target BDP" is privileged ground truth from the simulator, not an estimate.
- Base RTT is calibrated as `1.035 × min observed RTT`; the 84 ms default is only a pre-calibration fallback.
- Python adds a synthetic "virtual queue" delay on top of the real RTT when cWnd exceeds target BDP.

### Python env (`CustomTcpEnv`)
- Agent observation is 8 features, all relative to the target BDP: coarse/fine (tanh) cWnd error, under/sweet-spot/over-BDP flags (±0.6 MSS), RTT inflation, RTT gradient, and a loss flag. **The README's description of the observation and its 0.15 delay threshold are out of date** — the code uses these 8 features and a 0.25 excess-delay threshold. Trust `tcp_env.py`.
- 7 discrete actions map to cWnd deltas `[-4, -2, -0.5, 0, +0.5, +2, +4]` MSS, but `_compute_proposed_cwnd` rewrites the chosen delta with hard-coded guardrails: damping near target, a forced increase when below BDP and uncongested, a forced decrease when above BDP, and a cap during loss/recovery. Python then clamps cWnd to [8, 40] MSS. Observed behaviour is therefore a mix of policy and heuristics. Keep this in mind when evaluating or changing either one.
- ssThresh sent to ns-3 is always the target BDP.
- `reset()` does the ZMQ handshake only once. Later resets just push an initial cWnd via `step` and do not restart the sim, so an episode ends only when ns-3 finishes. `virtual_episode_steps` is accepted but unused.
- `set_training_mode(False)` is called through `env_method` before evaluation. It resets the BDP schedule to the nominal 36.25 MSS.

### Training (`train.py`)
`DummyVecEnv` → `VecNormalize(norm_obs=False, norm_reward=True)` → PPO `MlpPolicy` (SiLU, separate pi/vf [128,128]). Outputs: `ppo_tcp_normalized.zip`, `vec_normalize.pkl`, `checkpoints_ppo_tcp/` (every 25k steps), and `rl_tcp_traces.csv`. `.zip`, `.pkl`, and `.csv` files are gitignored. The trace CSV's `cwnd`/`ssThresh` columns are rescaled to bytes at 1448 B/MSS (`plot_segment_size`) for plotting.

### Known stale code
`check_env.py` prints `obs[0]` as cWnd and `obs[5]` as RTT µs. Those indices are from an older raw-observation layout; they are now normalized features.
