"""Runs comparison experiments (seeds x algorithms) as ns-3 subprocesses."""
import asyncio
import json
import os
import re
import shutil
import socket
import sys
import time

from metrics import aggregate, load_trace, summarize_run

DASHBOARD = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(DASHBOARD)
NS3_DIR = os.environ.get("NS3_DIR", os.path.join(REPO, "..", "ns-allinone-3.40", "ns-3.40"))
SIM_BIN = os.path.join(NS3_DIR, "build", "contrib", "opengym", "examples", "ns3.40-rl-tcp")
EXPERIMENTS = os.path.join(REPO, "experiments")
SCRATCH = os.path.join(EXPERIMENTS, "_scratch")
PYTHON = sys.executable

ALGORITHMS = ["rl", "newreno", "cubic", "bbr"]
PARAMS = {
    # name: (default, min, max)
    "runs": (3, 1, 10),
    "seed": (1, 1, 100000),
    "duration": (60, 20, 300),
    "minRate": (3.5, 1.0, 50.0),
    "maxRate": (11.5, 1.0, 50.0),
    "minHold": (3.5, 1.0, 30.0),
    "maxHold": (9.0, 1.0, 30.0),
    "queueSize": (60, 10, 500),
    "bottleneckDelay": (20.0, 5.0, 100.0),
}
# Conditions the shipped PPO model was trained under
TRAINING = {"minRate": 3.5, "maxRate": 11.5, "minHold": 3.5, "maxHold": 9.0,
            "queueSize": 60, "bottleneckDelay": 20.0}


def available_models():
    models = []
    if os.path.exists(os.path.join(REPO, "ppo_tcp_normalized.zip")):
        models.append("ppo_tcp_normalized.zip")
    ckpt = os.path.join(REPO, "checkpoints_ppo_tcp")
    if os.path.isdir(ckpt):
        models += sorted(f"checkpoints_ppo_tcp/{f}" for f in os.listdir(ckpt) if f.endswith(".zip"))
    return models


def validate(raw):
    cfg = {}
    for key, (default, lo, hi) in PARAMS.items():
        val = raw.get(key, default)
        try:
            val = float(val)
        except (TypeError, ValueError):
            raise ValueError(f"{key} must be a number")
        if not lo <= val <= hi:
            raise ValueError(f"{key} must be between {lo} and {hi}")
        cfg[key] = int(val) if isinstance(default, int) else val
    if cfg["minRate"] >= cfg["maxRate"]:
        raise ValueError("Minimum capacity must be below maximum capacity")
    if cfg["minHold"] > cfg["maxHold"]:
        raise ValueError("Minimum hold time must not exceed maximum hold time")
    algos = [a for a in raw.get("algorithms", ALGORITHMS) if a in ALGORITHMS]
    if not algos:
        raise ValueError("Pick at least one algorithm")
    cfg["algorithms"] = algos
    model = raw.get("model") or "ppo_tcp_normalized.zip"
    if "rl" in algos:
        if model not in available_models():
            raise ValueError("PPO model not found. Train the agent first (python train.py).")
    cfg["model"] = model
    return cfg


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def experiment_dir(exp_id):
    for base in (EXPERIMENTS, SCRATCH):
        path = os.path.join(base, exp_id)
        if exp_id != "_scratch" and os.path.isdir(path):
            return path
    return None


class Job:
    def __init__(self, cfg, name=None):
        self.id = time.strftime("%Y%m%d-%H%M%S")
        self.cfg = cfg
        self.name = name or time.strftime("Experiment %d %b %H:%M")
        self.dir = os.path.join(SCRATCH, self.id)
        self.started = time.time()
        self.status = "running"
        self.error = None
        self.version = 0
        seeds = [cfg["seed"] + i for i in range(cfg["runs"])]
        self.runs = [{"seed": s, "algo": a, "progress": 0.0, "status": "queued"}
                     for s in seeds for a in cfg["algorithms"]]
        self._procs = set()
        self._task = None

    def snapshot(self):
        return {"id": self.id, "name": self.name, "status": self.status, "error": self.error,
                "elapsed": time.time() - self.started, "runs": self.runs, "config": self.cfg}

    def _touch(self):
        self.version += 1

    async def run(self):
        os.makedirs(os.path.join(self.dir, "runs"), exist_ok=True)
        with open(os.path.join(self.dir, "config.json"), "w") as f:
            json.dump(self.cfg, f, indent=2)
        sem = asyncio.Semaphore(max(1, (os.cpu_count() or 2) // 2))

        async def guarded(run):
            async with sem:
                if self.status == "running":
                    await self._run_one(run)

        try:
            await asyncio.gather(*(guarded(r) for r in self.runs))
            if self.status == "cancelled":
                return
            failed = [r for r in self.runs if r["status"] == "failed"]
            if len(failed) == len(self.runs):
                raise RuntimeError(failed[0].get("error") or "All runs failed")
            self._write_summary()
            self.status = "done"
        except Exception as exc:  # surfaced to the UI
            self.status = "failed"
            self.error = str(exc)
        finally:
            self._touch()

    async def _run_one(self, run):
        cfg = self.cfg
        trace = os.path.join(self.dir, "runs", f"{run['seed']}_{run['algo']}.csv")
        args = [SIM_BIN, f"--variant={run['algo']}", f"--seed={run['seed']}",
                f"--duration={cfg['duration']}", f"--trace={trace}",
                f"--minRate={cfg['minRate']}", f"--maxRate={cfg['maxRate']}",
                f"--minHold={cfg['minHold']}", f"--maxHold={cfg['maxHold']}",
                f"--queueSize={cfg['queueSize']}", f"--bottleneckDelay={cfg['bottleneckDelay']}"]
        agent = None
        run["status"] = "running"
        self._touch()
        try:
            if run["algo"] == "rl":
                port = _free_port()
                args.append(f"--openGymPort={port}")
                agent = await asyncio.create_subprocess_exec(
                    PYTHON, os.path.join(DASHBOARD, "rl_eval.py"), "--port", str(port),
                    "--model", os.path.join(REPO, cfg["model"]),
                    stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.PIPE)
                self._procs.add(agent)
            sim = await asyncio.create_subprocess_exec(
                *args, cwd=NS3_DIR, stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.PIPE)
            self._procs.add(sim)
            tail = []
            async for line in sim.stderr:
                text = line.decode(errors="replace").strip()
                m = re.match(r"PROGRESS ([\d.]+) ([\d.]+)", text)
                if m:
                    run["progress"] = min(1.0, float(m.group(1)) / float(m.group(2)))
                    self._touch()
                elif text:
                    tail = (tail + [text])[-5:]
            code = await sim.wait()
            if agent:
                agent_err = (await agent.stderr.read()).decode(errors="replace")
                if await agent.wait() != 0 and self.status == "running":
                    raise RuntimeError("PPO agent crashed: " + agent_err.strip().splitlines()[-1]
                                       if agent_err.strip() else "PPO agent crashed")
            if self.status != "running":
                run["status"] = "cancelled"
            elif code != 0 or not os.path.exists(trace):
                raise RuntimeError(" ".join(tail) or f"ns-3 exited with code {code}")
            else:
                run["status"], run["progress"] = "done", 1.0
        except Exception as exc:
            run["status"], run["error"] = "failed", str(exc)
        finally:
            for p in (agent,):
                if p and p.returncode is None:
                    p.kill()
            self._procs = {p for p in self._procs if p.returncode is None}
            self._touch()

    def cancel(self):
        if self.status != "running":
            return
        self.status = "cancelled"
        for p in list(self._procs):
            if p.returncode is None:
                p.kill()
        self._touch()

    def _write_summary(self):
        per_algo = {a: [] for a in self.cfg["algorithms"]}
        runs = []
        for r in self.runs:
            if r["status"] != "done":
                continue
            path = os.path.join(self.dir, "runs", f"{r['seed']}_{r['algo']}.csv")
            s = summarize_run(load_trace(path, self.cfg), self.cfg)
            if s:
                per_algo[r["algo"]].append(s)
                runs.append({"seed": r["seed"], "algo": r["algo"], **s})
        summary = {"id": self.id, "name": self.name, "created": self.started,
                   "saved": False, "config": self.cfg, "runs": runs,
                   "algorithms": {a: aggregate(v) for a, v in per_algo.items() if v}}
        with open(os.path.join(self.dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)


class JobManager:
    def __init__(self):
        self.jobs = {}

    def active(self):
        return next((j for j in self.jobs.values() if j.status == "running"), None)

    def start(self, cfg):
        if self.active():
            raise ValueError("An experiment is already running")
        if not os.path.exists(SIM_BIN):
            raise ValueError("ns-3 binary not found. Run dashboard/build_sim.sh first.")
        # Unsaved results only live until the next run
        if os.path.isdir(SCRATCH):
            shutil.rmtree(SCRATCH)
        job = Job(cfg)
        self.jobs = {job.id: job}
        job._task = asyncio.create_task(job.run())
        return job


def list_saved():
    out = []
    if not os.path.isdir(EXPERIMENTS):
        return out
    for name in os.listdir(EXPERIMENTS):
        path = os.path.join(EXPERIMENTS, name, "summary.json")
        if name != "_scratch" and os.path.exists(path):
            with open(path) as f:
                s = json.load(f)
            out.append({"id": s["id"], "name": s["name"], "created": s["created"],
                        "algorithms": s["config"]["algorithms"], "runs": s["config"]["runs"],
                        "duration": s["config"]["duration"]})
    return sorted(out, key=lambda e: e["created"], reverse=True)


def load_summary(exp_id):
    d = experiment_dir(exp_id)
    if not d or not os.path.exists(os.path.join(d, "summary.json")):
        return None
    with open(os.path.join(d, "summary.json")) as f:
        return json.load(f)


def load_seed_traces(exp_id, seed):
    summary = load_summary(exp_id)
    if not summary:
        return None
    d = experiment_dir(exp_id)
    out = {}
    for algo in summary["config"]["algorithms"]:
        path = os.path.join(d, "runs", f"{seed}_{algo}.csv")
        if os.path.exists(path):
            out[algo] = load_trace(path, summary["config"])
    return out


def save_experiment(exp_id, name):
    src = experiment_dir(exp_id)
    if not src:
        return None
    summary = load_summary(exp_id)
    summary["name"] = name.strip() or summary["name"]
    summary["saved"] = True
    with open(os.path.join(src, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    if os.path.dirname(src) == SCRATCH:
        shutil.move(src, os.path.join(EXPERIMENTS, exp_id))
    return summary


def delete_experiment(exp_id):
    d = experiment_dir(exp_id)
    if d and os.path.dirname(d) == EXPERIMENTS:
        shutil.rmtree(d)
        return True
    return False
