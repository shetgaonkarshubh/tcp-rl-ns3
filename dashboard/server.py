"""Comparison dashboard: FastAPI backend serving the static UI and the experiment API."""
import asyncio
import json
import os

from fastapi import Body, FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

import runner

STATIC = os.path.join(runner.DASHBOARD, "static")
app = FastAPI(title="TCP-RL Compare")
jobs = runner.JobManager()


@app.get("/")
def index():
    return FileResponse(os.path.join(STATIC, "index.html"))


@app.get("/api/defaults")
def defaults():
    return {
        "params": {k: {"default": d, "min": lo, "max": hi} for k, (d, lo, hi) in runner.PARAMS.items()},
        "algorithms": runner.ALGORITHMS,
        "models": runner.available_models(),
        "training": runner.TRAINING,
        "simReady": os.path.exists(runner.SIM_BIN),
        "active": jobs.active().id if jobs.active() else None,
    }


@app.post("/api/experiments")
async def start(raw: dict = Body(...)):
    try:
        job = jobs.start(runner.validate(raw))
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    return job.snapshot()


@app.get("/api/experiments/{exp_id}/events")
async def events(exp_id: str):
    job = jobs.jobs.get(exp_id)
    if not job:
        raise HTTPException(404, "No running experiment with that id")

    async def stream():
        seen = -1
        while True:
            if job.version != seen:
                seen = job.version
                yield f"data: {json.dumps(job.snapshot())}\n\n"
                if job.status != "running":
                    return
            await asyncio.sleep(0.2)

    return StreamingResponse(stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache"})


@app.get("/api/experiments/{exp_id}/status")
def status(exp_id: str):
    job = jobs.jobs.get(exp_id)
    if not job:
        raise HTTPException(404, "No running experiment with that id")
    return job.snapshot()


@app.post("/api/experiments/{exp_id}/cancel")
def cancel(exp_id: str):
    job = jobs.jobs.get(exp_id)
    if not job:
        raise HTTPException(404, "No running experiment with that id")
    job.cancel()
    return job.snapshot()


@app.get("/api/experiments")
def library():
    return runner.list_saved()


@app.get("/api/experiments/{exp_id}")
def summary(exp_id: str):
    s = runner.load_summary(exp_id)
    if not s:
        raise HTTPException(404, "Experiment not found")
    return s


@app.get("/api/experiments/{exp_id}/traces/{seed}")
def traces(exp_id: str, seed: int):
    t = runner.load_seed_traces(exp_id, seed)
    if t is None:
        raise HTTPException(404, "Experiment not found")
    return t


@app.post("/api/experiments/{exp_id}/save")
def save(exp_id: str, body: dict = Body(...)):
    s = runner.save_experiment(exp_id, str(body.get("name", "")))
    if not s:
        raise HTTPException(404, "Experiment not found")
    return s


@app.delete("/api/experiments/{exp_id}")
def delete(exp_id: str):
    if not runner.delete_experiment(exp_id):
        raise HTTPException(404, "Saved experiment not found")
    return {"ok": True}


app.mount("/static", StaticFiles(directory=STATIC), name="static")
