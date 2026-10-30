"""Turn rl-tcp.cc trace CSVs into replay series and summary statistics."""
import csv
import math

MSS = 1448
STEP_S = 0.1


def base_rtt_ms(config):
    # Two 0.5 ms access links plus the bottleneck, both directions
    return 2.0 * (config["bottleneckDelay"] + 1.0)


def load_trace(path, config):
    """Return column arrays for one run, with derived throughput and BDP."""
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    base = base_rtt_ms(config)
    t, cwnd, rtt, cap, thr, queue, drops, bdp = [], [], [], [], [], [], [], []
    prev_rx = None
    for r in rows:
        rx = int(r["rx_bytes"])
        mbps = float(r["bottleneck_mbps"])
        t.append(round(float(r["t"]), 1))
        cwnd.append(round(int(r["cwnd_bytes"]) / MSS, 2))
        rtt.append(round(int(r["rtt_us"]) / 1000.0, 2))
        cap.append(mbps)
        thr.append(0.0 if prev_rx is None else round((rx - prev_rx) * 8 / STEP_S / 1e6, 3))
        queue.append(int(r["queue_pkts"]))
        drops.append(int(r["drops"]))
        bdp.append(round(mbps * 1e6 * base / 1000.0 / 8 / MSS, 2))
        prev_rx = rx
    return {
        "t": t, "cwnd": cwnd, "rtt": rtt, "capacity": cap, "throughput": thr,
        "queue": queue, "drops": drops, "bdp": bdp,
        "rx_total": int(rows[-1]["rx_bytes"]) - int(rows[0]["rx_bytes"]) if rows else 0,
    }


def _percentile(values, p):
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * p / 100.0
    lo, hi = math.floor(k), math.ceil(k)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def summarize_run(trace, config):
    n = len(trace["t"])
    if n < 2:
        return None
    duration = trace["t"][-1] - trace["t"][0]
    throughput = trace["rx_total"] * 8 / duration / 1e6
    capacity = sum(trace["capacity"]) / n
    rtts = [v for v in trace["rtt"] if v > 0]
    base = base_rtt_ms(config)
    return {
        "throughput": throughput,
        "utilization": 100.0 * throughput / capacity if capacity else 0.0,
        "rtt": sum(rtts) / len(rtts) if rtts else 0.0,
        "rtt_p95": _percentile(rtts, 95),
        "queue_delay": max(0.0, (sum(rtts) / len(rtts) if rtts else base) - base),
        "queue": sum(trace["queue"]) / n,
        "drops": trace["drops"][-1],
        "cwnd_error": sum(abs(c - b) for c, b in zip(trace["cwnd"], trace["bdp"])) / n,
    }


def aggregate(per_run):
    """per_run: list of summary dicts for one algorithm -> mean and std per metric."""
    out = {}
    if not per_run:
        return out
    for key in per_run[0]:
        vals = [r[key] for r in per_run]
        mean = sum(vals) / len(vals)
        std = math.sqrt(sum((v - mean) ** 2 for v in vals) / len(vals)) if len(vals) > 1 else 0.0
        out[key] = {"mean": mean, "std": std}
    return out
