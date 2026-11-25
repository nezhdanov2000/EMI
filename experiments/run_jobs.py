"""
Run the heavy computations in small, restartable pieces.

The work is cut into JOBS. One job is one call of `compare_baselines.py`
for a few splits of ONE configuration with one group of methods; its
results go to the per-split cache as soon as each split is computed. A job
that crashes, is killed or runs out of time loses at most the split it was
on: every other job is unaffected, and running the same command again
continues from the cache. The state of every job is kept in
`experiments/results/jobs/<plan>.json` and its output in
`experiments/logs/jobs/<job id>.log`.

Commands (run from the repository root, with the experiments Python):

    python experiments/run_jobs.py hardware              # what this machine has, suggested --workers
    python experiments/run_jobs.py plan                  # write the job list of the fixed (tau, m) grid
    python experiments/run_jobs.py status                # done / failed / pending per size class and dataset
    python experiments/run_jobs.py run --size small      # run one part (see PARTS below)
    python experiments/run_jobs.py run --retry-failed    # run again whatever failed
    python experiments/run_jobs.py summarize             # compare_grid_<dataset>.csv and pairwise_grid.csv

The certified selection at the grid's purity floors is a second plan with
its own state file; every command takes `--plan grid_certified`:

    python experiments/run_jobs.py plan --plan grid_certified
    python experiments/run_jobs.py run --plan grid_certified --size small
    python experiments/run_jobs.py summarize --plan grid_certified   # pairwise_grid_certified.csv

The large recorded tables (fars, shuttle; experiments/build_pmlb_benchmarks.py)
have two plans of their own, `large_real` (observed-purity rule, tau x m) and
`large_real_certified`; their summaries are compare_large_real[_certified]_<dataset>.csv
and pairwise_large_real[_certified].csv.

PARTS. Jobs are grouped twice, and `run` takes either or both filters:

    --stage  core      full grid, partial centres, greedy chain, rules, disjoint rules, CART   (default)
             wracc     WRAcc lists (top-up of cached splits)
             ssdpp     SSD++ (needs the patched `rulelist`; minutes to hours per split on large tables)
             tree_opt  optimal tree (needs `pydl8.5`; up to 40 minutes per split)
    --size   small     tables that take seconds per split
             medium    tens of seconds per split
             heavy     adult, chess_krkp, splice: minutes to half an hour per split
             huge      connect_4: hours per split, several GB of memory per job
    --only   one dataset name

Other options of `run`: --workers N (default: from the hardware), --hours H
(start no new job after H hours; running jobs finish), --max-jobs N,
--timeout-hours H per job (default 12), --dry-run.

Standard library only; Python 3.8+.
"""
from __future__ import annotations

import argparse
import ctypes
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments"
STATE_DIR = EXPERIMENTS / "results" / "jobs"
LOG_DIR = EXPERIMENTS / "logs" / "jobs"
sys.path.insert(0, str(EXPERIMENTS))

STAGES: Dict[str, Tuple[str, ...]] = {
    "core": ("vsf", "vsf_partial", "vsf_greedy", "rules", "rules_pure", "rules_disjoint", "tree"),
    "wracc": ("wracc", "wracc_raw"),
    "ssdpp": ("ssdpp",),
    "tree_opt": ("tree_opt",),
}
#: Size class of a dataset: (splits per job, memory per job in GB, rough core-seconds per split of the core stage).
#: The seconds are the measured means of the paper's runs (results/comparison_summary.csv); they only feed the
#: time estimate printed by `status`, nothing depends on them.
SIZE_CLASSES: Dict[str, Tuple[int, float]] = {"small": (25, 1.0), "medium": (5, 2.0), "heavy": (1, 4.0), "huge": (1, 10.0)}
DATASET_SIZE: Dict[str, Tuple[str, float]] = {
    "breast_cancer": ("small", 0.5), "car_evaluation": ("small", 0.3), "titanic": ("small", 0.4),
    "breast_cancer_wisconsin": ("small", 2.5), "hmda": ("small", 3.0), "mlc_churn": ("small", 5.5),
    "thyroid_recurrence": ("small", 6.0), "credit": ("small", 8.0), "nursery": ("small", 1.2),
    "mushroom": ("medium", 57.0), "wa_churn": ("medium", 23.0), "attrition": ("medium", 34.0), "resume": ("medium", 39.0),
    "splice": ("heavy", 250.0), "adult": ("heavy", 1500.0), "chess_krkp": ("heavy", 1700.0),
    "connect_4": ("huge", 8600.0),
    # large recorded tables (2026-11-25): seconds per split are guesses, not measurements
    "shuttle": ("heavy", 120.0), "fars": ("heavy", 1800.0),
}
#: Rough cost of the other stages relative to the core stage of the same split.
STAGE_FACTOR: Dict[str, float] = {"core": 1.0, "wracc": 0.4, "ssdpp": 1.0, "tree_opt": 0.5}


# --------------------------------------------------------------------------
# Hardware
# --------------------------------------------------------------------------
def memory_gb() -> Tuple[Optional[float], Optional[float]]:
    """(total, available) physical memory in GB, or (None, None) when the platform gives no answer."""
    try:
        if os.name == "nt":
            class MemoryStatus(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("sullAvailExtendedVirtual", ctypes.c_ulonglong)]
            status = MemoryStatus()
            status.dwLength = ctypes.sizeof(MemoryStatus)
            if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):  # type: ignore[attr-defined]
                return None, None
            return status.ullTotalPhys / 2 ** 30, status.ullAvailPhys / 2 ** 30
        info: Dict[str, float] = {}
        with open("/proc/meminfo", encoding="utf-8") as fh:
            for line in fh:
                key, _, rest = line.partition(":")
                info[key] = float(rest.split()[0]) / 2 ** 20
        return info.get("MemTotal"), info.get("MemAvailable")
    except Exception:
        return None, None


def cpu_name() -> str:
    if os.name == "nt":
        return os.environ.get("PROCESSOR_IDENTIFIER", platform.processor() or "unknown")
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def suggested_workers() -> int:
    """
    Half of the logical processors (on a hybrid laptop CPU the other half are
    hyper-threads and efficiency cores that add little and add heat), capped
    by available memory at 2 GB per worker, at least 1.
    """
    logical = os.cpu_count() or 2
    workers = max(1, logical // 2)
    _, available = memory_gb()
    if available is not None:
        workers = min(workers, max(1, int(available // 2)))
    return workers


def hardware_report() -> int:
    total, available = memory_gb()
    disk = shutil.disk_usage(str(ROOT))
    print(f"system        : {platform.system()} {platform.release()} ({platform.machine()})")
    print(f"processor     : {cpu_name()}")
    print(f"logical CPUs  : {os.cpu_count()}")
    print("memory        : " + ("unknown" if total is None else f"{total:.1f} GB total, {available:.1f} GB available now"))
    print(f"disk (repo)   : {disk.free / 2 ** 30:.0f} GB free of {disk.total / 2 ** 30:.0f} GB")
    print(f"python        : {sys.version.split()[0]} ({sys.executable})")
    for module, needed_for in (("numpy", "everything"), ("pandas", "everything"), ("scipy", "certified selection, summaries"),
                               ("sklearn", "tree (core stage)"), ("rulelist", "ssdpp stage"), ("pydl85", "tree_opt stage")):
        try:
            mod = __import__(module)
            print(f"  {module:9s}: {getattr(mod, '__version__', 'installed')}")
        except Exception as exc:  # a package that fails to import is as good as absent
            print(f"  {module:9s}: MISSING ({needed_for}): {type(exc).__name__}")
    workers = suggested_workers()
    print(f"suggested     : --workers {workers}")
    if available is not None and available < 12:
        print("note          : under 12 GB available: run --size huge (connect_4) with --workers 1, or leave it out")
    print("note          : on a laptop, keep it on mains power, switch sleep off and keep the vents clear; "
          "use --hours to stop a session cleanly")
    return 0


# --------------------------------------------------------------------------
# Plan
# --------------------------------------------------------------------------
def state_path(plan: str) -> Path:
    return STATE_DIR / f"{plan}.json"


def load_state(plan: str) -> Dict[str, Any]:
    path = state_path(plan)
    if not path.exists():
        raise SystemExit(f"no plan at {path}: run `python experiments/run_jobs.py plan` first")
    return json.loads(path.read_text(encoding="utf-8"))


def save_state(plan: str, state: Dict[str, Any]) -> None:
    """Atomic: a crash while saving leaves the previous file intact."""
    path = state_path(plan)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=1), encoding="utf-8")
    for _ in range(20):
        try:
            os.replace(str(tmp), str(path))
            return
        except OSError:
            time.sleep(0.25)
    raise RuntimeError(f"could not write {path}")


#: Plans this runner knows: the plan's name is the `--config-set` of
#: compare_baselines.py / pairwise_partial.py its jobs run with.
PLANS: Tuple[str, ...] = ("grid", "grid_certified", "large_real", "large_real_certified")


def build_plan(plan: str, stages: Sequence[str], force: bool) -> int:
    from compare_baselines import N_SPLITS, configs_of  # imported here: needs numpy

    path = state_path(plan)
    old: Dict[str, Any] = {}
    if path.exists():
        old = {j["id"]: j for j in json.loads(path.read_text(encoding="utf-8"))["jobs"]}
        if not force:
            print(f"{path} exists: its finished jobs are kept, new jobs are added (use --force to start from nothing)")
    jobs: List[Dict[str, Any]] = []
    for stage in stages:
        for cfg in configs_of(plan):
            size, seconds = DATASET_SIZE.get(cfg.dataset, ("heavy", 600.0))
            chunk, mem = SIZE_CLASSES[size]
            total = N_SPLITS * cfg.n_repeats
            for start in range(0, total, chunk):
                splits = list(range(start, min(total, start + chunk)))
                job_id = (f"{stage}__{cfg.dataset}__{cfg.positive}__t{cfg.tau}__m{cfg.min_samples}"
                          f"__s{splits[0]:02d}-{splits[-1]:02d}").replace(">", "gt").replace("<", "lt").replace(" ", "_")
                job = {
                    "id": job_id, "config_set": plan, "stage": stage, "size": size, "dataset": cfg.dataset, "positive": cfg.positive,
                    "tau": cfg.tau, "min_samples": cfg.min_samples, "splits": splits, "mem_gb": mem,
                    "estimate_s": round(seconds * len(splits) * STAGE_FACTOR[stage], 1),
                    "status": "pending", "attempts": 0, "seconds": None, "returncode": None,
                }
                if not force and job_id in old and old[job_id]["status"] == "done":
                    job = old[job_id]
                jobs.append(job)
    save_state(plan, {"plan": plan, "created": time.strftime("%Y-%m-%d %H:%M:%S"), "jobs": jobs})
    print(f"wrote {path}: {len(jobs)} jobs")
    return print_status(plan)


def job_command(job: Dict[str, Any]) -> List[str]:
    return [sys.executable, str(EXPERIMENTS / "compare_baselines.py"), "--config-set", job.get("config_set", "grid"),
            "--only", job["dataset"], "--positive", job["positive"], "--tau", str(job["tau"]),
            "--min-samples", str(job["min_samples"]), "--splits", ",".join(str(s) for s in job["splits"]),
            "--methods", ",".join(STAGES[job["stage"]]), "--max-seconds", "1000000000", "--no-csv"]


# --------------------------------------------------------------------------
# Status
# --------------------------------------------------------------------------
def print_status(plan: str) -> int:
    state = load_state(plan)
    jobs = state["jobs"]
    print(f"plan {plan}: {len(jobs)} jobs")
    print(f"{'stage':9s} {'size':7s} {'dataset':26s} {'done':>5s} {'failed':>6s} {'left':>5s}  {'est. core-hours left':>20s}")
    keys = sorted({(j["stage"], j["size"], j["dataset"]) for j in jobs},
                  key=lambda k: (list(STAGES).index(k[0]), list(SIZE_CLASSES).index(k[1]), k[2]))
    grand = 0.0
    for stage, size, dataset in keys:
        sel = [j for j in jobs if (j["stage"], j["size"], j["dataset"]) == (stage, size, dataset)]
        done = sum(j["status"] == "done" for j in sel)
        failed = sum(j["status"] == "failed" for j in sel)
        left = [j for j in sel if j["status"] != "done"]
        hours = sum(j["estimate_s"] for j in left) / 3600
        grand += hours
        print(f"{stage:9s} {size:7s} {dataset:26s} {done:5d} {failed:6d} {len(left) - failed:5d}  {hours:20.1f}")
    spent = sum(j["seconds"] or 0.0 for j in jobs if j["status"] == "done") / 3600
    print(f"computed so far: {spent:.1f} core-hours; estimated left: {grand:.1f} core-hours "
          f"(rough: measured on the paper's floors; a size floor of 1 on a large table can cost several times more)")
    for j in jobs:
        if j["status"] == "failed":
            print(f"FAILED {j['id']} (exit {j['returncode']}, {j['attempts']} attempt(s)): see {LOG_DIR / (j['id'] + '.log')}")
    return 0


# --------------------------------------------------------------------------
# Run
# --------------------------------------------------------------------------
def select_jobs(jobs: List[Dict[str, Any]], args: argparse.Namespace) -> List[Dict[str, Any]]:
    wanted_status = {"pending", "failed"} if args.retry_failed else {"pending"}
    out = [j for j in jobs if j["status"] in wanted_status]
    if args.stage:
        out = [j for j in out if j["stage"] in args.stage.split(",")]
    if args.size:
        out = [j for j in out if j["size"] in args.size.split(",")]
    if args.only:
        out = [j for j in out if j["dataset"] in args.only.split(",")]
    # cheap jobs first: an early failure shows up in minutes, not after a night
    out.sort(key=lambda j: (j["estimate_s"], j["id"]))
    if args.max_jobs:
        out = out[: args.max_jobs]
    return out


def run_jobs(plan: str, args: argparse.Namespace) -> int:
    state = load_state(plan)
    jobs = state["jobs"]
    for j in jobs:  # a previous session that was killed leaves jobs marked running
        if j["status"] == "running":
            j["status"] = "pending"
    if args.stage is None and not args.retry_failed:
        args.stage = "core"
    queue = select_jobs(jobs, args)
    workers = args.workers or suggested_workers()
    total_mem, _ = memory_gb()
    mem_budget = 0.8 * total_mem if total_mem else float("inf")
    hours = sum(j["estimate_s"] for j in queue) / 3600
    print(f"{len(queue)} job(s) selected, about {hours:.1f} core-hours; {workers} worker(s)"
          + (f"; memory budget {mem_budget:.0f} GB" if total_mem else ""))
    if args.dry_run:
        for j in queue[:40]:
            print("  " + " ".join(job_command(j)[1:]))
        if len(queue) > 40:
            print(f"  ... and {len(queue) - 40} more")
        return 0
    if not queue:
        return 0
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    started = time.time()
    running: List[Tuple[Dict[str, Any], subprocess.Popen, Any, float]] = []
    finished = failed = 0
    env = dict(os.environ)
    # one thread per job: the parallelism is across jobs, not inside numpy
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env.setdefault(var, "1")
    env["PYTHONIOENCODING"] = "utf-8"
    try:
        while queue or running:
            accepting = args.hours is None or (time.time() - started) < args.hours * 3600
            used = sum(j["mem_gb"] for j, _, _, _ in running)
            while accepting and queue and len(running) < workers:
                pick = next((i for i, j in enumerate(queue) if used + j["mem_gb"] <= mem_budget), None)
                if pick is None:
                    if running:
                        break  # wait for memory to free up
                    pick = 0   # a job larger than the budget still runs, alone
                job = queue.pop(pick)
                log = open(str(LOG_DIR / (job["id"] + ".log")), "a", encoding="utf-8")
                log.write(f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S')} attempt {job['attempts'] + 1}: "
                          f"{' '.join(job_command(job)[1:])}\n")
                log.flush()
                proc = subprocess.Popen(job_command(job), cwd=str(ROOT), stdout=log, stderr=subprocess.STDOUT, env=env)
                job["status"], job["attempts"] = "running", job["attempts"] + 1
                running.append((job, proc, log, time.time()))
                used += job["mem_gb"]
                save_state(plan, state)
            if not accepting and queue and not running:
                break
            time.sleep(1.0)
            still: List[Tuple[Dict[str, Any], subprocess.Popen, Any, float]] = []
            for job, proc, log, t0 in running:
                code = proc.poll()
                if code is None and (time.time() - t0) > args.timeout_hours * 3600:
                    proc.kill()
                    proc.wait()
                    code = -9
                    log.write(f"\n=== killed after {args.timeout_hours} hours\n")
                if code is None:
                    still.append((job, proc, log, t0))
                    continue
                log.close()
                job["returncode"] = code
                job["seconds"] = round(time.time() - t0, 1)
                job["status"] = "done" if code == 0 else "failed"
                finished += code == 0
                failed += code != 0
                # jobs not finished yet: the queue plus every running job whose process is still alive
                left = len(queue) + sum(1 for other, other_proc, _, _ in running if other is not job and other_proc.poll() is None)
                print(f"[{time.strftime('%H:%M:%S')}] {'ok    ' if code == 0 else 'FAILED'} {job['id']} "
                      f"({job['seconds']:.0f} s, exit {code}); {left} left in this run", flush=True)
                save_state(plan, state)
            running = still
    except KeyboardInterrupt:
        print("\ninterrupted: stopping the running jobs (their finished splits are already cached)")
        for job, proc, log, _ in running:
            proc.kill()
            proc.wait()
            log.close()
            job["status"] = "pending"
        save_state(plan, state)
        return 130
    not_started = len(queue)
    print(f"finished: {finished} ok, {failed} failed" + (f", {not_started} not started (time limit)" if not_started else ""))
    if failed:
        print("failed jobs keep their logs; run again with --retry-failed")
    return 1 if failed else 0


def summarize(plan: str, only: Optional[str]) -> int:
    state = load_state(plan)
    datasets = sorted({j["dataset"] for j in state["jobs"]})
    if only:
        datasets = [d for d in datasets if d in only.split(",")]
    methods = ",".join(STAGES["core"])
    for dataset in datasets:
        print(f"--- {dataset}", flush=True)
        subprocess.call([sys.executable, str(EXPERIMENTS / "compare_baselines.py"), "--config-set", plan,
                         "--only", dataset, "--methods", methods, "--max-seconds", "0"], cwd=str(ROOT))
    print("--- paired comparison over the grid", flush=True)
    return subprocess.call([sys.executable, str(EXPERIMENTS / "pairwise_partial.py"), "--config-set", plan], cwd=str(ROOT))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=("hardware", "plan", "status", "run", "summarize"))
    ap.add_argument("--plan", default="grid", choices=PLANS,
                    help="the job list: grid (observed-purity rule on the (tau, m) grid, default), "
                         "grid_certified (certified selection at the grid's purity floors), large_real and "
                         "large_real_certified (the same two selections on the large recorded tables fars and shuttle)")
    ap.add_argument("--stages", default="core", help="plan: comma-separated stages to list (default: core)")
    ap.add_argument("--force", action="store_true", help="plan: forget the state of an existing plan")
    ap.add_argument("--stage", default=None, help="run: comma-separated stages (default: core)")
    ap.add_argument("--size", default=None, help="run: comma-separated size classes: small, medium, heavy, huge")
    ap.add_argument("--only", default=None, help="run, summarize: comma-separated dataset names")
    ap.add_argument("--workers", type=int, default=None, help="run: parallel jobs (default: from the hardware)")
    ap.add_argument("--hours", type=float, default=None, help="run: start no new job after this many hours")
    ap.add_argument("--max-jobs", type=int, default=None, help="run: at most this many jobs")
    ap.add_argument("--timeout-hours", type=float, default=12.0, help="run: kill a job after this many hours")
    ap.add_argument("--retry-failed", action="store_true", help="run: also run jobs that failed before")
    ap.add_argument("--dry-run", action="store_true", help="run: print the commands, start nothing")
    args = ap.parse_args()
    if args.command == "hardware":
        return hardware_report()
    if args.command == "plan":
        stages = [s for s in args.stages.split(",") if s]
        unknown = set(stages) - set(STAGES)
        if unknown:
            raise SystemExit(f"unknown stage(s) {sorted(unknown)}; known: {list(STAGES)}")
        return build_plan(args.plan, stages, args.force)
    if args.command == "status":
        return print_status(args.plan)
    if args.command == "run":
        return run_jobs(args.plan, args)
    return summarize(args.plan, args.only)


if __name__ == "__main__":
    raise SystemExit(main())
