#!/usr/bin/env python3
"""
uprof.py — AMD uProfPcm integration for the benchmark framework.

Usage:
    python3 caribou.py --apps hpl --profiling
    python3 caribou.py --apps hpl --profiling --metrics ipc fp
    python3 caribou.py --apps hpl --profiling --metrics ipc l1 --profile-duration 30

Binary:
    /home/reddy/AMDuProf_Linux_x64_5.2.606/bin/AMDuProfPcm

Supported Core Metrics (from --help):
    ipc  →  Instructions Per Cycle
    fp   →  Floating Point utilization
    l1   →  L1 cache metrics
    l2   →  L2 cache metrics
    tlb  →  TLB metrics

Command per metric (system-wide, profiles running HPL process):
    AMDuProfPcm -m <metric> --html -a -d <duration> -O <outdir/>

NOTE: Uses -O <dir> (not -o <file>) so uProf auto-names CSV and HTML output.
      No '-- binary' is passed. uProf profiles system-wide (-a) for -d seconds
      while HPL is already running in the background.

Output per metric:
    profiles/uprof/HPL/<timestamp>/ipc/   ← HTML + CSV auto-generated here
"""

import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

UPROF_PCM        = Path("/home/reddy/AMDuProf_Linux_x64_5.2.606/bin/AMDuProfPcm")
PROFILES_ROOT    = Path("profiles/uprof")
ALL_METRICS      = ["ipc", "fp", "l1", "l2", "tlb"]
DEFAULT_METRICS  = ["ipc", "l2"]
PROFILE_DURATION = 10   # seconds

# -m flag per metric (as supported by this uProf version)
METRIC_FLAGS = {
    "ipc": "ipc",
    "pipeline_util": "pipeline_util", # pipeline utilization topdown
    "avx_imix": "avx_imix",
    "fp":  "fp",
    "l1":  "l1",
    "l2":  "l2",
    "tlb": "tlb",
}


# ---------------------------------------------------------------------------
# Prerequisite check
# ---------------------------------------------------------------------------

def check_uprof():
    """Check AMDuProfPcm exists. Called once at startup when --profiling is set."""
    print("\n[PREPROCESS — AMD uProfPcm]")
    if not UPROF_PCM.exists():
        print(f"  ✗  AMDuProfPcm not found at {UPROF_PCM}")
        sys.exit(1)
    if not os.access(UPROF_PCM, os.X_OK):
        print(f"  ✗  AMDuProfPcm not executable: {UPROF_PCM}")
        sys.exit(1)
    # Check perf_event_paranoid
    paranoid_path = Path("/proc/sys/kernel/perf_event_paranoid")
    if paranoid_path.exists():
        val = int(paranoid_path.read_text().strip())
        if val > 0:
            print(f"  ✗  perf_event_paranoid = {val}  (must be 0)")
            print(f"     Fix: echo 0 | sudo tee /proc/sys/kernel/perf_event_paranoid")
            sys.exit(1)
        else:
            print(f"  ✔  perf_event_paranoid = {val}  (OK)")
    print(f"  ✔  AMDuProfPcm found at {UPROF_PCM}\n")


# ---------------------------------------------------------------------------
# Core profiling function
# ---------------------------------------------------------------------------

def run_uprof(
    benchmark: str,
    metrics:   list = None,
    duration:  int  = PROFILE_DURATION,
):
    """
    Profile system-wide with AMDuProfPcm while a benchmark is already running.

    Runs one AMDuProfPcm invocation per metric:
        AMDuProfPcm -m <flag> --html -a -d <duration> -O <outdir/>

    Uses -O <dir> so uProf auto-names CSV + HTML inside that directory.
    No binary is launched — profiles system-wide (-a) for exactly <duration>s.

    Args:
        benchmark : name shown in output and used in output path, e.g. "HPL"
        metrics   : subset of ALL_METRICS — default: ["ipc", "fp", "l1"]
        duration  : profiling duration in seconds per metric (default 10)
    """
    metrics   = metrics or DEFAULT_METRICS
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print(f"\n{'='*60}")
    print(f"  AMD uProfPcm — {benchmark.upper()}")
    print(f"  Mode     : system-wide (-a), no binary launch")
    print(f"  Metrics  : {', '.join(metrics)}")
    print(f"  Duration : {duration}s per metric")
    print(f"{'='*60}")

    html_reports = {}   # metric → html path or None

    for metric in metrics:
        m_flag = METRIC_FLAGS.get(metric)
        if not m_flag:
            print(f"  ✗  Unknown metric '{metric}' — skipping.")
            continue

        # Output dir for this metric — uProf writes CSV + HTML here
        out_dir = PROFILES_ROOT / benchmark.upper() / timestamp / metric
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n  [{metric.upper()}]  -m {m_flag}  (profiling for {duration}s ...)")

        # Use -O <dir> so uProf auto-names its output files
        cmd = [
            str(UPROF_PCM),
            "-m",  m_flag,
            "--html",
            "-a",                   # system-wide — captures running HPL
            "-d", str(duration),
            "-O", str(out_dir),     # directory, not file — uProf names files
        ]

        print(f"  ▶  CMD: {' '.join(cmd)}")

        r = subprocess.run(cmd, capture_output=True, text=True)

        # Show uProf stdout
        if r.stdout.strip():
            for line in r.stdout.strip().splitlines():
                print(f"     {line}")

        if r.returncode != 0:
            print(f"  ✗  Failed (exit {r.returncode})")
            if r.stderr.strip():
                for line in r.stderr.strip().splitlines():
                    print(f"     {line}")
            html_reports[metric] = None
            continue

        # Find HTML report written by uProf inside out_dir
        found_html = list(out_dir.rglob("*.html"))
        if found_html:
            html_reports[metric] = found_html[0]
            print(f"  ✔  HTML report → {found_html[0]}")
        else:
            html_reports[metric] = None
            print(f"  ✗  HTML not found in {out_dir}")

    # ── Summary ────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  AMD uProfPcm Summary — {benchmark.upper()}")
    print(f"{'='*60}")
    all_ok = True
    for metric in metrics:
        html = html_reports.get(metric)
        if html:
            win_path = _to_windows_path(html)
            print(f"  ✔  {metric.upper():5s} → {html}")
            if win_path:
                print(f"         Open in browser : {win_path}")
        else:
            print(f"  ✗  {metric.upper():5s} → no HTML report generated")
            all_ok = False

    if all_ok:
        print(f"\n  Paste the Windows paths above into Chrome/Edge to view reports.")
    print()

    return html_reports


# ---------------------------------------------------------------------------
# WSL path helper
# ---------------------------------------------------------------------------

def _to_windows_path(path: Path) -> str:
    """Convert a WSL Linux path to a Windows path for opening in a browser."""
    try:
        r = subprocess.run(["wslpath", "-w", str(path)],
                           capture_output=True, text=True)
        if r.returncode == 0:
            return r.stdout.strip()
    except FileNotFoundError:
        pass
    return ""


# ---------------------------------------------------------------------------
# Argparse helpers — called from caribou.py
# ---------------------------------------------------------------------------

def add_uprof_args(parser: argparse.ArgumentParser):
    """
    Add --profiling, --metrics, --profile-duration to your ArgumentParser.
    """
    parser.add_argument(
        "--profiling",
        action="store_true",
        default=False,
        help="Enable AMD uProfPcm HTML profiling during benchmark run."
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        choices=ALL_METRICS,
        default=DEFAULT_METRICS,
        metavar="METRIC",
        help=(
            f"uProf metrics to collect. Choices: {ALL_METRICS}. "
            f"Default: {DEFAULT_METRICS}. Example: --metrics ipc fp l1"
        )
    )
    parser.add_argument(
        "--profile-duration",
        type=int,
        default=PROFILE_DURATION,
        metavar="SECONDS",
        help=f"Profiling duration per metric in seconds (default: {PROFILE_DURATION})."
    )


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="uprof.py self-test")
    add_uprof_args(parser)
    args = parser.parse_args()

    if args.profiling:
        check_uprof()
        run_uprof(
            benchmark = "TEST",
            metrics   = args.metrics,
            duration  = args.profile_duration,
        )
    else:
        print("Pass --profiling to run the self-test.")
