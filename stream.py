#!/usr/bin/env python3
import os
import sys
import math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from slurm import make_sbatch, write_script, sbatch_submit

STREAM_URL  = "https://www.cs.virginia.edu/stream/FTP/Code/stream.c"
STREAM_VER  = "stream_5.10"


def calc_stream_array(mem_mb):
    if mem_mb >= 16384: return 80000000
    elif mem_mb >= 8192: return 40000000
    elif mem_mb >= 4096: return 20000000
    else:                return 10000000


def _make_sbatch_with_err(job_name, output_log, error_log,
                           ntasks, time_limit, commands):
    header = f"""\
#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --nodes=1
#SBATCH --ntasks={ntasks}
#SBATCH --cpus-per-task=1
#SBATCH --time={time_limit}
#SBATCH --output={output_log}
#SBATCH --error={error_log}
"""
    banner_start = f"""\
echo "============================================"
echo "Job  : {job_name}"
echo "Host : $(hostname)"
echo "Date : $(date)"
echo "============================================"
"""
    banner_end = """\
echo "============================================"
echo "Done : $(date)"
echo "============================================"
"""
    body = "\n".join(commands)
    return header + "\n" + banner_start + "\n" + body + "\n\n" + banner_end


def build(build_dir, cores, compiler_info, info,
          out_log=None, err_log=None, workspace=None):
    print("\n[BUILD PHASE — STREAM]")
    binary = os.path.join(build_dir, "stream")

    if os.path.exists(binary):
        print(f"  ✔  Binary exists, skipping build : {binary}")
        return "skip"

    os.makedirs(build_dir, exist_ok=True)

    log        = out_log or os.path.join(build_dir, "build_stream.log")
    sbatch_err = err_log or os.path.join(build_dir, "build_stream.err")
    script     = os.path.join(build_dir, "build_stream.sh")

    cc    = compiler_info["cc"]
    flags = compiler_info["flags"]
    link  = compiler_info.get("link", "-lm")
    arr_n = calc_stream_array(info.get("mem_mb", 4096))
    src   = os.path.join(build_dir, "stream.c")

    commands = [
        f"echo '[STREAM BUILD]'",
        f"mkdir -p {build_dir}",
        f"cd {build_dir}",
        "",
        f"echo '--- Download: fetching real STREAM source ---'",
        f"if [ ! -f {src} ]; then",
        f"  wget -q {STREAM_URL} -O {src} || {{",
        f"    echo 'wget failed, using mirror ...'",
        f"    curl -sL {STREAM_URL} -o {src}",
        f"  }}",
        f"fi",
        f"ls -lh {src}",
        "",
        f"echo '--- Configure: checking {cc} ---'",
        f"which {cc} && {cc} --version | head -1",
        "",
        f"echo '--- Make: compiling STREAM with {cc} ---'",
        f"echo 'Array size = {arr_n} elements'",
        f"echo 'Link flags = {link}'",
        f"{cc} {flags} -DSTREAM_ARRAY_SIZE={arr_n} {src} -o {binary} {link}",
        "",
        f"echo '--- Verify binary ---'",
        f"if [ -f {binary} ]; then",
        f"  echo 'Binary OK : {binary}'",
        f"  ls -lh {binary}",
        f"  file {binary}",
        f"else",
        f"  echo 'ERROR: compile failed!' && exit 1",
        f"fi",
    ]

    write_script(script, _make_sbatch_with_err(
        "build_stream", log, sbatch_err, 1, "00:10:00", commands))

    print(f"  Array size : {arr_n} elements  ({arr_n*8//1024//1024} MB per array)")
    print(f"  Link flags : {link}")
    print(f"\n  [SLURM] Submitting BUILD (stream) ...")
    return sbatch_submit(script)


def run(build_dir, run_dir, cores, build_jid, compiler_info,
        out_log=None, err_log=None, workspace=None):
    print("\n[RUN PHASE — STREAM]")
    binary = os.path.join(build_dir, "stream")
    os.makedirs(run_dir, exist_ok=True)

    slurm_out  = out_log or os.path.join(run_dir, "stream.out")
    slurm_err  = err_log or os.path.join(run_dir, "stream.err")
    tee_log    = slurm_out
    script     = os.path.join(run_dir, "run_stream.sh")

    commands = [
        f"echo '[STREAM RUN]'",
        f"mkdir -p {run_dir}",
        f"cd {run_dir}",
        f"export OMP_NUM_THREADS={cores}",
        f"export OMP_PROC_BIND=close",
        f"export OMP_PLACES=cores",
        f"echo 'OMP_NUM_THREADS={cores}'",
        f"echo 'Compiler used : {compiler_info['cc']}'",
        f"echo 'Running real STREAM benchmark ...'",
        f"echo '--- Thread verification ---'",
        f"echo 'OMP_NUM_THREADS='$OMP_NUM_THREADS",
        f"echo 'OMP_PROC_BIND='$OMP_PROC_BIND",
        f"echo 'OMP_PLACES='$OMP_PLACES",
        f"{binary}",
        f"echo ''",
        f"echo 'Results saved to : {tee_log}'",
    ]

    write_script(script, _make_sbatch_with_err(
        "run_stream", slurm_out, slurm_err, cores, "00:10:00", commands))

    print(f"  Threads    : {cores}")
    print(f"  [SLURM] out → {slurm_out}")
    print(f"  [SLURM] err → {slurm_err}")
    print(f"\n  [SLURM] Submitting RUN (stream) ...")

    if build_jid == "skip":
        print(f"  ✔  Binary already existed — no build dependency")
        jid = sbatch_submit(script)
    else:
        jid = sbatch_submit(script, depend_id=build_jid)

    return jid, slurm_out
