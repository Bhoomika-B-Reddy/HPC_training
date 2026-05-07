#!/usr/bin/env python3
"""
hpcg.py — Mini-HPCG benchmark for Mini Caribou.

Public interface (matches hpl.py / stream.py pattern):
    build(build_dir, cores, compiler_info, compiler_name, out_log, err_log, workspace)
        → (job_id, binary_path)

    run(build_dir, run_dir, np, omp, build_jid, binary, compiler_info, info,
        out_log, err_log, workspace)
        → (job_id, out_log_path)
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from slurm import write_script, sbatch_submit


# ─────────────────────────────────────────────────────────────
# EMBEDDED C SOURCE  (self-contained mini-HPCG)
# ─────────────────────────────────────────────────────────────

HPCG_C_SOURCE = r"""
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <mpi.h>

#define NX 64
#define NY 64
#define NZ 64
#define N  (NX*NY*NZ)
#define MAX_ITER 50
#define TOL 1e-8

static double r[N], p[N], q[N], z[N], x[N], diag[N];

static double mysecond() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static void spmv(const double *v, double *out) {
    int ix, iy, iz, idx;
    for (iz=0;iz<NZ;iz++) for (iy=0;iy<NY;iy++) for (ix=0;ix<NX;ix++) {
        idx = iz*NY*NX + iy*NX + ix;
        double s = 26.0 * v[idx];
        if (ix>0)    s -= v[idx-1];
        if (ix<NX-1) s -= v[idx+1];
        if (iy>0)    s -= v[idx-NX];
        if (iy<NY-1) s -= v[idx+NX];
        if (iz>0)    s -= v[idx-NY*NX];
        if (iz<NZ-1) s -= v[idx+NY*NX];
        out[idx] = s;
    }
}

static double dot(const double *a, const double *b) {
    double s = 0.0;
    for (int i=0;i<N;i++) s += a[i]*b[i];
    return s;
}

int main(int argc, char **argv) {
    int rank, size, i, k;
    double t0, t1, gflops, rz, rz_new, alpha, beta, norm;
    long flops;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
        printf("================================================================================\n");
        printf("Mini-HPCG Benchmark  |  Grid: %dx%dx%d  N=%d  MPI ranks=%d\n",NX,NY,NZ,N,size);
        printf("================================================================================\n");

        for (i=0;i<N;i++) { r[i]=1.0; x[i]=0.0; diag[i]=1.0/26.0; }
        for (i=0;i<N;i++) z[i]=diag[i]*r[i];
        for (i=0;i<N;i++) p[i]=z[i];
        rz = dot(r,z);

        t0 = mysecond();
        for (k=0;k<MAX_ITER;k++) {
            spmv(p,q);
            double pq = dot(p,q);
            alpha = rz/pq;
            for (i=0;i<N;i++) x[i] += alpha*p[i];
            for (i=0;i<N;i++) r[i] -= alpha*q[i];
            norm = sqrt(dot(r,r));
            if (norm < TOL) { k++; break; }
            for (i=0;i<N;i++) z[i]=diag[i]*r[i];
            rz_new = dot(r,z);
            beta = rz_new/rz;
            for (i=0;i<N;i++) p[i]=z[i]+beta*p[i];
            rz = rz_new;
        }
        t1 = mysecond();

        flops = (long)k * (2L*7*N + 15L*N);
        gflops = (double)flops / (t1-t0) / 1e9;
        printf("Iterations  : %d\n", k);
        printf("Residual    : %.6e\n", norm);
        printf("Time        : %.4f seconds\n", t1-t0);
        printf("Performance : %.4f GFlops\n", gflops);
        printf("================================================================================\n");
        printf("HPCG_result : iters=%d  residual=%.4e  GFlops=%.4f\n", k, norm, gflops);
        printf("================================================================================\n");
    }
    MPI_Finalize();
    return 0;
}
"""


# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────

def _banner(tag):
    return (
        f'echo "======================================================"\n'
        f'echo "{tag}"\n'
        f'echo "Host : $(hostname)"\n'
        f'echo "Date : $(date)"\n'
        f'echo "======================================================"'
    )


def _write_source(build_dir):
    src = os.path.join(build_dir, "hpcg.c")
    if os.path.exists(src):
        print(f"  ✔  Source exists  : {src}")
    else:
        with open(src, "w") as f:
            f.write(HPCG_C_SOURCE)
        print(f"  ✔  Source written : {src}")
    return src


def _get_mpicc(compiler_name):
    if compiler_name == "icc":
        return "/opt/intel/oneapi/mpi/2021.17/bin/mpicc"
    return "/usr/bin/mpicc"


def _get_mpirun(compiler_name):
    if compiler_name == "icc":
        return "/opt/intel/oneapi/mpi/2021.17/bin/mpirun"
    return "/usr/bin/mpirun"


# ─────────────────────────────────────────────────────────────
# BUILD  (public entry point)
# ─────────────────────────────────────────────────────────────

def build(build_dir, cores, compiler_info, compiler_name,
          out_log=None, err_log=None, workspace=None):
    """
    Compile mini-HPCG via SLURM sbatch.
    Returns (job_id, binary_path).
    Returns ("skip", binary_path) if binary already exists.
    """
    print("\n[BUILD PHASE — HPCG]")

    os.makedirs(build_dir, exist_ok=True)
    binary = os.path.join(build_dir, "xhpcg")

    if os.path.exists(binary):
        print(f"  ✔  Binary exists, skipping build : {binary}")
        return "skip", binary

    src    = _write_source(build_dir)
    mpicc  = _get_mpicc(compiler_name)
    log    = out_log or os.path.join(build_dir, "hpcg_build.out")
    err    = err_log or os.path.join(build_dir, "hpcg_build.err")
    script = os.path.join(build_dir, f"build_hpcg_{compiler_name}.sh")

    # Compiler flags — use same optimisation level as the rest of the framework
    cc_flags = compiler_info.get("flags", "-O3 -march=native")
    # Strip OpenMP flag — mpicc handles threading
    cc_flags = cc_flags.replace("-fopenmp", "").replace("-qopenmp", "").strip()

    lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name=build_hpcg_{compiler_name}",
        "#SBATCH --nodes=1",
        "#SBATCH --ntasks=1",
        "#SBATCH --time=00:10:00",
        f"#SBATCH --output={log}",
        f"#SBATCH --error={err}",
        "",
        "set -euo pipefail",
        "",
        _banner(f"BUILD START — mini-HPCG [{compiler_name.upper()}]"),
        "",
        f"cd {build_dir}",
        f"echo 'Compiling hpcg.c with {mpicc} ...'",
        f"{mpicc} {cc_flags} -o {binary} {src} -lm",
        "",
        f"if [ -f {binary} ]; then",
        f"  echo 'Binary OK : {binary}'",
        "else",
        "  echo 'ERROR: compile failed' >&2",
        "  exit 1",
        "fi",
        "",
        _banner("BUILD DONE"),
    ]

    write_script(script, "\n".join(lines) + "\n")
    print(f"  ✔  Script written  : {script}")
    print(f"  [SLURM] out → {log}")
    print(f"  [SLURM] err → {err}")
    print(f"\n  [SLURM] Submitting BUILD (hpcg/{compiler_name}) ...")

    return sbatch_submit(script), binary


# ─────────────────────────────────────────────────────────────
# RUN  (public entry point)
# ─────────────────────────────────────────────────────────────

def run(build_dir, run_dir, np, omp, build_jid, binary,
        compiler_info, info,
        out_log=None, err_log=None, workspace=None):
    """
    Run mini-HPCG via SLURM sbatch.
    Returns (job_id, out_log_path).
    """
    print("\n[RUN PHASE — HPCG]")

    compiler_name = info.get("compiler_name", "gcc")
    mpirun        = _get_mpirun(compiler_name)

    os.makedirs(run_dir, exist_ok=True)

    slurm_out = out_log or os.path.join(run_dir, "hpcg.out")
    slurm_err = err_log or os.path.join(run_dir, "hpcg.err")
    script    = os.path.join(run_dir, "run_hpcg.sh")

    if compiler_name == "icc":
        mpi_flags = f"-np {np} -bind-to core -map-by socket"
    else:
        mpi_flags = f"-np {np} --bind-to none --map-by slot"

    lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name=run_hpcg",
        "#SBATCH --nodes=1",
        f"#SBATCH --ntasks={np}",
        f"#SBATCH --cpus-per-task={omp}",
        "#SBATCH --time=01:00:00",
        f"#SBATCH --output={slurm_out}",
        f"#SBATCH --error={slurm_err}",
        "",
        "set -euo pipefail",
        "",
        _banner("RUN START — mini-HPCG"),
        "",
        f"export OMP_NUM_THREADS={omp}",
        f"export OPENBLAS_NUM_THREADS={omp}",
        "",
        f"cd {run_dir}",
        f"{mpirun} {mpi_flags} {binary}",
        "",
        _banner("RUN DONE"),
    ]

    write_script(script, "\n".join(lines) + "\n")
    print(f"  Script written : {script}")
    print(f"  [SLURM] out → {slurm_out}")

    if build_jid and build_jid not in ("skip", "bash", None):
        jid = sbatch_submit(script, depend_id=build_jid)
    else:
        jid = sbatch_submit(script)

    return jid, slurm_out
