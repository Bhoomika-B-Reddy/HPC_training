#!/usr/bin/env python3
import os
import sys
import math
import subprocess
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from slurm import write_script, sbatch_submit




# ─────────────────────────────────────────────────────────────
# MPI CONFIG
# ─────────────────────────────────────────────────────────────

def get_mpi_config(compiler_name):
    if compiler_name == "icc":
        mpdir  = "/opt/intel/oneapi/mpi/2021.17"
        mpinc  = f"-I{mpdir}/include"
        mplib  = f"-L{mpdir}/lib -Xlinker --enable-new-dtags -Xlinker -rpath -Xlinker {mpdir}/lib -lmpifort -lmpi -ldl -lrt -lpthread"
        mpirun = f"{mpdir}/bin/mpirun"
    else:
        mpdir  = "/usr/lib/x86_64-linux-gnu/openmpi"
        mpinc  = f"-I{mpdir}/include"
        mplib  = f"-L{mpdir}/lib -lmpi"
        mpirun = "/usr/bin/mpirun"
    return mpinc, mplib, mpirun


# ─────────────────────────────────────────────────────────────
# AUTO-TUNE  —  N, NB, P×Q from available RAM + np (MPI ranks)
# ─────────────────────────────────────────────────────────────

def calc_hpl_params(avail_mb, np):
    """np controls P×Q and matrix size — independent of OMP threads."""
    usable = 0.80 * avail_mb * 1024 * 1024
    n_raw  = int(math.sqrt(usable / 8.0))

    if   avail_mb >= 32768: nb = 256
    elif avail_mb >= 2048:  nb = 192
    else:                   nb = 128

    N = (n_raw // nb) * nb

    if   avail_mb <  512:  N = min(N, 2048)
    elif avail_mb <  1024: N = min(N, 4000)
    elif avail_mb <  2048: N = min(N, 8000)
    elif avail_mb <  4096: N = min(N, 16000)

    best_p, best_q, best_diff = 1, np, np
    for p in range(1, int(math.sqrt(np)) + 1):
        if np % p == 0:
            q = np // p
            if q >= p and (q - p) < best_diff:
                best_diff = q - p
                best_p, best_q = p, q

    return N, nb, best_p, best_q



# ─────────────────────────────────────────────────────────────
# MAKEFILE
# ─────────────────────────────────────────────────────────────

def make_hpl_makefile(build_dir, compiler_info, compiler_name):
    cc  = compiler_info["cc"]
    mpi = compiler_info["mpi"]
    mpinc, mplib, _ = get_mpi_config(compiler_name)

    if compiler_name == "icc":
        cc_flags = "-O3 -march=native -fp-model fast=2"
        omp_flag = "-qopenmp"
    else:
        cc_flags = "-O3 -march=native -funroll-loops -ffast-math"
        omp_flag = "-fopenmp"

    arch_name = f"Linux_HPL_{compiler_name}"
    archiver  = get_archiver(compiler_name)

    makefile = f"""
SHELL        = /bin/sh
CD           = cd
CP           = cp
LN_S         = ln -fs
MKDIR        = mkdir -p
RM           = /bin/rm -f
TOUCH        = touch

ARCH         = {arch_name}

TOPdir       = {build_dir}/{HPL_DIR}
INCdir       = $(TOPdir)/include
BINdir       = $(TOPdir)/bin/$(ARCH)
LIBdir       = $(TOPdir)/lib/$(ARCH)
HPLlib       = $(LIBdir)/libhpl.a

MPinc        = {mpinc}
MPlib        = {mplib}

LAdir        = /usr/lib/x86_64-linux-gnu
LAinc        = -I/usr/include
LAlib        = -lopenblas -llapacke

F2CDEFS      =

HPL_INCLUDES = -I$(INCdir) -I$(INCdir)/$(ARCH) $(LAinc) $(MPinc)
HPL_LIBS     = -Wl,--start-group $(HPLlib) $(LAlib) $(MPlib) -Wl,--end-group -lm

HPL_OPTS     = -DHPL_CALL_CBLAS
HPL_DEFS     = $(F2CDEFS) $(HPL_OPTS) $(HPL_INCLUDES)

CC           = {cc}
CCNOOPT      = $(HPL_DEFS)
CCFLAGS      = $(HPL_DEFS) {cc_flags} {omp_flag}

LINKER       = {mpi}
LINKFLAGS    = {cc_flags} {omp_flag}

ARCHIVER     = {archiver}
ARFLAGS      = r
RANLIB       = ranlib

"""
    path = os.path.join(build_dir, f"Make.{arch_name}")
    with open(path, "w") as f:
        f.write(makefile)
    return path


# ─────────────────────────────────────────────────────────────
# HPL.dat
# ─────────────────────────────────────────────────────────────

def make_hpl_dat(run_dir, N, nb, P, Q):
    lines = [
        "HPLinpack benchmark input file",
        "Innovative Computing Laboratory, University of Tennessee",
        "HPL.out                output file name (if any)",
        "6                      device out (6=stdout,7=stderr,file)",
        "1                      # of problems sizes (Ns)",
        f"{N}                   Ns",
        "1                      # of NBs",
        f"{nb}                  NBs",
        "0                      PMAP process mapping (0=Row-,1=Col-major)",
        "1                      # of process grids (P x Q)",
        f"{P}                   Ps",
        f"{Q}                   Qs",
        "16.0                   threshold",
        "1                      # of panel fact",
        "2                      PFACTs (0=left, 1=Crout, 2=Right)",
        "1                      # of recursive stopping criterium",
        "4                      NBMINs (>= 1)",
        "1                      # of panels in recursion",
        "2                      NDIVs",
        "1                      # of recursive panel fact.",
        "1                      RFACTs (0=left, 1=Crout, 2=Right)",
        "1                      # of broadcast",
        "1                      BCASTs (0=1rg,1=1rM,2=2rg,3=2rM,4=Lng,5=LnM)",
        "1                      # of lookahead depth",
        "1                      DEPTHs (>=0)",
        "2                      SWAP (0=bin-exch,1=long,2=mix)",
        f"{nb}                  swapping threshold",
        "0                      L1 in (0=transposed,1=no-transposed) form",
        "0                      U  in (0=transposed,1=no-transposed) form",
        "1                      Equilibration (0=no,1=yes)",
        "8                      memory alignment in double (> 0)",
    ]
    content = "\n".join(lines) + "\n"
    path = os.path.join(run_dir, "HPL.dat")
    with open(path, "w") as f:
        f.write(content)
    return path


# ─────────────────────────────────────────────────────────────
# SBATCH HEADER BUILDER
# ─────────────────────────────────────────────────────────────

def _sbatch_header(
    job_name, out_log, err_log, ntasks,
    time_limit, mem_mb=None, partition=None,
    cpus_per_task=1, exclusive=False,
):
    lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name={job_name}",
        f"#SBATCH --nodes=1",
        f"#SBATCH --ntasks={ntasks}",
        f"#SBATCH --cpus-per-task={cpus_per_task}",
        f"#SBATCH --time={time_limit}",
        f"#SBATCH --output={out_log}",
        f"#SBATCH --error={err_log}",
    ]
    if mem_mb is not None:
        lines.append(f"#SBATCH --mem={mem_mb}M")
    if partition:
        lines.append(f"#SBATCH --partition={partition}")
    if exclusive:
        lines.append("#SBATCH --exclusive")
    return "\n".join(lines)


def _banner(tag):
    return (
        f'echo "======================================================"\n'
        f'echo "{tag}"\n'
        f'echo "Host : $(hostname)"\n'
        f'echo "Date : $(date)"\n'
        f'echo "======================================================"'
    )


# ─────────────────────────────────────────────────────────────
# BUILD SBATCH SCRIPT
# ─────────────────────────────────────────────────────────────

def _make_build_sbatch(
    job_name, out_log, err_log,
    tar, srcdir, makefile_path, arch_name, binary,
    compiler_name,
):
    env_setup = []
    if compiler_name == "icc":
        env_setup = [
            "# Source Intel oneAPI environment",
            "export OCL_ICD_FILENAMES=''",
            "source /opt/intel/oneapi/setvars.sh --force || true",
            "",
        ]

    mod_dir = os.path.join(os.path.dirname(os.path.dirname(srcdir)), "modules")

    patch_makefile_cmd = []
    if compiler_name != "icc":
        patch_makefile_cmd = [
            "# ── patch archiver in copied Makefile (ensure no xiar) ──────",
            f"sed -i 's/^ARCHIVER.*=.*xiar/ARCHIVER     = llvm-ar/' {srcdir}/Make.{arch_name}",
            "",
        ]

    lines = [
        _sbatch_header(
            job_name, out_log, err_log,
            ntasks=1, time_limit="00:30:00",
            mem_mb=None, exclusive=False,
        ),
        "",
        "set -euo pipefail",
        "",
        _banner(f"BUILD START — HPL {HPL_VERSION} [{compiler_name}]"),
        "",
        *env_setup,
        "# ── lmod module info ─────────────────────────────────",
        f'echo "  [LMOD] module use {mod_dir}"',
        f'echo "  [LMOD] module load hpl/{compiler_name}/1.0"',
        "",
        "# ── download ─────────────────────────────────────────",
        f"if [ ! -f {tar} ]; then",
        f"  echo 'Downloading HPL {HPL_VERSION} ...'",
        f"  wget -q --show-progress {HPL_URL} -O {tar} \\",
        f"    || curl -L --progress-bar {HPL_URL} -o {tar}",
        "fi",
        "",
        "# ── extract ──────────────────────────────────────────",
        f"if [ ! -d {srcdir} ]; then",
        f"  echo 'Extracting ...'",
        f"  tar -xf {tar} -C $(dirname {srcdir})",
        "fi",
        "",
        "# ── configure ────────────────────────────────────────",
        f"cp -f {makefile_path} {srcdir}/Make.{arch_name}",
        "",
        *patch_makefile_cmd,
        "# ── build ────────────────────────────────────────────",
        f"cd {srcdir}",
        f"echo 'Building library ...'",
        f"make -f Make.top build_src arch={arch_name} 2>&1",
        f"echo 'Building binary ...'",
        f"make -f Make.top build_tst arch={arch_name} 2>&1",
        "",
        "# ── verify ───────────────────────────────────────────",
        f"if [ -f {binary} ]; then",
        f"  echo '✔  Binary OK : {binary}'",
        "else",
        "  echo '✘  ERROR: xhpl binary not found — check build log' >&2",
        "  exit 1",
        "fi",
        "",
        _banner("BUILD DONE"),
    ]
    return "\n".join(lines) + "\n"


# ─────────────────────────────────────────────────────────────
# RUN SBATCH SCRIPT  (normal SLURM path)
# ─────────────────────────────────────────────────────────────

def _make_run_sbatch(
    job_name, out_log, err_log,
    run_dir, binary, np, omp, mpirun,
    compiler_name, avail_mb,
    N, nb, P, Q,
):
    env_setup = []
    if compiler_name == "icc":
        env_setup = [
            "# Source Intel oneAPI / MKL environment",
            "export OCL_ICD_FILENAMES=''",
            "source /opt/intel/oneapi/setvars.sh --force || true",
            "",
        ]

    if compiler_name == "icc":
        mpi_flags = f"-np {np} -bind-to core -map-by socket"
    else:
        mpi_flags = f"-np {np} --bind-to none --map-by slot"

    lines = [
        _sbatch_header(
            job_name, out_log, err_log,
            ntasks=np, cpus_per_task=omp,
            time_limit="02:00:00",
        ),
        "",
        "set -euo pipefail",
        "",
        _banner(f"RUN START — HPL {HPL_VERSION} [{compiler_name}]"),
        "",
        *env_setup,
        "# ── thread environment ───────────────────────────────",
        f"export OMP_NUM_THREADS={omp}",
        f"export OPENBLAS_NUM_THREADS={omp}",
        f"export MKL_NUM_THREADS={omp}",
        f"export GOTO_NUM_THREADS={omp}",
        "",
        "# ── run ──────────────────────────────────────────────",
        f"cd {run_dir}",
        f"{mpirun} {mpi_flags} {binary}",
        "",
        _banner("RUN DONE"),
    ]
    return "\n".join(lines) + "\n"


# ─────────────────────────────────────────────────────────────
# BUILD  (public entry point)
# ─────────────────────────────────────────────────────────────

def build(build_dir, cores, compiler_info, compiler_name,
          out_log=None, err_log=None, workspace=None):
    print("\n[BUILD PHASE — HPL]")

    arch_name = f"Linux_HPL_{compiler_name}"
    binary    = os.path.join(build_dir, HPL_DIR, "bin", arch_name, "xhpl")

    if os.path.exists(binary):
        print(f"  ✔  Binary exists, skipping build : {binary}")
        return "skip", binary

    os.makedirs(build_dir, exist_ok=True)

    makefile_path = make_hpl_makefile(build_dir, compiler_info, compiler_name)
    print(f"  ✔  Makefile written : {makefile_path}")

    log    = out_log or os.path.join(build_dir, "hpl_build.out")
    err    = err_log or os.path.join(build_dir, "hpl_build.err")
    script = os.path.join(build_dir, f"build_hpl_{compiler_name}.sh")
    tar    = os.path.join(build_dir, HPL_TAR)
    srcdir = os.path.join(build_dir, HPL_DIR)

    content = _make_build_sbatch(
        job_name      = f"build_hpl_{compiler_name}",
        out_log       = log,
        err_log       = err,
        tar           = tar,
        srcdir        = srcdir,
        makefile_path = makefile_path,
        arch_name     = arch_name,
        binary        = binary,
        compiler_name = compiler_name,
    )
    write_script(script, content)

    print(f"  ✔  Script written  : {script}")
    print(f"  [SLURM] out → {log}")
    print(f"  [SLURM] err → {err}")
    print(f"\n  [SLURM] Submitting BUILD (hpl/{compiler_name}) ...")

    return sbatch_submit(script), binary


# ─────────────────────────────────────────────────────────────
# RUN  (public entry point — SLURM path)
# ─────────────────────────────────────────────────────────────

def run(build_dir, run_dir, np, omp, build_jid, binary,
        compiler_info, info,
        out_log=None, err_log=None, workspace=None):
    print("\n[RUN PHASE — HPL]")

    avail_mb      = info.get("avail_mb", 1024)
    compiler_name = info.get("compiler_name", "gcc")
    N, nb, P, Q   = calc_hpl_params(avail_mb, np)
    _, _, mpirun  = get_mpi_config(compiler_name)

    os.makedirs(run_dir, exist_ok=True)
    make_hpl_dat(run_dir, N, nb, P, Q)

    slurm_out = out_log or os.path.join(run_dir, "hpl.out")
    slurm_err = err_log or os.path.join(run_dir, "hpl.err")
    script    = os.path.join(run_dir, "run_hpl.sh")

    content = _make_run_sbatch(
        job_name      = "run_hpl",
        out_log       = slurm_out,
        err_log       = slurm_err,
        run_dir       = run_dir,
        binary        = binary,
        np            = np,
        omp           = omp,
        mpirun        = mpirun,
        compiler_name = compiler_name,
        avail_mb      = avail_mb,
        N=N, nb=nb, P=P, Q=Q,
    )
    write_script(script, content)

    if build_jid == "skip":
        jid = sbatch_submit(script)
    else:
        jid = sbatch_submit(script, depend_id=build_jid)

    return jid, slurm_out


# ─────────────────────────────────────────────────────────────
# RUN DIRECT  (profiling path — bypasses SLURM)
# Uses --use-hwthread-cpus instead of --bind-to/--map-by
# which fails outside SLURM when slot count isn't pre-allocated.
# ─────────────────────────────────────────────────────────────

def run_direct(build_dir, run_dir, np, omp, binary,
               compiler_info, info,
               out_log=None, err_log=None):
    print("\n[RUN PHASE — HPL] (direct / profiling mode)")

    avail_mb      = info.get("avail_mb", 1024)
    compiler_name = info.get("compiler_name", "gcc")
    N, nb, P, Q   = calc_hpl_params(avail_mb, np)
    _, _, mpirun  = get_mpi_config(compiler_name)

    os.makedirs(run_dir, exist_ok=True)
    make_hpl_dat(run_dir, N, nb, P, Q)

    direct_out = out_log or os.path.join(run_dir, "hpl.out")
    direct_err = err_log or os.path.join(run_dir, "hpl.err")

    # Use --use-hwthread-cpus so mpirun works outside SLURM without
    # needing --oversubscribe or pre-allocated slots
    if compiler_name == "icc":
        mpi_flags = ["-np", str(np), "-bind-to", "core", "-map-by", "socket"]
    else:
        mpi_flags = ["-np", str(np), "--use-hwthread-cpus"]  # ← only change vs SLURM path

    cmd = [mpirun] + mpi_flags + [binary]

    env = os.environ.copy()
    env["OMP_NUM_THREADS"]      = str(omp)
    env["OPENBLAS_NUM_THREADS"] = str(omp)
    env["MKL_NUM_THREADS"]      = str(omp)
    env["GOTO_NUM_THREADS"]     = str(omp)

    if compiler_name == "icc":
        env["OCL_ICD_FILENAMES"] = ""

    print(f"  CMD: {' '.join(cmd)}")
    print(f"  OUT: {direct_out}")

    out_f = open(direct_out, "w")
    err_f = open(direct_err, "w")

    proc = subprocess.Popen(
        cmd,
        cwd=run_dir,
        env=env,
        stdout=out_f,
        stderr=err_f,
    )

    print(f"  ✔  HPL started (PID {proc.pid}) — uProf will profile now ...")
    return proc, direct_out, out_f, err_f
