#!/usr/bin/env python3
import os, sys, math, subprocess
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from slurm import write_script, sbatch_submit

# ── MPI config ────────────────────────────────────────────────────────────────

MPI = {
    "icc": {
        "dir":    "/opt/intel/oneapi/mpi/2021.17",
        "run":    "/opt/intel/oneapi/mpi/2021.17/bin/mpirun",
        "flags":  "-np {np} -bind-to core -map-by socket",
    },
    "gcc": {
        "dir":    "/usr/lib/x86_64-linux-gnu/openmpi",
        "run":    "/usr/bin/mpirun",
        "flags":  "-np {np} --bind-to none --map-by slot",
    },
    "aocc": {
        "dir":   "/usr/lib/x86_64-linux-gnu/openmpi",
        "run":   "/usr/bin/mpirun",
        "flags": "-np {np} --bind-to none --map-by slot",
    },
}

def mpi_inc(c):  d = MPI[c]["dir"]; return f"-I{d}/include"
def mpi_lib(c):  d = MPI[c]["dir"]; return (
    f"-L{d}/lib -Xlinker --enable-new-dtags -Xlinker -rpath -Xlinker {d}/lib -lmpifort -lmpi -ldl -lrt -lpthread"
    if c == "icc" else f"-L{d}/lib -lmpi"
)

# ── Auto-tune N, NB, P×Q ─────────────────────────────────────────────────────

def calc_hpl_params(avail_mb, np):
    usable = 0.80 * avail_mb * 1024 * 1024
    nb     = 256 if avail_mb >= 32768 else (192 if avail_mb >= 2048 else 128)
    N      = (int(math.sqrt(usable / 8)) // nb) * nb
    caps   = [(512, 2048), (1024, 4000), (2048, 8000), (4096, 16000)]
    for cap_mb, cap_n in caps:
        if avail_mb < cap_mb:
            N = min(N, cap_n)
            break

    best_p, best_q, best_diff = 1, np, np
    for p in range(1, int(math.sqrt(np)) + 1):
        if np % p == 0:
            q = np // p
            if q >= p and (q - p) < best_diff:
                best_diff, best_p, best_q = q - p, p, q
    return N, nb, best_p, best_q

# ── Makefile ──────────────────────────────────────────────────────────────────

def make_hpl_makefile(build_dir, compiler_info, c):
    arch = f"Linux_HPL_{c}"
    cc_flags = "-O3 -march=native -fp-model fast=2" if c == "icc" else "-O3 -march=native -funroll-loops -ffast-math"
    omp      = "-qopenmp" if c == "icc" else "-fopenmp"
    archiver = "xiar" if c == "icc" else "llvm-ar"

    content = f"""\
ARCH    = {arch}
TOPdir  = {build_dir}/{HPL_DIR}
INCdir  = $(TOPdir)/include
BINdir  = $(TOPdir)/bin/$(ARCH)
LIBdir  = $(TOPdir)/lib/$(ARCH)
HPLlib  = $(LIBdir)/libhpl.a

MPinc   = {mpi_inc(c)}
MPlib   = {mpi_lib(c)}
LAlib   = -lopenblas -llapacke

HPL_INCLUDES = -I$(INCdir) -I$(INCdir)/$(ARCH) -I/usr/include $(MPinc)
HPL_LIBS     = -Wl,--start-group $(HPLlib) $(LAlib) $(MPlib) -Wl,--end-group -lm
HPL_OPTS     = -DHPL_CALL_CBLAS
HPL_DEFS     = $(HPL_OPTS) $(HPL_INCLUDES)

CC        = {compiler_info["cc"]}
CCFLAGS   = $(HPL_DEFS) {cc_flags} {omp}
LINKER    = {compiler_info["mpi"]}
LINKFLAGS = {cc_flags} {omp}
ARCHIVER  = {archiver}
ARFLAGS   = r
RANLIB    = ranlib
"""
    path = os.path.join(build_dir, f"Make.{arch}")
    with open(path, "w") as f:
        f.write(content)
    return path

# ── HPL.dat ───────────────────────────────────────────────────────────────────

def make_hpl_dat(run_dir, N, nb, P, Q):
    lines = [
        "HPLinpack benchmark input file",
        "Innovative Computing Laboratory, University of Tennessee",
        "HPL.out", "6", "1", f"{N}", "1", f"{nb}",
        "0", "1", f"{P}", f"{Q}", "16.0",
        "1", "2", "1", "4", "1", "2", "1", "1", "1", "1", "1", "1",
        "2", f"{nb}", "0", "0", "1", "8",
    ]
    path = os.path.join(run_dir, "HPL.dat")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return path

# ── SLURM helpers ─────────────────────────────────────────────────────────────

def sbatch_header(job, out, err, ntasks, time, cpus=1, mem_mb=None, partition=None, exclusive=False):
    lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name={job}",
        f"#SBATCH --nodes=1",
        f"#SBATCH --ntasks={ntasks}",
        f"#SBATCH --cpus-per-task={cpus}",
        f"#SBATCH --time={time}",
        f"#SBATCH --output={out}",
        f"#SBATCH --error={err}",
    ]
    if mem_mb:   lines.append(f"#SBATCH --mem={mem_mb}M")
    if partition: lines.append(f"#SBATCH --partition={partition}")
    if exclusive: lines.append("#SBATCH --exclusive")
    return "\n".join(lines)

def banner(tag):
    return (f'echo "================================================"\n'
            f'echo "{tag}  Host:$(hostname)  Date:$(date)"\n'
            f'echo "================================================"')

def intel_env():
    return ["export OCL_ICD_FILENAMES=''", "source /opt/intel/oneapi/setvars.sh --force || true", ""]

# ── Build script ──────────────────────────────────────────────────────────────

def make_build_script(job, out, err, tar, srcdir, makefile, arch, binary, c):
    env  = intel_env() if c == "icc" else []
    patch = [f"sed -i 's/^ARCHIVER.*=.*xiar/ARCHIVER = llvm-ar/' {srcdir}/Make.{arch}", ""] if c != "icc" else []
    return "\n".join([
        sbatch_header(job, out, err, ntasks=1, time="00:30:00"),
        "", "set -euo pipefail", "", banner(f"BUILD — HPL {HPL_VERSION} [{c}]"), "",
        *env,
        f"[ -f {tar} ] || wget -q {HPL_URL} -O {tar} || curl -L {HPL_URL} -o {tar}",
        f"[ -d {srcdir} ] || tar -xf {tar} -C $(dirname {srcdir})",
        f"cp -f {makefile} {srcdir}/Make.{arch}", "",
        *patch,
        f"cd {srcdir}",
        f"make -f Make.top build_src arch={arch} 2>&1",
        f"make -f Make.top build_tst arch={arch} 2>&1", "",
        f'[ -f {binary} ] && echo "✔ {binary}" || {{ echo "✘ xhpl not found" >&2; exit 1; }}', "",
        banner("BUILD DONE"),
    ]) + "\n"

# ── Run script ────────────────────────────────────────────────────────────────

def make_run_script(job, out, err, run_dir, binary, np, omp, c):
    env      = intel_env() if c == "icc" else []
    mpirun   = MPI[c]["run"]
    flags    = MPI[c]["flags"].format(np=np)
    omp_vars = "\n".join(f"export {k}={omp}" for k in
                         ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "GOTO_NUM_THREADS"])
    return "\n".join([
        sbatch_header(job, out, err, ntasks=np, cpus=omp, time="02:00:00"),
        "", "set -euo pipefail", "", banner(f"RUN — HPL {HPL_VERSION} [{c}]"), "",
        *env, omp_vars, "",
        f"cd {run_dir}",
        f"{mpirun} {flags} {binary}", "",
        banner("RUN DONE"),
    ]) + "\n"

# ── Public: build / run / run_direct ─────────────────────────────────────────

def build(build_dir, cores, compiler_info, c, out_log=None, err_log=None, workspace=None):
    print("\n[BUILD — HPL]")
    arch   = f"Linux_HPL_{c}"
    binary = os.path.join(build_dir, HPL_DIR, "bin", arch, "xhpl")

    if os.path.exists(binary):
        print(f"  ✔ Skip (exists): {binary}")
        return "skip", binary

    os.makedirs(build_dir, exist_ok=True)
    makefile = make_hpl_makefile(build_dir, compiler_info, c)
    tar      = os.path.join(build_dir, HPL_TAR)
    srcdir   = os.path.join(build_dir, HPL_DIR)
    script   = os.path.join(build_dir, f"build_hpl_{c}.sh")
    log      = out_log or os.path.join(build_dir, "hpl_build.out")
    err      = err_log or os.path.join(build_dir, "hpl_build.err")

    write_script(script, make_build_script(f"build_hpl_{c}", log, err, tar, srcdir, makefile, arch, binary, c))
    print(f"  ✔ Script: {script}")
    return sbatch_submit(script), binary


def run(build_dir, run_dir, np, omp, build_jid, binary, compiler_info, info,
        out_log=None, err_log=None, workspace=None):
    print("\n[RUN — HPL]")
    c        = info.get("compiler_name", "gcc")
    avail_mb = info.get("avail_mb", 1024)
    N, nb, P, Q = calc_hpl_params(avail_mb, np)

    os.makedirs(run_dir, exist_ok=True)
    make_hpl_dat(run_dir, N, nb, P, Q)

    script = os.path.join(run_dir, "run_hpl.sh")
    out    = out_log or os.path.join(run_dir, "hpl.out")
    err    = err_log or os.path.join(run_dir, "hpl.err")

    write_script(script, make_run_script("run_hpl", out, err, run_dir, binary, np, omp, c))
    jid = sbatch_submit(script) if build_jid == "skip" else sbatch_submit(script, depend_id=build_jid)
    return jid, out


def run_direct(build_dir, run_dir, np, omp, binary, compiler_info, info,
               out_log=None, err_log=None):
    print("\n[RUN — HPL] (direct/profiling)")
    c        = info.get("compiler_name", "gcc")
    avail_mb = info.get("avail_mb", 1024)
    N, nb, P, Q = calc_hpl_params(avail_mb, np)

    os.makedirs(run_dir, exist_ok=True)
    make_hpl_dat(run_dir, N, nb, P, Q)

    mpirun  = MPI[c]["run"]
    flags   = (["-np", str(np), "-bind-to", "core", "-map-by", "socket"] if c == "icc"
               else ["-np", str(np), "--use-hwthread-cpus"])
    cmd     = [mpirun] + flags + [binary]
    env     = {**os.environ,
               "OMP_NUM_THREADS":      str(omp),
               "OPENBLAS_NUM_THREADS": str(omp),
               "MKL_NUM_THREADS":      str(omp),
               "GOTO_NUM_THREADS":     str(omp),
               **({"OCL_ICD_FILENAMES": ""} if c == "icc" else {})}

    out = out_log or os.path.join(run_dir, "hpl.out")
    err = err_log or os.path.join(run_dir, "hpl.err")
    print(f"  CMD: {' '.join(cmd)}\n  OUT: {out}")

    proc = subprocess.Popen(cmd, cwd=run_dir, env=env,
                            stdout=open(out, "w"), stderr=open(err, "w"))
    print(f"  ✔ HPL started (PID {proc.pid})")
    return proc, out, proc.stdout, proc.stderr
