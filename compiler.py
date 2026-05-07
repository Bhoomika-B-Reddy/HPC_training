#!/usr/bin/env python3
import subprocess
import sys
import os

COMPILERS = {
    "aocc": {
        "cc":    "/opt/AMD/aocc-compiler-5.1.0/bin/clang",
        "mpi":   "/usr/bin/mpicc",
        "flags": "-O3 -march=native -funroll-loops -ffast-math -fopenmp",
        "link":  "-lm -lomp",          # AOCC uses libomp (not libgomp)
        "install": "/opt/AMD/aocc-compiler-5.1.0/setenv_AOCC.sh",
    },
    "icc": {
        "cc":    "icx",
        "mpi":   "mpiicx",
        "flags": "-O3 -xHost -ipo -funroll-loops -qopenmp",
        "link":  "-lm -liomp5",        # Intel uses libiomp5
        "install": "/opt/intel/oneapi/setvars.sh",
    },
    "gcc": {
        "cc":    "gcc",
        "mpi":   "/usr/bin/mpicc",
        "flags": "-O3 -march=native -funroll-loops -ffast-math -fopenmp",
        "link":  "-lm -lgomp",         # GCC uses libgomp
        "install": "sudo apt install -y gcc openmpi-bin libopenmpi-dev",
    },
}

def check_tool(name):
    found = subprocess.run(["which", name], capture_output=True).returncode == 0
    print(f"  {'✔' if found else '✗'}  {name}")
    return found

def install_pkg(*pkgs):
    for pkg in pkgs:
        print(f"  ▶  Installing {pkg} ...")
        r = subprocess.run(
            ["sudo", "apt", "install", "-y", pkg],
            capture_output=True, text=True)
        print(f"  {'✔' if r.returncode==0 else '✗'}  {pkg}")

def detect_compiler():
    print("\n  Detecting best compiler:")
    for name, info in COMPILERS.items():
        cc  = info["cc"]
        mpi = info["mpi"]
        if (subprocess.run(["which", cc],  capture_output=True).returncode == 0 and
            subprocess.run(["which", mpi], capture_output=True).returncode == 0):
            print(f"  ✔  Using {name.upper()} : {cc} + {mpi}")
            r = subprocess.run([cc, "--version"], capture_output=True, text=True)
            if r.returncode == 0:
                print(f"      {r.stdout.splitlines()[0]}")
            return name, info
    print("  ✗  No compiler found.")
    sys.exit(1)

def preprocess_compiler():
    print("\n[PREPROCESS — COMPILER]")

    print("  Step 1 — Scanning for compilers (AOCC > ICC > GCC):")
    for name, info in COMPILERS.items():
        cc_found  = check_tool(info["cc"])
        mpi_found = check_tool(info["mpi"])
        if cc_found and mpi_found:
            print(f"  ✔  {name.upper()} available")

    print("\n  Step 2 — Selecting best compiler:")
    compiler_name, compiler_info = detect_compiler()

    print("\n  Step 3 — Checking libraries:")
    missing_libs = []
    for lib, pkg in [
        ("libopenblas-dev", "libopenblas-dev"),
        ("liblapacke-dev",  "liblapacke-dev"),
    ]:
        r = subprocess.run(["dpkg", "-l", lib], capture_output=True, text=True)
        found = r.returncode == 0 and "ii" in r.stdout
        print(f"  {'✔' if found else '✗'}  {lib}")
        if not found:
            missing_libs.append(pkg)

    if missing_libs:
        print("\n  Step 4 — Installing missing libraries:")
        for pkg in missing_libs:
            install_pkg(pkg)
    else:
        print("\n  Step 4 — All libraries present.")

    print("\n  Step 5 — Final verification:")
    cc  = compiler_info["cc"]
    mpi = compiler_info["mpi"]
    check_tool(cc)
    check_tool(mpi)
    check_tool("make")
    check_tool("wget")
    check_tool("tar")

    print(f"\n  ✔  Compiler preprocessing complete.")
    print(f"  ✔  Selected : {compiler_name.upper()}")
    print(f"      CC    : {cc}")
    print(f"      MPI   : {mpi}")
    print(f"      FLAGS : {compiler_info['flags']}")

    return compiler_name, compiler_info

def preprocess_compiler_forced(compiler_name, compiler_info):
    print(f"\n[PREPROCESS — COMPILER — FORCED: {compiler_name.upper()}]")
    cc  = compiler_info["cc"]
    mpi = compiler_info["mpi"]

    print(f"\n  Step 1 — Checking {compiler_name.upper()} tools:")
    cc_ok  = check_tool(cc)
    mpi_ok = check_tool(mpi)

    if not cc_ok or not mpi_ok:
        print(f"\n  ✗  {compiler_name.upper()} not found.")
        print(f"      Install : {compiler_info['install']}")
        sys.exit(1)

    print(f"\n  Step 2 — Checking libraries:")
    missing_libs = []
    for lib, pkg in [
        ("libopenblas-dev", "libopenblas-dev"),
        ("liblapacke-dev",  "liblapacke-dev"),
    ]:
        r = subprocess.run(["dpkg", "-l", lib], capture_output=True, text=True)
        found = r.returncode == 0 and "ii" in r.stdout
        print(f"  {'✔' if found else '✗'}  {lib}")
        if not found:
            missing_libs.append(pkg)

    if missing_libs:
        print(f"\n  Step 3 — Installing missing libraries:")
        for pkg in missing_libs:
            install_pkg(pkg)
    else:
        print(f"\n  Step 3 — All libraries present.")

    print(f"\n  Step 4 — Final verification:")
    check_tool(cc)
    check_tool(mpi)
    check_tool("make")
    check_tool("wget")
    check_tool("tar")

    print(f"\n  Compiler version:")
    r = subprocess.run([cc, "--version"], capture_output=True, text=True)
    if r.returncode == 0:
        print(f"  ✔  {r.stdout.splitlines()[0]}")

    print(f"\n  ✔  {compiler_name.upper()} ready.")
    print(f"      CC    : {cc}")
    print(f"      MPI   : {mpi}")
    print(f"      FLAGS : {compiler_info['flags']}")

    return compiler_name, compiler_info
