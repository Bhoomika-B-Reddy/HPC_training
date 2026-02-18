Objective:
To measure runtime for accessing integer arrays of sizes:
10^4
10^5
10^6
using different numbers of processes (np = 1, 2, 4).

Compilation:
mpicc array_access.c -o array_access

Execution:
mpirun -np 1 ./array_access
mpirun -np 2 ./array_access
mpirun -np 4 ./array_access

Observations:
Runtime increases slightly with more processes for simple memory access.
Parallel overhead affects small workloads.
Larger workloads show better scalability.

Conclusion:
Parallelism is beneficial for computation-heavy tasks, but simple memory access does not scale well due to communication overhead.
