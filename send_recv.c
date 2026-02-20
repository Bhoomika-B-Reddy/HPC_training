#include <stdio.h>
#include <mpi.h>

int main(int argc, char *argv[])
{
    int rank, nprocs;
    int value, total = 0;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    value = rank * rank;

    if (rank == 0) {
        total = value;  // include its own value

        for (int i = 1; i < nprocs; i++) {
            int temp;
            MPI_Recv(&temp, 1, MPI_INT, i, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            total += temp;
        }

        printf("Total = %d\n", total);
    } else {
        MPI_Send(&value, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return 0;
}