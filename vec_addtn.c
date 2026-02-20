#include <stdio.h>
#include <mpi.h>

int main(int argc, char *argv[])
{
    int rank, size;
    int A[8], B[8], C[8];
    int local_A[2], local_B[2], local_C[2];

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Assume np = 4 (8 elements / 4 processes = 2 each)

    if (rank == 0) {
        int tempA[8] = {1,2,3,4,5,6,7,8};
        int tempB[8] = {8,7,6,5,4,3,2,1};

        for (int i = 0; i < 8; i++) {
            A[i] = tempA[i];
            B[i] = tempB[i];
        }
    }

    // Scatter A and B
    MPI_Scatter(A, 2, MPI_INT, local_A, 2, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Scatter(B, 2, MPI_INT, local_B, 2, MPI_INT, 0, MPI_COMM_WORLD);

    // Local computation
    for (int i = 0; i < 2; i++)
        local_C[i] = local_A[i] + local_B[i];

    // Gather result
    MPI_Gather(local_C, 2, MPI_INT, C, 2, MPI_INT, 0, MPI_COMM_WORLD);

    // Print final result
    if (rank == 0) {
        printf("Result vector C:\n");
        for (int i = 0; i < 8; i++)
            printf("%d ", C[i]);
        printf("\n");
    }

    MPI_Finalize();
    return 0;
}