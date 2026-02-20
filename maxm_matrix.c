#include <stdio.h>
#include <mpi.h>

int main(int argc, char *argv[])
{
    int rank, size;
    int matrix[4][4];
    int local_row[4];
    int local_max, global_max;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size != 4) {
        if (rank == 0)
            printf("Run with 4 processes.\n");
        MPI_Finalize();
        return 0;
    }

    // Initialize matrix at rank 0
    if (rank == 0) {
        int temp[4][4] = {
            {1, 22, 3, 4},
            {5, 6, 70, 8},
            {9, 10, 11, 12},
            {13, 14, 15, 16}
        };

        for (int i = 0; i < 4; i++)
            for (int j = 0; j < 4; j++)
                matrix[i][j] = temp[i][j];
    }

    // Scatter rows
    MPI_Scatter(matrix, 4, MPI_INT, local_row, 4, MPI_INT,0, MPI_COMM_WORLD);

    // Find local maximum
    local_max = local_row[0];
    for (int i = 1; i < 4; i++)
        if (local_row[i] > local_max)
            local_max = local_row[i];

    printf("Rank %d: Local max = %d\n", rank, local_max);

    // Reduce using MPI_MAX
    MPI_Reduce(&local_max, &global_max, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Global maximum = %d\n", global_max);
    }

    MPI_Finalize();
    return 0;
}