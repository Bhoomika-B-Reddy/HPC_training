#include <stdio.h>
#include <mpi.h>

int main(int argc, char *argv[])
{
    int rank, size;
    int matrix[4][4];
    int vector[4];
    int local_row[4];
    int local_result;
    int result[4];

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size != 4) {
        if (rank == 0)
            printf("Run with 4 processes.\n");
        MPI_Finalize();
        return 0;
    }

    // Initialize matrix and vector at rank 0
    if (rank == 0) {
        int temp[4][4] = {
            {1, 2, 3, 4},
            {5, 6, 7, 8},
            {9,10,11,12},
            {13,14,15,16}
        };

        int temp_vec[4] = {1, 1, 1, 1};

        for (int i = 0; i < 4; i++) {
            vector[i] = temp_vec[i];
            for (int j = 0; j < 4; j++)
                matrix[i][j] = temp[i][j];
        }
    }

    // Scatter matrix rows
    MPI_Scatter(matrix, 4, MPI_INT, local_row, 4, MPI_INT, 0, MPI_COMM_WORLD);

    // Broadcast vector
    MPI_Bcast(vector, 4, MPI_INT, 0, MPI_COMM_WORLD);

    // Compute local dot product
    local_result = 0;
    for (int i = 0; i < 4; i++)
        local_result += local_row[i] * vector[i];

    // Gather results
    MPI_Gather(&local_result, 1, MPI_INT,result, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Print final result
    if (rank == 0) {
        printf("Result vector y:\n");
        for (int i = 0; i < 4; i++)
            printf("%d ", result[i]);
        printf("\n");
    }

    MPI_Finalize();
    return 0;
}