#include <stdio.h>
#include <mpi.h>

int main(int argc, char *argv[])
{
    int rank, size;
    int matrix[4][4];
    int local_row[4];
    int local_sum = 0, total_sum = 0;

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
            {1, 2, 3, 4}, {5, 6, 7, 8},{9,10,11,12},{13,14,15,16} 
         };

        for (int i = 0; i < 4; i++)
            for (int j = 0; j < 4; j++)
                matrix[i][j] = temp[i][j];
    }

    // Scatter rows (each process gets 4 elements = one row)
    MPI_Scatter(matrix, 4, MPI_INT,local_row, 4, MPI_INT, 0, MPI_COMM_WORLD);

    // Compute local row sum
    for (int i = 0; i < 4; i++)
        local_sum += local_row[i];

    printf("Rank %d: Row sum = %d\n", rank, local_sum);

    // Reduce to get total matrix sum
    MPI_Reduce(&local_sum, &total_sum, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Total matrix sum = %d\n", total_sum);
    }

    MPI_Finalize();
    return 0;
}