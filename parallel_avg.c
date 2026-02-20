#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int main(int argc, char *argv[])
{
    int rank, size;
    int N;
    int *data = NULL;
    int local_n;
    int *local_data;
    int local_sum = 0, global_sum = 0;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
        printf("Enter number of elements: ");
        scanf("%d", &N);

        data = (int*)malloc(N * sizeof(int));

        printf("Enter %d numbers:\n", N);
        for (int i = 0; i < N; i++)
            scanf("%d", &data[i]);
    }

    // Broadcast N
    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Divide data equally
    local_n = N / size;

    local_data = (int*)malloc(local_n * sizeof(int));

    // Scatter data
    MPI_Scatter(data, local_n, MPI_INT, local_data, local_n, MPI_INT, 0, MPI_COMM_WORLD);

    // Compute local sum
    for (int i = 0; i < local_n; i++)
        local_sum += local_data[i];

    // Reduce to global sum
    MPI_Reduce(&local_sum, &global_sum, 1,
               MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    // Rank 0 computes average
    if (rank == 0) {
        double avg = (double)global_sum / N;
        printf("Average = %.2f\n", avg);
    }

    MPI_Finalize();
    return 0;
}