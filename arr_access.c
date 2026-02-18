#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

void test_array(long N) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    long local_n = N / size;
    long start_index = rank * local_n;
    long end_index = start_index + local_n;

    int *array = NULL;

    if (rank == 0) {
        array = (int *)malloc(N * sizeof(int));
        for (long i = 0; i < N; i++)
            array[i] = i;
    }

    int *local_array = (int *)malloc(local_n * sizeof(int));

    MPI_Scatter(array, local_n, MPI_INT,
                local_array, local_n, MPI_INT,
                0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    double start_time = MPI_Wtime();

    long long local_sum = 0;
    for (long i = 0; i < local_n; i++) {
        local_sum += local_array[i];
    }

    long long global_sum = 0;

    MPI_Reduce(&local_sum, &global_sum, 1,
               MPI_LONG_LONG, MPI_SUM,
               0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    double end_time = MPI_Wtime();

    if (rank == 0) {
        printf("Array Size = %ld, Processes = %d, Time = %f seconds\n",
               N, size, end_time - start_time);
    }

    free(local_array);
    if (rank == 0)
        free(array);
}

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    test_array(10000);
    test_array(100000);
    test_array(1000000);

    MPI_Finalize();
    return 0;
}
