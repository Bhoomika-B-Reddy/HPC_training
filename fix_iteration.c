#include <stdio.h>
#include <mpi.h>

int main(int argc, char **argv)
{
    int i, rank, nprocs, count, start, stop, nloops, total_nloops;

    MPI_Init(&argc, &argv);

    // get the number of processes, and the id of this process
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    // we want to perform 1000 iterations in total. Work out the
    // number of iterations to perform per process...
    count = 1000 / (nprocs-1);

    // we use the rank of this process to work out which
    // iterations to perform.
    if(rank!=0){
    start = (rank-1) * count;
    stop = start + count;
    }

    // now perform the loop
    nloops = 0;
     if(rank!=0){
    for (i=start; i<stop; ++i)
    {
        ++nloops;
    }
    }

    printf("Process %d performed %d iterations of the loop.\n",
           rank, nloops);

//p1,2,3 ==> nloops 333
    if (rank != 0)
    {
        // if we are not the master process (rank 0) then we need to
        // tell that process how many iterations we performed
        MPI_Send( &nloops, 1, MPI_INT, 0, 0, MPI_COMM_WORLD );
    }
    else
    {
        // if we are the master process, then we need to wait to be
        // told how many iterations were performed by each other
        // process
        total_nloops = nloops;

        for (i=1; i<nprocs; ++i)
        {
            MPI_Recv( &nloops, 1, MPI_INT, i, 0, MPI_COMM_WORLD, 0 );

            total_nloops += nloops;
        }

        // ok - are there any more loops to run?
        nloops = 0;

        for (i=total_nloops; i<1000; ++i)
        {
            ++nloops;
        }

        printf("Process 0 performed the remaining %d iterations of the loop\n",
               nloops);
    }

    MPI_Finalize();

    return 0;
}
