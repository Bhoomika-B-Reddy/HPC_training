#include<stdio.h>
#include<mpi.h>

int main(int argc,char **argv)
{
    int rank,size;;
    int send,recv;
    MPI_Init(&argc,&argv);
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD,&size);

    int next=(rank + 1) % size;
    int prev=(rank - 1 + size) % size;

    send=rank;
    MPI_Sendrecv(&send,1,MPI_INT,next,0,&recv,1,MPI_INT,prev,0,MPI_COMM_WORLD,0);
    printf("Rank %d received %d ",rank,recv);
    MPI_Finalize();         
    return 0;

}