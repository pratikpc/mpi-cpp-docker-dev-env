#include<mpi.h>
#include<iostream>

int main(int argc, char**argv)
{
        int rank, size;
        MPI_Init(&argc, &argv);                                                          // initialize
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);  // get rank of each process
        MPI_Comm_size(MPI_COMM_WORLD, &size);  // get no of process in this communicator

        std::cout<<"I am " <<rank<<" of "<< size <<" process"<<std::endl;
        MPI_Finalize();
}