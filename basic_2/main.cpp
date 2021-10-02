#include<mpi.h>
#include<iostream>

int main(int argc, char**argv)
{
	int rank, size;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	// We should use even number of processes
	// Otherwise abort the program
	if( size%2 != 0 )
	{
		std::cout <<"Use even number of processes "<< std::endl;
		MPI_Abort(MPI_COMM_WORLD, 1);
	}

	// Sending messages from even numbered processors
	// processor n sends message to processor n+1
	if( rank %2 == 0)
	{
		int send_data = rank * 100;
		int destination = (rank + 1) % size;
		int tag = rank;
		MPI_Send( &send_data, 1, MPI_INT, destination, tag, MPI_COMM_WORLD );
	}

	// receiving messages in odd numbered processors
	// processor n+1 receives message from processor n
	if( rank %2 != 0)
	{
		int recv_data;
		int source = (rank -1) ;
                if (source < 0)
                        source = size - 1;
		int tag = source;
		MPI_Recv( &recv_data, 1, MPI_INT, source, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE );
		std::cout <<"Received "<< recv_data <<" from process "
							<<source <<". I am process "<< rank << std::endl;
	}

	MPI_Finalize();
}
