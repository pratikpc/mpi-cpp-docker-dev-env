#include <mpi.h>
#include <iostream>
#include <vector>

int main(int argc, char **argv)
{
	int rank, size;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	// --- TASK 1: MPI_Scatter
	{
		int my_value;
		int root_rank = 0;
		if (rank == 0)
		{
			int buffer[4] = {0, 100, 200, 300};
			MPI_Scatter(buffer, 1, MPI_INT, &my_value, 1, MPI_INT, root_rank, MPI_COMM_WORLD);
		}
		else
		{
			MPI_Scatter(NULL, 1, MPI_INT, &my_value, 1, MPI_INT, root_rank, MPI_COMM_WORLD);
		}
		//std::cout<<"scatter "<< rank <<" value = "<< my_value <<std::endl;
	}

	// --- TASK 2: MPI_Barrier
	// processes wait until all processes arrived here
	if( rank == 0 )
	{
		MPI_Barrier( MPI_COMM_WORLD );
	}

	// TASK 3: MPI_Allreduce
	double locval = rank * 10;
	double gloval = 0.0;
	MPI_Allreduce(&locval, &gloval, 1, MPI_DOUBLE, MPI_SUM,
				  MPI_COMM_WORLD);
	//std::cout<<"allreduce "<< rank <<" value = "<< gloval <<std::endl;

	MPI_Finalize();
}
