//=============================================================================
// Solution of unsteady 1D diffusion equation withinin the domain [0,1] using
// finite difference method.
// MPI libraries are used for parallelization
// Homogeneous Dirichlet BC on both boundary points.
//
// This is a part of teaching material provided for the following workshop
// "MPI in action: Parallelization of unsteady heat conduction solvers"
// conducted by IIT Goa - Nodal centre of National Supercomputing Mission
// Date: 2--3 October 2021
//
// Written by Y Sudhakar, IIT Goa
// email: sudhakar@iitgoa.ac.in
//
// TODO list for the participants
// 1. Fix the norm computation -- think from parallel computation perspective
// 2. Why do we get different results than serial code?
// (Hint: data transfer missing between processor)
// 3. bring entire final solution to processor0 and write output
// 4. Use dynamic memory allocation and deallocation
//=============================================================================

#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <cmath> // value of PI is defined as M_PI
#include <mpi.h> // header for MPI

int main(int argc, char **argv)
{
  int nglobal = 32;                  // global number of nodes
  int np = 2;                        // number of processors
  int nlocal = nglobal / np + 2;     // +2 are used to store ghost nodes
                                     // communication happens on these nodes
  double dt = 0.0001;                // time step
  double dx = 1.0 / (nglobal - 1.0); // mesh size
  double dxinv2 = 1.0 / dx / dx;     // 1.0/(dx)^2 which is used often

  // each processor holds only local data
  double xx[nlocal], uold[nlocal], unew[nlocal];
  double g[nlocal], f[nlocal];

  // ----- MPI related stuff
  MPI_Init(&argc, &argv); // Initialization of MPI
  //int procsize; // number of processors
  //MPI_Comm_size(MPI_COMM_WORLD, &procsize);
  int my_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank); // processor id of each processor

  // store the starting and end points of local indices
  int ibegin = my_rank * (nlocal - 2);
  int iend = ibegin + nlocal - 1;

  // iloc -- local index -- local node number
  // iglo -- global index -- global node number
  // initialize
  int iglo = 0;
  for (int iloc = 0; iloc < nlocal; iloc++)
  {
    iglo = iloc + ibegin;
    xx[iloc] = (iglo - 1) * dx;
    uold[iloc] = 0.0;
    unew[iloc] = 0.0;
    g[iloc] = M_PI * M_PI * sin(M_PI * xx[iloc]);
  }

  // write convergence history in a outfile
  std::ofstream outconv;
  if (my_rank == 0)
    outconv.open("convergence-par.txt");

  double norm = 100.0;
  bool converged = false;
  int itercount = 0;
  // --- time integration begins
  while (not converged)
  {
    itercount++;
    // update all interior nodes
    for (int iloc = 1; iloc < nlocal - 1; iloc++)
    {
      if (my_rank == 0 and iloc == 1)
      {
        unew[iloc] = 0.0;
        continue;
      }
      if (my_rank == np - 1 and iloc == nlocal - 2)
      {
        unew[iloc] = 0.0;
        continue;
      }
      f[iloc] = g[iloc] + dxinv2 * (uold[iloc + 1] - 2.0 * uold[iloc] + uold[iloc - 1]);
      unew[iloc] = uold[iloc] + dt * f[iloc];
    }

    // compute norm
    norm = 0.0;
    for (int iloc = 1; iloc < nlocal - 1; iloc++)
    {
      if (abs(f[iloc]) > norm)
        norm = fabs(f[iloc]);
    }

    // write convergence history
    if (my_rank == 0)
    {
      outconv << itercount << " " << norm << std::endl;
      std::cout << itercount << " " << norm << std::endl;
    }

    // check for convergence
    if (norm < 1e-6)
    {
      converged = true;
    }

    // update the interior nodes
    for (int iloc = 1; iloc < nlocal - 1; iloc++)
    {
      uold[iloc] = unew[iloc];
    }
  }

  // Write output of result in a file
  std::stringstream sstm;
  sstm << "output_" << my_rank << ".txt";
  std::string file_local = sstm.str();
  std::ofstream outfile;
  outfile.open(file_local);
  double uexact;
  for (int iloc = 1; iloc < nlocal - 1; iloc++)
  {
    uexact = sin(M_PI * xx[iloc]);
    outfile << xx[iloc] << "\t" << unew[iloc] << "\t" << uexact << std::endl;
  }
  outfile.close();

  // properly close MPI processes
  MPI_Finalize();
}
