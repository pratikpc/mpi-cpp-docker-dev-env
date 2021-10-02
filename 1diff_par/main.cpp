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
#include <cfloat> // Value of DBL_MAX
#include <cmath>  // value of PI is defined as M_PI
#include <mpi.h>  // header for MPI

void update_interior_nodes(std::size_t const nlocal, int const rank, int const np, double dxinv2, double dt, double *unew, double *uold, double *f, double *g)
{
  // update all interior nodes
  for (std::size_t iloc = 1; iloc < nlocal - 1; iloc++)
  {
    // 0th value
    if (rank == 0 and iloc == 1)
    {
      unew[iloc] = 0.0;
      continue;
    }
    // Last value
    if (rank == np - 1 and iloc == nlocal - 2)
    {
      unew[iloc] = 0.0;
      continue;
    }
    g[iloc] = f[iloc] + dxinv2 * (uold[iloc + 1] - 2.0 * uold[iloc] + uold[iloc - 1]);
    unew[iloc] = uold[iloc] + dt * f[iloc];
    std::cout << unew[iloc] << "\n";
  }
}

double norm(std::size_t nlocal, double const *const f, double norm = 0.0)
{
  for (std::size_t iloc = 1; iloc < nlocal - 1; iloc++)
    if (fabs(f[iloc]) > norm)
      norm = fabs(f[iloc]);
  return norm;
}

int proc_size()
{
  int procsize; // number of processors
  MPI_Comm_size(MPI_COMM_WORLD, &procsize);
  return procsize;
}
int mpi_processor_id()
{
  int processor_id;
  MPI_Comm_rank(MPI_COMM_WORLD, &processor_id); // processor id of each processor
  return processor_id;
}
int main(int argc, char **argv)
{
  int const nglobal = 64 * 2;              // global number of nodes
                                           // communication happens on these nodes
  static constexpr double dt = 0.0001;     // time step
  double const dx = 1.0 / (nglobal - 1.0); // mesh size

  // ----- MPI related stuff
  MPI_Init(&argc, &argv);        // Initialization of MPI
  int const np = proc_size();    // number of processors

  int const processor_id = mpi_processor_id();

  // store the starting and end points of local indices
  int nlocal = nglobal / np + 2; // +2 are used to store ghost nodes
  int ibegin = processor_id * (nlocal - 2);
  int iend = ibegin + nlocal - 1;

  if (processor_id == np - 1 && iend != nglobal + 1)
  {
    iend = nglobal + 1;
    nlocal = iend - ibegin + 1;
  }

  // Each processor holds only local data
  // Problem 4 solved as this is a VLA
  // VLAs are Dynamically allocated
  // They are not Standard C++ which would render this code incompatible
  // With MSVC anyways
  // But are supported by Clang and GCC
  double uold[nlocal], unew[nlocal];
  double g[nlocal], f[nlocal];

  // iloc -- local index -- local node number
  // iglo -- global index -- global node number
  // initialize
  int iglo = 0;
  for (int iloc = 0; iloc < nlocal; iloc++)
  {
    iglo = iloc + ibegin;
    auto const xx = (iglo - 1) * dx;
    uold[iloc] = 0.0;
    unew[iloc] = 0.0;
    g[iloc] = M_PI * M_PI * sin(M_PI * xx);
  }

  // write convergence history in a outfile
  std::ofstream outconv;
  if (processor_id == 0)
    outconv.open("convergence-par.txt");

  double const dxinv2 = 1.0 / dx / dx; // 1.0/(dx)^2 which is used often
  bool converged = false;
  int itercount = 0;
  std::cout << dx << " : " << dt * dxinv2 << "\n";
  // --- time integration begins
  while (not converged)
  {
    itercount++;

    update_interior_nodes(nlocal, processor_id, np, dxinv2, dt, unew /*Passed as ref*/, uold /*Passed as ref*/, f /*Passed as ref*/, g /*Passed as ref*/);

    // compute local norm
    double local_norm = norm(nlocal, f);
    {
      double norm = DBL_MAX;
      MPI_Allreduce(&local_norm, &norm, 1, MPI_DOUBLE, MPI_MAX,
                    MPI_COMM_WORLD);
      // write convergence history
      if (processor_id == 0)
      {
        outconv << itercount << " " << norm << std::endl;
        // std::cout << itercount << " " << norm << std::endl;
      }

      // check for convergence
      if (norm < 1e-6)
        converged = true;
    }

    {
      // if (my_rank == 0 && itercount % 100 == 0)

      // Send 1 -> n - 1
      // Because the value of 1 is calculated here
      // However at n - 1 it is last element making calcualtion impossible
      if (processor_id > 0)
        MPI_Send(&unew[1], 1, MPI_DOUBLE, processor_id - 1, processor_id, MPI_COMM_WORLD);
      if (processor_id + 1 < np) // If not last
        MPI_Recv(&uold[nlocal - 1], 1, MPI_DOUBLE, processor_id + 1, processor_id + 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

      // update the interior nodes
      // update the interior nodes
      for (int iloc = 1; iloc < nlocal - 1; iloc++)
        uold[iloc] = unew[iloc];
    }
    {
      // Send n - 2 -> 0
      // Because the value of n - 2 is calculated here
      // However at n - 2 it is first element making calcualtion impossible
      if (processor_id != 0)
        MPI_Recv(&uold[0], 1, MPI_DOUBLE, processor_id - 1, processor_id - 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      if (processor_id + 1 < np)
        MPI_Send(&uold[nlocal - 2], 1, MPI_DOUBLE, processor_id + 1, processor_id, MPI_COMM_WORLD);
    }
  }

  double usave[nglobal + 1];
  {
    static constexpr int TAG_DATA_TRANSFER = 10;
    if (processor_id != 0)
    {
      // Send size first
      int const size = nlocal - 1;
      MPI_Send(&size, 1, MPI_DOUBLE, 0, TAG_DATA_TRANSFER, MPI_COMM_WORLD);
      MPI_Send(&unew[1], size, MPI_DOUBLE, 0, TAG_DATA_TRANSFER, MPI_COMM_WORLD);
    }
    else
    {
      int source_begin = 0;
      for (int source = 1; source < np; ++source)
      {
        // Send size first
        int size;
        MPI_Recv(&size, 1, MPI_DOUBLE, source, TAG_DATA_TRANSFER, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        std::cout << size << " : " << source_begin << '\n';
        source_begin += size - 1;
        MPI_Recv(&usave[1] + source_begin, size, MPI_DOUBLE, source, TAG_DATA_TRANSFER, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      }
    }
  }
  // update the interior nodes
  for (int iloc = 1; iloc < nlocal - 1; iloc++)
    usave[iloc] = unew[iloc];

  if (processor_id == 0)
  {
    // Write output of result in a file
    std::stringstream sstm;
    sstm << "output.txt";
    std::string file_local = sstm.str();
    std::ofstream outfile;
    outfile.open(file_local);
    double uexact;
    double mse = 0;

    // Display for all elements
    for (int iloc = 1; iloc < nglobal; iloc++)
    {
      // Calculate WRT global (that is if all elements were present here)
      auto const xx = (iloc - 1) * dx;
      uexact = sin(M_PI * xx);
      mse += pow(usave[iloc] - uexact, 2);

      outfile << iloc << " " << xx << "\t" << usave[iloc] << "\t" << uexact << std::endl;
    }
    outfile.close();
    std::cout << "Mean Squared Error is " << mse / nlocal << std::endl;
  }
  // properly close MPI processes
  MPI_Finalize();
}
