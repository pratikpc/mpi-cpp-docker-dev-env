//=============================================================================
// Solution of unsteady 1D diffusion equation withinin the domain [0,1] using
// finite difference method.
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
// 1. There is a bug in this code: locate and fix
// 2. write iteration count and norm in a file, and plot convergence
// 3. Use dynamic memory allocation and deallocation
//=============================================================================

#include <iostream>
#include <fstream>
#include <cmath> // value of PI is defined as M_PI

int main()
{
  int n = 20;                       // number of nodes in the mesh
  double dt = 0.001;                // time step
  double xx[n], uold[n], unew[n];   // xx--locaiton of nodes, uold--u^{n}, unew--u^{n+1}
  double g[n], f[n];
  double dx = 1.0/(n-1.0);         // mesh size
  double dxinv2 = 1.0/dx/dx;       // 1.0/(dx)^2 which is used often

  // initialize
  for( int ii=0; ii<n; ii++ )
  {
    xx[ii] = ii*dx;
    uold[ii] = 0.0;
    unew[ii] = 0.0;
    f[ii] = M_PI*M_PI*sin(M_PI*xx[ii]);   // source term
  }

  double norm  = 100.0;
  bool converged = false;
  int itercount = 0;
  // --- time integation begin here
  while( not converged )
  {
    itercount++;
    for( int ii=1; ii<n-1; ii++ )
    {
      g[ii] = f[ii] + dxinv2 * ( uold[ii+1]-2.0*uold[ii]+uold[ii-1] );
      unew[ii] = uold[ii] + dt * g[ii];
    }

    // compute L_infinity norm -- to check convergence to steady state
    norm  = 0.0;
    for( int ii=0; ii<n; ii++ )
    {
      if( fabs(g[ii]) > norm )
        norm = fabs(g[ii]);
    }

    // stopping criterion
    if( norm < 1e-6 )
    {
      converged = true;
    }

    // store new values as old before moving for next time step
    for( int ii=0; ii<n; ii++ )
    {
      uold[ii] = unew[ii];
    }
  }

  // Write output of result in a file
  std::ofstream outfile;
  outfile.open("output.txt");
  double mse = 0;
  for( int ii=0; ii<n; ii++ )
  {
    double uex = sin(M_PI*xx[ii]);
	mse += pow(unew[ii] - uex, 2);
    outfile << xx[ii] << "\t" << unew[ii] <<"\t" << uex <<std::endl;
  }
  std::cout << "Mean Squared Error is " << mse/n;
  outfile.close();
}
