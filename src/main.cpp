/********************************************************************
 * Simple multigrid solver on unit square, using Jacobi smoother
 * (c) 2014 Philipp Neumann, TUM I-5
 *
 * With explicit permission to use in the BEAST lab WS 20/21
 *******************************************************************/

#include "Multigrid.h"
#include "ComputeError.h"
#include <chrono>

int main(int argc, char *argv[])
{
  if (argc != 5)
  {
    std::cout << "ERROR main: Please call program by"
              << std::endl
              << " ./multigrid preSmoothingSteps postSmoothingSteps numberGridLevels numberMultigridIterations"
              << std::endl;
    return -1;
  }

  const unsigned int preSmoothing = (unsigned int)atoi(argv[1]);
  const unsigned int postSmoothing = (unsigned int)atoi(argv[2]);
  const unsigned int jacobiStepsCoarsestLevel = 1;
  const unsigned int levels = (unsigned int)atoi(argv[3]);
  const unsigned int multigridCycles = (unsigned int)atoi(argv[4]);

  // determine resolution
  const unsigned int nx = ((unsigned int)(1 << levels)) - 1;

  // initialise fields and set boundary values
  FLOAT *field1 = new FLOAT[(nx + 2) * (nx + 2)];
  if (field1 == NULL)
  {
    std::cout << "ERROR field1==NULL!" << std::endl;
    return -1;
  }
  FLOAT *field2 = new FLOAT[(nx + 2) * (nx + 2)];
  if (field2 == NULL)
  {
    std::cout << "ERROR field2==NULL!" << std::endl;
    return -1;
  }
  FLOAT *rhs = new FLOAT[(nx + 2) * (nx + 2)];
  if (rhs == NULL)
  {
    std::cout << "ERROR rhs   ==NULL!" << std::endl;
    return -1;
  }
  //implement first touchh policy on field 1 & 2
  /*
  int tile = 128;
  unsigned int pos = 0;
  
  // set pointers of 5-point stencil (only neighbour values) to very first inner grid point
  const FLOAT *readPtr_S = field2 + 1  
  const FLOAT *readPtr_W = field2 + (_nx + 2);
  const FLOAT *readPtr_E = field2 + (_nx + 4);
  const FLOAT *readPtr_N = field2 + (2 * _nx + 5);
  FLOAT *writePtr = field1 + (_nx + 3);
  #pragma omp parallel for num_threads(tile) schedule(static, 10)
  for (unsigned int yy= 0; yy < _ny+1;yy+= (unsigned int)((_ny+1)/tile)){
    pos=yy*(_nx+2) ;
    unsigned int miny = yy+(_ny+1)/tile > _ny+1?_ny+1:yy+_ny/tile;
    for (unsigned int y = yy+1; y<miny; y++)
    {
      for (unsigned int x = 1; x < _nx + 1; x++)
      {
        writePtr[pos] = 0;
        readPtr_W[pos] = 0; 
	readPtr_E[pos] = 0;
        readPtr_S[pos] = 0;
	readPtr_N[pos] = 0;

        // update pos along x-axis
        pos++;
      }

      // update pos along y-axis; therefore just jump over the two boundary values
      pos += 2;
    }
  }
  */
  const SetBoundary setBoundary(nx, nx);
  for (unsigned int i = 0; i < (nx + 2) * (nx + 2); i++)
  {
    field1[i] = 0.0;
    field2[i] = 0.0;
    rhs[i] = 0.0;
  }
  setBoundary.iterate(field1);
  setBoundary.iterate(field2);

  Multigrid multigrid(nx, nx);
  FLOAT *currentSolution = field1;

  ComputeError computeError(nx, nx);
  const VTKPlotter vtkPlotter;
  
  auto t0 = std::chrono::high_resolution_clock::now();
  for (unsigned int i = 0; i < multigridCycles; i++)
  {
    // always work on the correct field (either field1 or field2);
    // this switch is required due to the Jacobi method
    if (currentSolution == field1)
    {
      currentSolution = multigrid.solve(
          preSmoothing, postSmoothing, jacobiStepsCoarsestLevel,
          field1, field2, rhs);
    }
    else
    {
      currentSolution = multigrid.solve(
          preSmoothing, postSmoothing, jacobiStepsCoarsestLevel,
          field2, field1, rhs);
    }

    computeError.computePointwiseError(currentSolution);
    std::cout << "Iteration " << i << ", Max-error: " << computeError.getMaxError() << std::endl;
  }
  auto t1 = std::chrono::high_resolution_clock::now();
  using dsec = std::chrono::duration<double>;
  double dur = std::chrono::duration_cast<dsec>(t1-t0).count();
  printf("time: %f \n", dur);
 

  // commented out: result not interesting here
  //computeError.plotPointwiseError();
  //vtkPlotter.writeFieldData(currentSolution,nx,nx,"result.vtk");

  return 0;
}
