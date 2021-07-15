/* Simple multigrid solver on unit square, using Jacobi smoother
 * (c) 2014 Philipp Neumann, TUM I-5
 */

#ifndef _JACOBI_H_
#define _JACOBI_H_
#include <cstdio> 
#include "Definitions.h"
// carries out a jacobi step, reading from val and writing to _writeGrid.
class Jacobi
{
public:
  Jacobi(const unsigned int nx, const unsigned int ny)
      : _nx(nx), _ny(ny), _X(getX(nx, ny)), _Y(getY(nx, ny)), _RHS(getRHS(nx, ny)) {}
  ~Jacobi() {}

  void iterate(const FLOAT *const readField, FLOAT *const writeField, const FLOAT *const rhs) const
  {
    // set pointers of 5-point stencil (only neighbour values) to very first inner grid point
    const FLOAT *readPtr_S = readField + 1;
    const FLOAT *readPtr_W = readField + (_nx + 2);
    const FLOAT *readPtr_E = readField + (_nx + 4);
    const FLOAT *readPtr_N = readField + (2 * _nx + 5);
    const FLOAT *rhsPtr = rhs + (_nx + 3);
    FLOAT *writePtr = writeField + (_nx + 3);



    //set parameters
    int tile = 128;
    int nthreads =128;
    unsigned int t=(unsigned int)((_ny+1)/tile);
    
    
    // when whe use 128 threads with stripe tiling stripes will become smaller than 1 which causes crashes
    if (t>1){ 
      //assign stripes to threads
      #pragma omp parallel for num_threads(nthreads) schedule(static)
      for (unsigned int yy= 1; yy < _ny+1;yy+=t){
        
	//adjust pos to current tile and compute limit	   
        unsigned int pos=(yy-1)*(_nx+2);
        unsigned int miny = yy+t > _ny+1?_ny+1:yy+(_ny+1)/tile;
        
	for (unsigned int y = yy; y<miny; y++)
        {
          for (unsigned int x = 1; x < _nx + 1; x++)
          {
            // do Jacobi update and write to writePtr
            writePtr[pos] = _RHS * rhsPtr[pos];
            writePtr[pos] += _X * (readPtr_W[pos] + readPtr_E[pos]);
            writePtr[pos] += _Y * (readPtr_S[pos] + readPtr_N[pos]);
            // update pos along x-axis
            pos++;
          }
          // update pos along y-axis; therefore just jump over the two boundary values
          pos += 2;
        }
      }
    }
//alternative soloution either sweeps or original
  else{
   
    unsigned int pos = 0;
    for (unsigned int y = 1; y < _ny + 1; y++)
    {
      for (unsigned int x = 1; x < _nx + 1; x++)
      {
        // do Jacobi update and write to writePtr
        writePtr[pos] = _RHS * rhsPtr[pos];
        writePtr[pos] += _X * (readPtr_W[pos] + readPtr_E[pos]);
        writePtr[pos] += _Y * (readPtr_S[pos] + readPtr_N[pos]);

        // update pos along x-axis
        pos++;
      }

      // update pos along y-axis; therefore just jump over the two boundary values
      pos += 2;
    }
  }
}

private:
  // returns the prefactor for the Jacobi stencil in x-direction
  FLOAT getX(const unsigned int nx, const unsigned int ny) const
  {
    const FLOAT hx = 1.0 / (nx + 1);
    const FLOAT hy = 1.0 / (ny + 1);
    return hy * hy / (2.0 * (hx * hx + hy * hy));
  }
  // returns the prefactor for the Jacobi stencil in y-direction
  FLOAT getY(const unsigned int nx, const unsigned int ny) const
  {
    const FLOAT hx = 1.0 / (nx + 1);
    const FLOAT hy = 1.0 / (ny + 1);
    return hx * hx / (2.0 * (hx * hx + hy * hy));
  }
  // returns the prefactor for the right hand side in Jacobi computation
  FLOAT getRHS(const unsigned int nx, const unsigned int ny) const
  {
    const FLOAT hx = 1.0 / (nx + 1);
    const FLOAT hy = 1.0 / (ny + 1);
    return -1.0 / (2.0 / hx / hx + 2.0 / hy / hy);
  }

  // number of inner grid points
  const unsigned int _nx;
  const unsigned int _ny;
  // prefactors in Jacobi computations
  const FLOAT _X;
  const FLOAT _Y;
  const FLOAT _RHS;
};

#endif // _JACOBI_H_
