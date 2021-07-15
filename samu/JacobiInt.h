/* Simple multigrid solver on unit square, using Jacobi smoother
 * (c) 2014 Philipp Neumann, TUM I-5
 */

#ifndef _JACOBI_H_
#define _JACOBI_H_

#include <stdio.h>
#include "Definitions.h"
#include <immintrin.h>
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

    // use pos to advance access through the whole grid without any expensive index computations
    unsigned int pos = 0;
    for (unsigned int y = 1; y < _ny + 1; y++)
    {
        unsigned int ub=_nx-(_nx%4);
        __m256d vrhsPtr, vreadPtr_W, vreadPtr_E, vreadPtr_N, vreadPtr_S,tmp1,tmp2, k_X,k_Y,k_RHS,vwritePtr;
        k_X=_mm256_set1_pd(_X);
        k_Y=_mm256_set1_pd(_Y);
        k_RHS=_mm256_set1_pd(_RHS);
        for(unsigned int x=0;x<ub;x+=4){
         vrhsPtr=_mm256_loadu_pd(&rhsPtr[pos]);
         vreadPtr_W=_mm256_loadu_pd(&readPtr_W[pos]);
         vreadPtr_N=_mm256_loadu_pd(&readPtr_N[pos]);
         vreadPtr_E=_mm256_loadu_pd(&readPtr_E[pos]);
         vreadPtr_S=_mm256_loadu_pd(&readPtr_S[pos]);
         vwritePtr=_mm256_mul_pd(k_RHS,vrhsPtr);
         tmp1=_mm256_add_pd(vreadPtr_W,vreadPtr_E);
         tmp2=_mm256_mul_pd(k_X,tmp1);
         vwritePtr=_mm256_add_pd(tmp2,vwritePtr);
         tmp1=_mm256_add_pd(vreadPtr_S,vreadPtr_N);
         tmp2=_mm256_mul_pd(k_Y,tmp1);
         vwritePtr=_mm256_add_pd(tmp2,vwritePtr);
         _mm256_storeu_pd(&writePtr[pos],vwritePtr);
         pos+=4;
         //printf("%d\n",pos);
        }
        
        for(unsigned int x=ub;x<_nx;x++){      
       //for (unsigned int x = 1; x < _nx + 1; x++)
       //{
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

