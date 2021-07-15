/* Simple multigrid solver on unit square, using Jacobi smoother
 * (c) 2014 Philipp Neumann, TUM I-5
 */

#ifndef _RESTRICTION_H_
#define _RESTRICTION_H_

#include "Definitions.h"

// carries out restriction by injection; we assume looping over the coarse
// grid and reading from the fine grid
class Restriction
{
public:
  Restriction(const unsigned int nxCoarseGrid, const unsigned int nyCoarseGrid)
      : _nxCoarseGrid(nxCoarseGrid), _nyCoarseGrid(nyCoarseGrid),
        _nxFineGrid(2 * nxCoarseGrid + 1),
        _nyFineGrid(2 * nyCoarseGrid + 1),
        _hxFineGrid2((1.0 / (2.0 * (nxCoarseGrid + 1))) * (1.0 / (2.0 * (nxCoarseGrid + 1)))),
        _hyFineGrid2((1.0 / (2.0 * (nyCoarseGrid + 1))) * (1.0 / (2.0 * (nyCoarseGrid + 1))))
  {}

  ~Restriction() {}

  void injection(const FLOAT *const fineGrid, const FLOAT *const rhsFineGrid, FLOAT *const rhsCoarseGrid, FLOAT *const coarseGridValues1, FLOAT *const coarseGridValues2) const
  {
    // set pointers for 4-point stencil to very first inner point on fine
    // grid which is to be restricted (-> coordinates (2,2))
    const FLOAT *fineVal_C = fineGrid + (2 * (_nxFineGrid + 2) + 2);
    const FLOAT *fineVal_S = fineVal_C - (_nxFineGrid + 2);
    const FLOAT *fineVal_W = fineVal_C - 1;
    const FLOAT *fineVal_E = fineVal_C + 1;
    const FLOAT *fineVal_N = fineVal_C + (_nxFineGrid + 2);
    const FLOAT *fineRhs = rhsFineGrid + (2 * (_nxFineGrid + 2) + 2);

    // set pointer to coarse grid-rhs to first inner grid point (-> coordinates (1,1))
    FLOAT *coarseRhs = rhsCoarseGrid + (_nxCoarseGrid + 3);
    FLOAT *coarseVal1 = coarseGridValues1 + (_nxCoarseGrid + 3);
    FLOAT *coarseVal2 = coarseGridValues2 + (_nxCoarseGrid + 3);

    // use pos-values to always access the right values on fine and coarse grid
    // and avoid any index computations (better directly update pointers?)
    unsigned int coarsePos = 0;
    unsigned int finePos = 0;
    for (unsigned int y = 1; y < _nyCoarseGrid + 1; y++)
    {
      for (unsigned int x = 1; x < _nxCoarseGrid + 1; x++)
      {
        // do injection of error: therefore, compute residual on fine grid
        // and write the value into the coarse grid right hand side
        coarseRhs[coarsePos] = fineRhs[finePos];
        coarseRhs[coarsePos] -= (fineVal_W[finePos] - 2.0 * fineVal_C[finePos] + fineVal_E[finePos]) / _hxFineGrid2;
        coarseRhs[coarsePos] -= (fineVal_S[finePos] - 2.0 * fineVal_C[finePos] + fineVal_N[finePos]) / _hyFineGrid2;
        coarseVal1[coarsePos] = 0.0;
        coarseVal2[coarsePos] = 0.0;

        // update pos-values along x-axis
        coarsePos++;
        finePos += 2;
      }

      // update pos-values across y-axis:
      // for "coarsePos", the loop for the x-direction will set coarsePos
      // right onto the last point located on the right boundary.
      //    -> To advance to the first inner grid point, we thus need to add 2.
      // for finePos, we will also end on the point located on the right boundary.
      // However, we need to go the second inner grid point (+3) AND
      // jump over one line of fine grid points (+_nxFineGrid+2).
      coarsePos += 2;
      finePos += (_nxFineGrid + 5);
    }
  }

private:
  // inner grid points on coarse grid
  const unsigned int _nxCoarseGrid;
  const unsigned int _nyCoarseGrid;
  // inner grid points on fine grid
  const unsigned int _nxFineGrid;
  const unsigned int _nyFineGrid;
  // squared mesh size of fine grid
  const FLOAT _hxFineGrid2;
  const FLOAT _hyFineGrid2;
};

#endif // _RESTRICTION_H_
