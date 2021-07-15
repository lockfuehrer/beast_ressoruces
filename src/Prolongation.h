/* Simple multigrid solver on unit square, using Jacobi smoother
 * (c) 2014 Philipp Neumann, TUM I-5
 */

#ifndef _PROLONGATION_H_
#define _PROLONGATION_H_

#include "Definitions.h"
#include <iostream>

// standard prolongation algorithm.
class Prolongation
{
public:
  Prolongation(const unsigned int nxCoarseGrid, const unsigned int nyCoarseGrid)
      : _nxCoarseGrid(nxCoarseGrid), _nyCoarseGrid(nyCoarseGrid),
        _nxFineGrid(2 * nxCoarseGrid + 1),
        _nyFineGrid(2 * nyCoarseGrid + 1)
  {}

  ~Prolongation() {}

  void interpolation(const FLOAT *const coarseGrid, FLOAT *const fineGrid) const
  {
    // define pointers to all neighbouring grid points centered around the
    // inner fine grid point at (2,2)
    FLOAT *fine_SW = fineGrid + (_nxFineGrid + 3);
    FLOAT *fine_S = fine_SW + 1;
    FLOAT *fine_SE = fine_S + 1;
    FLOAT *fine_W = fine_SW + (_nxFineGrid + 2);
    FLOAT *fine_C = fine_W + 1;
    FLOAT *fine_E = fine_C + 1;
    FLOAT *fine_NW = fine_W + (_nxFineGrid + 2);
    FLOAT *fine_N = fine_NW + 1;
    FLOAT *fine_NE = fine_N + 1;

    // define pointer to coarse grid point at (1,1)
    const FLOAT *coarse = coarseGrid + (_nxCoarseGrid + 3);

    // use pos-values for direct access to fine and coarse grid point data
    unsigned int finePos = 0;
    unsigned int coarsePos = 0;
    for (unsigned int y = 1; y < _nyCoarseGrid + 1; y++)
    {
      for (unsigned int x = 1; x < _nxCoarseGrid + 1; x++)
      {
        // the points on the "diagonal" contribute to 0.25 to the error
        // interpolation (since 4 values are used for bilinear interpolation of the mid-point).
        // for the central point (fine_C), we directly use the corresponding value.
        // for the other points, the bi-linear interpolation reduces to averaging
        // the two closest values (e.g. South- and North-values)
        fine_SW[finePos] += 0.25 * coarse[coarsePos];
        fine_S[finePos] += 0.5 * coarse[coarsePos];
        fine_SE[finePos] += 0.25 * coarse[coarsePos];
        fine_W[finePos] += 0.5 * coarse[coarsePos];
        fine_C[finePos] += coarse[coarsePos];
        fine_E[finePos] += 0.5 * coarse[coarsePos];
        fine_NW[finePos] += 0.25 * coarse[coarsePos];
        fine_N[finePos] += 0.5 * coarse[coarsePos];
        fine_NE[finePos] += 0.25 * coarse[coarsePos];

        // go to next coarse grid point to the right
        coarsePos++;
        // go to second fine grid point to the right
        finePos += 2;
      }

      // for index-updates, see Restriction.h
      coarsePos += 2;
      finePos += (_nxFineGrid + 5);
    }
  }

private:
  // number of inner coarse grid points
  const unsigned int _nxCoarseGrid;
  const unsigned int _nyCoarseGrid;
  // number of inner fine grid points
  const unsigned int _nxFineGrid;
  const unsigned int _nyFineGrid;
};

#endif // _PROLONGATION_H_
