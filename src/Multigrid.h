/* Simple multigrid solver on unit square, using Jacobi smoother
 * (c) 2014 Philipp Neumann, TUM I-5
 */

#ifndef _MULTIGRID_H_
#define _MULTIGRID_H_

#include "Jacobi.h"
#include "SetBoundary.h"
#include "Restriction.h"
#include "Prolongation.h"
#include <cstdio>
#include <iostream>
#include <cstdlib>

/* Multigrid algorithm.
 *
 * The Multigrid class is defined recursively: for each grid level,
 * there is one multigrid object which contains the multigrid object
 * for the next grid level (except for the coarsest level where the
 * recursion is stopped). Besides, the coarse grid buffers for function
 * values and right hand side are stored in it, as well as the functional
 * objects for smoothing (Jacobi), restriction (Restriction) and
 * prolongation (Prolongation).
 */
class Multigrid
{
public:
  // - recursion should be stopped for inner grid points <= 2.
  // - _callRecursion must be initialised as very first one,
  //    as we re-use it in the constructor.
  // - nx/yCourseGrid must be initialised before the coarseGrid-fields are created
  Multigrid(const unsigned int nxFine, const unsigned int nyFine)
      : _callRecursion((nxFine > 2) && (nyFine > 2)),
        _nxFineGrid(nxFine), _nyFineGrid(nyFine),
        _nxCoarseGrid((nxFine - 1) / 2), _nyCoarseGrid((nyFine - 1) / 2),
        _coarseGridValues1(_callRecursion ? new FLOAT[(_nxCoarseGrid + 2) * (_nyCoarseGrid + 2)] : NULL),
        _coarseGridValues2(_callRecursion ? new FLOAT[(_nxCoarseGrid + 2) * (_nyCoarseGrid + 2)] : NULL),
        _coarseGridRhs(_callRecursion ? new FLOAT[(_nxCoarseGrid + 2) * (_nyCoarseGrid + 2)] : NULL),
        _multigrid(_callRecursion ? new Multigrid(_nxCoarseGrid, _nyCoarseGrid) : NULL),
        _jacobi(nxFine, nyFine),
        _restriction(_nxCoarseGrid, _nyCoarseGrid),
        _prolongation(_nxCoarseGrid, _nyCoarseGrid)
  {
    // do error checks and initialise data
    if (_callRecursion)
    {
      if (_coarseGridValues1 == NULL)
      {
        std::cout << "ERROR Multigrid(): _coarseGridValues1==NULL!" << std::endl;
        exit(EXIT_FAILURE);
      }
      if (_coarseGridValues2 == NULL)
      {
        std::cout << "ERROR Multigrid(): _coarseGridValues2==NULL!" << std::endl;
        exit(EXIT_FAILURE);
      }
      if (_coarseGridRhs == NULL)
      {
        std::cout << "ERROR Multigrid(): _coarseGridRhs    ==NULL!" << std::endl;
        exit(EXIT_FAILURE);
      }
      if (_multigrid == NULL)
      {
        std::cout << "ERROR Multigrid(): _multigrid        ==NULL!" << std::endl;
        exit(EXIT_FAILURE);
      }

      // initialise fields
      for (unsigned int i = 0; i < (_nxCoarseGrid + 2) * (_nyCoarseGrid + 2); i++)
      {
        _coarseGridValues1[i] = 0.0;
        _coarseGridValues2[i] = 0.0;
        _coarseGridRhs[i] = 0.0;
      }
    }
  }

  ~Multigrid()
  {
    if (_callRecursion)
    {
      delete[] _coarseGridValues1;
      delete[] _coarseGridValues2;
      delete[] _coarseGridRhs;
      delete _multigrid;
    }
  }

  FLOAT *solve(
      const unsigned int preSmoothing, const unsigned int postSmoothing,
      const unsigned int jacobiStepsCoarsestLevel,
      FLOAT *const fineGridValues1, FLOAT *const fineGridValues2,
      const FLOAT *const fineGridRhs)
  {
    FLOAT *readField = fineGridValues1;
    FLOAT *writeField = fineGridValues2;

    // if we are on the coarsest level, just do Jacobi iterations and return (do swapping of read- and write fields correspondingly)
    if (!_callRecursion)
    {
      for (unsigned int i = 0; i < jacobiStepsCoarsestLevel; i++)
      {
        _jacobi.iterate(readField, writeField, fineGridRhs);
        FLOAT *swap = readField;
        readField = writeField;
        writeField = swap;
      }
      return readField;
    }

    // otherwise: do multigrid scheme --------------------------------------

    // pre-smoothing
    for (unsigned int i = 0; i < preSmoothing; i++)
    {
      _jacobi.iterate(readField, writeField, fineGridRhs);
      FLOAT *swap = readField;
      readField = writeField;
      writeField = swap;
    }
    // restriction done by injection
    _restriction.injection(readField, fineGridRhs, _coarseGridRhs, _coarseGridValues1, _coarseGridValues2);
    // do recursive multigrid solving procedure; the returned pointer points to _coarseGridValues1 or _coarseGridValues2,
    // depending on which of those fields holds the current solution
    FLOAT *coarseSolutionField = _multigrid->solve(
        preSmoothing, postSmoothing, jacobiStepsCoarsestLevel,
        _coarseGridValues1, _coarseGridValues2, _coarseGridRhs);
    // prolongate error values to the fine grid
    _prolongation.interpolation(coarseSolutionField, readField);

    // post-smoothing
    for (unsigned int i = 0; i < postSmoothing; i++)
    {
      _jacobi.iterate(readField, writeField, fineGridRhs);
      FLOAT *swap = readField;
      readField = writeField;
      writeField = swap;
    }

    // return pointer to current solution
    return readField;
  }

private:
  const bool _callRecursion;
  const unsigned int _nxFineGrid;
  const unsigned int _nyFineGrid;
  const unsigned int _nxCoarseGrid;
  const unsigned int _nyCoarseGrid;
  FLOAT *_coarseGridValues1;
  FLOAT *_coarseGridValues2;
  FLOAT *_coarseGridRhs;

  Multigrid *_multigrid;
  const Jacobi _jacobi;
  const Restriction _restriction;
  const Prolongation _prolongation;
};
#endif // _MULTIGRID_H_
