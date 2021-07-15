/* Simple multigrid solver on unit square, using Jacobi smoother
 * (c) 2014 Philipp Neumann, TUM I-5
 */

#ifndef _SETBOUNDARY_H_
#define _SETBOUNDARY_H_

#include "Definitions.h"

// set boundary conditions for function f(x,y) = x*y defined on unit square.
class SetBoundary
{
public:
  SetBoundary(const unsigned int nx, const unsigned int ny)
      : _nx(nx), _ny(ny), _meshsize2(1.0 / (nx + 1) / (ny + 1)) {}
  ~SetBoundary() {}

  void iterate(FLOAT *const field) const
  {
    unsigned int pos = 0;

    // init lower and upper boundary
    for (unsigned int x = 0; x < _nx + 2; x++)
    {
      field[x] = 0.0;
      field[x + (_nx + 2) * (_ny + 1)] = x * (_ny + 1) * _meshsize2;
    }

    // init left and right boundary
    for (unsigned int y = 1; y < _ny + 1; y++)
    {
      field[y * (_nx + 2)] = 0.0;
      field[y * (_nx + 2) + (_nx + 1)] = (_nx + 1) * y * _meshsize2;
    }
  }

private:
  const unsigned int _nx;
  const unsigned int _ny;
  const FLOAT _meshsize2;
};

#endif // _SETBOUNDARY_H_
