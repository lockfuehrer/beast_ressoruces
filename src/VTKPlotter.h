/* Simple multigrid solver on unit square, using Jacobi smoother
 * (c) 2014 Philipp Neumann, TUM I-5
 */

#ifndef _VTKPLOTTER_H_
#define _VTKPLOTTER_H_

#include "Definitions.h"
#include <string>
#include <iostream>
#include <sstream>
#include <fstream>

// vtk plotting.
class VTKPlotter
{
public:
  VTKPlotter() {}
  ~VTKPlotter() {}

  /**
     * field         - data field
     * nx            - number of inner grid points in x-direction
     * ny            - number of inner grid points in y-direction
     * filename      - name of output file (should have .vtk suffix)
     * fieldName     - name of the field to be written
     * writeBoundary - if true, all grid point data are written to the file.
     *                 If false, only inner grid points are processed.
     */
  void writeFieldData(const FLOAT *const field, const unsigned int nx,
                      const unsigned int ny, std::string filename,
                      std::string fieldName = "values", bool writeBoundary = true) const
  {
    const FLOAT meshsizeX = 1.0 / (nx + 1);
    const FLOAT meshsizeY = 1.0 / (ny + 1);

    std::stringstream coords;
    std::stringstream fieldValues;

    extractData(field, writeBoundary, coords, fieldValues, nx, ny, meshsizeX, meshsizeY);
    writeToFile(filename, fieldName, coords, fieldValues, writeBoundary, nx, ny);
  }

private:
  // extracts data from field and writes them to stringstreams
  void extractData(const FLOAT *const field, bool writeBoundary,
                   std::stringstream &coords, std::stringstream &fieldValues,
                   const unsigned int nx, const unsigned int ny,
                   const FLOAT meshsizeX, const FLOAT meshsizeY) const
  {
    const unsigned int leftBoundary = (const unsigned int)(!writeBoundary);
    const unsigned int rightBoundary = nx + 2 - leftBoundary;
    const unsigned int lowerBoundary = leftBoundary;
    const unsigned int upperBoundary = ny + 2 - lowerBoundary;

    for (unsigned int y = lowerBoundary; y < upperBoundary; y++)
    {
      for (unsigned int x = leftBoundary; x < rightBoundary; x++)
      {
        coords << x * meshsizeX << " " << y * meshsizeY << " 0.0" << std::endl;
        fieldValues << field[x + y * (nx + 2)] << std::endl;
      }
    }
  }

  // takes data from stringstreams and writes them to a file
  void writeToFile(std::string filename, std::string fieldName,
                   std::stringstream &coords, std::stringstream &fieldValues,
                   const bool writeBoundary, const unsigned int nx, const unsigned int ny) const
  {
    std::ofstream file(filename.c_str());
    if (!file.is_open())
    {
      std::cout << "ERROR VTKPlotter::writeToFile: Could not open file " << filename << "!" << std::endl;
      return;
    }
    const unsigned int NX = nx + 2 * ((const unsigned int)writeBoundary);
    const unsigned int NY = ny + 2 * ((const unsigned int)writeBoundary);

    file << "# vtk DataFile Version 2.0" << std::endl
         << "Philipps MG code" << std::endl
         << "ASCII" << std::endl;
    file << std::endl
         << "DATASET STRUCTURED_GRID" << std::endl
         << "DIMENSIONS " << NX << " " << NY << " 1" << std::endl;
    file << "POINTS " << NX * NY << " float" << std::endl;
    file << coords.str() << std::endl
         << std::endl
         << "POINT_DATA " << NX * NY << std::endl;
    file << "SCALARS " << fieldName << " float 1" << std::endl
         << "LOOKUP_TABLE default" << std::endl;
    file << fieldValues.str() << std::endl
         << std::endl;

    file.close();
  }
};
#endif // _VTKPLOTTER_H_
