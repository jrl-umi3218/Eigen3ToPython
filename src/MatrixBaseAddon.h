// This file is part of Eigen3ToPython.
// 
// Eigen3ToPython is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
// 
// Eigen3ToPython is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Lesser General Public License for more details.
// 
// You should have received a copy of the GNU Lesser General Public License
// along with Eigen3ToPython.  If not, see <http://www.gnu.org/licenses/>.

inline Scalar getItem(unsigned int i) const
{
  if(i >= this->size())
  {
    PyErr_SetString(PyExc_IndexError, "Container index out of range");
    return 0.;
  }
  return this->operator()(i);
}

/// return -1 on error and 0 on success to fulfill the PySequence_SetItem
/// convention
inline int setItem(unsigned int i, const Scalar& val)
{
  if(i >= this->size() || i < 0)
  {
    PyErr_SetString(PyExc_IndexError, "Container index out of range");
    return -1;
  }
  this->operator()(i) = val;
  return 0;
}

inline Scalar getItem(unsigned int i, unsigned int j) const
{
  if(i >= this->rows() || j >= this->cols())
  {
    PyErr_SetString(PyExc_IndexError, "Container index out of range");
    return 0.;
  }
  return this->operator()(i, j);
}

/// return -1 on error and 0 on success to fulfill the PySequence_SetItem
/// convention
inline int setItem(unsigned int i, unsigned int j, const Scalar& val)
{
  if(i >= this->rows() || j >= this->cols() || i < 0 || j < 0)
  {
    PyErr_SetString(PyExc_IndexError, "Container index out of range");
    return -1;
  }
  this->operator()(i, j) = val;
  return 0;
}

