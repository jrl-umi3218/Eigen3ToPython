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

inline Scalar getItem(uint i) const
{
  if(i >= this->size())
  {
    PyErr_SetString(PyExc_IndexError, "Container index out of range");
    return 0.;
  }
  return this->operator()(i);
}

inline void setItem(uint i, const Scalar& val)
{
  if(i >= this->size())
  {
    PyErr_SetString(PyExc_IndexError, "Container index out of range");
    return;
  }
  this->operator()(i) = val;
}

inline Scalar getItem(uint i, uint j) const
{
  if(i >= this->rows() || j >= this->cols())
  {
    PyErr_SetString(PyExc_IndexError, "Container index out of range");
    return 0.;
  }
  return this->operator()(i, j);
}

inline void setItem(uint i, uint j, const Scalar& val)
{
  if(i >= this->rows() || j >= this->cols())
  {
    PyErr_SetString(PyExc_IndexError, "Container index out of range");
    return;
  }
  this->operator()(i, j) = val;
}

