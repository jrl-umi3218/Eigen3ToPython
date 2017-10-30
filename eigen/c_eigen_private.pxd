# Copyright 2012-2017 CNRS-UM LIRMM, CNRS-AIST JRL
#
# This file is part of Eigen3ToPython.
#
# Eigen3ToPython is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Eigen3ToPython is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with Eigen3ToPython.  If not, see <http://www.gnu.org/licenses/>.

from libcpp.string cimport string

from c_eigen cimport *

# The wrapper is needed for templated static functions, string conversion and disambiguating some operators
cdef extern from "eigen_wrapper.hpp" namespace "Eigen":
  cdef cppclass DynStride:
    DynStride(int, int)

  cdef cppclass DynMap[M]:
    DynMap(double*, int, int, DynStride)

cdef extern from "eigen_wrapper.hpp":
  Matrix[T,nRow,nCol] MatrixFromDynMap[T,nRow,nCol](const DynMap[Matrix[T,nRow,nCol]]&)

  Matrix[T,nRow,opCol] EigenMul[T,nRow,nCol,opRow,opCol](const Matrix[T,nRow,nCol]&, const Matrix[T,opRow,opCol]&)
  Matrix[T,opRow,opCol] EigenFixedMul[T,nRow,nCol,opRow,opCol](const Matrix[T,nRow,nCol]&, const Matrix[T,opRow,opCol]&)
  Vector3d EigenEulerAngles(const Matrix3d &, int, int, int)
  Matrix[T,nRow,nCol] EigenZero[T,nRow,nCol]()
  Matrix[T,nRow,nCol] EigenZero[T,nRow,nCol](int)
  Matrix[T,nRow,nCol] EigenZero[T,nRow,nCol](int, int)
  Matrix[T,nRow,nCol] EigenRandom[T,nRow,nCol]()
  Matrix[T,nRow,nCol] EigenRandom[T,nRow,nCol](int)
  Matrix[T,nRow,nCol] EigenRandom[T,nRow,nCol](int, int)
  Matrix[T,nRow,nCol] EigenIdentity[T,nRow,nCol]()
  Matrix[T,nRow,nCol] EigenIdentity[T,nRow,nCol](int,int)
  void EigenSetValue[T,nRow,nCol](const Matrix[T,nRow,nCol]&, int, int, const T&)
  string toString[T,nRow,nCol](const Matrix[T,nRow,nCol]&)
  string AAtoString[T](const AngleAxis[T]&)
  string QtoString[T](const Quaternion[T]&)
  T poly_eval[T](const Matrix[T,dynamic,one]&, const T&)
  AngleAxis[T] EigenAAFromQ[T](const Quaternion[T] &)
  Quaternion[T] QuaternionFromM3[T](const Matrix[T, three, three]&)
