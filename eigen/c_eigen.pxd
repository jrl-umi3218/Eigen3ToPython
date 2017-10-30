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
from libcpp cimport bool

cdef extern from *:
  ctypedef int dynamic "-1"
  ctypedef int one "1"
  ctypedef int two "2"
  ctypedef int three "3"
  ctypedef int four "4"
  ctypedef int six "6"

cdef extern from "<Eigen/Dense>" namespace "Eigen":
  cdef cppclass Map[M]:
    Map(double*, int, int)

  cdef cppclass Matrix[T,nRow,nCol]:
    Matrix()
    Matrix(const Matrix[T,nRow,nCol] &)
    Matrix(const Map[Matrix[T,nRow,nCol]]&)
    T& operator[](int)
    Matrix[T,nRow,nCol] operator+(const Matrix[T,nRow,nCol] &)
    Matrix[T,nRow,nCol] operator-()
    Matrix[T,nRow,nCol] operator-(const Matrix[T,nRow,nCol] &)
    # Matrix x Scalar
    Matrix[T,nRow,nCol] scalar_mul "operator*"(const T&)
    # Matrix / Scalar
    Matrix[T,nRow,nCol] scalar_div "operator/"(const T&)
    bool operator==(const Matrix[T,nRow,nCol]&)
    bool operator!=(const Matrix[T,nRow,nCol]&)
    T& operator()(int)
    T& operator()(int,int)
    Matrix[T,dynamic,dynamic] block(int, int, int, int)
    Matrix[T,nRow,nCol] inverse()

    T& coeff(int, int)
    int cols()
    Matrix[T,nRow,nCol] cross(const Matrix[T,nRow,nCol] &)
    T dot(const Matrix[T,nRow,nCol] &)
    T norm()
    void normalize()
    Matrix[T,nRow,nCol] normalized()
    int rows()
    int size()
    T squaredNorm()
    Matrix[T,nCol,nRow] transpose()
    void setZero()
    Matrix[T,nRow,nCol] Zero()
    T* data()

  ctypedef Matrix[double, two, one] Vector2d
  ctypedef Matrix[double, three, one] Vector3d
  ctypedef Matrix[double, four, one] Vector4d
  ctypedef Matrix[double, six, one] Vector6d
  ctypedef Matrix[double, dynamic, one] VectorXd

  ctypedef Matrix[double, two, two] Matrix2d
  ctypedef Matrix[double, three, three] Matrix3d
  ctypedef Matrix[double, four, four] Matrix4d
  ctypedef Matrix[double, six, six] Matrix6d
  ctypedef Matrix[double, dynamic, dynamic] MatrixXd

  cdef cppclass AngleAxis[T]:
    AngleAxis()
    AngleAxis(const AngleAxis[T] &)
    AngleAxis(T, const Matrix[T, three, one] &)
    AngleAxis(const Matrix[T, three, three] &)
    Matrix[T, three, three] matrix()
    Matrix[T,three,three] toRotationMatrix()
    AngleAxis[T] inverse()
    T angle()
    Matrix[T, three, one] axis()
 #   Quaternion[T] operator*(const AngleAxis[T]&)
 #   Quaternion[T] operator*(const Quaternion[T]&)

  ctypedef AngleAxis[double] AngleAxisd

  cdef cppclass Quaternion[T]:
    Quaternion()
    Quaternion(const Quaternion[T] &)
    Quaternion(const Matrix[T, four, one] &)
    # Quaternion(const Matrix[T, three, three] &)
    Quaternion(const AngleAxis[T] &)
    Quaternion(T,T,T,T)
    T x()
    T y()
    T z()
    T w()
    Matrix[T,three,one] vec()
    Matrix[T,four,one] coeffs()
    void setIdentity()
    void setFromTwoVectors(const Vector3d &, const Vector3d &)
    T angularDistance(const Quaternion[T] &)
    Quaternion[T] conjugate()
    Quaternion[T] operator*(const Quaternion[T]&)
    T dot(const Quaternion[T] &)
    Quaternion[T] inverse()
    bool isApprox(const Quaternion[T] &)
    bool isApprox(const Quaternion[T] &, double)
    Matrix[T,three,three] matrix()
    Matrix[T,three,three] toRotationMatrix()
    T norm()
    void normalize()
    Quaternion[T] normalized()
    Quaternion[T] slerp(double, const Quaternion[T] &)
    T squaredNorm()

  ctypedef Quaternion[double] Quaterniond
