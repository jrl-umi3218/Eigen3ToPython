#
# Copyright 2012-2019 CNRS-UM LIRMM, CNRS-AIST JRL
#

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
  string EigenVersion()
