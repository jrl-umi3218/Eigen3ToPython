/* Copyright 2012-2017 CNRS-UM LIRMM, CNRS-AIST JRL
*
* This file is part of Eigen3ToPython.
*
* Eigen3ToPython is free software: you can redistribute it and/or modify
* it under the terms of the GNU Lesser General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* Eigen3ToPython is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
* GNU Lesser General Public License for more details.
*
* You should have received a copy of the GNU Lesser General Public License
* along with Eigen3ToPython.  If not, see <http://www.gnu.org/licenses/>.
*/

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <unsupported/Eigen/Polynomials>
#include <sstream>

namespace Eigen
{

typedef Matrix<double, 6, 1> Vector6d;
typedef Matrix<double, 6, 6> Matrix6d;
typedef Stride<Dynamic, Dynamic> DynStride;

template <typename Mtype>
#if EIGEN_VERSION_AT_LEAST(3, 3, 0)
using DynMap = Map<Mtype, AlignmentType::Aligned16, DynStride>;
#elif EIGEN_VERSION_AT_LEAST(3, 2, 90)
using DynMap = Map<Mtype, Aligned, DynStride>;
#elif EIGEN_VERSION_AT_LEAST(3, 2, 9)
using DynMap = Map<Mtype, AlignmentType::Aligned, DynStride>;
#else
using DynMap = Map<Mtype, Aligned, DynStride>;
#endif

}

template<typename T, int r, int c>
Eigen::Matrix<T, r, c> MatrixFromDynMap(const Eigen::DynMap<Eigen::Matrix<T, r, c>> & map)
{
  return Eigen::Matrix<T, r, c>(map);
}

template<typename T, int r, int c, int opr, int opc>
Eigen::Matrix<T, r, opc> EigenMul(const Eigen::Matrix<T, r, c> & lhs, const Eigen::Matrix<T, opr, opc> & rhs)
{
  return lhs*rhs;
}

template<typename T, int r, int c, int opr, int opc>
Eigen::Matrix<T, opr, opc> EigenFixedMul(const Eigen::Matrix<T, r, c> & lhs, const Eigen::Matrix<T, opr, opc> & rhs)
{
  return lhs*rhs;
}

Eigen::Vector3d EigenEulerAngles(const Eigen::Matrix3d & m, int a0, int a1, int a2)
{
  return m.eulerAngles(a0, a1, a2);
}

template<typename T, int r, int c>
Eigen::Matrix<T, r, c> EigenZero()
{
  return Eigen::Matrix<T,r,c>::Zero();
}

template<typename T, int r, int c>
Eigen::Matrix<T, r, c> EigenZero(int row)
{
  return Eigen::Matrix<T,r,c>::Zero(row);
}

template<typename T, int r, int c>
Eigen::Matrix<T, r, c> EigenZero(int row, int col)
{
  return Eigen::Matrix<T,r,c>::Zero(row,col);
}

template<typename T, int r, int c>
Eigen::Matrix<T, r, c> EigenRandom()
{
  return Eigen::Matrix<T,r,c>::Random();
}

template<typename T, int r, int c>
Eigen::Matrix<T, r, c> EigenRandom(int row)
{
  return Eigen::Matrix<T,r,c>::Random(row);
}

template<typename T, int r, int c>
Eigen::Matrix<T, r, c> EigenRandom(int row, int col)
{
  return Eigen::Matrix<T,r,c>::Random(row,col);
}

template<typename T, int r, int c>
Eigen::Matrix<T, r, c> EigenIdentity()
{
  return Eigen::Matrix<T,r,c>::Identity();
}

template<typename T, int r, int c>
Eigen::Matrix<T, r, c> EigenIdentity(int row, int col)
{
  return Eigen::Matrix<T,r,c>::Identity(row, col);
}

template<typename T, int r, int c>
void EigenSetValue(Eigen::Matrix<T,r,c> & m, int row, int col, const T & v)
{
  m(row,col) = v;
}

template<typename T, int r, int c>
std::string toString(const Eigen::Matrix<T,r,c> &m)
{
  std::stringstream ss;
  ss << m;
  return ss.str();
}

template<typename T>
std::string QtoString(const Eigen::Quaternion<T> &q)
{
  std::stringstream ss;
  ss << q.coeffs().transpose();
  return ss.str();
}

template<typename T>
std::string AAtoString(const Eigen::AngleAxis<T> &aa)
{
  std::stringstream ss;
  ss << "Angle: " << aa.angle() << ", Axis: " << aa.axis().transpose();
  return ss.str();
}

template<typename T>
T poly_eval(const Eigen::Matrix<T, -1, 1> & m, const T & x)
{
  return Eigen::poly_eval(m, x);
}

template<typename T>
Eigen::AngleAxis<T> EigenAAFromQ(const Eigen::Quaternion<T> & q)
{
  return Eigen::AngleAxis<T>(q);
}

template<typename T>
Eigen::Quaternion<T> QuaternionFromM3(const Eigen::Matrix<T, 3, 3> & m)
{
  return Eigen::Quaternion<T>(m);
}
