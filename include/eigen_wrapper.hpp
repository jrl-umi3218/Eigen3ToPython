#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <unsupported/Eigen/Polynomials>
#include <sstream>

namespace Eigen
{

typedef Matrix<double, 6, 1> Vector6d;
typedef Matrix<double, 6, 6> Matrix6d;

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

Eigen::Quaterniond EigenQFromM(const Eigen::Matrix3d & m)
{
  return Eigen::Quaterniond(m);
}

template<typename T>
T poly_eval(const Eigen::Matrix<T, -1, 1> & m, const T & x)
{
  return Eigen::poly_eval(m, x);
}
