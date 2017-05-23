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
  cdef cppclass Matrix[T,nRow,nCol]:
    Matrix()
    Matrix(const Matrix[T,nRow,nCol] &)
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
    AngleAxis(T, const Matrix[T, three, one] &)
    Matrix[T, three, three] matrix()
    AngleAxis[T] inverse()

  ctypedef AngleAxis[double] AngleAxisd

  cdef cppclass Quaternion[T]:
    Quaternion()
    Quaternion(const Quaternion[T] &)
    Quaternion(const Matrix[T, four, one] &)
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

# The wrapper is needed for templated static functions, string conversion and 
cdef extern from "eigen_wrapper.hpp":
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
  Quaterniond EigenQFromM(const Matrix3d &)
  T poly_eval[T](const Matrix[T,dynamic,one]&, const T&)
