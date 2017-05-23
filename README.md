[![Build Status](https://travis-ci.org/jrl-umi3218/Eigen3ToPython.svg?branch=master)](https://travis-ci.org/jrl-umi3218/Eigen3ToPython)

Eigen3ToPython
======

Eigen3ToPython is a Python library that aims to make a bidirectional bridge between Numpy and Eigen3.

The goal is not to provide a full Eigen3 Python binding but to provide easy conversion between Numpy and Eigen3.

Documentation
------

This library allows to:
 * Make operations on fixed size Eigen3 matrices and vectors
 * Make operations on dynamic size Eigen3 matrices and vectors
 * Use Eigen::Quaterniond in Python
 * Use Eigen::AngleAxisd in Python
 * Convert to/from Numpy arrays (`np.array`) in a transparent manner (however, memory is not shared between both representations)

If you want more features you can open an issue or just fork the project :)

### Fixed size Eigen3 Matrix operations

```Python
import eigen as e3

# Fixed size vector constructors (Vector2d, Vector3d, Vector4d, Vector6d)
e3.Vector3d.Zero() # Eigen::Vector3d::Zero()
v3d = e3.Vector3d.Random() # Eigen::Vector3d::Random()
e3.Vector4d.UnitX() # Eigen::Vector4d::UnitX() (also Unit{Y,Z,W} when it makes sense)
e3.Vector3d() # Eigen::Vector3d::Zero() (no uninitialized values)
e3.Vector3d(v3d) # Eigen::Vector3d(Eigen::Vector3d) (copy constructor)
v4d = e3.Vector4d(1., 2., 3., 4.) # Eigen::Vector4d(double, double, double, double)

# Fixed size vector getters
v4d.x(), v4d.y(), v4d.z(), v4d.w() # Eigen::Vector4d::{x,y,z,w}()
for i in xrange(4):
  # Eigen::Vector4d::getItem(int, int)
  v4d.coeff(i,0) 
  v4d[i]
v4d.rows(), v4d.cols() # Eigen::Vector4d::{rows,cols}()
len(v4d) # Eigen::Vector4d::size()
# Slice getter
v4d[1:3] # No equivalent in C++

# Fixed size vector setters
# Eigen::Vector4d::setItem(int, int, double)
v4d.coeff(1, 0, 0.4)
v4d[1] = 0.4
# Slice setter
v4d[1:3] = [0., 1.]

# Fixed size vector operations
v4d.norm() # Eigen::Vector4d::norm()
v4d.squaredNorm() # Eigen::Vector4d::squaredNorm()
v4d.normalize() # Eigen::Vector4d::normalize()
v4d.normalized() # Eigen::Vector4d::normalized()
v4d.transpose() # Eigen::Vector4d::transpose() (return a MatrixXd)
v4d.dot(v4d)
v3d.cross(v3d) # Returns an eigen.Vector3d object

# Arithmetic operations
v4d + v4d, v4d - v4d
2.*-v4d*2., v4d/2.
v4d += v4d
v4d -= v4d

# Fixed size matrix constructors (Matrix2d, Matrix3d, Matrix6d)
e3.Matrix3d.Zero() # Eigen::Matrix3d::Zero()
e3.Matrix3d.Random() # Eigen::Matrix3d::Random()
m3d = e3.Matrix3d.Identity() # Eigen::Matrix3d::Identity()
e3.Matrix3d(m3d) # Eigen::Matrix3d(Eigen::Matrix3d) (copy constructor)
e3.Matrix3d() # Eigen::Matrix3d() (uninitialized values)

# Fixed size matrix getters
for row in xrange(3):
  for col in xrange(3):
    # Eigen::Matrix3d::getItem(int, int)
    m3d.coeff(row, col)
    m3d[row + col*3]
m3d.rows(), m3d.cols() # Eigen::Matrix3d::{rows,cols}()
len(m3d) # Eigen::Matrix3d::size()

# Fixed size matrix setters
# Eigen::Matrix4d::setItem(int, int, double)
m3d.coeff(1,2,1.)
m3d[1 + 2*3] = 1.

# Fixed size matrix operations
m3d.norm() # Eigen::Matrix3d::norm()
m3d.squaredNorm() # Eigen::Matrix3d::squaredNorm()
m3d.normalize() # Eigen::Matrix3d::normalize()
m3d.normalized() # Eigen::Matrix3d::normalized()
m3d.transpose() # Eigen::Matrix3d::transpose() (return a Matrix3d)
m3d.inverse() # Eigen::Matrix3d::inverse()
m3d.eulerAngles(0,1,2) # Eigen::Matrix3d::eulerAngles(int, int, int)

# Arithmetic operations
m3d + m3d, m3d - m3d
2.*-m3d*2., m3d/2.
m3d += m3d
m3d -= m3d

m3d*v3d # give a eigen.Vector3d
m3d*m3d # give a eigen.Matrix3d
```

### Quaterniond operations

```Python
import eigen as e3

# constructors
e3.Quaterniond() # Eigen::Quaterniond() (uninitialized values)
e3.Quaterniond(e3.Vector4d(0., 0., 0., 1.)) # Eigen::Quaterniond(Eigen::Vector4d)
quat = e3.Quaterniond(1., 0., 0., 0.) # Eigen::Quaterniond(double w, double x, double y, double z)
e3.Quaterniond(quat) # Eigen::Quaterniond(Eigen::Quaterniond) (copy constructor)
e3.Quaterniond(0.1, e3.Vector3d.UnitX()) # Eigen::Quaterniond(Eigen::AngleAxisd(double, Eigen::Vector3d));
e3.Quaterniond(e3.Matrix3d.Identity()) # Eigen::Quaterniond(Eigen::AngleAxisd(Eigen::Matrix3d))
e3.Quaterniond.Identity() # Eigen::Quaterniond::Identity()

# getters
quat.x(), quat.y(), quat.z(), quat.w() # Eigen::Quaterniond::{x,y,z,w}()
quat.vec() # Eigen::Quaterniond.vec()
quat.coeffs() # Eigen::Quaterniond.coeffs()

# setters
quat.setIdentity() # Eigen::Quaterniond::setIdentity()
# Eigen::Quaterniond::setFromTwoVectors(Eigen::Vector3d, Eigen::Vector3d)
quat.setFromTwoVectors(e3.Vector3d.UnitX(), e3.Vector3d.UnitY())

# operations
quat.angularDistance(quat) # Eigen::Quaterniond::angularDistance(Eigen::Quaterniond)
quat.conjugate() # Eigen::Quaterniond::conjugate()
quat.dot(quat) # Eigen::Quaterniond::dot(Eigen::Quaterniond)
quat.inverse() # Eigen::Quaterniond::inverse()
quat.isApprox(quat) # Eigen::Quaterniond::isApprox(Eigen::Quaternion)
quat.isApprox(quat, 1e-8) # Eigen::Quaterniond::isApprox(Eigen::Quaterniond, double)
quat.matrix() # Eigen::Quaterniond::matrix()
quat.toRotationMatrix() # Eigen::Quaterniond::toRotationMatrix()
quat.normalize() # Eigen::Quaterniond::normalize()
quat.normalized() # Eigen::Quaterniond.normalized()
quat.slerp(0.5, quat) # Eigen::Quaterniond::slerp(double, Eigen::Quaterniond)
quat.squaredNorm() # Eigen::Quaterniond::squaredNorm()

quat*quat
```

### Dynamic size Eigen3 Matrix operations
Few operations are defined for dynamic size vector/matrix.
It's recommended to convert them into Numpy matrix.

```Python
import eigen3 as e3

# constructors
m10d = e3.MatrixXd(10, 10) # Eigen::MatrixXd(double, double)
e3.MatrixXd() # Eigen::MatrixXd()
e3.MatrixXd(m10d) # Eigen::MatrixXd(Eigen::MatrixXd)

e3.MatrixXd.Identity(10,10)
e3.MatrixXd.Zero(10,10)
e3.MatrixXd.Random(10,10)

e3.VectorXd(10) # Eigen::VectorXd(double)
v10d = e3.VectorXd.Zero(10)
e3.VectorXd.Random(10)

# getters
for i in xrange(10):
  # Eigen::MatrixXd::getItem(int, int)
  m10d.coeff(i, i)
  m10d[i + i*10]
  v10d.coeff(i, 0)
  v10d[i]
m10d.rows(), m10d.cols() # Eigen::MatrixXd::{rows,cols}()
len(m10d) # Eigen::MatrixXd::size()

# setters
# Eigen::MatrixXd::setItem(int, int, double)
m10d.coeff(1,2,1.)
m10d[1 + 2*10] = 1.
v10d.coeff(1,0,1.)
v10d[1] = 1.

# operations
m10d.norm() # Eigen::MatrixXd::norm()
m10d.squaredNorm() # Eigen::MatrixXd::squaredNorm()
m10d.normalize() # Eigen::MatrixXd::normalize()
m10d.normalized() # Eigen::MatrixXd::normalized()
m10d.transpose() # Eigen::MatrixXd::transpose() (return a MatrixXd)
```

Installing
------

### Manual

#### Dependencies

To compile you need the following tools:

 * [Git](https://git-scm.com/)
 * [pkg-config]()
 * [Eigen](http://eigen.tuxfamily.org/index.php?title=Main_Page) >= 3.2
 * [pip](https://pypi.python.org/pypi/pip)
 * [Cython](http://cython.org/) >= 0.25

#### Building

```sh
git clone https://github.com/jrl-umi3218/Eigen3ToPython.git
cd Eigen3ToPython
pip install -r requirements.txt
pip install .
```

Pulling git subtree
------

To update/sync .travis directory with their upstream git repository:

```sh
git fetch git://github.com/jrl-umi3218/jrl-travis.git master
git subtree pull --prefix .travis git://github.com/jrl-umi3218/jrl-travis.git master --squash
```
