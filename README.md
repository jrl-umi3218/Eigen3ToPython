[![Build Status](https://travis-ci.org/jrl-umi3218/Eigen3ToPython.svg?branch=master)](https://travis-ci.org/jrl-umi3218/Eigen3ToPython)

Eigen3ToPython
======

Eigen3ToPython is a Python interface for [Eigen3](eigen.tuxfamily.org/) with support for [NumPy](http://www.numpy.org/).

This library supports:
 * operations on fixed size Eigen3 matrices and vectors
 * operations on dynamic size Eigen3 matrices and vectors
 * Use Eigen::Quaterniond in Python
 * Use Eigen::AngleAxisd in Python
 * Transparent conversion to/from Numpy arrays (`np.array`)
     * Note that memory is not shared between numpy and eigen

If you want more features feel free to open an issue or submit a pull request. :-)

Documentation
------

### Eigen <-> Numpy conversions

```Python
import numpy as np
import eigen

A = np.random.random((2000, 50))
B = eigen.MatrixXd(A)

n = np.linalg.norm(B) # Implicit conversion to numpy object

# Note:
# Because of the difference in default storage order between Eigen and Numpy,
# conversions of big matrix/arrays can be a bit expensive, e.g
%timeit eigen.MatrixXd(A)
10000 loops, best of 3: 107 µs per loop
%timeit np.array(B)
10000 loops, best of 3: 53.1 µs per loop
```

### Fixed size Eigen3 Matrix operations

```Python
import eigen as e

# Fixed size vector constructors (Vector2d, Vector3d, Vector4d, Vector6d)
e.Vector3d.Zero() # Eigen::Vector3d::Zero()
v3d = e.Vector3d.Random() # Eigen::Vector3d::Random()
e.Vector4d.UnitX() # Eigen::Vector4d::UnitX() (also Unit{Y,Z,W} when it makes sense)
e.Vector3d() # Eigen::Vector3d::Zero() (no uninitialized values)
e.Vector3d(v3d) # Eigen::Vector3d(Eigen::Vector3d) (copy constructor)
v4d = e.Vector4d(1., 2., 3., 4.) # Eigen::Vector4d(double, double, double, double)

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
e.Matrix3d.Zero() # Eigen::Matrix3d::Zero()
e.Matrix3d.Random() # Eigen::Matrix3d::Random()
m3d = e.Matrix3d.Identity() # Eigen::Matrix3d::Identity()
e.Matrix3d(m3d) # Eigen::Matrix3d(Eigen::Matrix3d) (copy constructor)
e.Matrix3d() # Eigen::Matrix3d() (uninitialized values)

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

### Quaternions via Quaterniond

[Unit Quaternions](https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation) are used to represent rigid body rotations in 3D. Unit Quaternions are the ideal representation for numerical calculations on rotations because of their stability compared to other representations. The Quaterniond class is a python interface for the C++ [Eigen::Quaterniond](https://eigen.tuxfamily.org/dox/classEigen_1_1Quaternion.html).

```Python
import eigen as e

# constructors
e.Quaterniond() # Eigen::Quaterniond() (uninitialized values)
# Important: coefficients are in xyzw order, while the scalar constructor is in wxyz order!
e.Quaterniond(e.Vector4d(0., 0., 0., 1.)) # Eigen::Quaterniond(Eigen::Vector4d) (x, y, z, w)
quat = e.Quaterniond(1., 0., 0., 0.) # Eigen::Quaterniond(double w, double x, double y, double z)
e.Quaterniond(quat) # Eigen::Quaterniond(Eigen::Quaterniond) (copy constructor)
e.Quaterniond(0.1, e.Vector3d.UnitX()) # Eigen::Quaterniond(Eigen::AngleAxisd(double, Eigen::Vector3d));
e.Quaterniond(e.Matrix3d.Identity()) # Eigen::Quaterniond(Eigen::AngleAxisd(Eigen::Matrix3d))
e.Quaterniond.Identity() # Eigen::Quaterniond::Identity()

# getters
quat.x(), quat.y(), quat.z(), quat.w() # Eigen::Quaterniond::{x,y,z,w}()
quat.vec() # Eigen::Quaterniond.vec()
quat.coeffs() # Eigen::Quaterniond.coeffs()

# setters (in-place)
quat.setIdentity() # Eigen::Quaterniond::setIdentity()
# Eigen::Quaterniond::setFromTwoVectors(Eigen::Vector3d, Eigen::Vector3d)
quat.setFromTwoVectors(e.Vector3d.UnitX(), e.Vector3d.UnitY())

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
import eigen3 as e

# constructors
m10d = e.MatrixXd(10, 10) # Eigen::MatrixXd(double, double)
e.MatrixXd() # Eigen::MatrixXd()
e.MatrixXd(m10d) # Eigen::MatrixXd(Eigen::MatrixXd)

e.MatrixXd.Identity(10,10)
e.MatrixXd.Zero(10,10)
e.MatrixXd.Random(10,10)

e.VectorXd(10) # Eigen::VectorXd(double)
v10d = e.VectorXd.Zero(10)
e.VectorXd.Random(10)

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

A demonstration of how interacting with Quaterniond affects the values:


```
$ q = e.Quaterniond()
$ q.setIdentity()
# coefficients are in x, y, z, w order
$ print(np.array(q.coeffs()))
[[ 0.]
 [ 0.]
 [ 0.]
 [ 1.]]
# scalar constructor is w, x, y, z
# this is also the identity
$ q2 = e.Quaterniond(1,0,0,0)
# coefficients are in x, y, z, w order
$ print(np.array(q2.coeffs()))
[[ 0.]
 [ 0.]
 [ 0.]
 [ 1.]]
$ q.angularDistance(q2)
0.0
# This is what is expected
$ q.w()
1.0
$ q.x()
0.0
$ q.y()
0.0
$ q.z()
0.0


# BAD EXAMPLE illustrating a common mistake:
# The angular distance between two identity quaternions is zero
# The test below shows how the bad example does
# not create two identity quaternions as intended:
$ q2 = e.Quaterniond(qcoeffs[0,0],qcoeffs[1,0],qcoeffs[2,0],qcoeffs[3,0])
$ print(np.array(q2.coeffs()))
[[ 0.]
 [ 0.]
 [ 1.]
 [ 0.]]
$ q.angularDistance(q2)
3.141592653589793
$ vq2 = e.Vector4d(np.array([1,0,0,0]))
$ q2 = e.Quaterniond(vq2)
$ np.array(q2.coeffs())
array([[ 1.],
       [ 0.],
       [ 0.],
       [ 0.]])
$ q.angularDistance(q2)
3.141592653589793
# END BAD EXAMPLE
```


### Angle Axis representation

The [Angle Axis](https://en.wikipedia.org/wiki/Axis%E2%80%93angle_representation) representation of 3D rotations is useful because it is easy for humans to understand to define rotations which can then be converted to a more numerically stable representation.

```Python

  aa = e.AngleAxisd()
  # quaternion xyzw coefficient order
  q = e.Quaterniond(e.Vector4d(0., 0., 0., 1.))
  aa = e.AngleAxisd(q)
  aa.angle()
  v = e.Vector3d.UnitX()
  # construct with angle in radians and axis vector v
  aa = e.AngleAxisd(0.1, v)
  aa.axis()
  aa.angle()
  aa.inverse()
  q = e.Quaterniond(aa)
```

### Converting from C++

Eigen3ToPython doesn't allow reference-based access.

The following code is valid in C++, where `pt.translation()` returns a reference to the object:

```C++
auto pt = sva::PTransformd::Identity();
pt.translation().z() = 1.0;
// or
pt.translation() = Eigen::Vector3d::UnitZ();
```
However the equivalent Python code is not valid, and `pt.translation()` is a copy of the PTransform translation:

```python
pt = sva.PTransformd.Identity()
pt.translation().z() = 1.0 # SyntaxError: can't assign to function call
pt.translation() = eigen.Vector3d.UnitZ() # SyntaxError: can't assign to function call
```

Instead you might construct a new object with the updated values.

Installing
------

### Ubuntu 14.04 and 16.04 binary ppa install

Use the [multi-contact-unstable](https://launchpad.net/~pierre-gergondet+ppa/+archive/ubuntu/multi-contact-unstable) ppa:
```bash
sudo add-apt-repository ppa:pierre-gergondet+ppa/multi-contact-unstable
sudo apt-get update
sudo apt-get install python-eigen python3-eigen
```

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
