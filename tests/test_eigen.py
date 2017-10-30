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

import eigen as e
import numpy as np
from nose import with_setup
from nose.tools import assert_raises

precision = 1e-6

vectors = {}

vector_types = {'2': e.Vector2d,
                '3': e.Vector3d,
                '4': e.Vector4d,
                '6': e.Vector6d,
                'X': e.VectorXd}

vector_slices = {'2': slice(0,2),
                 '3': slice(1,3),
                 '4': slice(1,4),
                 '6': slice(2,5),
                 'X': slice(4,10)}

vector_args = {str(i) : range(i) for i in [2,3,4,6]}
vector_args['X'] = range(12)

vector_lists = {str(i) : [[j] for j in range(i)] for i in [2,3,4,6]}
vector_lists['X'] = [[i] for i in range(12)]

vector_arrays = {k: np.array(v) for k, v in vector_lists.items()}

matrices = {}

matrix_types = {'2': e.Matrix2d,
                '3': e.Matrix3d,
                '4': e.Matrix4d,
                '6': e.Matrix6d,
                'X': e.MatrixXd}

matrix_slices = {'2': (slice(0,2), slice(0,2)),
                 '3': (slice(1,3), slice(1,3)),
                 '4': (slice(1,4), slice(0,4)),
                 '6': (slice(2,5), slice(1,3)),
                 'X': (slice(2,6), slice(6,10))}

matrix_args = {str(i) : [range(i)]*i for i in [2,3,4,6]}
matrix_args['X'] = [range(12)]*10

matrix_lists = {str(i) : [range(i)]*i for i in [2,3,4,6]}
matrix_lists['X'] = [range(12)]*10

matrix_arrays = {k: np.array(v) for k, v in matrix_lists.items()}

def create(eType, obj):
  return eType(obj)

def generate_vectors():
  global vectors
  vectors = {k: create(v, vector_args[k]) for k, v in vector_types.items()}

def teardown_vectors():
  global vectors
  vectors = {}

def generate_matrices():
  global matrices
  matrices = {k: create(m, matrix_args[k]) for k, m in matrix_types.items()}

def teardown_matrices():
  global matrices
  matrices = {}

def setup():
  global vectors, matrices
  generate_vectors()
  generate_matrices()

def teardown():
  teardown_vectors()
  teardown_matrices()

def test_create_from_args():
  for k, v in vector_types.items():
    #Check Vector3d(0, 1, 2)
    v(*vector_args[k])
    #Check Vector3d([0, 1, 2])
    v(vector_args[k])

  for k, m in matrix_types.items():
    #Only matrix([[0, 0], [0, 0]]) is supported
    m(matrix_args[k])

def test_create_from_list():
  for k, v in vector_types.items():
    #Check Vector3d([[0], [1], [2]])
    v(vector_lists[k])

  for k, m in matrix_types.items():
    #Check Matrix2d([[0, 0], [0, 0]])
    m(matrix_lists[k])

def test_create_from_array():
  for k, v in vector_types.items():
    #Check Vector3d(np.array((3,1)))
    v(vector_arrays[k])

  for k, m in matrix_types.items():
    #Check Matrix2d(np.array((2,2)))
    m(matrix_arrays[k])

@with_setup(generate_vectors, teardown_vectors)
def test_getitem_vector():
  yield getitem_other, vectors, vector_lists
  yield getitem_other, vectors, vector_arrays

@with_setup(generate_vectors, teardown_vectors)
def test_getslice_vector():
  yield getitem_other, vectors, vector_lists, vector_slices
  yield getitem_other, vectors, vector_arrays, vector_slices

@with_setup(generate_matrices, teardown_matrices)
def test_getitem_matrix():
  yield getitem_other, matrices, matrix_lists
  yield getitem_other, matrices, matrix_arrays

@with_setup(generate_matrices, teardown_matrices)
def test_getslice_matrix():
  yield getitem_other, matrices, matrix_lists, matrix_slices
  yield getitem_other, matrices, matrix_arrays, matrix_slices

def getitem_other(first_container, other_container, slicing=None):
  for k, obj in first_container.items():
    other = other_container[k]
    if slicing is None:
      for i in range(obj.rows()):
        if obj.cols() == 1:
          #Vector
          assert(obj[i] == other[i][0])
        else:
          for j in range(obj.cols()):
            assert(obj[i][j] == other[i][j])
    else:
      if isinstance(slicing, dict):
        left = obj[slicing[k]]
        right = obj[slicing[k]]
      else:
        left = obj[slicing]
        right = obj[slicing]
      for i in range(len(left)):
        for j in range(len(left[i])):
          assert(left[i][j] == right[i][j])

@with_setup(generate_vectors, teardown_vectors)
def test_setitem_vector():
  yield setitem_other, vectors, vector_lists
  yield setitem_other, vectors, vector_arrays

@with_setup(generate_vectors, teardown_vectors)
def test_setslice_vector():
  yield setitem_other, vectors, vector_arrays, vector_slices

@with_setup(generate_vectors, teardown_vectors)
def test_setslice_vector_vector():
  for k, v in matrices.items():
    vec = e.VectorXd.Zero(v.rows()+2)
    vec[1:v.rows()] = v
    assert((v[1:v.rows()+1] == np.array(v)).all())

@with_setup(generate_matrices, teardown_matrices)
def test_setitem_matrix():
  yield setitem_other, matrices, matrix_lists
  yield setitem_other, matrices, matrix_arrays

@with_setup(generate_matrices, teardown_matrices)
def test_setslice_matrix():
  yield setitem_other, matrices, matrix_arrays, matrix_slices

@with_setup(generate_matrices, teardown_matrices)
def test_setslice_matrix_matrix():
  for k, m in matrices.items():
    mat = e.MatrixXd.Zero(m.rows()+2, m.cols()+2)
    mat[1:m.rows()+1, 1:m.cols()+1] = m
    assert((mat[1:m.rows()+1, 1:m.cols()+1] == np.array(m)).all())

def setitem_other(first_container, other_container, slicing=None):
  for k, obj in first_container.items():
    obj.setZero()
    other = other_container[k]
    if slicing is None:
      for i in range(obj.rows()):
        if obj.cols() == 1:
          obj[i] = other[i]
        else:
          for j in range(obj.cols()):
            obj[i, j] = other[i][j]
    else:
      if isinstance(slicing, dict):
        obj[slicing[k]] = other[slicing[k]]
      else:
        obj[slicing] = other[slicing]
  getitem_other(first_container, other_container, slicing)

@with_setup(generate_vectors, teardown_vectors)
def test_convert_tonumpy_vector():
  for k, v in vectors.items():
    assert((np.array(v) == vector_arrays[k]).all())

@with_setup(generate_matrices, teardown_matrices)
def test_convert_tonumpy_matrix():
  for k, m in matrices.items():
    assert((np.array(m) == matrix_arrays[k]).all())

@with_setup(generate_vectors, teardown_vectors)
def test_convert_fromnumpy_vector():
  for k, v in vectors.items():
    # Test both storage orders
    assert((v - vector_types[k](np.ascontiguousarray(vector_arrays[k]))).norm() == 0.0)
    assert((v - vector_types[k](np.asfortranarray(vector_arrays[k]))).norm() == 0.0)

@with_setup(generate_matrices, teardown_matrices)
def test_convert_fromnumpy_matrix():
  for k, m in matrices.items():
    # Test both storage orders
    assert((m - matrix_types[k](np.ascontiguousarray(matrix_arrays[k]))).norm() == 0.0)
    assert((m - matrix_types[k](np.asfortranarray(matrix_arrays[k]))).norm() == 0.0)

def test_arithmetic():
  scalars = [0, 1, -1, 2.34, np.pi]
  for k, v in vector_types.items():
    check_op_scalar('*', v, vector_lists[k], scalars)
    check_op_scalar('*=', v, vector_lists[k], scalars)
    check_op_scalar('/', v, vector_lists[k], scalars)
    check_op_scalar('/=', v, vector_lists[k], scalars)
    check_op_vec('+', v, vector_lists[k], scalars)
    check_op_vec('-', v, vector_lists[k], scalars)
    check_op_vec('+=', v, vector_lists[k], scalars)
    check_op_vec('-=', v, vector_lists[k], scalars)

  for k, m in matrix_types.items():
    check_op_scalar('*', m, matrix_lists[k], scalars)
    check_op_scalar('/', m, matrix_lists[k], scalars)
    check_op_scalar('*=', m, matrix_lists[k], scalars)
    check_op_scalar('/=', m, matrix_lists[k], scalars)
    check_op_vec('+', m, matrix_lists[k], scalars)
    check_op_vec('-', m, matrix_lists[k], scalars)
    check_op_vec('+=', m, matrix_lists[k], scalars)
    check_op_vec('-=', m, matrix_lists[k], scalars)

def check_op_scalar(op, objtype, objlist, scalars):
  for scalar in scalars:
    obj = objtype(objlist)
    if op == '*' or op == '*=':
      svec = objtype([[coeff*scalar for coeff in arg] for arg in objlist])
      if op == '*':
        assert((obj*scalar - svec).norm() < precision)
      else:
        obj *= scalar
        assert((obj - svec).norm() < precision)
    elif op == '/' or op == '/=':
      try:
        svec = objtype([[coeff/scalar for coeff in arg] for arg in objlist])
      except ZeroDivisionError:
        continue
      if op == '/':
        assert((obj/scalar - svec).norm() < precision)
      else:
        obj /= scalar
        assert((obj - svec).norm() < precision)
    else:
      raise ValueError("Only scalar multiplication/division is supported")

def check_op_vec(op, objtype, objlist, scalars):
  for scalar in scalars:
    obj = objtype(objlist)
    if op == '+' or op == '+=':
        opvec = objtype([[scalar+(obj[line, col]) for col in range(obj.cols())] for line in range(obj.rows())])
        svec = objtype([[scalar for col in range(obj.cols())] for line in range(obj.rows())])
        if op == '+':
          assert((obj+svec - opvec).norm() < precision)
        else:
          obj += svec
          assert((obj - opvec).norm() < precision)
    elif op == '-' or op == '-=':
        svec = objtype([[scalar for col in range(obj.cols())] for line in range(obj.rows())])
        opvec = objtype([[obj[line, col]-scalar for col in range(obj.cols())] for line in range(obj.rows())])
        if op == '-':
          assert((obj-svec - opvec).norm() < precision)
        else:
          obj -= svec
          assert((obj - opvec).norm() < precision)
    else:
      raise ValueError("Only vector addition/substraction is supported")

@with_setup(setup, teardown)
def test_slicing_eigen():
  for k, v in vectors.items():
    start, stop = vector_slices[k].start, vector_slices[k].stop
    eigen_vblock = v.block(start, 0, stop - start, 1)
    assert(isinstance(eigen_vblock, vector_types['X']))
    assert(np.allclose(np.array(eigen_vblock), v[vector_slices[k]], precision))

  for k, m in matrices.items():
      row_slice, col_slice = matrix_slices[k]
      eigen_mblock = m.block(row_slice.start, col_slice.start,
                             row_slice.stop - row_slice.start,
                             col_slice.stop - col_slice.start)
      assert(isinstance(eigen_mblock, matrix_types['X']))
      assert(np.allclose(np.array(eigen_mblock), m[matrix_slices[k]], precision))
      eigen_mblock = m.block(row_slice.start, col_slice.start,
                             row_slice.stop - row_slice.start,
                             1)
      assert(isinstance(eigen_mblock, vector_types['X']))
      assert(np.allclose(np.array(eigen_mblock), m[matrix_slices[k][0],  col_slice.start], precision))

@with_setup(generate_vectors, teardown_vectors)
def test_norm():
  for k, v in vectors.items():
    assert(np.sqrt(v.transpose()*v) == np.linalg.norm(v))
    assert(v.norm() == np.linalg.norm(v))
    assert(abs(v.squaredNorm() - np.linalg.norm(v)**2) < precision)

@with_setup(setup, teardown)
def test_mat_vec_mult():
  for k, v in vectors.items():
    m = matrices[k]
    res = m*v
    assert(isinstance(res, vector_types[k]))
    if k == 'X':
      assert(res.rows() == m.rows())
    assert((abs(np.array(res) - np.array(m).dot(np.array(v))) < precision).all())

@with_setup(generate_vectors, teardown_vectors)
def test_vec_tvec_mult():
  for k, v in vectors.items():
    res = v*v.transpose()
    assert(isinstance(res, matrix_types[k]))
    if k == 'X':
      assert(res.rows() == v.rows())
      assert(res.cols() == v.rows())
    assert((abs(np.array(res) - np.array(v).dot(np.array(v).T)) < precision).all())

@with_setup(generate_vectors, teardown_vectors)
def test_vec_mat_mult():
  cols = {'2': 5,
          '3': 7,
          '4': 13,
          '6': 19,
          'X': 4}
  for k, v in vectors.items():
    m = e.MatrixXd(1, cols[k])
    res = v*m
    assert(isinstance(res, e.MatrixXd))
    assert(res.rows() == v.rows())
    assert(res.cols() == cols[k])
    assert((abs(np.array(res) - np.array(v).dot(np.array(m))) < precision).all())

@with_setup(generate_vectors, teardown_vectors)
def test_fail_vec_mat_mult():
  dims = {'2': (2, 5),
          '3': (3, 7),
          '4': (4, 13),
          '6': (13, 19),
          'X': (6, 4)}
  for k, v in vectors.items():
    m = e.MatrixXd(*dims[k])
    assert_raises(TypeError, lambda x,y: x*y, v, m)

@with_setup(generate_matrices, teardown_matrices)
def test_mat_mat_mult():
  for k, m in matrices.items():
    res = m.transpose()*m
    assert(isinstance(res, matrix_types[k]))
    if k == 'X':
      assert(res.rows() == m.cols())
      assert(res.cols() == m.cols())
    assert((abs(np.array(res) - np.array(m).T.dot(np.array(m))) < precision).all())

@with_setup(generate_matrices, teardown_matrices)
def test_mat_dynvec_mult():
  for k, m in matrices.items():
    m = matrices[k]
    v = e.VectorXd.Random(m.cols())
    res = m*v
    assert(isinstance(res, vector_types[k]))
    assert(res.rows() == m.rows())
    assert((abs(np.array(res) - np.array(m).dot(np.array(v))) < precision).all())

@with_setup(generate_matrices, teardown_matrices)
def test_mat_dynmat_mult():
  ncols = {'2': 17,
           '3': 13,
           '4': 7,
           '6': 27,
           'X': 2}

  for k, m in matrices.items():
    m = matrices[k]
    m2 = e.MatrixXd.Random(m.cols(), ncols[k])
    res = m*m2
    assert(isinstance(res, e.MatrixXd))
    assert(res.rows() == m.rows())
    assert(res.cols() == ncols[k])
    assert((abs(np.array(res) - np.array(m).dot(np.array(m2))) < precision).all())

@with_setup(generate_matrices, teardown_matrices)
def test_dynmat_mat_mult():
  ncols = {'2': 17,
           '3': 13,
           '4': 7,
           '6': 27,
           'X': 2}

  for k, m in matrices.items():
    m = matrices[k]
    m2 = e.MatrixXd.Random(ncols[k], m.rows())
    res = m2*m
    assert(isinstance(res, e.MatrixXd))
    assert(res.rows() == ncols[k])
    assert(res.cols() == m.cols())
    assert((abs(np.array(res) - np.array(m2).dot(np.array(m))) < precision).all())

  for k, m in matrices.items():
    m = matrices[k]
    m2 = e.MatrixXd.Random(m.cols(), m.rows())
    res = m2*m
    assert(isinstance(res, type(m)))
    assert(res.rows() == m.cols())
    assert(res.cols() == m.cols())
    assert((abs(np.array(res) - np.array(m2).dot(np.array(m))) < precision).all())

def test_access():
  for k, v in vector_types.items():
    vec = v(vector_args[k])
    yield check_negative_vec_access, vec
    yield check_oob_access, vec

  for k, m in matrix_types.items():
    mat = m(matrix_args[k])
    yield check_negative_mat_access, mat
    yield check_oob_access, mat

def check_oob_access(obj):
  get = lambda v, i : v[i]
  assert_raises(IndexError, get, obj, obj.rows())
  assert_raises(IndexError, get, obj, (0, obj.cols()))
  assert_raises(IndexError, get, obj, -(obj.rows()+1))
  assert_raises(IndexError, get, obj, (0, -(obj.cols()+1)))

def check_negative_vec_access(vec):
  for index in range(1, vec.rows()+1):
    assert(vec[-index] == vec[len(vec)-index])

def check_negative_mat_access(mat):
  for i in range(1, mat.rows()+1):
    for j in range(1, mat.cols()+1):
      assert(mat[-i, -j] == mat[mat.rows()-i, mat.cols()-j])

def check_quaternion_almost_equals(q1, q2):
  assert((q1.coeffs() - q2.coeffs()).norm() < 1e-6)

def test_quaternion():
  q = e.Quaterniond()
  # Identity, xyzw coefficient order.
  id_v = e.Vector4d(0., 0., 0., 1.)
  # Identity, wxyz scalar constructor order.
  q = e.Quaterniond(1., 0., 0., 0.)
  q2 = e.Quaterniond(id_v)
  # Both identity quaternions must be equal.
  assert(q.angularDistance(q2) == 0)
  q3 = q2*q
  assert(q.angularDistance(q3) == 0)
  v4 = e.Vector4d.Random()
  while v4 == id_v:
    v4 = e.Vector4d.Random()
  v4.normalize()
  q = e.Quaterniond(v4) # Vector4d ctor
  assert(q.coeffs() == v4)
  q = e.Quaterniond(v4[3], v4[0], v4[1], v4[2]) # 4 doubles ctor
  assert(q.coeffs() == v4)
  q_copy = e.Quaterniond(q) # Copy ctor
  assert(q_copy.coeffs() == q.coeffs())
  q_copy.setIdentity()
  assert(q_copy.coeffs() != q.coeffs()) # Check the two objects are actually different
  q = e.Quaterniond(np.pi, e.Vector3d.UnitZ()) # Angle-Axis
  assert((q.coeffs() - e.Vector4d(0., 0., 1., 0.)).norm() < 1e-6)
  q = e.Quaterniond(e.AngleAxisd(np.pi, e.Vector3d.UnitZ()))
  assert((q.coeffs() - e.Vector4d(0., 0., 1., 0.)).norm() < 1e-6)
  q = e.Quaterniond(e.Matrix3d.Identity())
  assert(q.coeffs() == id_v)
  q = e.Quaterniond.Identity()
  assert(q.coeffs() == id_v)

  # Check getters
  assert(q.x() == 0)
  assert(q.y() == 0)
  assert(q.z() == 0)
  assert(q.w() == 1)
  assert(q.vec() == e.Vector3d.Zero())
  assert(q.coeffs() == id_v)

  # Check setters
  q = e.Quaterniond(v4) # Rebuild from something other than Identity
  q.setIdentity()
  assert(q.coeffs() == id_v)
  q.setFromTwoVectors(e.Vector3d.UnitX(), e.Vector3d.UnitY())
  assert(q.isApprox(e.Quaterniond(np.pi/2, e.Vector3d.UnitZ())))
  q.setIdentity()

  # Operations
  assert(e.Quaterniond(id_v).angularDistance(e.Quaterniond(np.pi, e.Vector3d.UnitZ())) == np.pi)
  assert(e.Quaterniond(v4).conjugate().coeffs() == e.Vector4d(-v4.x(), -v4.y(), -v4.z(), v4.w()))
  assert(e.Quaterniond(id_v).dot(e.Quaterniond(np.pi, e.Vector3d.UnitZ())) == np.cos(np.pi/2))
  check_quaternion_almost_equals(e.Quaterniond(v4).inverse(), e.Quaterniond(v4).conjugate())
  assert(q.isApprox(q))
  assert(q.isApprox(q, 1e-8))
  assert(q.matrix() == e.Matrix3d.Identity())
  assert(q.toRotationMatrix() == e.Matrix3d.Identity())
  v4_2 = e.Vector4d.Random()
  v4_2 = 2 * v4_2 / v4_2.norm()
  q = e.Quaterniond(v4_2)
  assert(q.norm() != 1.0)
  q_n = q.normalized()
  assert(q.norm() != 1.0 and q_n.norm() == 1.0)
  q.normalize()
  assert(q.norm() == 1.0)
  check_quaternion_almost_equals(e.Quaterniond.Identity().slerp(1.0, q), q)
  assert(q.squaredNorm() == 1.0)

  # Static method
  q = e.Quaterniond.UnitRandom()
  assert(q.norm() == 1.0)


def test_angle_axis():
  aa = e.AngleAxisd()
  # quaternion xyzw coefficient order
  q = e.Quaterniond(e.Vector4d(0., 0., 0., 1.))
  aa = e.AngleAxisd(q)
  assert(aa.angle() == 0)
  v = e.Vector3d.UnitX()
  aa = e.AngleAxisd(0.1, v)
  assert(aa.axis() == v)
  assert(aa.angle() == 0.1)
  aa2 = aa.inverse()
  assert(aa2.axis() == v)
  assert(aa2.angle() == -0.1)
