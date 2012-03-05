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

from pybindgen import *
import sys


def makeMatrixBase(mb, dim):
  mb.add_constructor([])

  # if is a vector
  if dim[1] == 1:
    mb.add_constructor([param('double', 'val')]*dim[0])
    if dim[0] > 0:
      mb.add_method('UnitX', retval(mb.full_name), [], is_static=True)
    if dim[0] > 1:
      mb.add_method('UnitY', retval(mb.full_name), [], is_static=True)
    if dim[0] > 2:
      mb.add_method('UnitZ', retval(mb.full_name), [], is_static=True)
    if dim[0] > 3:
      mb.add_method('UnitW', retval(mb.full_name), [], is_static=True)

  # if is a matrix
  if dim[1] > 1:
    mb.add_method('Identity', retval(mb.full_name), [], is_static=True)

  mb.add_method('Zero', retval(mb.full_name), [], is_static=True)

  mb.add_method('rows', retval('int'), [], is_const=True)
  mb.add_method('cols', retval('int'), [], is_const=True)

  mb.add_method('getItem', retval('double'), [param('int', 'id')], is_const=True,
                custom_name='__getitem__')
  mb.add_method('setItem', None, [param('int', 'row'), param('double', 'id')], custom_name='__setitem__')

  mb.add_method('getItem', retval('double'), [param('int', 'row'), param('int', 'cols')], is_const=True, custom_name='coeff')
  mb.add_method('setItem', None, [param('int', 'row'), param('int', 'cols'), param('double', 'val')], custom_name='coeff')

  mb.add_method('size', retval('int'), [], is_const=True, custom_name='__len__')

  mb.add_output_stream_operator()

def add_quaternion(mod):
  q = mod.add_class('Quaterniond')

  q.add_copy_constructor()

  q.add_constructor([])
  q.add_constructor([param('Eigen::Vector4d', 'vec')])
  q.add_constructor([param('double', 'w'), param('double', 'x'),
                     param('double', 'y'), param('double', 'z'),])
  q.add_function_as_constructor('::createFromAngleAxis', 'Quaterniond*',
                                [param('double', 'angle'), param('Vector3d', 'axis')])
  q.add_function_as_constructor('::createFromMatrix', 'Quaterniond*', [param('Matrix3d', 'axis')])

  q.add_method('angularDistance', retval('double'), [param('Quaterniond', 'other')],
               is_const=True)
  q.add_method('conjugate', retval('Eigen::Quaterniond'), [], is_const=True)
  q.add_method('dot', retval('double'), [param('Eigen::Quaterniond', 'other')])
  q.add_method('inverse', retval('Eigen::Quaterniond'), [], is_const=True)
  q.add_method('isApprox', retval('bool'), [param('Eigen::Quaterniond', 'other')], is_const=True)
  q.add_method('isApprox', retval('bool'),
               [param('Eigen::Quaterniond', 'other'), param('double', 'prec')],
               is_const=True)
  q.add_method('matrix', retval('Eigen::Matrix3d'), [], is_const=True)
  q.add_method('normalize', None, [])
  q.add_method('normalized', retval('Eigen::Quaterniond'), [], is_const=True)
  q.add_method('setIdentity', retval('Eigen::Quaterniond'), [])
  q.add_method('slerp', retval('Eigen::Quaterniond'),
               [param('double', 't'), param('Eigen::Quaterniond', 'other')], is_const=True)
  q.add_method('squaredNorm', retval('double'), [], is_const=True)
  q.add_method('toRotationMatrix', retval('Eigen::Matrix3d'), [], is_const=True)

  q.add_method('coeffs', retval('Eigen::Vector4d'), [], is_const=True)
  q.add_method('vec', retval('Eigen::Vector3d'), [], is_const=True)
  q.add_method('w', retval('double'), [], is_const=True)
  q.add_method('x', retval('double'), [], is_const=True)
  q.add_method('y', retval('double'), [], is_const=True)
  q.add_method('z', retval('double'), [], is_const=True)

  q.add_method('Identity', retval('Eigen::Quaterniond'), [], is_static=True)

  q.add_binary_numeric_operator('*')


  q.add_output_stream_operator()

if __name__ == '__main__':
  if len(sys.argv) < 2:
    sys.exit(1)

  eigen3 = Module('_eigen3', cpp_namespace='::Eigen')
  eigen3.add_include('<stdexcept>')
  eigen3.add_include('<Eigen/Core>')
  eigen3.add_include('<Eigen/Geometry>')
  eigen3.add_include('"EigenTypedef.h"')
  eigen3.add_include('"EigenUtils.h"')

  # Vector3d
  vector2d = eigen3.add_class('Vector2d')
  makeMatrixBase(vector2d, (2, 1))

  # Matrix2d
  Matrix2d = eigen3.add_class('Matrix2d')
  makeMatrixBase(Matrix2d, (2, 2))

  # Vector3d
  vector3d = eigen3.add_class('Vector3d')
  makeMatrixBase(vector3d, (3, 1))

  # Matrix3d
  Matrix3d = eigen3.add_class('Matrix3d')
  makeMatrixBase(Matrix3d, (3, 3))

  # Vector4d
  vector4d = eigen3.add_class('Vector4d')
  makeMatrixBase(vector4d, (4, 1))

  # Matrix4d
  Matrix4d = eigen3.add_class('Matrix4d')
  makeMatrixBase(Matrix4d, (4, 4))

  # Vector6d
  vector6d = eigen3.add_class('Vector6d')
  makeMatrixBase(vector6d, (6, 1))

  # Matrix3d
  Matrix6d = eigen3.add_class('Matrix6d')
  makeMatrixBase(Matrix6d, (6, 6))

  # Quaterniond
  add_quaternion(eigen3)

  with open(sys.argv[1], 'w') as f:
    eigen3.generate(f)

