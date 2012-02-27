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

  mb.add_method('rows', retval('int'), [], is_const=True)
  mb.add_method('cols', retval('int'), [], is_const=True)

  mb.add_method('getItem', retval('double'), [param('int', 'id')], is_const=True,
                custom_name='__getitem__')
  mb.add_method('setItem', None, [param('int', 'row'), param('double', 'id')], custom_name='__setitem__')

  mb.add_method('getItem', retval('double'), [param('int', 'row'), param('int', 'cols')], is_const=True, custom_name='coeff')
  mb.add_method('setItem', None, [param('int', 'row'), param('int', 'cols'), param('double', 'val')], custom_name='coeff')

  mb.add_method('size', retval('int'), [], is_const=True, custom_name='__len__')

  mb.add_output_stream_operator()


if __name__ == '__main__':
  if len(sys.argv) < 2:
    sys.exit(1)

  eigen3 = Module('_eigen3', cpp_namespace='::Eigen')
  eigen3.add_include('<stdexcept>')
  eigen3.add_include('<Eigen/Core>')
  eigen3.add_include('"EigenTypedef.h"')

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

  with open(sys.argv[1], 'w') as f:
    eigen3.generate(f)

