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
from collections import defaultdict
import sys



mbByRows = defaultdict(list)
mbByShape = {}



def createMatrixBase(name, shape):
  type = eigen3.add_class(name)
  setattr(type, 'shape', shape)
  mbByShape[shape] = type
  return type



def isVector(mb):
  return mb.shape[0] == 1 or mb.shape[1] == 1



def isSquareMatrix(mb):
  return mb.shape[0] == mb.shape[1]



def makeCommonMatrixBase(mb):
  # constructor
  mb.add_constructor([])
  mb.add_copy_constructor()

  # accessors
  mb.add_method('rows', retval('int'), [], is_const=True)
  mb.add_method('cols', retval('int'), [], is_const=True)

  mb.add_method('getItem', retval('double'), [param('int', 'id')], is_const=True,
                custom_name='__getitem__')
  # set a int value as return type to fullfil the PySequence_SetItem convention
  mb.add_method('setItem', retval('int'), [param('int', 'row'), param('double', 'id')], custom_name='__setitem__')

  mb.add_method('getItem', retval('double'), [param('int', 'row'), param('int', 'cols')], is_const=True, custom_name='coeff')
  mb.add_method('setItem', None, [param('int', 'row'), param('int', 'cols'), param('double', 'val')], custom_name='coeff')

  mb.add_method('size', retval('int'), [], is_const=True, custom_name='__len__')

  # norm functions
  mb.add_method('norm', retval('double'), [], is_const=True)
  mb.add_method('squaredNorm', retval('double'), [], is_const=True)
  mb.add_method('normalize', None, [])
  mb.add_method('normalized', retval(mb.full_name), [], is_const=True)

  # output
  mb.add_output_stream_operator()



def makeDynamicMatrixBase(mb):
  makeCommonMatrixBase(mb)

  mb.add_constructor([param('int', 'rows'), param('int', 'cols')])

  mb.add_method('Zero', retval(mb.full_name),
                [param('int', 'rows'), param('int', 'cols')], is_static=True)
  mb.add_method('Random', retval(mb.full_name),
                [param('int', 'rows'), param('int', 'cols')], is_static=True)
  mb.add_method('Identity', retval(mb.full_name),
                [param('int', 'rows'), param('int', 'cols')], is_static=True)

  mb.add_method('transpose', retval(mb.full_name), [], is_const=True)



def makeDynamicVectorBase(mb):
  makeCommonMatrixBase(mb)

  mb.add_constructor([param('int', 'rows')])

  mb.add_method('Zero', retval(mb.full_name),
                [param('int', 'rows')], is_static=True)
  mb.add_method('Random', retval(mb.full_name),
                [param('int', 'rows')], is_static=True)



def makeFixedMatrixBase(mb):
  makeCommonMatrixBase(mb)

  mb.add_method('Zero', retval(mb.full_name), [], is_static=True)
  mb.add_method('Random', retval(mb.full_name), [], is_static=True)

  if isVector(mb):
    elems = max(mb.shape)
    mb.add_constructor([param('double', 'val')]*elems)
    if elems > 0:
      mb.add_method('UnitX', retval(mb.full_name), [], is_static=True)
      mb.add_method('x', 'double', [], is_const=True)
    if elems > 1:
      mb.add_method('UnitY', retval(mb.full_name), [], is_static=True)
      mb.add_method('y', 'double', [], is_const=True)
    if elems > 2:
      mb.add_method('UnitZ', retval(mb.full_name), [], is_static=True)
      mb.add_method('z', 'double', [], is_const=True)
    if elems > 3:
      mb.add_method('UnitW', retval(mb.full_name), [], is_static=True)
      mb.add_method('w', 'double', [], is_const=True)

  if isSquareMatrix(mb):
    mb.add_method('Identity', retval(mb.full_name), [], is_static=True)
    mb.add_method('inverse', retval(mb.full_name), [], is_const=True)

  # transpose
  try:
    tShape = (mb.shape[1], mb.shape[0])
    mbRet = mbByShape[tShape]
    mb.add_method('transpose', retval(mbRet.full_name), [], is_const=True)
  except KeyError:
    mb.add_method('transpose', retval(matrixXd.full_name), [], is_const=True)

  # numeric operator
  mb.add_binary_numeric_operator('*', left_cppclass=Parameter.new('double', 'scalar'))
  mb.add_binary_numeric_operator('*', right=param('double', 'scalar'))

  mb.add_binary_numeric_operator('/', right=param('double', 'scalar'))

  mb.add_binary_numeric_operator('+')
  mb.add_binary_numeric_operator('-')

  mb.add_inplace_numeric_operator('+=')
  mb.add_inplace_numeric_operator('-=')

  mb.add_unary_numeric_operator('-')

  if isVector(mb) and mb.shape[0] == 3:
    mb.add_method('cross', retval(mb.full_name), [param(mb.full_name, 'v2')], is_const=True)
  if isVector(mb):
    mb.add_method('dot', retval('double'), [param(mb.full_name, 'v2')], is_const=True)

  # add multiply with all compatible type
  for mb2 in mbByRows[mb.shape[1]]:
    nShape = (mb.shape[0], mb2.shape[1])
    try:
      mb3 = mbByShape[nShape]
      mb.add_binary_numeric_operator('*', result_cppclass=mb3, right=param(mb2.full_name, 't2'))
    except KeyError:
      mb.add_binary_numeric_operator('*', result_cppclass=matrixXd, right=param(mb2.full_name, 't2'))

  if isSquareMatrix(mb):
    mb.add_inplace_numeric_operator('*=')

  if mb.shape[0] == 3 and mb.shape[1] == 3:
    mb.add_method('eulerAngles', retval('Eigen::Vector3d'), [param('int', 'a0'), param('int', 'a1'), param('int', 'a2')], is_const=True)


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
  q.add_method('setFromTwoVectors', retval('Eigen::Quaterniond'), [param('Eigen::Vector3d', 'v1'),
                                                                   param('Eigen::Vector3d', 'v2')])
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

  # declare type
  vector2d = createMatrixBase('Vector2d', (2,1))
  matrix2d = createMatrixBase('Matrix2d', (2,2))
  vector3d = createMatrixBase('Vector3d', (3,1))
  matrix3d = createMatrixBase('Matrix3d', (3,3))
  vector4d = createMatrixBase('Vector4d', (4,1))
  matrix4d = createMatrixBase('Matrix4d', (4,4))
  vector6d = createMatrixBase('Vector6d', (6,1))
  matrix6d = createMatrixBase('Matrix6d', (6,6))

  matrixXd = eigen3.add_class('MatrixXd')
  vectorXd = eigen3.add_class('VectorXd')

  for k, v in mbByShape.iteritems():
    mbByRows[k[0]].append(v)

  # make dynamic matrix base
  makeDynamicMatrixBase(matrixXd)
  makeDynamicVectorBase(vectorXd)

  # make fixed matrix base type
  for k, v in mbByShape.iteritems():
    makeFixedMatrixBase(v)

  # Quaterniond
  add_quaternion(eigen3)

  with open(sys.argv[1], 'w') as f:
    eigen3.generate(f)
