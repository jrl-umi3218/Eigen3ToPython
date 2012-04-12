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

import numpy as np
import numpy.matlib
import _eigen3 as ei


def toNumpy(eiM):
  num = np.matlib.zeros((eiM.rows(), eiM.cols()))
  for i in xrange(0, eiM.rows()):
    for j in xrange(0, eiM.cols()):
      num[i,j] = eiM.coeff(i,j)
  return num


eigenFact = {(2,1):ei.Vector2d,
             (2,2):ei.Matrix2d,
             (3,1):ei.Vector3d,
             (3,3):ei.Matrix3d,
             (4,1):ei.Vector4d,
             (4,4):ei.Matrix4d,
             (6,1):ei.Vector6d,
             (6,6):ei.Matrix6d}

def toEigen(num):
  try:
    eiM = eigenFact[num.shape]()
  except KeyError:
    if num.shape[1] == 1:
      eiM = ei.VectorXd(num.shape[0])
    else:
      eiM = ei.MatrixXd(*num.shape)

  for i in xrange(0, eiM.rows()):
    for j in xrange(0, eiM.cols()):
      eiM.coeff(i, j, num[i,j])
  return eiM

