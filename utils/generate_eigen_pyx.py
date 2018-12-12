#!/usr/bin/env python

#
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

import hashlib
import os

GEN_DIMS = [2,3,4,6]

def n2c(n):
  t = ""
  if n == -1:
    t = "dynamic"
  elif n == 1:
    t = "one"
  elif n == 2:
    t = "two"
  elif n == 3:
    t = "three"
  elif n == 4:
    t = "four"
  elif n == 6:
    t = "six"
  else:
    raise ValueError("Cannot convert %s to ctypedef")
  return "c_eigen."+t

def generateDeclaration(className):
  ret="""cdef class {0}(object):
  cdef c_eigen.{0} impl

cdef {0} {0}FromC(const c_eigen.{0} &)

cdef class {0}Vector(object):
  cdef vector[c_eigen.{0}] v

""".format(className)
  return ret

def generateBaseBinding(className, type, nRow, nCol):
  ret="""cdef class {0}(object):
  def __copyctor__(self, {0} other):
    self.impl = other.impl
  def __arrayctor(self, numpy.ndarray[dtype=numpy.double_t, ndim=2] array):
    self.impl = c_eigen_private.MatrixFromDynMap[{1}, {2}, {3}](c_eigen_private.DynMap[c_eigen.{0}](<double*>array.data, array.shape[0], array.shape[1], c_eigen_private.DynStride(array.strides[1]/array.itemsize, array.strides[0]/array.itemsize)))
  def __copy__(self):
    return {0}(self)
  def __deepcopy__(self, memo):
    return {0}(self)
  def __str__(self):
    return c_eigen_private.toString[{1}, {2}, {3}](self.impl).decode('utf-8')
  def __repr__(self):
    return "{0} [%sx%s]"%(self.rows(), self.cols())
  def __len__(self):
    return self.impl.rows()
  def __add(self, {0} other):
    ret = {0}()
    ret.impl = self.impl + other.impl
    return ret
  def __add__(self, {0} other):
    assert(isinstance(self, {0}))
    return self.__add(other)
  def __iadd__(self, {0} other):
    self.impl = self.impl + other.impl
    return self
  def __neg(self):
    ret = {0}()
    ret.impl = -self.impl
    return ret
  def __neg__(self):
    return self.__neg()
  def __sub__(self, {0} other):
    return self.__add__(other.__neg__())
  def __isub__(self, {0} other):
    self.impl = self.impl - other.impl
    return self
  def __mul(self, double s):
    ret = {0}()
    ret.impl = self.impl.scalar_mul(s)
    return ret
  def __div(self, double s):
    ret = {0}()
    ret.impl = self.impl.scalar_div(s)
    return ret
  def __div__(self, other):
    if isinstance(self, {0}):
      return self.__truediv__(other)
    else:
      return other.__div__(self)
  def __truediv__(self, other):
    if isinstance(self, {0}):
      return self.__div(other)
    else:
      return other.__truediv__(self)
  def __idiv__(self, other):
    return self.__itruediv__(other)
  def __itruediv__(self, other):
    self.impl = self.impl.scalar_div(other)
    return self
  def __richcmp__({0} self, {0} other, int op):
    if op == 2:
      return self.impl.rows() == other.rows() and self.impl.cols() == other.cols() and self.impl == other.impl
    elif op == 3:
      return self.impl.rows() != other.rows() or self.impl.cols() != other.cols() or self.impl != other.impl
    else:
      raise NotImplementedError("This comparison is not supported")
  def __vblock(self, int i, int j, int p, int q):
    cdef VectorXd ret = VectorXd(p)
    ret.impl = <c_eigen.VectorXd>(self.impl.block(i,j,p,q))
    return ret
  def __mblock(self, int i, int j, int p, int q):
    cdef MatrixXd ret = MatrixXd(p, q)
    ret.impl = <c_eigen.MatrixXd>(self.impl.block(i,j,p,q))
    return ret
  def __array__(self):
    A = numpy.asarray(<numpy.double_t[:self.impl.cols(), :self.impl.rows()]> self.impl.data()).T
    return A
  def block(self, int i, int j, int p, int q):
    if q == 1:
      return self.__vblock(i,j,p,q)
    else:
      return self.__mblock(i,j,p,q)
  def cols(self):
    return self.impl.cols()
  def norm(self):
    return self.impl.norm()
  @property
  def size(self):
    return self.impl.size()
  @property
  def shape(self):
    return (self.impl.rows(), self.impl.cols())
  def normalize(self):
    self.impl.normalize()
  def normalized(self):
    ret = {0}()
    ret.impl = self.impl.normalized()
    return ret
  def rows(self):
    return self.impl.rows()
  def squaredNorm(self):
    return self.impl.squaredNorm()
  def setZero(self):
    self.impl.setZero()
""".format(className, type, n2c(nRow), n2c(nCol))
  return ret

# Bindings differ significantly for the following functions
# - Constructor
# - Transpose function
# - Multiplication (Vector only handle scalar, Matrix handle Matrix/Vector/Scalar)
# - Some static methods (UnitX/Y/Z/W/Identity)

def generateMatrixBinding(className, type, nRow, nCol):
  ret = generateBaseBinding(className, type, nRow, nCol)
  # Constructors,accessors,repr and mult operations are different here
  ret += """  def __cinit__(self, *args):
"""
  if nRow > 0:
    ret += """    if len(args) == 0:
      self.impl = c_eigen_private.EigenZero[{1},{2},{3}]()
    elif len(args) == 1 and isinstance(args[0], {0}):
      self.__copyctor__(args[0])
    elif len(args) == 1 and len(args[0]) == {4}:
      self.impl = c_eigen_private.EigenZero[{1},{2},{3}]()
      for i, row in enumerate(args[0]):
        assert(len(args[0][i]) == {5})
        for j, el in enumerate(row):
          self.coeff(i,j,el)
    else:
      raise TypeError("Unsupported argument types passed to {0} ctor: " + ", ".join([str(type(x)) for x in args]))
""".format(className, type, n2c(nRow), n2c(nCol), nRow, nCol)
  else:
    ret += """    if len(args) == 0:
      self.impl = c_eigen_private.EigenZero[{1},{2},{3}](0,0)
    elif len(args) == 2:
      self.impl = c_eigen_private.EigenZero[{1},{2},{3}](args[0], args[1])
    elif len(args) == 1 and isinstance(args[0], {0}):
      self.__copyctor__(args[0])
    elif len(args) == 1 and isinstance(args[0], int):
      self.impl = c_eigen_private.EigenZero[{1},{2},{3}](args[0],1)
    elif len(args) == 1:
      self.__arrayctor(numpy.asanyarray(args[0], dtype=numpy.double))
    else:
      raise TypeError("Unsupported argument types passed to {0} ctor: " + ", ".join([str(type(x)) for x in args]))
""".format(className, type, n2c(nRow), n2c(nCol))
  ret += """  def __getitem__(self, pos):
    if isinstance(pos, tuple):
      r,c = pos
      if isinstance(r, slice) and isinstance(c, slice):
        ri = r.indices(self.impl.rows())
        ci = c.indices(self.impl.cols())
        return [[self.__getitem__((rix, cix)) for cix in range(*ci)]
                 for rix in range(*ri)]
      if isinstance(r, slice):
        indices = r.indices(self.impl.rows())
        return [[self.__getitem__((i, c))] for i in range(*indices)]
      if isinstance(c, slice):
        indices = c.indices(self.impl.cols())
        return [[self.__getitem__((r, i)) for i in range(*indices)]]
      if (abs(r) < self.impl.rows() or r == -self.impl.rows()) and\
          (abs(c) < self.impl.cols() or c == -self.impl.cols()):
        if r < 0:
          r = self.impl.rows() + r
        if c < 0:
          c = self.impl.cols() + c
        return self.impl(r,c)
      else:
        raise IndexError("Index out of bounds")
    if isinstance(pos, slice):
      indices = pos.indices(len(self))
      return [self.__getitem__(i) for i in range(*indices)]
    else:
      if abs(pos) < self.impl.rows():
        if pos < 0:
          pos = self.impl.rows() + pos
        return [self.impl(pos, j) for j in range(self.impl.cols())]
      else:
        raise IndexError("Index larger than rows")
  def __setitem__(self, pos, v):
    r,c = pos
    if isinstance(r, slice) and isinstance(c, slice):
      ri = r.indices(self.impl.rows())
      ci = c.indices(self.impl.cols())
      for scx, cix in enumerate(range(*ci)):
        for srx, rix in enumerate(range(*ri)):
          self.__setitem__((rix, cix), v[srx][scx])
    elif isinstance(r, slice):
      indices = r.indices(self.impl.rows())
      for rsx, rix in enumerate(range(*indices)):
        self.__setitem__((rix, c), v[rsx])
    elif isinstance(c, slice):
      indices = c.indices(self.impl.cols())
      for csx, cix in enumerate(range(*indices)):
        self.__setitem__((r, cix), v[csx])
    elif r < self.impl.rows() and c < self.impl.cols():
      c_eigen_private.EigenSetValue[{1},{2},{3}](self.impl, r, c, v)
    else:
      raise IndexError("Index out of bounds")
  def transpose(self):
    ret = {0}()
    ret.impl = self.impl.transpose()
    return ret
  def inverse(self):
    ret = {0}()
    ret.impl = self.impl.inverse()
    return ret
""".format(className, type, n2c(nRow), n2c(nCol))
  if nRow > 0:
    ret += """  def __fixvecmul(self, Vector{4}d other):
    ret = Vector{4}d()
    ret.impl = c_eigen_private.EigenMul[{1},{2},{3},{2},c_eigen.one](self.impl, other.impl)
    return ret
  def __fixmatmul(self, {0} other):
    ret = {0}()
    ret.impl = c_eigen_private.EigenMul[{1},{2},{3},{2},{3}](self.impl, other.impl)
    return ret
""".format(className, type, n2c(nRow), n2c(nCol), nRow)
  else:
    for i in  GEN_DIMS:
      ret += """  def __fixedvec{0}mul(self,Vector{0}d other):
    ret = Vector{0}d()
    ret.impl = c_eigen_private.EigenFixedMul[{2},c_eigen.dynamic,c_eigen.dynamic,{1},c_eigen.one](self.impl,other.impl)
    return ret
  def __dynvec{0}mul(self,Vector{0}d other):
    ret = VectorXd()
    ret.impl = c_eigen_private.EigenMul[{2},c_eigen.dynamic,c_eigen.dynamic,{1},c_eigen.one](self.impl,other.impl)
    return ret
  def __vec{0}mul(self, Vector{0}d other):
    if self.cols() == {0}:
      if self.rows() == {0}:
        return self.__fixedvec{0}mul(other)
      else:
        return self.__dynvec{0}mul(other)
    else:
      raise TypeError("Incompatible multiplications")
""".format(i, n2c(i), type)
    for i in  GEN_DIMS:
      ret += """  def __fixedmat{0}mul(self,Matrix{0}d other):
    ret = Matrix{0}d()
    ret.impl = c_eigen_private.EigenFixedMul[{2},c_eigen.dynamic,c_eigen.dynamic,{1},{1}](self.impl,other.impl)
    return ret
  def __dynmat{0}mul(self,Matrix{0}d other):
    ret = MatrixXd()
    ret.impl = <c_eigen.MatrixXd>(c_eigen_private.EigenMul[{2},c_eigen.dynamic,c_eigen.dynamic,{1},{1}](self.impl,other.impl))
    return ret
  def __mat{0}mul(self, Matrix{0}d other):
    if self.cols() == {0}:
      if self.rows() == {0}:
        return self.__fixedmat{0}mul(other)
      else:
        return self.__dynmat{0}mul(other)
    else:
      raise TypeError("Incompatible multiplications")
""".format(i, n2c(i), type)
  ret += """  def __vecmul(self, VectorXd other):
    if other.rows() == self.cols():
"""
  if nRow > 0:
    ret += """      ret = Vector{0}d()
""".format(nRow)
  else:
    ret += """      ret = VectorXd()
"""
  ret += """      ret.impl = c_eigen_private.EigenMul[{1},{2},{3},c_eigen.dynamic,c_eigen.one](self.impl, other.impl)
      return ret
    else:
      raise TypeError("Vector size incompatible with this matrix")
  def coeff(self, int row, int col, value = None):
    if value is None:
      return self.impl.coeff(row, col)
    else:
      c_eigen_private.EigenSetValue[{1},{2},{3}](self.impl, row, col, value)
  def __matmul(self, MatrixXd other):
    if other.rows() == self.cols():
      ret = MatrixXd()
      ret.impl = c_eigen_private.EigenFixedMul[{1},{2},{3},c_eigen.dynamic,c_eigen.dynamic](self.impl,other.impl)
      return ret
    else:
      raise TypeError("Multiplying incompatible matrices")
  def __mul__(self, other):
    if isinstance(self, {0}):
      if isinstance(other, {0}):
""".format(className, type, n2c(nRow), n2c(nCol), nRow)
  if nRow > 0:
    ret += """        return self.__fixmatmul(other)
    """
  else:
    ret += """        return self.__matmul(other)
    """
  ret+="""  elif isinstance(other, VectorXd):
        return self.__vecmul(other)
""".format(className, type, n2c(nRow), n2c(nCol), nRow)
  if nRow > 0:
    ret += """      elif isinstance(other, MatrixXd):
        return self.__matmul(other)
      elif isinstance(other, Vector{4}d):
        return self.__fixvecmul(other)
      else:
        return self.__mul(other)
    else:
      return other.__mul__(self)
""".format(className, type, n2c(nRow), n2c(nCol), nRow)
  else:
    for i in GEN_DIMS:
      ret += """      elif isinstance(other, Vector{0}d):
        return self.__vec{0}mul(other)
""".format(i)
    for i in GEN_DIMS:
      ret += """      elif isinstance(other, Matrix{0}d):
        return self.__mat{0}mul(other)
""".format(i)
    ret += """      else:
        return self.__mul(other)
    else:
      return other.__mul__(self)
""".format(i)
  if nRow == 3 and nCol == 3:
    ret += """  def eulerAngles(self, int a0, int a1, int a2):
    ret = Vector3d()
    ret.impl = c_eigen_private.EigenEulerAngles(self.impl, a0, a1, a2)
    return ret
"""
  if nRow > 0:
    ret += """  @staticmethod
  def Random():
    ret = {0}()
    ret.impl = c_eigen_private.EigenRandom[{1},{2},{3}]()
    return ret
  @staticmethod
  def Zero():
    ret = {0}()
    ret.impl = c_eigen_private.EigenZero[{1},{2},{3}]()
    return ret
  @staticmethod
  def Identity():
    ret = {0}()
    ret.impl = c_eigen_private.EigenIdentity[{1},{2},{3}]()
    return ret
""".format(className, type, n2c(nRow), n2c(nCol))
  else:
    ret += """  @staticmethod
  def Random(int row, int col):
    ret = {0}()
    ret.impl = c_eigen_private.EigenRandom[{1},{2},{3}](row, col)
    return ret
  @staticmethod
  def Zero(int row, int col):
    ret = {0}()
    ret.impl = c_eigen_private.EigenZero[{1},{2},{3}](row, col)
    return ret
  @staticmethod
  def Identity(int row, int col):
    ret = {0}()
    ret.impl = c_eigen_private.EigenIdentity[{1},{2},{3}](row, col)
    return ret
""".format(className, type, n2c(nRow), n2c(nCol))
  ret += """cdef {0} {0}FromC(const c_eigen.{0} & arg):
  cdef {0} ret = {0}()
  ret.impl = arg
  return ret

cdef class {0}Vector(object):
  def __add{0}(self, {0} pt):
    self.v.push_back(pt.impl)
  def __cinit__(self, *args):
    if len(args) == 1 and isinstance(args[0], list):
      for pt in args[0]:
        self.__add{0}(pt)
    elif len(args) == 1 and isinstance(args[0], {0}):
      self.__add{0}(args[0])
    else:
      for pt in args:
        self.__add{0}(pt)

""".format(className)
  return ret

def generateVectorBinding(className, type, nRow, nCol):
  ret = generateBaseBinding(className, type, nRow, nCol)
  # Constructors, accessors, and mult operations differ from base binding
  ret += """  def __vctor(self, v):
    assert(len(v) <= self.rows())
    for (i,vi) in enumerate(v):
      if hasattr(vi, '__len__'):
        if(len(vi) == 1):
          self.impl[i] = vi[0]
        else:
          raise IndexError("Setting an element with a sequence")
      else:
        self.impl[i] = vi
  def __cinit__(self, *args):
"""
  if nRow > 0:
    ret += """
    if len(args) == 0:
      self.impl = c_eigen_private.EigenZero[{1},{2},{3}]()
    elif len(args) == 1:
      assert(len(args[0]) == {4})
      self.__vctor(args[0])
    elif len(args) == {4}:
      self.__vctor(args)
""".format(className, type, n2c(nRow), n2c(nCol), nRow)
  else:
    ret += """
    if len(args) == 0:
      self.impl = c_eigen_private.EigenZero[{1},{2},{3}](0)
    elif len(args) > 1:
      self.impl = c_eigen_private.EigenZero[{1},{2},{3}](len(args))
      self.__vctor(args)
""".format(className, type, n2c(nRow), n2c(nCol))
  ret += """    elif len(args) == 1:
        if isinstance(args[0], {0}):
          self.__copyctor__(args[0])
        elif isinstance(args[0], int):
          self.impl = c_eigen_private.EigenZero[{1},{2},{3}](args[0])
        else:
          self.impl = c_eigen_private.EigenZero[{1},{2},{3}](len(args[0]))
          self.__vctor(args[0])
    else:
      raise TypeError("Unsupported argument types passed to {0} ctor: " + ", ".join([str(type(x)) for x in args]))
  def __getitem__(self, idx):
    if isinstance(idx, tuple):
      if idx[1] == 0:
        return self.__getitem__(idx[0])
      else:
        raise IndexError("Colum index can only be zero in vector")
    if isinstance(idx, slice):
      indices = idx.indices(self.impl.rows())
      return [[self.__getitem__(i)] for i in range(*indices)]
    if abs(idx) < self.impl.rows() or idx == -self.impl.rows():
      if idx < 0:
        idx = self.impl.rows() + idx
      return self.impl(idx)
    else:
      raise IndexError("Index larger than number of rows")
  def __setitem__(self, idx, value):
    if isinstance(idx, slice):
      indices = idx.indices(self.impl.rows())
      for j, i in enumerate(range(*indices)):
        self.__setitem__(i, value[j])
    elif idx < self.impl.rows():
      if(hasattr(value, '__len__')):
          if(len(value) == 1):
            self.impl[idx] = value[0]
          else:
            raise IndexError("Trying to set an element with a sequence")
      else:
        self.impl[idx] = value
  def __mul__(self, other):
    if isinstance(self, {0}):
      if isinstance(other, MatrixXd):
        return self.__matmul(other)
      else:
        return self.__mul(other)
    else:
      return other.__mul__(self)
  def __imul__(self, other):
    self.impl = self.impl.scalar_mul(other)
    return self
  def transpose(self):
    ret = MatrixXd()
    ret.impl = <c_eigen.MatrixXd>self.impl.transpose()
    return ret
  def dot(self, {0} other):
    return self.impl.dot(other.impl)
  def coeff(self, int row, value = None):
    if value is None:
      return self.impl.coeff(row, 0)
    else:
      c_eigen_private.EigenSetValue[{1},{2},{3}](self.impl, row, 0, value)
""".format(className, type, n2c(nRow), n2c(nCol))
  if nRow == 3:
    ret += """  def cross(self, {0} other):
    ret = {0}()
    ret.impl = self.impl.cross(other.impl)
    return ret
""".format(className)
  if nRow > 0:
    ret += """  def x(self):
    return self.impl[0]
  @staticmethod
  def UnitX():
    return {0}({1})
""".format(className, ','.join([str(float(x == 0)) for x in range(nRow)]))
  if nRow > 1:
    ret += """  def y(self):
    return self.impl[1]
  @staticmethod
  def UnitY():
    return {0}({1})
""".format(className, ','.join([str(float(x == 1)) for x in range(nRow)]))
  if nRow > 2:
    ret += """  def z(self):
    return self.impl[2]
  @staticmethod
  def UnitZ():
    return {0}({1})
""".format(className, ','.join([str(float(x == 2)) for x in range(nRow)]))
  if nRow > 3:
    ret += """  def w(self):
    return self.impl[3]
  @staticmethod
  def UnitW():
    return {0}({1})
""".format(className, ','.join([str(float(x == 3)) for x in range(nRow)]))
  if nRow > 0:
    ret += """  @staticmethod
  def Random():
    ret = {0}()
    ret.impl = c_eigen_private.EigenRandom[{1},{2},{3}]()
    return ret
  @staticmethod
  def Zero():
    ret = {0}()
    ret.impl = c_eigen_private.EigenZero[{1},{2},{3}]()
    return ret
""".format(className, type, n2c(nRow), n2c(nCol))
  else:
    ret += """  @staticmethod
  def Random(int row):
    ret = {0}()
    ret.impl = c_eigen_private.EigenRandom[{1},{2},{3}](row)
    return ret
  @staticmethod
  def Zero(int row):
    ret = {0}()
    ret.impl = c_eigen_private.EigenZero[{1},{2},{3}](row)
    return ret
""".format(className, type, n2c(nRow), n2c(nCol))
  ret += """  def __matmul(self, MatrixXd m):
    if not m.impl.rows() == {0}:
      raise TypeError("Vector can only be multiplied by 1-row matrices")""".format(nCol)
  if(nRow >= 0):
    ret+="""
    if m.impl.cols() == {4}:
      ret = Matrix{4}dFromC(<c_eigen.Matrix{4}d>c_eigen_private.EigenMul[{1},{2},{3},c_eigen.dynamic,c_eigen.dynamic](self.impl, m.impl))
    else:
      ret = MatrixXdFromC(<c_eigen.MatrixXd>c_eigen_private.EigenMul[{1},{2},{3},c_eigen.dynamic,c_eigen.dynamic](self.impl, m.impl))""".format(className, type, n2c(nRow), n2c(nCol), nRow)
  else:
    ret +="""
    ret = MatrixXdFromC(<c_eigen.MatrixXd>c_eigen_private.EigenMul[{1},c_eigen.dynamic, {3}, c_eigen.dynamic, c_eigen.dynamic](self.impl, m.impl))""".format(className, type, n2c(nRow), n2c(nCol))
  ret+= """
    return ret

"""
  ret += """cdef {0} {0}FromC(const c_eigen.{0} & arg):
  cdef {0} ret = {0}()
  ret.impl = arg
  return ret

cdef class {0}Vector(object):
  def __add{0}(self, {0} pt):
    self.v.push_back(pt.impl)
  def __cinit__(self, *args):
    if len(args) == 1 and isinstance(args[0], list):
      for pt in args[0]:
        self.__add{0}(pt)
    elif len(args) == 1 and isinstance(args[0], {0}):
      self.__add{0}(args[0])
    else:
      for pt in args:
        self.__add{0}(pt)

""".format(className)
  return ret

def generate_eigen_pyx(out_path, utils_path):
  if os.path.exists(out_path + '/eigen.pyx'):
    assert(os.path.exists(out_path + '/eigen.pxd'))
    sha512 = hashlib.sha512()
    src_files = [out_path + '/eigen.pyx', out_path + '/eigen.pxd']
    for f in src_files:
      chunk = 2**16
      with open(f, 'r') as fd:
        while True:
          data = fd.read(chunk)
          if data:
            sha512.update(data.encode('ascii'))
          else:
            break
    output_hash = sha512.hexdigest()[:7]
  else:
    output_hash = None
  with open("{}/__init__.py".format(out_path), 'w') as fd:
    fd.write("from .eigen import *\n")
  with open("{}/eigen.pyx.tmp".format(out_path), 'w') as fd:
    fd.write("""# distutils: language = c++

from __future__ import division
import numpy
from cython.operator cimport dereference as deref
from libcpp.vector cimport vector
cimport c_eigen
cimport c_eigen_private

""")
    fd.write("# This file was automatically generated, do not modify it\n")
    fd.write(generateMatrixBinding("MatrixXd", "double", -1, -1))
    fd.write(generateMatrixBinding("Matrix2d", "double", 2, 2))
    fd.write(generateMatrixBinding("Matrix3d", "double", 3, 3))
    fd.write(generateMatrixBinding("Matrix4d", "double", 4, 4))
    fd.write(generateMatrixBinding("Matrix6d", "double", 6, 6))
    fd.write(generateVectorBinding("VectorXd", "double", -1, 1))
    fd.write(generateVectorBinding("Vector2d", "double", 2, 1))
    fd.write(generateVectorBinding("Vector3d", "double", 3, 1))
    fd.write(generateVectorBinding("Vector4d", "double", 4, 1))
    fd.write(generateVectorBinding("Vector6d", "double", 6, 1))
    with open('{}/quaternion.in.pyx'.format(utils_path),'r') as qfd:
      fd.write(qfd.read())
    with open('{}/angleaxis.in.pyx'.format(utils_path),'r') as effd:
      fd.write(effd.read())
  with open("{}/eigen.pxd.tmp".format(out_path), 'w') as fd:
    fd.write("# This file was automatically generated, do not modify it\n")
    fd.write("cimport numpy\n")
    fd.write("from cython.view cimport array as cvarray\n")
    fd.write("from libcpp.vector cimport vector\n")
    fd.write("cimport c_eigen\n")
    fd.write(generateDeclaration("MatrixXd"))
    fd.write(generateDeclaration("Matrix2d"))
    fd.write(generateDeclaration("Matrix3d"))
    fd.write(generateDeclaration("Matrix4d"))
    fd.write(generateDeclaration("Matrix6d"))
    fd.write(generateDeclaration("VectorXd"))
    fd.write(generateDeclaration("Vector2d"))
    fd.write(generateDeclaration("Vector3d"))
    fd.write(generateDeclaration("Vector4d"))
    fd.write(generateDeclaration("Vector6d"))
    fd.write(generateDeclaration("Quaterniond"))
    fd.write(generateDeclaration("AngleAxisd"))
  if output_hash is not None:
    sha512 = hashlib.sha512()
    src_files = [out_path + '/eigen.pyx.tmp', out_path + '/eigen.pxd.tmp']
    for f in src_files:
      chunk = 2**16
      with open(f, 'r') as fd:
        while True:
          data = fd.read(chunk)
          if data:
            sha512.update(data.encode('ascii'))
          else:
            break
    new_output_hash = sha512.hexdigest()[:7]
    if output_hash == new_output_hash:
      os.unlink('{}/eigen.pyx.tmp'.format(out_path))
      os.unlink('{}/eigen.pxd.tmp'.format(out_path))
      return
  if os.path.exists('{}/eigen.pyx'.format(out_path)):
    os.unlink('{}/eigen.pyx'.format(out_path))
  if os.path.exists('{}/eigen.pxd'.format(out_path)):
    os.unlink('{}/eigen.pxd'.format(out_path))
  os.rename('{}/eigen.pyx.tmp'.format(out_path), '{}/eigen.pyx'.format(out_path))
  os.rename('{}/eigen.pxd.tmp'.format(out_path), '{}/eigen.pxd'.format(out_path))
