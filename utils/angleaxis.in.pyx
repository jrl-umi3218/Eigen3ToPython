cdef class AngleAxisd(object):
  def __aaxisctor__(self, angle, Vector3d axis):
    self.impl = c_eigen.AngleAxisd(angle, axis.impl)
  def __quatctor__(self, Quaterniond other):
    self.impl = c_eigen_private.EigenAAFromQ[double](other.impl)
  def __cinit__(self, *args):
    if len(args) == 0:
      self.impl = c_eigen.AngleAxisd()
    elif len(args) == 1 and isinstance(args[0], Quaterniond):
      self.__quatctor__(*args)
    elif len(args) == 2 and isinstance(args[1], Vector3d):
      self.__aaxisctor__(*args)
    else:
      raise TypeError("Wrong arguments passed to AngleAxisd ctor")
  def matrix(self):
    return Matrix3dFromC(<c_eigen.Matrix3d>(self.impl.matrix()))
  def inverse(self):
    return AngleAxisdFromC(self.impl.inverse())
  def angle(self):
    return self.impl.angle()
  def axis(self):
    return Vector3dFromC(<c_eigen.Vector3d>(self.impl.axis()))

cdef AngleAxisd AngleAxisdFromC(const c_eigen.AngleAxisd & arg):
  cdef AngleAxisd ret = AngleAxisd()
  ret.impl = arg
  return ret
