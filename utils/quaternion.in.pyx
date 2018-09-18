cdef class Quaterniond(object):
  def __copyctor__(self, Quaterniond other):
    self.impl = other.impl
  def __v4ctor__(self, Vector4d other):
    self.impl = c_eigen.Quaterniond(other.impl)
  def __aaxisctor__(self, double a, Vector3d ax):
    self.impl = c_eigen.Quaterniond(c_eigen.AngleAxisd(a, ax.impl))
  def __aaxisdctor__(self, AngleAxisd aax):
    self.impl = c_eigen.Quaterniond(aax.impl)
  def __m3ctor__(self, Matrix3d other):
    self.impl = c_eigen_private.QuaternionFromM3[double](other.impl)
  def __cinit__(self, *args):
    if len(args) == 0:
      self.impl = c_eigen.Quaterniond()
    elif len(args) == 1 and isinstance(args[0], Quaterniond):
      self.__copyctor__(args[0])
    elif len(args) == 1 and isinstance(args[0], Vector4d):
      self.__v4ctor__(args[0])
    elif len(args) == 1 and isinstance(args[0], AngleAxisd):
      self.__aaxisdctor__(args[0])
    elif len(args) == 1 and isinstance(args[0], Matrix3d):
      self.__m3ctor__(args[0])
    elif len(args) == 2 and isinstance(args[1], Vector3d):
      self.__aaxisctor__(args[0], args[1])
    elif len(args) == 4:
      self.impl = c_eigen.Quaterniond(args[0], args[1], args[2], args[3])
    else:
      raise TypeError("Unsupported argument types passed to Quaterniond ctor: {0}".format(", ".join([str(type(x)) for x in args])))
  def x(self):
    return self.impl.x()
  def y(self):
    return self.impl.y()
  def z(self):
    return self.impl.z()
  def w(self):
    return self.impl.w()
  def vec(self):
    ret = Vector3d()
    ret.impl = <c_eigen.Vector3d>(self.impl.vec())
    return ret
  def coeffs(self):
    ret = Vector4d()
    ret.impl = <c_eigen.Vector4d>(self.impl.coeffs())
    return ret
  def setIdentity(self):
    self.impl.setIdentity()
  def setFromTwoVectors(self, Vector3d v1, Vector3d v2):
    self.impl.setFromTwoVectors(v1.impl, v2.impl)
  def angularDistance(self, Quaterniond other):
    return self.impl.angularDistance(other.impl)
  def conjugate(self):
    ret = Quaterniond()
    ret.impl = self.impl.conjugate()
    return ret
  def __str__(self):
    return c_eigen_private.QtoString[double](self.impl).decode('utf-8')
  def __q_mul(self, Quaterniond other):
    return QuaterniondFromC(self.impl*other.impl)
  def __mul__(self, other):
    if isinstance(self, Quaterniond):
      if isinstance(other, Quaterniond):
        return self.__q_mul(other)
      else:
        raise TypeError("Unsupported operands Quaterniond and {0}".format(type(other)))
    else:
      return other.__mul__(self)
  def dot(self, Quaterniond other):
    return self.impl.dot(other.impl)
  def inverse(self):
    ret = Quaterniond()
    ret.impl = self.impl.inverse()
    return ret
  def isApprox(self, Quaterniond other, double prec = -1):
    if prec < 0:
      return self.impl.isApprox(other.impl)
    else:
      return self.impl.isApprox(other.impl, prec)
  def matrix(self):
    ret = Matrix3d()
    ret.impl = <c_eigen.Matrix3d>(self.impl.matrix())
    return ret
  def toRotationMatrix(self):
    ret = Matrix3d()
    ret.impl = <c_eigen.Matrix3d>(self.impl.toRotationMatrix())
    return ret
  def norm(self):
    return self.impl.norm()
  def normalize(self):
    self.impl.normalize()
  def normalized(self):
    ret = Quaterniond()
    ret.impl = self.impl.normalized()
    return ret
  def slerp(self, double s, Quaterniond q):
    ret = Quaterniond()
    ret.impl = self.impl.slerp(s, q.impl)
    return ret
  def squaredNorm(self):
    return self.impl.squaredNorm()
  @staticmethod
  def Identity():
    ret = Quaterniond()
    ret.setIdentity()
    return ret
  @staticmethod
  def UnitRandom():
    u1 = numpy.random.random()
    u2 = 2 * numpy.pi * numpy.random.random()
    u3 = 2 * numpy.pi * numpy.random.random()
    a = numpy.sqrt(1 - u1)
    b = numpy.sqrt(u1)
    return Quaterniond(a*numpy.sin(u2), a*numpy.cos(u2),
                       b*numpy.sin(u3), b*numpy.cos(u3)).normalized()

cdef Quaterniond QuaterniondFromC(const c_eigen.Quaterniond & arg):
  cdef Quaterniond ret = Quaterniond()
  ret.impl = arg
  return ret

def poly_eval(VectorXd poly, x):
  return c_eigen_private.poly_eval[double](poly.impl, x)
