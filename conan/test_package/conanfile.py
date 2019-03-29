from conans import ConanFile, tools

class Eigen3ToPythonTestConan(ConanFile):
    requires = "Eigen3ToPython/1.0.0@gergondet/stable"

    def test(self):
      # self.conanfile_directory
      with tools.pythonpath(self):
          import eigen
          print("Random Vector3d: %s" % eigen.Vector3d.Random())
