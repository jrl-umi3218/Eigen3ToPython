# -*- coding: utf-8 -*-
#
# Copyright 2012-2019 CNRS-UM LIRMM, CNRS-AIST JRL
#

from conans import ConanFile, CMake, tools
import os
import shutil
import subprocess
import sys

def get_python_version():
    # Get the version of the Python executable, not this one
    return '.'.join(subprocess.check_output('python -V'.split(), stderr = subprocess.STDOUT).strip().split()[1].decode().split('.')[0:2])

class Eigen3ToPythonConan(ConanFile):
    name = "Eigen3ToPython"
    version = "1.0.0"
    description = "Python bindings for the Eigen library"
    # topics can get used for searches, GitHub topics, Bintray tags etc. Add here keywords about the library
    topics = ("eigen", "python")
    url = "https://github.com/jrl-umi3218/Eigen3ToPython"
    homepage = "https://github.com/jrl-umi3218/Eigen3ToPython"
    author = "Pierre Gergondet <pierre.gergondet@gmail.com>"
    license = "BSD-2-Clause"  # Indicates license type of the packaged library; please use SPDX Identifiers https://spdx.org/licenses/
    exports = ["LICENSE"]      # Packages the license for the conanfile.py
    # Remove following lines if the target lib does not use cmake.
    exports_sources = ["CMakeLists.txt", "requirements.txt", "setup.in.py", "conan/CMakeLists.txt", "eigen/*", "include/*", "tests/*", "utils/*"]
    generators = "cmake"
    options = { "python_version": ["2.7", "3.3", "3.4", "3.5", "3.6", "3.7"] }
    default_options = { "python_version": get_python_version() }

    # Options may need to change depending on the packaged library.
    settings = "os", "arch"

    requires = (
        "eigen/3.3.4@conan/stable"
    )

    def source(self):
        # Wrap the original CMake file to call conan_basic_setup
        shutil.move("CMakeLists.txt", "CMakeListsOriginal.txt")
        shutil.move(os.path.join("conan", "CMakeLists.txt"), "CMakeLists.txt")

    def _extra_path(self):
        return os.path.join(self.package_folder, 'bin')

    def _extra_python_path(self):
        return os.path.join(self.package_folder, 'lib', 'python{}'.format(get_python_version()), 'site-packages')

    def _configure_cmake(self):
        os.environ['PATH'] =  self._extra_path() + os.pathsep + os.environ.get('PATH', '')
        os.environ['PYTHONPATH'] =  self._extra_python_path() + os.pathsep + os.environ.get('PYTHONPATH', '')
        cmake = CMake(self)
        cmake.definitions['DISABLE_TESTS'] = True
        cmake.definitions['PIP_INSTALL_PREFIX'] = self.package_folder
        cmake.configure()
        return cmake

    def build(self):
        cmake = self._configure_cmake()
        # This recipe is used down the line but we only want to run this in this package
        if isinstance(self, Eigen3ToPythonConan):
            subprocess.run(["pip", "install", "-r", "{}/requirements.txt".format(self.source_folder)])
        cmake.build()

    def package(self):
        cmake = self._configure_cmake()
        cmake.install()

    def deploy(self):
        self.copy("*")
        self.copy_deps("*")

    def package_info(self):
        self.env_info.PATH.append(self._extra_path())
        self.env_info.PYTHONPATH.append(self._extra_python_path())
