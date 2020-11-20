# -*- coding: utf-8 -*-
#
# Copyright 2012-2019 CNRS-UM LIRMM, CNRS-AIST JRL
#

from conans import ConanFile, CMake, tools
from conans.tools import os_info, SystemPackageTool
import os
import shutil
import subprocess
import sys

def get_python_version(cmd = 'python'):
    # Get the version of the `cmd` command assumed to be a python interpreter
    try:
        return '.'.join(subprocess.check_output('{} -V'.format(cmd).split(), stderr = subprocess.STDOUT).strip().split()[1].decode().split('.')[0:2])
    except OSError:
        return None

def get_default_options():
    if os_info.is_windows:
        return { "python2_version": None, "python3_version": get_python_version() }
    else:
        return { "python2_version": get_python_version('python2'), "python3_version": get_python_version('python3') }

def enable_python2_and_python3(options):
    return options['python2_version'] is not None and options['python3_version'] is not None and not os_info.is_windows

class Eigen3ToPythonConan(ConanFile):
    name = "Eigen3ToPython"
    version = "1.0.2"
    description = "Python bindings for the Eigen library"
    # topics can get used for searches, GitHub topics, Bintray tags etc. Add here keywords about the library
    topics = ("eigen", "python")
    url = "https://github.com/jrl-umi3218/Eigen3ToPython"
    homepage = "https://github.com/jrl-umi3218/Eigen3ToPython"
    author = "Pierre Gergondet <pierre.gergondet@gmail.com>"
    license = "BSD-2-Clause"  # Indicates license type of the packaged library; please use SPDX Identifiers https://spdx.org/licenses/
    exports = ["LICENSE"]      # Packages the license for the conanfile.py
    exports_sources = ["CMakeLists.txt", "requirements.txt", "setup.in.py", "conan/CMakeLists.txt", "eigen/*", "include/*", "tests/*", "utils/*"]
    generators = "cmake"
    options = {
            "python2_version": [None, "2.7"],
            "python3_version": [None, "3.3", "3.4", "3.5", "3.6", "3.7", "3.8", "3.9"]
    }
    default_options = get_default_options()

    settings = "os", "arch", "compiler"

    requires = (
        "eigen/3.3.4@conan/stable"
    )

    def system_requirements(self):
        if os_info.is_linux:
            installer = SystemPackageTool()
            packages = ''
            if self.default_options['python2_version'] is not None:
                packages = 'cython python-coverage python-nose python-numpy '
            if self.default_options['python3_version'] is not None:
                packages += 'cython3 python3-coverage python3-nose python3-numpy'
            if len(packages):
                installer.install(packages)
        else:
            if enable_python2_and_python3(self.default_options):
                subprocess.run("pip2 install --user Cython>=0.2 coverage nose numpy>=1.8.2".split())
                subprocess.run("pip3 install --user Cython>=0.2 coverage nose numpy>=1.8.2".split())
            else:
                subprocess.run("pip install --user Cython>=0.2 coverage nose numpy>=1.8.2".split())

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
        cmake.definitions['CMAKE_BUILD_TYPE'] = self.settings.get_safe("build_type", "Release")
        cmake.definitions['PIP_INSTALL_PREFIX'] = self.package_folder
        cmake.definitions['PYTHON_BINDING_BUILD_PYTHON2_AND_PYTHON3'] = enable_python2_and_python3(self.options)
        cmake.configure()
        return cmake

    def build(self):
        cmake = self._configure_cmake()
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

    def package_id(self):
        del self.info.settings.compiler.runtime
        if self.options.python2_version == "None":
            self.info.options.python2_version = "2.7"
        if self.options.python3_version == "None":
            for v3 in ["3.9", "3.8", "3.7", "3.6", "3.5", "3.4", "3.3"]:
                compatible_pkg = self.info.clone()
                compatible_pkg.options.python3_version = v3
                self.compatible_packages.append(compatible_pkg)
