import os
import sys
import platform
import subprocess

from pprint import pprint

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext


class CMakeExtension(Extension):
    def __init__(self, name, cmake_lists_dir=''):
        Extension.__init__(self, name, sources=[])
        self.cmake_lists_dir = os.path.abspath(cmake_lists_dir)


class CMakeBuild(build_ext):
    def run(self):
        try:
            subprocess.check_output(['cmake', '--version'])
        except OSError:
            raise RuntimeError('Cannot find CMake executable')

        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):

        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))

        debug = int(os.environ.get("DEBUG", 0)) if self.debug is None else self.debug
        cfg = "Debug" if debug else "Release"
        build_args = ['--config', cfg]

        cmake_args = [
            '-DSTK_BUILD_PYTHON_WRAPPER=ON',
            '-DSTK_BUILD_TESTS=OFF',
            '-DSTK_BUILD_WITH_DEBUG_INFO=%s' % ('ON' if cfg == 'Debug' else 'OFF'),
            '-DCMAKE_BUILD_TYPE=%s' % cfg,
            '-DPYTHON_EXECUTABLE=' + sys.executable,
            '-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + extdir,
        ]

        if platform.system() == "Windows":
            cmake_args += ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{}={}'.format(cfg.upper(), extdir)]

        # Adding CMake arguments set as environment variable
        if "CMAKE_ARGS" in os.environ:
            cmake_args += [item for item in os.environ["CMAKE_ARGS"].split(" ") if item]

        pprint(cmake_args)

        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)

        subprocess.check_call(['cmake', ext.cmake_lists_dir] + cmake_args, cwd=self.build_temp)
        subprocess.check_call(['cmake', '--build', '.'] + build_args, cwd=self.build_temp)


setup(
    ext_modules=[CMakeExtension('_stk', '.')],
    cmdclass={'build_ext': CMakeBuild},
)

