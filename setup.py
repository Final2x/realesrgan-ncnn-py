# coding:utf-8

from setuptools import setup
from wheel.bdist_wheel import bdist_wheel as _bdist_wheel
import sys

_version = '1.3.0'


class bdist_wheel(_bdist_wheel):
    def finalize_options(self):
        _bdist_wheel.finalize_options(self)
        self.root_is_pure = False


_options = {}
if sys.platform == "darwin":
    _options = {
        "bdist_wheel": {"plat_name": "macosx_11_0_universal2"},
    } if sys.version_info[1] >= 9 else {
        "bdist_wheel": {"plat_name": "macosx_10_15_x86_64"},
    }
elif sys.platform == "linux":
    _options = {
        "bdist_wheel": {"plat_name": "manylinux1_x86_64"},
    }
    

setup(
    name='realesrgan_ncnn_py',
    version=_version,
    description='',
    long_description=open('README.md', encoding='utf-8').read(),
    long_description_content_type='text/markdown',
    author='Tohrusky',
    url='https://github.com/Tohrusky/realesrgan-ncnn-py',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: BSD License',
        # 目标 Python 版本
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],
    packages=['realesrgan_ncnn_py'],
    include_package_data=True,
    python_requires='>=3.6',
    install_requires=['pillow', 'opencv_python'],
    cmdclass={'bdist_wheel': bdist_wheel},
    options=_options,
)
