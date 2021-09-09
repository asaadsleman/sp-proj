from setuptools import setup, find_packages, Extension

setup(
    name='spkmeans',
    version='0.1.0',
    description='C - API extension',
    author='Asaad Sleman',
    install_requires=['invoke'],
    packages=find_packages(),
    license = 'GPL-2',
    ext_modules=[
        Extension(
            'spkmeans',
            ['spkmeansmodule.c', 'spkmeans.c'],
            ),
    ]
)