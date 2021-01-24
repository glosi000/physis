from setuptools import setup

setup(
    name='physis',
    version=0.0.0,
    description='Modules for physical and chemical sciences.'
    long_description=open('README.md').read(),
    url='https://github.com/glosi000/physis'
    author='Gabriele Losi'
    author_email='gabriele.losi@outlook.com',
    maintainer='Gabriele Losi'
    maintainer_email='gabriele.losi@outlook.com',
    license='BSD-4-Clause License',
    install_requires=['numpy', 'matplotlib', 'numba']
)
