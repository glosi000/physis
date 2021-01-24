from setuptools import setup

with open("README.md", 'r') as f:
    long_description = f.read()

setup(
    name='physis',
    version='0.0.0',
    description='Modules for physical and chemical sciences.',
    long_description=long_description,
    url='https://github.com/glosi000/physis',
    author='Gabriele Losi',
    author_email='gabriele.losi@outlook.com',
    maintainer='Gabriele Losi',
    maintainer_email='gabriele.losi@outlook.com',
    license='BSD-4-Clause License',
    install_requires=['numpy', 'matplotlib', 'numba'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD-4-Clause License",
        "Operating System :: OS Independent",
    ]
)
