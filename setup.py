from distutils.core import setup

setup(
    name='matrix_factorization',
    version='0.0.1',
    description='Movie recommendation based on matrix factorization',
    long_description=readme,
    author='Keisuke Yanagi',
    author_email='k.yanagi07@gmail.com',
    install_requires=['numpy', 'pandas', 'matplotlib'],
    url='https://github.com/keisuke-yanagi/matrix_factorization',
    license=license,
    packages=find_packages(exclude=('tests'))
)
