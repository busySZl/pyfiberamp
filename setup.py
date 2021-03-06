from setuptools import setup

VERSION = '0.3.0'

long_description = '''PyFiberAmp is a powerful simulation library for modeling 
rare-earth-doped fiber lasers and amplifiers using rate equations.'''

setup(
    name='PyFiberAmp',
    version=VERSION,
    author='Joona Rissanen',
    author_email='joona.m.rissanen@gmail.com',
    url='https://github.com/Jomiri/pyfiberamp',
    description='Fiber amplifier modeling library',
    long_description=long_description,
    license='MIT',
    install_requires=[
        'matplotlib',
        'numpy',
        'scipy'
    ],
    packages=[
        'pyfiberamp',
        'pyfiberamp.fibers',
        'pyfiberamp.dynamic',
        'pyfiberamp.steady_state',
        'pyfiberamp.steady_state.models',
        'pyfiberamp.spectroscopies',
        'pyfiberamp.util'
    ],
    package_data={'pyfiberamp': ['spectroscopies/fiber_spectra/*.dat',
                                 'dynamic/fiber_simulation_pybindings.cp36-win_amd64.pyd']}
)
