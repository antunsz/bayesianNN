from setuptools import setup, find_packages

setup(
    name='bayesianNN',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'tensorflow',
        'pgmpy',
    ],
)

