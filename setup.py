import os
from setuptools import setup, find_packages


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name="gim_cv",

    description="Computer vision tools for informal settlement delineation on rasters",

    author="Laurent Nlemba, Nicolas S. Matton",

    packages=find_packages(exclude=['data', 'figures', 'output', 'notebooks']),

    long_description=read('README.md'),
)
