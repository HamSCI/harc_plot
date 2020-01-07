#!/usr/bin/env python

from distutils.core import setup

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(name='harc_plot',
      version='0.1',
      description='HamSCI Amateur Radio Communications Database Plotting Tools',
      author='Nathaniel A. Frissell',
      author_email='nathaniel.frissell@scranton.edu',
      url='https://hamsci.org',
      packages=['harc_plot'],
      install_requires=requirements
     )
