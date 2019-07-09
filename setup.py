from setuptools import setup

def readme():
    with open('README.md') as f:
        return f.read()
        
setup(name='pyrobust',
      version='0.1',
      description='Robust Optimizer',
      long_description=readme(),
      classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3 :: Only',
        'Topic :: Scientific/Engineering',
        'Operating System :: OS Independent',
      ],
      keywords='optimization, optimisation, optimizer, optimiser, robust, genetic algorithms, evolutionary algorithms, GA, genetic',
      url='https://github.com/EPCCed/pyrobust/wiki/HOME',
      author='Neelofer Banglawala',
      author_email='n.banglawala@epcc.ed.ac.uk',
      license='MIT',
      packages=['pyrobust'],
      install_requires=[
          'numpy',
          'scipy',
          'pyDOE',
          'sobol_seq'
      ],
      test_suite='nose.collector',
      tests_require=['nose'],
      zip_safe=False)
