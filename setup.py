from setuptools import setup, find_packages

setup(
  name = 'simple-parallel-transformer',
  packages = find_packages(exclude=['examples']),
  version = '0.0.1',
  license='GPL-3',
  description = 'Simple Parallel Transformer',
  author = 'Charles Foster',
  author_email = 'cfoster0@alumni.stanford.edu',
  url = 'https://github.com/cfoster0/simple-parallel-transformer',
  keywords = [
    'artificial intelligence',
    'attention mechanism',
    'transformers'
  ],
  install_requires=[
    'torch>=1.6',
    'einops>=0.3',
    'hydra-core',
  ],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.6',
  ],
)
