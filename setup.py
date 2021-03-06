from distutils.core import setup
from setuptools import find_packages

setup(
  name = 'pytorchUtils',
  packages = find_packages(),
  version = '0.0.3',
  license='MIT',
  description = 'Some utility classes for use with pytorch',
  author = 'Alexandros Benetatos',
  author_email = 'alexandrosbene@gmail.com',
  url = 'https://github.com/alex-bene/pytorch-utils',
  download_url = 'https://github.com/alex-bene/pytorch-utils/releases/download/v0.0.3-beta/pytorchUtils-0.0.3.tar.gz',
  keywords = ['pytorch', 'utilities'],
  install_requires=[
          'tqdm',
          'numpy',
          'torch',
          'matplotlib',
          'torchvision',
                   ],
  classifiers=[
    'Development Status :: 4 - Beta', # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
    'Intended Audience :: Developers',
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3',
    'Operating System :: OS Independent',
              ],
)
