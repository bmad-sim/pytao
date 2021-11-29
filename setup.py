from setuptools import setup, find_packages
from os import path, environ
from codecs import open

import versioneer

cur_dir = path.abspath(path.dirname(__file__))

with open(path.join(cur_dir, 'requirements.txt'), 'r') as f:
    requirements = f.read().split()

with open(path.join(cur_dir, 'README.md'), 'r', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='pytao',
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    packages=find_packages(),  
    package_dir={'pytao':'pytao'},
    url='https://www.classe.cornell.edu/bmad/tao.html',
    long_description=long_description,
    long_description_content_type='text/markdown',
    install_requires=requirements,
    include_package_data=True,
    python_requires='>=3.6'
)
