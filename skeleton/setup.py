import os

import setuptools

# Get version
cur_dir = os.path.dirname(__file__)
version_file = os.path.join(cur_dir, 'skeleton', '__version__')
with open(version_file, 'rt') as f:
    version = f.read().strip()

install_requires = [
    'numpy~=1.18',
    'opencv-contrib-python~=3.4.1',
    'pyyaml~=5.3',
    'tensorflow==1.13.2',
]

setuptools.setup(
    name='skeleton',
    version=version,
    description='Skeleton package for DL',
    packages=setuptools.find_packages(exclude=['tests', 'tests.*']),
    install_requires=install_requires,
    include_package_data=True,
    zip_safe=False
)
