import os

import setuptools

# Get version
cur_dir = os.path.dirname(__file__)
version_file = os.path.join(cur_dir, u'skeleton', u'__version__')
with open(version_file, u'rt') as f:
    version = f.read().strip()

# Platform agnostic requirements
install_requires = [
]

setuptools.setup(
    name=u'skeleton',
    version=version,
    description=u'Skeleton package for DL',
    author=u'Issey Masuda Mora',
    packages=setuptools.find_packages(exclude=[u'tests', u'tests.*']),
    install_requires=install_requires,
    include_package_data=True,
    zip_safe=False
)
