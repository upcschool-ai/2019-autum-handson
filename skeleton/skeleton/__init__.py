import pkg_resources

__version__ = pkg_resources.resource_string(__package__, '__version__').strip()
