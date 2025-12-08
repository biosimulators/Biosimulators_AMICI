import amici

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("biosimulators-amici")
except PackageNotFoundError:
    # package is not installed
    pass

from .core import exec_sed_task, preprocess_sed_task, exec_sed_doc, exec_sedml_docs_in_combine_archive  # noqa: F401

__all__ = [
    '__version__',
    'get_simulator_version',
    'exec_sed_task',
    'preprocess_sed_task',
    'exec_sed_doc',
    'exec_sedml_docs_in_combine_archive',
]


def get_simulator_version():
    """ Get the version of AMICI

    Returns:
        :obj:`str`: version
    """
    return amici.__version__
