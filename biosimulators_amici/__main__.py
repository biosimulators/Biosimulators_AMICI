""" BioSimulators-compliant command-line interface to the `AMICI <https://github.com/AMICI-dev/AMICI>`_ simulation program.

:Author: Jonathan Karr <karr@mssm.edu>
:Date: 2020-12-16
:Copyright: 2020, BioSimulators Team
:License: MIT
"""

from . import get_simulator_version
from ._version import __version__
from .core import exec_sedml_docs_in_combine_archive
from biosimulators_utils.simulator.cli import build_cli

App = build_cli('biosimulators-amici', __version__,
                'AMICI', get_simulator_version(), 'https://github.com/AMICI-dev/AMICI',
                exec_sedml_docs_in_combine_archive)


def main():
    with App() as app:
        app.run()
