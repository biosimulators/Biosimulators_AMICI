""" Mappings from KiSAO terms to methods and arguments

:Author: Jonathan Karr <karr@mssm.edu>
:Date: 2020-12-16
:Copyright: 2020, BioSimulators Team
:License: MIT
"""

from biosimulators_utils.data_model import ValueType

__all__ = ['KISAO_ALGORITHMS_MAP', 'KISAO_PARAMETERS_MAP']

KISAO_ALGORITHMS_MAP = {
    'KISAO_0000496': 'CVODES',
}

KISAO_PARAMETERS_MAP = {
    'KISAO_0000209': {
        'name': 'setRelativeTolerance',
        'type': ValueType.float,
        'default': 1e-8,
    },
    'KISAO_0000211': {
        'name': 'setAbsoluteTolerance',
        'type': ValueType.float,
        'default': 1e-16,
    },
    'KISAO_0000415': {
        'name': 'setMaxSteps',
        'type': ValueType.integer,
        'default': 10000,
    },
    'KISAO_0000543': {
        'name': 'setStabilityLimitFlag',
        'type': ValueType.boolean,
        'default': True,
    },
}
