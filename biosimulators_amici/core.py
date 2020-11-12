""" Methods for executing SED tasks in COMBINE archives and saving their outputs

:Author: Jonathan Karr <karr@mssm.edu>
:Date: 2020-10-29
:Copyright: 2020, Center for Reproducible Biomedical Modeling
:License: MIT
"""

from Biosimulations_utils.simulation.data_model import TimecourseSimulation
from Biosimulations_utils.simulator.utils import exec_simulations_in_archive
import amici
import importlib.util
import numpy
import os.path
import pandas
import sys
import tempfile

__all__ = ['exec_combine_archive', 'exec_simulation']

KISAO_ALGORITHMS_MAP = {
    'KISAO_0000496': 'CVODES',
}

KISAO_PARAMETERS_MAP = {
    'KISAO_0000209': 'setRelativeTolerance',
    'KISAO_0000211': 'setAbsoluteTolerance',
    'KISAO_0000415': 'setMaxSteps',
    'KISAO_0000543': 'setStabilityLimitFlag',
}


def exec_combine_archive(archive_file, out_dir):
    """ Execute the SED tasks defined in a COMBINE archive and save the outputs

    Args:
        archive_file (:obj:`str`): path to COMBINE archive
        out_dir (:obj:`str`): directory to store the outputs of the tasks
    """
    exec_simulations_in_archive(archive_file, exec_simulation, out_dir, apply_model_changes=True)


def exec_simulation(model_filename, model_sed_urn, simulation, working_dir, out_filename, out_format):
    ''' Execute a simulation and save its results

    Args:
       model_filename (:obj:`str`): path to the model
       model_sed_urn (:obj:`str`): SED URN for the format of the model (e.g., `urn:sedml:language:sbml`)
       simulation (:obj:`TimecourseSimulation`): simulation
       working_dir (:obj:`str`): directory of the SED-ML file
       out_filename (:obj:`str`): path to save the results of the simulation
       out_format (:obj:`str`): format to save the results of the simulation (e.g., `csv`)
    '''
    if not isinstance(simulation, TimecourseSimulation):
        raise ValueError('{} is not supported'.format(simulation.__class__.__name__))

    # check that model parameter changes have already been applied (because handled by :obj:`exec_simulations_in_archive`)
    if simulation.model_parameter_changes:
        raise ValueError('Model parameter changes are not supported')

    # Read the model located at `os.path.join(working_dir, model_filename)` in the format
    # with the SED URN `model_sed_urn`.
    if model_sed_urn != "urn:sedml:language:sbml":
        raise ValueError("Model language with URN '{}' is not supported".format(model_sed_urn))

    sbml_importer = amici.SbmlImporter(model_filename)
    sbml_model = sbml_importer.sbml    

    model_output_dir = tempfile.mkdtemp()
    model_name = 'biosimulators_amici_model_' + os.path.basename(model_output_dir)
    constant_parameters = [param.getId() for param in sbml_model.parameters]
    observables = {var.id: {'name': var.name or var.id, 'formula': var.id} for var in simulation.model.variables}
    sbml_importer.sbml2amici(model_name,
                             model_output_dir,
                             observables=observables,
                             constant_parameters=constant_parameters)

    model_module_spec = importlib.util.spec_from_file_location(model_name, os.path.join(model_output_dir, model_name, '__init__.py'))
    model_module = importlib.util.module_from_spec(model_module_spec)
    sys.modules[model_name] = model_module
    model_module_spec.loader.exec_module(model_module)
    model = model_module.getModel()

    # Simulate the model from `simulation.start_time` to `simulation.end_time`
    model.setT0(simulation.start_time)
    model.setTimepoints(numpy.linspace(simulation.output_start_time, simulation.end_time, simulation.num_time_points + 1))

    # Load the algorithm specified by `simulation.algorithm`
    algorithm_name = KISAO_ALGORITHMS_MAP.get(simulation.algorithm.kisao_term.id, None)
    if algorithm_name is None:
        raise ValueError(
            "Algorithm with KiSAO id '{}' is not supported".format(simulation.algorithm.kisao_term.id))

    solver = model.getSolver()

    # Apply the algorithm parameter changes specified by `simulation.algorithm_parameter_changes`
    for parameter_change in simulation.algorithm_parameter_changes:
        param_setter_name = KISAO_PARAMETERS_MAP.get(parameter_change.parameter.kisao_term.id, None)
        if param_setter_name is None:
            raise ValueError(
                "Algorithm parameter with KiSAO id '{}' is not supported".format(parameter_change.parameter.kisao_term.id))
        param_setter = getattr(solver, param_setter_name)
        param_setter(parameter_change.value)

    # Run simulation using default model parameters and solver options
    results = amici.runAmiciSimulation(model, solver)

    # Save a report of the results of the simulation with `simulation.num_time_points` time points
    # beginning at `simulation.output_start_time` to `out_filename` in `out_format` format.
    # This should save all of the variables specified by `simulation.model.variables`.
    results_matrix = numpy.zeros((simulation.num_time_points + 1, len(simulation.model.variables) + 1))
    results_matrix[:, 0] = results['ts']

    state_ids = model.getStateIds()
    var_ids = sorted([var.id for var in simulation.model.variables])

    unpredicted_vars = set(var_ids).difference(state_ids)
    if unpredicted_vars:
        raise ValueError('The simulation did not record the following required outputs:\n  - {}'.format(
            '\n  - '.join(sorted(unpredicted_vars))))

    for i_var, var_id in enumerate(var_ids):
        i_state = state_ids.index(var_id)
        results_matrix[:, i_var + 1] = results['x'][:, i_state]

    results_df = pandas.DataFrame(results_matrix, columns=var_ids)
    results_df.to_csv(out_filename, index=False)

    # cleanup module and temporary directory
    sys.modules.pop(model_name)
    shutils.rmdir(model_output_dir)
