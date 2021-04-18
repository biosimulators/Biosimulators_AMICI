""" Methods for executing SED tasks in COMBINE archives and saving their outputs

:Author: Jonathan Karr <karr@mssm.edu>
:Date: 2020-12-16
:Copyright: 2020, BioSimulators Team
:License: MIT
"""

from .data_model import KISAO_ALGORITHMS_MAP, KISAO_PARAMETERS_MAP
from biosimulators_utils.combine.exec import exec_sedml_docs_in_archive
from biosimulators_utils.log.data_model import CombineArchiveLog, TaskLog  # noqa: F401
from biosimulators_utils.plot.data_model import PlotFormat  # noqa: F401
from biosimulators_utils.report.data_model import ReportFormat, VariableResults  # noqa: F401
from biosimulators_utils.sedml.data_model import (Task, ModelLanguage, UniformTimeCourseSimulation,  # noqa: F401
                                                  Variable, Symbol)
from biosimulators_utils.sedml import validation
from biosimulators_utils.sedml.exec import exec_sed_doc
from biosimulators_utils.utils.core import validate_str_value, parse_value, raise_errors_warnings
import amici
import functools
import importlib.util
import numpy
import os.path
import shutil
import sys
import tempfile


__all__ = [
    'exec_sedml_docs_in_combine_archive',
    'exec_sed_task',
    'validate_sed_task',
    'import_model_from_sbml',
    'cleanup_model',
    'config_task',
    'exec_task',
    'extract_variables_from_results',
]


def exec_sedml_docs_in_combine_archive(archive_filename, out_dir,
                                       report_formats=None, plot_formats=None,
                                       bundle_outputs=None, keep_individual_outputs=None):
    """ Execute the SED tasks defined in a COMBINE/OMEX archive and save the outputs

    Args:
        archive_filename (:obj:`str`): path to COMBINE/OMEX archive
        out_dir (:obj:`str`): path to store the outputs of the archive

            * CSV: directory in which to save outputs to files
              ``{ out_dir }/{ relative-path-to-SED-ML-file-within-archive }/{ report.id }.csv``
            * HDF5: directory in which to save a single HDF5 file (``{ out_dir }/reports.h5``),
              with reports at keys ``{ relative-path-to-SED-ML-file-within-archive }/{ report.id }`` within the HDF5 file

        report_formats (:obj:`list` of :obj:`ReportFormat`, optional): report format (e.g., csv or h5)
        plot_formats (:obj:`list` of :obj:`PlotFormat`, optional): report format (e.g., pdf)
        bundle_outputs (:obj:`bool`, optional): if :obj:`True`, bundle outputs into archives for reports and plots
        keep_individual_outputs (:obj:`bool`, optional): if :obj:`True`, keep individual output files

    Returns:
        :obj:`CombineArchiveLog`: log
    """
    sed_doc_executer = functools.partial(exec_sed_doc, exec_sed_task)
    return exec_sedml_docs_in_archive(sed_doc_executer, archive_filename, out_dir,
                                      apply_xml_model_changes=True,
                                      report_formats=report_formats,
                                      plot_formats=plot_formats,
                                      bundle_outputs=bundle_outputs,
                                      keep_individual_outputs=keep_individual_outputs)


def exec_sed_task(task, variables, log=None):
    ''' Execute a task and save its results

    Args:
       task (:obj:`Task`): task
       variables (:obj:`list` of :obj:`Variable`): variables that should be recorded
       log (:obj:`TaskLog`, optional): log for the task

    Returns:
        :obj:`tuple`:

            :obj:`VariableResults`: results of variables
            :obj:`TaskLog`: log
    '''
    log = log or TaskLog()

    target_x_paths_ids = validate_sed_task(task, variables)

    # Read the model for the task
    model, sbml_model, model_name, model_dir = import_model_from_sbml(task.model.source, sorted(target_x_paths_ids.values()))

    # Configure task
    solver, solver_arguments = config_task(task, model)

    # Run simulation using default model parameters and solver options
    results = exec_task(model, solver)

    # Save a report of the results of the simulation with `simulation.num_time_points` time points
    # beginning at `simulation.output_start_time` to `out_filename` in `out_format` format.
    # This should save all of the variables specified by `simulation.model.variables`.
    variable_results = extract_variables_from_results(model, sbml_model, variables, target_x_paths_ids, results)

    # cleanup module and temporary directory
    cleanup_model(model_name, model_dir)

    # log action
    log.algorithm = task.simulation.algorithm.kisao_id
    arguments = solver_arguments
    arguments['solver'] = amici.CVodeSolver.__module__ + '.' + amici.CVodeSolver.__name__
    log.simulator_details = {
        'method': amici.runAmiciSimulation.__module__ + '.' + amici.runAmiciSimulation.__name__,
        'arguments': arguments,
    }

    # return results and log
    return variable_results, log


def validate_sed_task(task, variables):
    """ Validate that AMICI can support a SED task

    Args:
       task (:obj:`Task`): task
       variables (:obj:`list` of :obj:`Variable`): variables that should be recorded

    Returns:
        :obj:`dict` of :obj:`str` to :obj:`str`: dictionary that maps each XPath to the
            value of the attribute of the object in the XML file that matches the XPath
    """
    model = task.model
    sim = task.simulation

    raise_errors_warnings(validation.validate_task(task),
                          error_summary='Task `{}` is invalid.'.format(task.id))
    raise_errors_warnings(validation.validate_model_language(task.model.language, ModelLanguage.SBML),
                          error_summary='Language for model `{}` is not supported.'.format(model.id))
    raise_errors_warnings(validation.validate_model_change_types(task.model.changes, ()),
                          error_summary='Changes for model `{}` are not supported.'.format(model.id))
    raise_errors_warnings(validation.validate_model_changes(task.model),
                          error_summary='Changes for model `{}` are invalid.'.format(model.id))
    raise_errors_warnings(validation.validate_simulation_type(task.simulation, (UniformTimeCourseSimulation, )),
                          error_summary='{} `{}` is not supported.'.format(sim.__class__.__name__, sim.id))
    raise_errors_warnings(validation.validate_simulation(task.simulation),
                          error_summary='Simulation `{}` is invalid.'.format(sim.id))
    raise_errors_warnings(validation.validate_data_generator_variables(variables),
                          error_summary='Data generator variables for task `{}` are invalid.'.format(task.id))
    return validation.validate_variable_xpaths(variables, task.model.source, attr='id')


def import_model_from_sbml(filename, variables):
    """ Generate an AMICI model from a SBML file

    Args:
        filename (:obj:`str`): path to SBML file
        variables (:obj:`list` of :obj:`str`): ids of SBML objects to observe

    Returns:
        :obj:`tuple`:

            * :obj:`amici.amici.ModelPtr`: AMICI model
            * :obj:`libsbml.Model`: SBML model
            * :obj:`str`: name of the Python module for model
            * :obj:`str`: directory which contains the files for the model
    """
    sbml_importer = amici.SbmlImporter(filename)
    sbml_model = sbml_importer.sbml

    model_dir = tempfile.mkdtemp()
    model_name = 'biosimulators_amici_model_' + os.path.basename(model_dir)
    constant_parameters = [param.getId() for param in sbml_model.parameters if param.constant]
    observables = {var: {'name': var, 'formula': var} for var in variables}
    sbml_importer.sbml2amici(model_name,
                             model_dir,
                             observables=observables,
                             constant_parameters=constant_parameters)

    model_module_spec = importlib.util.spec_from_file_location(model_name, os.path.join(model_dir, model_name, '__init__.py'))
    model_module = importlib.util.module_from_spec(model_module_spec)
    sys.modules[model_name] = model_module
    model_module_spec.loader.exec_module(model_module)
    model = model_module.getModel()

    return (model, sbml_model, model_name, model_dir)


def cleanup_model(model_name, model_dir):
    """ Cleanup model created with :obj:`import_model_from_sbml`

    Args:
        model_name (:obj:`str`): name of the Python module for model
        model_dir (:obj:`str`): directory which contains the files for the model
    """
    sys.modules.pop(model_name)
    shutil.rmtree(model_dir)


def config_task(task, model):
    """ Configure an AMICI model for a SED task

    Args:
        task (:obj:`Task`): task
        model (:obj:`amici.amici.ModelPtr`): AMICI model

    Returns:
        :obj:`tuple`:

            * :obj:`amici.amici.SolverPtr`: solver
            * :obj:`dict`: dictionary of arguments for the solver

    Raises:
        :obj:`NotImplementedError`: the task involves and unsupported algorithm or parameter
        :obj:`ValueError`: the task involves an invalid value of a parameter
    """
    # Simulate the model from `initial_time` to `output_end_time`
    # record results from `output_start_time` to `output_end_time`
    sim = task.simulation
    model.setT0(sim.initial_time)
    model.setTimepoints(numpy.linspace(sim.output_start_time, sim.output_end_time, sim.number_of_points + 1))

    # Load the algorithm specified by `sim.algorithm`
    algorithm_name = KISAO_ALGORITHMS_MAP.get(sim.algorithm.kisao_id, None)
    if algorithm_name is None:
        raise NotImplementedError(
            "Algorithm with KiSAO id '{}' is not supported".format(sim.algorithm.kisao_id))

    solver = model.getSolver()
    args = {}

    # Apply the algorithm parameter changes specified by `sim.algorithm_parameter_changes`
    for change in sim.algorithm.changes:
        param_props = KISAO_PARAMETERS_MAP.get(change.kisao_id, None)
        if param_props is None:
            raise NotImplementedError(
                "Algorithm parameter with KiSAO id '{}' is not supported".format(change.kisao_id))
        param_setter = getattr(solver, 'set' + param_props['name'])

        value = change.new_value
        if not validate_str_value(value, param_props['type']):
            raise ValueError("'{}' is not a valid {} value for parameter {}".format(
                value, param_props['type'].name, change.kisao_id))

        param_setter(parse_value(value, param_props['type']))
        args[param_props['name']] = value

    # return solver
    return solver, args


def exec_task(model, solver):
    """ Execute a SED task for an AMICI model and return its results

    Args:
        model (:obj:`amici.amici.ModelPtr`): AMICI model
        solver (:obj:`amici.amici.SolverPtr`): solver

    Returns:
        :obj:`amici.numpy.ReturnDataView`: simulation results
    """
    return amici.runAmiciSimulation(model, solver)


def extract_variables_from_results(model, sbml_model, variables, target_x_paths_ids, results):
    """ Extract data generator variables from results

    Args:
        model (:obj:`amici.amici.ModelPtr`): AMICI model
        sbml_model (:obj:`libsbml.Model`): SBML model
        variables (:obj:`list` of :obj:`Variable`): variables that should be recorded
        target_x_paths_ids (:obj:`dict` of :obj:`str` to :obj:`str`): dictionary that maps each XPath to the
            value of the attribute of the object in the XML file that matches the XPath
        results (:obj:`amici.numpy.ReturnDataView`): simulation results

    Returns:
        :obj:`VariableResults`: results of variables

    Raises:
        :obj:`NotImplementedError`: if a symbol could not be recorded
        :obj:`ValueError`: if a target could not be recorded
    """
    var_id_to_state_index = {id: index for index, id in enumerate(model.getStateIds())}

    variable_results = VariableResults()
    unpredicted_symbols = []
    unpredicted_targets = []
    for variable in variables:
        if variable.symbol:
            if variable.symbol == Symbol.time:
                variable_results[variable.id] = results['ts']
            else:
                unpredicted_symbols.append(variable.symbol)

        else:
            var_id = target_x_paths_ids.get(variable.target, None)
            i_state = var_id_to_state_index.get(var_id, None)
            if i_state is None:
                unpredicted_targets.append(variable.target)
            else:
                variable_results[variable.id] = results['x'][:, i_state]

    if unpredicted_symbols:
        raise NotImplementedError("".join([
            "The following variable symbols are not supported:\n  - {}\n\n".format(
                '\n  - '.join(sorted(unpredicted_symbols)),
            ),
            "Symbols must be one of the following:\n  - {}".format(Symbol.time),
        ]))

    if unpredicted_targets:
        raise ValueError(''.join([
            'The following variable targets could not be recorded:\n  - {}\n\n'.format(
                '\n  - '.join(sorted(unpredicted_targets)),
            ),
            'Targets must have one of the following ids:\n  - {}'.format(
                '\n  - '.join(sorted((
                    [s.getId() for s in sbml_model.species]
                    + [p.getId() for p in sbml_model.parameters]
                ))),
            ),
        ]))

    # return the result of each variable
    return variable_results
