""" Methods for executing SED tasks in COMBINE archives and saving their outputs

:Author: Jonathan Karr <karr@mssm.edu>
:Date: 2020-12-16
:Copyright: 2020, BioSimulators Team
:License: MIT
"""

from .data_model import KISAO_ALGORITHMS_MAP, KISAO_PARAMETERS_MAP
from biosimulators_utils.combine.exec import exec_sedml_docs_in_archive
from biosimulators_utils.config import get_config, Config  # noqa: F401
from biosimulators_utils.log.data_model import CombineArchiveLog, TaskLog, StandardOutputErrorCapturerLevel  # noqa: F401
from biosimulators_utils.viz.data_model import VizFormat  # noqa: F401
from biosimulators_utils.report.data_model import ReportFormat, VariableResults  # noqa: F401
from biosimulators_utils.sedml.data_model import (Task, ModelLanguage, ModelAttributeChange, UniformTimeCourseSimulation,  # noqa: F401
                                                  Variable, Symbol)
from biosimulators_utils.sedml import validation
from biosimulators_utils.sedml.exec import exec_sed_doc as base_exec_sed_doc
from biosimulators_utils.simulator.utils import get_algorithm_substitution_policy
from biosimulators_utils.utils.core import validate_str_value, parse_value, raise_errors_warnings
from biosimulators_utils.warnings import warn, BioSimulatorsWarning
from kisao.data_model import AlgorithmSubstitutionPolicy, ALGORITHM_SUBSTITUTION_POLICY_LEVELS
from kisao.utils import get_preferred_substitute_algorithm_by_ids
import amici
import importlib.util
import lxml.etree
import numpy
import os.path
import shutil
import sys
import tempfile

# libSBML seems to need to be reloaded in some environments
import importlib
import libsbml  # noqa: F401
importlib.reload(libsbml)


__all__ = [
    'exec_sedml_docs_in_combine_archive',
    'exec_sed_doc',
    'exec_sed_task',
    'preprocess_sed_task',
    'validate_task',
    'import_model_from_sbml',
    'cleanup_model',
    'config_task',
    'exec_task',
    'validate_model_changes',
    'validate_variables',
    'extract_variables_from_results',
]


STATE_TYPES = [
    {
        'ids': 'getFixedParameterIds',
        'getter': 'getFixedParameters',
        'setter': 'setFixedParameters',
    },
    {
        'ids': 'getParameterIds',
        'getter': 'getParameters',
        'setter': 'setParameters',
    },
    {
        'ids': 'getStateIds',
        'getter': 'getInitialStates',
        'setter': 'setInitialStates',
    },
]


def exec_sedml_docs_in_combine_archive(archive_filename, out_dir, config=None):
    """ Execute the SED tasks defined in a COMBINE/OMEX archive and save the outputs

    Args:
        archive_filename (:obj:`str`): path to COMBINE/OMEX archive
        out_dir (:obj:`str`): path to store the outputs of the archive

            * CSV: directory in which to save outputs to files
              ``{ out_dir }/{ relative-path-to-SED-ML-file-within-archive }/{ report.id }.csv``
            * HDF5: directory in which to save a single HDF5 file (``{ out_dir }/reports.h5``),
              with reports at keys ``{ relative-path-to-SED-ML-file-within-archive }/{ report.id }`` within the HDF5 file

        config (:obj:`Config`, optional): BioSimulators common configuration

    Returns:
        :obj:`CombineArchiveLog`: log
    """
    return exec_sedml_docs_in_archive(exec_sed_doc, archive_filename, out_dir,
                                      apply_xml_model_changes=True,
                                      config=config)


def exec_sed_doc(doc, working_dir, base_out_path, rel_out_path=None,
                 apply_xml_model_changes=True,
                 log=None, indent=0, pretty_print_modified_xml_models=False,
                 log_level=StandardOutputErrorCapturerLevel.c, config=None):
    """ Execute the tasks specified in a SED document and generate the specified outputs

    Args:
        doc (:obj:`SedDocument` or :obj:`str`): SED document or a path to SED-ML file which defines a SED document
        working_dir (:obj:`str`): working directory of the SED document (path relative to which models are located)

        base_out_path (:obj:`str`): path to store the outputs

            * CSV: directory in which to save outputs to files
              ``{base_out_path}/{rel_out_path}/{report.id}.csv``
            * HDF5: directory in which to save a single HDF5 file (``{base_out_path}/reports.h5``),
              with reports at keys ``{rel_out_path}/{report.id}`` within the HDF5 file

        rel_out_path (:obj:`str`, optional): path relative to :obj:`base_out_path` to store the outputs
        apply_xml_model_changes (:obj:`bool`, optional): if :obj:`True`, apply any model changes specified in the SED-ML file before
            calling :obj:`task_executer`.
        log (:obj:`SedDocumentLog`, optional): log of the document
        indent (:obj:`int`, optional): degree to indent status messages
        pretty_print_modified_xml_models (:obj:`bool`, optional): if :obj:`True`, pretty print modified XML models
        log_level (:obj:`StandardOutputErrorCapturerLevel`, optional): level at which to log output
        config (:obj:`Config`, optional): BioSimulators common configuration
        simulator_config (:obj:`SimulatorConfig`, optional): tellurium configuration

    Returns:
        :obj:`tuple`:

            * :obj:`ReportResults`: results of each report
            * :obj:`SedDocumentLog`: log of the document
    """
    return base_exec_sed_doc(exec_sed_task, doc, working_dir, base_out_path,
                             rel_out_path=rel_out_path,
                             apply_xml_model_changes=apply_xml_model_changes,
                             log=log,
                             indent=indent,
                             pretty_print_modified_xml_models=pretty_print_modified_xml_models,
                             log_level=log_level,
                             config=config)


def exec_sed_task(task, variables, preprocessed_task=None, log=None, config=None):
    ''' Execute a task and save its results

    Args:
        task (:obj:`Task`): task
        variables (:obj:`list` of :obj:`Variable`): variables that should be recorded
        preprocessed_task (:obj:`dict`, optional): preprocessed information about the task, including possible
            model changes and variables. This can be used to avoid repeatedly executing the same initialization
            for repeated calls to this method.
        log (:obj:`TaskLog`, optional): log for the task
        config (:obj:`Config`, optional): BioSimulators common configuration

    Returns:
        :obj:`tuple`:

            :obj:`VariableResults`: results of variables
            :obj:`TaskLog`: log
    '''
    if not config:
        config = get_config()

    if config.LOG and not log:
        log = TaskLog()

    if preprocessed_task is None:
        preprocessed_task = preprocess_sed_task(task, variables, config=config)

    # get model
    amici_model = preprocessed_task['model']['model']

    # modify model
    if task.model.changes:
        raise_errors_warnings(validation.validate_model_change_types(task.model.changes, (ModelAttributeChange,)),
                              error_summary='Changes for model `{}` are not supported.'.format(task.model.id))
        model_change_setter_map = preprocessed_task['model']['model_change_setter_map']

        states = {
            state_type['setter']: {'modified': False, 'values': numpy.array(getattr(amici_model, state_type['getter'])())}
            for state_type in STATE_TYPES
        }

        for change in task.model.changes:
            change_setter, i_change = model_change_setter_map[change.target]
            states[change_setter]['modified'] = True
            states[change_setter]['values'][i_change] = float(change.new_value)

        for state_setter, state in states.items():
            if state['modified']:
                getattr(amici_model, state_setter)(state['values'])

    # Configure simulation
    # - Simulate the model from `initial_time` to `output_end_time`
    # - record results from `output_start_time` to `output_end_time`
    amici_model.setT0(task.simulation.initial_time)
    amici_model.setTimepoints(numpy.linspace(
        task.simulation.output_start_time,
        task.simulation.output_end_time,
        task.simulation.number_of_points + 1,
    ))

    # Run simulation using default model parameters and solver options
    solver = preprocessed_task['simulation']['solver']
    results = exec_task(amici_model, solver)

    # Save a report of the results of the simulation with `simulation.num_time_points` time points
    # beginning at `simulation.output_start_time` to `out_filename` in `out_format` format.
    # This should save all of the variables specified by `simulation.model.variables`.
    variable_observable_map = preprocessed_task['model']['variable_observable_map']
    variable_results = extract_variables_from_results(variables, variable_observable_map, results)

    # log action
    if config.LOG:
        log.algorithm = preprocessed_task['simulation']['algorithm_kisao_id']
        log.simulator_details = preprocessed_task['simulation']['simulator_details']

    # return results and log
    return variable_results, log


def validate_task(task, variables, config=None):
    """ Validate that AMICI can support a SED task

    Args:
       task (:obj:`Task`): task
       variables (:obj:`list` of :obj:`Variable`): variables that should be recorded
       config (:obj:`Config`, optional): BioSimulators common configuration

    Returns:
        :obj:`tuple`:

            * :obj:`dict` of :obj:`str` to :obj:`str`: dictionary that maps the XPath of each target of each
                model change to the SBML id of the associated model object
            * :obj:`dict` of :obj:`str` to :obj:`str`: dictionary that maps the XPath of each variable target
                to the SBML id of the associated model object
    """
    config = config or get_config()

    model = task.model
    sim = task.simulation

    if config.VALIDATE_SEDML:
        raise_errors_warnings(validation.validate_task(task),
                              error_summary='Task `{}` is invalid.'.format(task.id))
        raise_errors_warnings(validation.validate_model_language(model.language, ModelLanguage.SBML),
                              error_summary='Language for model `{}` is not supported.'.format(model.id))
        raise_errors_warnings(validation.validate_model_change_types(model.changes, (ModelAttributeChange,)),
                              error_summary='Changes for model `{}` are not supported.'.format(model.id))
        raise_errors_warnings(*validation.validate_model_changes(model),
                              error_summary='Changes for model `{}` are invalid.'.format(model.id))
        raise_errors_warnings(validation.validate_simulation_type(sim, (UniformTimeCourseSimulation, )),
                              error_summary='{} `{}` is not supported.'.format(sim.__class__.__name__, sim.id))
        raise_errors_warnings(*validation.validate_simulation(sim),
                              error_summary='Simulation `{}` is invalid.'.format(sim.id))
        raise_errors_warnings(*validation.validate_data_generator_variables(variables),
                              error_summary='Data generator variables for task `{}` are invalid.'.format(task.id))

    model_etree = lxml.etree.parse(model.source)
    return (
        validation.validate_target_xpaths(task.model.changes, model_etree, attr='id'),
        validation.validate_target_xpaths(variables, model_etree, attr='id'),
    )


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
    observables = {variable_id_to_observable_id(var): {'name': var, 'formula': var} for var in variables}
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


def config_task(task, model, config=None):
    """ Configure an AMICI model for a SED task

    Args:
        task (:obj:`Task`): task
        model (:obj:`amici.amici.ModelPtr`): AMICI model
        config (:obj:`Config`): configuration

    Returns:
        :obj:`tuple`:

            * :obj:`amici.amici.SolverPtr`: solver
            * :obj:`str`: KiSAO id of the solver
            * :obj:`dict`: dictionary of arguments for the solver

    Raises:
        :obj:`NotImplementedError`: the task involves and unsupported algorithm or parameter
        :obj:`ValueError`: the task involves an invalid value of a parameter
    """
    sim = task.simulation

    # Load the algorithm specified by `sim.algorithm`
    algorithm_substitution_policy = get_algorithm_substitution_policy(config=config)
    exec_kisao_id = get_preferred_substitute_algorithm_by_ids(
        sim.algorithm.kisao_id, KISAO_ALGORITHMS_MAP.keys(),
        substitution_policy=algorithm_substitution_policy)

    solver = model.getSolver()
    args = {}

    # Apply the algorithm parameter changes specified by `sim.algorithm_parameter_changes`
    if exec_kisao_id == sim.algorithm.kisao_id:
        for change in sim.algorithm.changes:
            param_props = KISAO_PARAMETERS_MAP.get(change.kisao_id, None)
            if param_props is None:
                if (
                    ALGORITHM_SUBSTITUTION_POLICY_LEVELS[algorithm_substitution_policy]
                    <= ALGORITHM_SUBSTITUTION_POLICY_LEVELS[AlgorithmSubstitutionPolicy.NONE]
                ):
                    msg = "Algorithm parameter with KiSAO id '{}' is not supported".format(change.kisao_id)
                    raise NotImplementedError(msg)
                else:
                    msg = "Algorithm parameter with KiSAO id '{}' was ignored because it is not supported".format(change.kisao_id)
                    warn(msg, BioSimulatorsWarning)
                    continue

            param_setter = getattr(solver, 'set' + param_props['name'])

            value = change.new_value
            if not validate_str_value(value, param_props['type']):
                if (
                    ALGORITHM_SUBSTITUTION_POLICY_LEVELS[algorithm_substitution_policy]
                    <= ALGORITHM_SUBSTITUTION_POLICY_LEVELS[AlgorithmSubstitutionPolicy.NONE]
                ):
                    msg = "'{}' is not a valid {} value for parameter {}".format(
                        value, param_props['type'].name, change.kisao_id)
                    raise ValueError(msg)
                else:
                    msg = "'{}' was ignored because it is not a valid {} value for parameter {}".format(
                        value, param_props['type'].name, change.kisao_id)
                    warn(msg, BioSimulatorsWarning)
                    continue

            param_setter(parse_value(value, param_props['type']))
            args[param_props['name']] = value

    # return solver
    return solver, exec_kisao_id, args


def exec_task(model, solver):
    """ Execute a SED task for an AMICI model and return its results

    Args:
        model (:obj:`amici.amici.ModelPtr`): AMICI model
        solver (:obj:`amici.amici.SolverPtr`): solver

    Returns:
        :obj:`amici.numpy.ReturnDataView`: simulation results
    """
    return amici.runAmiciSimulation(model, solver)


def validate_model_changes(model, changes, change_sbml_id_map):
    """ Validate model changes

    Args:
        model (:obj:`amici.amici.ModelPtr`): AMICI model
        changes (:obj:`list` of :obj:`ModelAttributeChange`): model changes
        change_sbml_id_map (:obj:`dict` of :obj:`str` to :obj:`str`): dictionary that maps each XPath to the
            value of the attribute of the object in the XML file that matches the XPath

    Returns:
        :obj:`dict`: dictionary that maps the targets of changes to AMICI setters

    Raises:
        :obj:`ValueError`: if a change could not be applied
    """
    change_setter_map = {}

    invalid_changes = []

    smbl_id_setter_map = {}
    for state_type in STATE_TYPES:
        for i_component, sbml_id in enumerate(getattr(model, state_type['ids'])()):
            smbl_id_setter_map[sbml_id] = (state_type['setter'], i_component)

    for change in changes:
        sbml_id = change_sbml_id_map.get(change.target, None)
        setter = smbl_id_setter_map.get(sbml_id, None)
        if setter is None:
            invalid_changes.append(change.target)
        else:
            change_setter_map[change.target] = setter

    if invalid_changes:
        raise ValueError(''.join([
            'The following change targets are invalid:\n  - {}\n\n'.format(
                '\n  - '.join(sorted(invalid_changes)),
            ),
            'Targets must have one of the following SBML ids:\n  - {}'.format(
                '\n  - '.join(sorted(smbl_id_setter_map.keys())),
            ),
        ]))

    # return a map from changes to AMICI setters
    return change_setter_map


def validate_variables(model, variables, variable_target_sbml_id_map):
    """ Validate variables

    Args:
        model (:obj:`amici.amici.ModelPtr`): AMICI model
        variables (:obj:`list` of :obj:`Variable`): variables that should be recorded
        variable_target_sbml_id_map (:obj:`dict` of :obj:`str` to :obj:`str`): dictionary that maps each XPath to the
            value of the attribute of the object in the XML file that matches the XPath

    Returns:
        :obj:`dict`: dictionary that maps the targets and symbols of variables to AMICI observables

    Raises:
        :obj:`NotImplementedError`: if a symbol could not be recorded
        :obj:`ValueError`: if a target could not be recorded
    """
    variable_observable_map = {}

    unpredicted_symbols = []
    unpredicted_targets = []

    obs_id_to_obs_index = {id: index for index, id in enumerate(model.getObservableIds())}

    for variable in variables:
        if variable.symbol:
            if variable.symbol == Symbol.time:
                variable_observable_map[(variable.target, variable.symbol)] = ('ts', None)
            else:
                unpredicted_symbols.append(variable.symbol)

        else:
            sbml_id = variable_target_sbml_id_map.get(variable.target, None)
            i_obs = obs_id_to_obs_index.get(variable_id_to_observable_id(sbml_id), None)
            if i_obs is None:
                unpredicted_targets.append(variable.target)
            else:
                variable_observable_map[(variable.target, variable.symbol)] = ('y', i_obs)

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
                '\n  - '.join(sorted(model.getObservableIds())),
            ),
        ]))

    # return a map from variables to AMICI observables
    return variable_observable_map


def extract_variables_from_results(variables, variable_observable_map, results):
    """ Extract data generator variables from results

    Args:
        variables (:obj:`list` of :obj:`Variable`): variables that should be recorded
        variable_observable_map (:obj:`dict`): dictionary that maps the targets and symbols of variables to AMICI observables
        results (:obj:`amici.numpy.ReturnDataView`): simulation results

    Returns:
        :obj:`VariableResults`: results of variables
    """
    variable_results = VariableResults()
    for variable in variables:
        obs_type, obs_index = variable_observable_map[(variable.target, variable.symbol)]
        result = results[obs_type]
        if obs_index is not None:
            result = result[:, obs_index]
        variable_results[variable.id] = result

    # return the result of each variable
    return variable_results


def preprocess_sed_task(task, variables, config=None):
    """ Preprocess a SED task, including its possible model changes and variables. This is useful for avoiding
    repeatedly initializing tasks on repeated calls of :obj:`exec_sed_task`.

    Args:
        task (:obj:`Task`): task
        variables (:obj:`list` of :obj:`Variable`): variables that should be recorded
        config (:obj:`Config`, optional): BioSimulators common configuration

    Returns:
        :obj:`dict`: preprocessed information about the task
    """
    if not config:
        config = get_config()

    model_change_sbml_id_map, variable_target_sbml_id_map = validate_task(task, variables, config=config)

    # Read the model for the task
    amici_model, sbml_model, python_module_name, python_model_dirname = import_model_from_sbml(
        task.model.source, sorted(variable_target_sbml_id_map.values()))

    # cleanup module and temporary directory
    cleanup_model(python_module_name, python_model_dirname)

    # validate model changes and variables
    model_change_setter_map = validate_model_changes(amici_model, task.model.changes, model_change_sbml_id_map)
    variable_observable_map = validate_variables(amici_model, variables, variable_target_sbml_id_map)

    # Configure task
    solver, solver_kisao_id, solver_args = config_task(task, amici_model, config=config)

    # return preprocessed information
    return {
        'model': {
            'model': amici_model,
            'model_change_setter_map': model_change_setter_map,
            'variable_observable_map': variable_observable_map,
        },
        'simulation': {
            'algorithm_kisao_id': solver_kisao_id,
            'solver': solver,
            'simulator_details': {
                'solver': amici.CVodeSolver.__module__ + '.' + amici.CVodeSolver.__name__,
                'method': amici.runAmiciSimulation.__module__ + '.' + amici.runAmiciSimulation.__name__,
                'arguments': solver_args,
            }
        },
    }


def variable_id_to_observable_id(variable_id: str) -> str:
    """Convert a variable ID to an observable ID.

    In AMICI, identifiers need to be globally unique. Therefore, we cannot use a species ID as an observable ID.
    Let's hope that this alias does not already exist in the model...
    """
    return f"___{variable_id}"
