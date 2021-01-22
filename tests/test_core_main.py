""" Tests of the command-line interface

:Author: Jonathan Karr <karr@mssm.edu>
:Date: 2020-10-29
:Copyright: 2020, Center for Reproducible Biomedical Modeling
:License: MIT
"""

from biosimulators_amici import __main__
from biosimulators_amici import core
from biosimulators_utils.combine import data_model as combine_data_model
from biosimulators_utils.combine.exceptions import CombineArchiveExecutionError
from biosimulators_utils.combine.io import CombineArchiveWriter
from biosimulators_utils.report import data_model as report_data_model
from biosimulators_utils.report.io import ReportReader
from biosimulators_utils.simulator.exec import exec_sedml_docs_in_archive_with_containerized_simulator
from biosimulators_utils.simulator.specs import gen_algorithms_from_specs
from biosimulators_utils.sedml import data_model as sedml_data_model
from biosimulators_utils.sedml.io import SedmlSimulationWriter
from biosimulators_utils.sedml.utils import append_all_nested_children_to_doc
from unittest import mock
import datetime
import dateutil.tz
import numpy
import numpy.testing
import os
import shutil
import tempfile
import unittest


class CliTestCase(unittest.TestCase):
    DOCKER_IMAGE = 'ghcr.io/biosimulators/biosimulators_amici/amici'

    def setUp(self):
        self.dirname = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.dirname)

    def test_exec_sed_task_successfully(self):
        task = sedml_data_model.Task(
            model=sedml_data_model.Model(
                source=os.path.join(os.path.dirname(__file__), 'fixtures', 'biomd0000000002.xml'),
                language=sedml_data_model.ModelLanguage.SBML.value,
                changes=[],
            ),
            simulation=sedml_data_model.UniformTimeCourseSimulation(
                algorithm=sedml_data_model.Algorithm(
                    kisao_id='KISAO_0000496',
                    changes=[
                        sedml_data_model.AlgorithmParameterChange(
                            kisao_id='KISAO_0000209',
                            new_value='2e-8',
                        ),
                    ],
                ),
                initial_time=5.,
                output_start_time=10.,
                output_end_time=20.,
                number_of_points=20,
            ),
        )

        variables = [
            sedml_data_model.Variable(id='time', symbol=sedml_data_model.Symbol.time, task=task),
            sedml_data_model.Variable(id='AL', target="/sbml:sbml/sbml:model/sbml:listOfSpecies/sbml:species[@id='AL']", task=task),
            sedml_data_model.Variable(id='BLL', target='/sbml:sbml/sbml:model/sbml:listOfSpecies/sbml:species[@id="BLL"]', task=task),
            sedml_data_model.Variable(id='IL', target="/sbml:sbml/sbml:model/sbml:listOfSpecies/sbml:species[@id='IL']", task=task),
        ]

        variable_results, _ = core.exec_sed_task(task, variables)

        self.assertTrue(sorted(variable_results.keys()), sorted([var.id for var in variables]))
        self.assertEqual(variable_results[variables[0].id].shape, (task.simulation.number_of_points + 1,))
        numpy.testing.assert_almost_equal(
            variable_results['time'],
            numpy.linspace(task.simulation.output_start_time, task.simulation.output_end_time, task.simulation.number_of_points + 1),
        )

        for results in variable_results.values():
            self.assertFalse(numpy.any(numpy.isnan(results)))

    def test_exec_sed_task_error_handling(self):
        task = sedml_data_model.Task(
            model=sedml_data_model.Model(
                source=os.path.join(os.path.dirname(__file__), 'fixtures', 'biomd0000000002.xml'),
                language=sedml_data_model.ModelLanguage.SBML.value,
                changes=[],
            ),
            simulation=sedml_data_model.UniformTimeCourseSimulation(
                algorithm=sedml_data_model.Algorithm(
                    kisao_id='KISAO_0000001',
                    changes=[
                        sedml_data_model.AlgorithmParameterChange(
                            kisao_id='KISAO_0000209',
                            new_value='2e-8',
                        ),
                    ],
                ),
                initial_time=5.,
                output_start_time=10.,
                output_end_time=20.,
                number_of_points=20,
            ),
        )

        variables = [
            sedml_data_model.Variable(id='time', symbol=sedml_data_model.Symbol.time, task=task),
            sedml_data_model.Variable(id='AL', target="/sbml:sbml/sbml:model/sbml:listOfSpecies/sbml:species[@id='AL']", task=task),
            sedml_data_model.Variable(id='BLL', target='/sbml:sbml/sbml:model/sbml:listOfSpecies/sbml:species[@id="BLL"]', task=task),
            sedml_data_model.Variable(id='IL', target="/sbml:sbml/sbml:model/sbml:listOfSpecies/sbml:species[@id='IL']", task=task),
        ]

        target_x_paths_ids = core.validate_sed_task(task, variables)

        # Read the model for the task
        model, sbml_model, model_name, model_dir = core.import_model_from_sbml(task.model.source, sorted(target_x_paths_ids.values()))

        # Configure task
        task.simulation.algorithm.kisao_id = 'KISAO_0000001'
        with self.assertRaisesRegex(NotImplementedError, 'is not supported'):
            core.config_task(task, model)

        task.simulation.algorithm.kisao_id = 'KISAO_0000496'
        task.simulation.algorithm.changes[0].kisao_id = 'KISAO_0000001'
        with self.assertRaisesRegex(NotImplementedError, 'is not supported'):
            core.config_task(task, model)

        task.simulation.algorithm.changes[0].kisao_id = 'KISAO_0000209'
        task.simulation.algorithm.changes[0].new_value = 'two e minus 8'
        with self.assertRaisesRegex(ValueError, 'is not a valid'):
            core.config_task(task, model)

        task.simulation.algorithm.changes[0].new_value = '2e-8'
        solver, _ = core.config_task(task, model)

        # Run simulation using default model parameters and solver options
        results = core.exec_task(model, solver)

        # Save a report of the results of the simulation with `simulation.num_time_points` time points
        # beginning at `simulation.output_start_time` to `out_filename` in `out_format` format.
        # This should save all of the variables specified by `simulation.model.variables`.
        variables[0].symbol = 'urn:sedml:symbol:undefined'
        with self.assertRaisesRegex(NotImplementedError, 'symbols are not supported'):
            core.extract_variables_from_results(model, sbml_model, variables, target_x_paths_ids, results)

        variables[0].symbol = sedml_data_model.Symbol.time
        variables[1].target = "/sbml:sbml/sbml:model/sbml:listOfSpecies/sbml:species[@id='undefined']"
        with self.assertRaisesRegex(ValueError, 'targets could not be recorded'):
            core.extract_variables_from_results(model, sbml_model, variables, target_x_paths_ids, results)

        variables[1].target = "/sbml:sbml/sbml:model/sbml:listOfSpecies/sbml:species[@id='AL']"
        variable_results = core.extract_variables_from_results(model, sbml_model, variables, target_x_paths_ids, results)

        self.assertTrue(sorted(variable_results.keys()), sorted([var.id for var in variables]))
        self.assertEqual(variable_results[variables[0].id].shape, (task.simulation.number_of_points + 1,))
        numpy.testing.assert_almost_equal(
            variable_results['time'],
            numpy.linspace(task.simulation.output_start_time, task.simulation.output_end_time, task.simulation.number_of_points + 1),
        )
        for results in variable_results.values():
            self.assertFalse(numpy.any(numpy.isnan(results)))

        # cleanup module and temporary directory
        core.cleanup_model(model_name, model_dir)

    def test_exec_sedml_docs_in_combine_archive_successfully(self):
        doc, archive_filename = self._build_combine_archive()

        out_dir = os.path.join(self.dirname, 'out')
        core.exec_sedml_docs_in_combine_archive(archive_filename, out_dir,
                                                report_formats=[
                                                    report_data_model.ReportFormat.h5,
                                                    report_data_model.ReportFormat.csv,
                                                ],
                                                bundle_outputs=True,
                                                keep_individual_outputs=True)

        self._assert_combine_archive_outputs(doc, out_dir)

    def _build_combine_archive(self, algorithm=None):
        doc = self._build_sed_doc(algorithm=algorithm)

        archive_dirname = os.path.join(self.dirname, 'archive')
        if not os.path.isdir(archive_dirname):
            os.mkdir(archive_dirname)

        model_filename = os.path.join(archive_dirname, 'model_1.xml')
        shutil.copyfile(
            os.path.join(os.path.dirname(__file__), 'fixtures', 'biomd0000000002.xml'),
            model_filename)

        sim_filename = os.path.join(archive_dirname, 'sim_1.sedml')
        SedmlSimulationWriter().run(doc, sim_filename)

        updated = datetime.datetime(2020, 1, 2, 1, 2, 3, tzinfo=dateutil.tz.tzutc())
        archive = combine_data_model.CombineArchive(
            contents=[
                combine_data_model.CombineArchiveContent(
                    'model_1.xml', combine_data_model.CombineArchiveContentFormat.SBML.value, updated=updated),
                combine_data_model.CombineArchiveContent(
                    'sim_1.sedml', combine_data_model.CombineArchiveContentFormat.SED_ML.value, updated=updated),
            ],
            updated=updated,
        )
        archive_filename = os.path.join(self.dirname,
                                        'archive.omex' if algorithm is None else 'archive-{}.omex'.format(algorithm.kisao_id))
        CombineArchiveWriter().run(archive, archive_dirname, archive_filename)

        return (doc, archive_filename)

    def _build_sed_doc(self, algorithm=None):
        if algorithm is None:
            algorithm = sedml_data_model.Algorithm(
                kisao_id='KISAO_0000496',
                changes=[
                    sedml_data_model.AlgorithmParameterChange(
                        kisao_id='KISAO_0000209',
                        new_value='2e-8',
                    ),
                ],
            )

        doc = sedml_data_model.SedDocument()
        doc.models.append(sedml_data_model.Model(
            id='model_1',
            source='model_1.xml',
            language=sedml_data_model.ModelLanguage.SBML.value,
            changes=[],
        ))
        doc.simulations.append(sedml_data_model.UniformTimeCourseSimulation(
            id='sim_1_time_course',
            algorithm=algorithm,
            initial_time=0.,
            output_start_time=0.1,
            output_end_time=0.2,
            number_of_points=20,
        ))
        doc.tasks.append(sedml_data_model.Task(
            id='task_1',
            model=doc.models[0],
            simulation=doc.simulations[0],
        ))
        doc.data_generators.append(sedml_data_model.DataGenerator(
            id='data_gen_time',
            variables=[
                sedml_data_model.Variable(
                    id='var_time',
                    symbol=sedml_data_model.Symbol.time,
                    task=doc.tasks[0],
                ),
            ],
            math='var_time',
        ))
        doc.data_generators.append(sedml_data_model.DataGenerator(
            id='data_gen_AL',
            variables=[
                sedml_data_model.Variable(
                    id='var_AL',
                    target="/sbml:sbml/sbml:model/sbml:listOfSpecies/sbml:species[@id='AL']",
                    task=doc.tasks[0],
                ),
            ],
            math='var_AL',
        ))
        doc.data_generators.append(sedml_data_model.DataGenerator(
            id='data_gen_BLL',
            variables=[
                sedml_data_model.Variable(
                    id='var_BLL',
                    target='/sbml:sbml/sbml:model/sbml:listOfSpecies/sbml:species[@id="BLL"]',
                    task=doc.tasks[0],
                ),
            ],
            math='var_BLL',
        ))
        doc.data_generators.append(sedml_data_model.DataGenerator(
            id='data_gen_IL',
            variables=[
                sedml_data_model.Variable(
                    id='var_IL',
                    target="/sbml:sbml/sbml:model/sbml:listOfSpecies/sbml:species[@id='IL']",
                    task=doc.tasks[0],
                ),
            ],
            math='var_IL',
        ))
        doc.outputs.append(sedml_data_model.Report(
            id='report_1',
            data_sets=[
                sedml_data_model.DataSet(id='data_set_time', label='Time', data_generator=doc.data_generators[0]),
                sedml_data_model.DataSet(id='data_set_AL', label='AL', data_generator=doc.data_generators[1]),
                sedml_data_model.DataSet(id='data_set_BLL', label='BLL', data_generator=doc.data_generators[2]),
                sedml_data_model.DataSet(id='data_set_IL', label='IL', data_generator=doc.data_generators[3]),
            ],
        ))

        append_all_nested_children_to_doc(doc)

        return doc

    def _assert_combine_archive_outputs(self, doc, out_dir):
        self.assertEqual(set(['reports.h5', 'reports.zip', 'sim_1.sedml']).difference(set(os.listdir(out_dir))), set())

        report = doc.outputs[0]

        # check HDF report
        report_results = ReportReader().run(report, out_dir, 'sim_1.sedml/report_1', format=report_data_model.ReportFormat.h5)

        self.assertEqual(sorted(report_results.keys()), sorted([d.id for d in doc.outputs[0].data_sets]))

        sim = doc.tasks[0].simulation
        self.assertEqual(len(report_results[report.data_sets[0].id]), sim.number_of_points + 1)
        numpy.testing.assert_almost_equal(
            report_results[report.data_sets[0].id],
            numpy.linspace(sim.output_start_time, sim.output_end_time, sim.number_of_points + 1),
        )

        for data_set_result in report_results.values():
            self.assertFalse(numpy.any(numpy.isnan(data_set_result)))

        # check CSV report
        report_results = ReportReader().run(report, out_dir, 'sim_1.sedml/report_1', format=report_data_model.ReportFormat.csv)

        self.assertEqual(sorted(report_results.keys()), sorted([d.id for d in doc.outputs[0].data_sets]))

        sim = doc.tasks[0].simulation
        self.assertEqual(len(report_results[report.data_sets[0].id]), sim.number_of_points + 1)
        numpy.testing.assert_almost_equal(
            report_results[report.data_sets[0].id],
            numpy.linspace(sim.output_start_time, sim.output_end_time, sim.number_of_points + 1),
        )

        for data_set_result in report_results.values():
            self.assertFalse(numpy.any(numpy.isnan(data_set_result)))

    def test_exec_sedml_docs_in_combine_archive_with_all_algorithms(self):
        for alg in gen_algorithms_from_specs(os.path.join(os.path.dirname(__file__), '..', 'biosimulators.json')).values():
            doc, archive_filename = self._build_combine_archive(algorithm=alg)

            out_dir = os.path.join(self.dirname, alg.kisao_id)
            core.exec_sedml_docs_in_combine_archive(archive_filename, out_dir,
                                                    report_formats=[
                                                        report_data_model.ReportFormat.h5,
                                                        report_data_model.ReportFormat.csv,
                                                    ],
                                                    bundle_outputs=True,
                                                    keep_individual_outputs=True)
            self._assert_combine_archive_outputs(doc, out_dir)

    def test_raw_cli(self):
        with mock.patch('sys.argv', ['', '--help']):
            with self.assertRaises(SystemExit) as context:
                __main__.main()
                self.assertRegex(context.Exception, 'usage: ')

    def test_exec_sedml_docs_in_combine_archive_with_cli(self):
        doc, archive_filename = self._build_combine_archive()
        out_dir = os.path.join(self.dirname, 'out')
        env = self._get_combine_archive_exec_env()

        with mock.patch.dict(os.environ, env):
            with __main__.App(argv=['-i', archive_filename, '-o', out_dir]) as app:
                app.run()

        self._assert_combine_archive_outputs(doc, out_dir)

    def _get_combine_archive_exec_env(self):
        return {
            'REPORT_FORMATS': 'h5,csv',
            'KEEP_INDIVIDUAL_OUTPUTS': '1',
        }

    def test_exec_sedml_docs_in_combine_archive_with_docker_image(self):
        doc, archive_filename = self._build_combine_archive()
        out_dir = os.path.join(self.dirname, 'out')
        docker_image = self.DOCKER_IMAGE
        env = self._get_combine_archive_exec_env()

        exec_sedml_docs_in_archive_with_containerized_simulator(
            archive_filename, out_dir, docker_image, environment=env, pull_docker_image=False)

        self._assert_combine_archive_outputs(doc, out_dir)
