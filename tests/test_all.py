""" Tests of the command-line interface

:Author: Jonathan Karr <karr@mssm.edu>
:Date: 2020-10-29
:Copyright: 2020, Center for Reproducible Biomedical Modeling
:License: MIT
"""

from Biosimulations_utils.simulator.testing import SimulatorValidator
from biosimulators_amici import __main__
import biosimulators_amici
import capturer
import docker
import os
import shutil
import tempfile
import unittest


class CliTestCase(unittest.TestCase):
    EXAMPLE_ARCHIVE_FILENAME = 'tests/fixtures/BIOMD0000000297.omex'
    DOCKER_IMAGE = 'ghcr.io/biosimulators/biosimulators_amici/amici'

    def setUp(self):
        self.dirname = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.dirname)

    def test_help(self):
        with self.assertRaises(SystemExit):
            with __main__.App(argv=['--help']) as app:
                app.run()

    def test_version(self):
        with __main__.App(argv=['-v']) as app:
            with capturer.CaptureOutput(merged=False, relay=False) as captured:
                with self.assertRaises(SystemExit):
                    app.run()
                self.assertIn(biosimulators_amici.__version__, captured.stdout.get_text())
                self.assertEqual(captured.stderr.get_text(), '')

        with __main__.App(argv=['--version']) as app:
            with capturer.CaptureOutput(merged=False, relay=False) as captured:
                with self.assertRaises(SystemExit):
                    app.run()
                self.assertIn(biosimulators_amici.__version__, captured.stdout.get_text())
                self.assertEqual(captured.stderr.get_text(), '')

    def test_sim_short_arg_names(self):
        with __main__.App(argv=['-i', self.EXAMPLE_ARCHIVE_FILENAME, '-o', self.dirname]) as app:
            app.run()
        self.assert_outputs_created(self.dirname)

    def test_sim_long_arg_names(self):
        with __main__.App(argv=['--archive', self.EXAMPLE_ARCHIVE_FILENAME, '--out-dir', self.dirname]) as app:
            app.run()
        self.assert_outputs_created(self.dirname)

    def test_build_docker_image(self):
        docker_client = docker.from_env()
        docker_client.images.build(
            path='.',
            dockerfile='Dockerfile',
            pull=True,
            rm=True,
        )

    def test_sim_with_docker_image(self):
        docker_client = docker.from_env()

        # setup input and output directories
        in_dir = os.path.join(self.dirname, 'in')
        out_dir = os.path.join(self.dirname, 'out')
        os.makedirs(in_dir)
        os.makedirs(out_dir)

        # copy model and simulation to temporary directory which will be mounted into container
        shutil.copyfile(self.EXAMPLE_ARCHIVE_FILENAME, os.path.join(in_dir, os.path.basename(self.EXAMPLE_ARCHIVE_FILENAME)))

        # run image
        docker_client.containers.run(
            image=self.DOCKER_IMAGE,
            volumes={
                in_dir: {
                    'bind': '/root/in',
                    'mode': 'ro',
                },
                out_dir: {
                    'bind': '/root/out',
                    'mode': 'rw',
                }
            },
            command=['-i', '/root/in/' + os.path.basename(self.EXAMPLE_ARCHIVE_FILENAME), '-o', '/root/out'],
            tty=True,
            remove=True)

        self.assert_outputs_created(out_dir)

    def assert_outputs_created(self, dirname):
        self.assertEqual(set(os.listdir(dirname)), set(['ex1', 'ex2']))
        self.assertEqual(set(os.listdir(os.path.join(dirname, 'ex1'))), set(['BIOMD0000000297']))
        self.assertEqual(set(os.listdir(os.path.join(dirname, 'ex2'))), set(['BIOMD0000000297']))

    def test_validator(self):
        validator = SimulatorValidator()
        validator.run(self.DOCKER_IMAGE)
