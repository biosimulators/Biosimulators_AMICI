from biosimulators_amici import core
import amici
import os.path
import unittest


class UtilsTestCase(unittest.TestCase):
    def test_import_successful(self):
        model_filename = os.path.join(os.path.dirname(__file__), 'fixtures', 'biomd0000000002.xml')
        self.assertTrue(os.path.isfile(model_filename))

        model, sbml_model, model_name, model_dir = core.import_model_from_sbml(
            model_filename, ['BLL', 'IL', 'AL'])

        self.assertTrue(os.path.isdir(model_dir))
        self.assertIsInstance(model, amici.amici.ModelPtr)

        core.cleanup_model(model_name, model_dir)
        self.assertFalse(os.path.isdir(model_dir))

    def test_import_models_with_events(self):
        model_filename = os.path.join(os.path.dirname(__file__), 'fixtures', 'BIOMD0000000297_url.xml')
        core.import_model_from_sbml(model_filename, [])
