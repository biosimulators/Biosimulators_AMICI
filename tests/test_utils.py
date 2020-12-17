from biosimulators_amici import core as utils
import amici
import os.path
import unittest


class UtilsTestCase(unittest.TestCase):
    def test_import_successful(self):
        model_filename = os.path.join(os.path.dirname(__file__), 'fixtures', 'biomd0000000002.xml')
        model, sbml_model, model_name, model_dir = utils.import_model_from_sbml(
            model_filename, ['BLL', 'IL', 'AL'])

        self.assertTrue(os.path.isdir(model_dir))
        self.assertIsInstance(model, amici.amici.ModelPtr)

        utils.cleanup_model(model_name, model_dir)
        self.assertFalse(os.path.isdir(model_dir))

    def test_import_error(self):
        with self.assertRaisesRegex(amici.sbml_import.SBMLException, 'Events are currently not supported'):
            model_filename = os.path.join(os.path.dirname(__file__), 'fixtures', 'BIOMD0000000297_url.xml')
            utils.import_model_from_sbml(model_filename, [])
