# 
# Copyright © 2020 Intel Corporation.
# 
# This software and the related documents are Intel copyrighted
# materials, and your use of them is governed by the express 
# license under which they were provided to you (License). Unless
# the License provides otherwise, you may not use, modify, copy, 
# publish, distribute, disclose or transmit  this software or the
# related documents without Intel's prior written permission.
# 
# This software and the related documents are provided as is, with
# no express or implied warranties, other than those that are 
# expressly stated in the License.

"""Tests DNN Tutorials"""
import glob
import unittest
from test import support
import nbformat
import tempfile

import subprocess

import os

import nxsdk


class TestDNNTutorials(unittest.TestCase):
    """Tests all the tutorials with dnn/tutorials directory"""

    @classmethod
    def setUpClass(cls):
        """Install SNN Toolbox"""
        TestDNNTutorials._call('pip install snntoolbox')

    @classmethod
    def tearDownClass(cls):
        """Uninstall SNN Toolbox"""
        TestDNNTutorials._call('pip uninstall -y snntoolbox')

    @staticmethod
    def _call(command):
        """Run a unix shell command"""
        subprocess.check_call(command, shell=True, env=os.environ)

    def test_dnn_jupyter_tutorials(self):
        """Test all ipython tutorials in tutorials directory"""
        self.find_and_test_ipython_tutorials()

    @staticmethod
    def _notebook_run(basedir, path):
        """
        Execute a notebook via nbconvert and collect output.

        :param basedir: Base directory of the IPython directory
        :param path: Path of this notebook relative to the IPython directory
        :return: (parsed nb object, execution errors)
        """

        cwd = os.getcwd()
        dirname, notebook = os.path.split(path)
        try:
            os.chdir(basedir + "/" + dirname)

            env = os.environ.copy()
            modulePath = nxsdk.__path__
            # The overall path is module path + parent directory of module +
            # any existing PYTHONPATH
            modulePath.extend(
                [os.path.dirname(modulePath[0]), env.get("PYTHONPATH", "")])
            env["PYTHONPATH"] = ":".join(modulePath)

            with tempfile.NamedTemporaryFile(mode="w+t", suffix=".ipynb") \
                    as fout:
                args = [
                    "jupyter",
                    "nbconvert",
                    "--to",
                    "notebook",
                    "--execute",
                    "--ExecutePreprocessor.timeout=-1",
                    "--output",
                    fout.name,
                    notebook]
                subprocess.check_call(args, env=env)

                fout.seek(0)
                nb = nbformat.read(fout, nbformat.current_nbformat)

            errors = []
            for cell in nb.cells:
                if 'outputs' in cell:
                    for output in cell['outputs']:
                        if output.output_type == 'error':
                            errors.append(output)

        except Exception as e:
            nb = None
            errors = str(e)
        finally:
            os.chdir(cwd)

        return nb, errors

    def find_and_test_ipython_tutorials(self):
        """Tests path planning tutorial executes without errors"""
        cwd = os.getcwd()
        testDirectory = os.path.dirname(os.path.realpath(__file__))
        parentDirectory = os.path.dirname(testDirectory)
        os.chdir(parentDirectory)

        errors_record = {}

        try:
            globPattern = "**/tutorials/*.ipynb"
            discoveredNotebooks = sorted(
                glob.glob(globPattern, recursive=True))

            for suite in discoveredNotebooks:
                nb, errors = self._notebook_run(parentDirectory, suite)
                errors_joined = "\n".join(errors) if isinstance(
                    errors, list) else errors
                if errors:
                    errors_record[suite] = (errors_joined, nb)

            self.assertFalse(
                errors_record, "Failed to execute Jupyter Notebooks with "
                               "errors: \n {}".format(errors_record))
        finally:
            os.chdir(cwd)


def main():
    """Invoke all unit tests within TestDNNTutorials"""
    support.run_unittest(TestDNNTutorials)


if __name__ == '__main__':
    main()
