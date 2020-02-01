#!/usr/bin/env python
#
# Tests the importers for various formats
#
# This file is part of Myokit.
# See http://myokit.org for copyright, sharing, and licensing details.
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals

import os
import unittest

import myokit
import myokit.formats as formats

from shared import DIR_FORMATS

# Unit testing in Python 2 and 3
try:
    unittest.TestCase.assertRaisesRegex
except AttributeError:  # pragma: no python 3 cover
    unittest.TestCase.assertRaisesRegex = unittest.TestCase.assertRaisesRegexp

# Strings in Python 2 and 3
try:
    basestring
except NameError:   # pragma: no python 2 cover
    basestring = str


class ImporterTest(unittest.TestCase):
    """ Test shared importer functionality. """

    def test_importer_interface(self):
        """ Test listing and creating importers. """
        ims = myokit.formats.importers()
        self.assertTrue(len(ims) > 0)
        for i in ims:
            self.assertIsInstance(i, basestring)
            i = myokit.formats.importer(i)
            self.assertTrue(isinstance(i, myokit.formats.Importer))

    def test_unknown(self):
        """ Test requesting an unknown importer. """
        # Test fetching using importer method
        self.assertRaisesRegex(
            KeyError, 'Importer not found', myokit.formats.importer, 'blip')


class SBMLTest(unittest.TestCase):
    """ Test SBML import. """

    def test_capability_reporting(self):
        """ Test if the right capabilities are reported. """
        i = formats.importer('sbml')
        self.assertFalse(i.supports_component())
        self.assertTrue(i.supports_model())
        self.assertFalse(i.supports_protocol())

    def test_model(self):
        i = formats.importer('sbml')

        def sbml(fname):
            m = i.model(os.path.join(DIR_FORMATS, 'sbml', fname))
            try:
                m.validate()
            except myokit.MissingTimeVariableError:
                # SBML models don't specify the time variable
                pass

        # Basic Hodgkin-Huxley
        sbml('HodgkinHuxley.xml')

        # Same but without a model name
        sbml('HodgkinHuxley-no-model-name-but-id.xml')
        sbml('HodgkinHuxley-no-model-name-or-id.xml')

        # Same but with funny variable names
        sbml('HodgkinHuxley-funny-names.xml')

        # Model with listOfInitialValues and unit with multiplier
        sbml('Noble1962-initial-assignments-and-weird-unit.xml')

    def test_info(self):
        i = formats.importer('sbml')
        self.assertIsInstance(i.info(), basestring)


class AxonTest(unittest.TestCase):
    """ Partially tests Axon formats importing. """

    def test_capability_reporting(self):
        """ Test if the right capabilities are reported. """
        i = formats.importer('abf')
        self.assertFalse(i.supports_component())
        self.assertFalse(i.supports_model())
        self.assertTrue(i.supports_protocol())

    def test_protocol(self):
        i = formats.importer('abf')
        self.assertTrue(i.supports_protocol())
        i.protocol(os.path.join(DIR_FORMATS, 'abf-v1.abf'))

    def test_info(self):
        i = formats.importer('abf')
        self.assertIsInstance(i.info(), basestring)


if __name__ == '__main__':
    unittest.main()