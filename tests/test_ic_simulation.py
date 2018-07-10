#!/usr/bin/env python
#
# Tests the ICSimulation class.
#
# This file is part of Myokit
#  Copyright 2011-2018 Maastricht University, University of Oxford
#  Licensed under the GNU General Public License v3.0
#  See: http://myokit.org
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals

import os
import unittest

import myokit

from shared import DIR_DATA, CancellingReporter


class ICSimulationTest(unittest.TestCase):
    """
    Tests the :class:`ICSimulation`.
    """
    def test_basic(self):
        """ Tests basic usage. """
        # Load model
        m = os.path.join(DIR_DATA, 'lr-1991.mmt')
        m, p, x = myokit.load(m)

        # Run a simulation
        s = myokit.ICSimulation(m, p)

        self.assertEqual(s.time(), 0)
        self.assertEqual(s.state(), m.state())
        self.assertEqual(s.default_state(), m.state())
        d, e = s.run(20, log_interval=5)
        self.assertEqual(s.time(), 20)
        self.assertNotEqual(s.state(), m.state())
        self.assertEqual(s.default_state(), m.state())

        # Create a datablock from the simulation log
        b = s.block(d, e)

        # Calculate eigenvalues
        b.eigenvalues('derivatives')

        # Log with missing time value
        d2 = d.clone()
        del(d2['engine.time'])
        self.assertRaisesRegexp(ValueError, 'time', s.block, d2, e)

        # Wrong size derivatives array
        self.assertRaisesRegexp(ValueError, 'shape', s.block, d, e[:-1])

        # Time can't be negative
        self.assertRaises(ValueError, s.run, -1)

        # Test running without a protocol
        s.set_protocol(None)
        s.run(1)

        # Test step size is > 0
        self.assertRaises(ValueError, s.set_step_size, 0)

        # Test negative log interval is ignored
        s.run(1, log_interval=-1)

    def test_progress_reporter(self):
        """ Test running with a progress reporter. """
        m, p, x = myokit.load(os.path.join(DIR_DATA, 'lr-1991.mmt'))

        # Test using a progress reporter
        s = myokit.ICSimulation(m, p)
        with myokit.PyCapture() as c:
            s.run(110, progress=myokit.ProgressPrinter())
        c = c.text().splitlines()
        self.assertTrue(len(c) > 0)

        # Not a progress reporter
        self.assertRaisesRegexp(
            ValueError, 'ProgressReporter', s.run, 5, progress=12)

        # Cancel from reporter
        self.assertRaises(
            myokit.SimulationCancelledError, s.run, 1,
            progress=CancellingReporter(0))

    def test_invalid_model(self):
        """ Tests running with an invalid model. """
        m = myokit.Model()
        self.assertRaises(
            myokit.MissingTimeVariableError, myokit.ICSimulation, m)


if __name__ == '__main__':
    unittest.main()
