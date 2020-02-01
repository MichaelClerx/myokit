#!/usr/bin/env python
#
# Tests the Chaste module.
#
# This file is part of Myokit.
# See http://myokit.org for copyright, sharing, and licensing details.
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals

import unittest

import myokit
import myokit.formats
import myokit.formats.chaste

from shared import TemporaryDirectory

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


class ChasteExporterTest(unittest.TestCase):
    """ Tests Chaste export. """

    def test_chaste_exporter(self):
        # Tests exporting a model

        m, p, _ = myokit.load('example')
        e = myokit.formats.chaste.ChasteExporter()

        with TemporaryDirectory() as d:
            path = d.path('chaste')

            # Test with simple model
            e.runnable(path, m, p)

            # Test with invalid model
            v = m.get('membrane.V')
            v.demote()
            v.set_rhs('2 * V')
            self.assertRaisesRegex(
                myokit.ExportError, 'valid model', e.runnable, path, m, p)

    def test_chaste_exporter_fetching(self):
        # Tests getting an Chaste exporter via the 'exporter' interface

        e = myokit.formats.exporter('chaste')
        self.assertIsInstance(e, myokit.formats.chaste.ChasteExporter)

    def test_chaste_exporter_info(self):
        # Tests the info() method returns a string

        e = myokit.formats.chaste.ChasteExporter()
        self.assertIsInstance(e.info(), basestring)

        # Test support
        self.assertFalse(e.supports_model())
        self.assertTrue(e.supports_runnable())


class ChasteExpressionWriterTest(unittest.TestCase):
    """ Tests Chaste expression writer functionality. """

    '''
    def test_all(self):
        w = myokit.formats.ewriter('chaste')

        model = myokit.Model()
        component = model.add_component('c')
        avar = component.add_variable('a')

        # Name
        a = myokit.Name(avar)
        self.assertEqual(w.ex(a), 'c.a')
        # Number with unit
        b = myokit.Number('12', 'pF')
        self.assertEqual(w.ex(b), '12.0')
        # Integer
        c = myokit.Number(1)
        self.assertEqual(w.ex(c), '1.0')
        # Integer

        # Prefix plus
        x = myokit.PrefixPlus(b)
        self.assertEqual(w.ex(x), '12.0')
        # Prefix minus
        x = myokit.PrefixMinus(b)
        self.assertEqual(w.ex(x), '(-12.0)')

        # Plus
        x = myokit.Plus(a, b)
        self.assertEqual(w.ex(x), 'c.a + 12.0')
        # Minus
        x = myokit.Minus(a, b)
        self.assertEqual(w.ex(x), 'c.a - 12.0')
        # Multiply
        x = myokit.Multiply(a, b)
        self.assertEqual(w.ex(x), 'c.a * 12.0')
        # Divide
        x = myokit.Divide(a, b)
        self.assertEqual(w.ex(x), 'c.a / 12.0')

        # Quotient
        x = myokit.Quotient(a, b)
        self.assertEqual(w.ex(x), 'floor(c.a / 12.0)')
        # Remainder
        x = myokit.Remainder(a, b)
        self.assertEqual(w.ex(x), 'c.a - 12.0 * (floor(c.a / 12.0))')

        # Power
        x = myokit.Power(a, b)
        self.assertEqual(w.ex(x), 'pow(c.a, 12.0)')
        # Sqrt
        x = myokit.Sqrt(b)
        self.assertEqual(w.ex(x), 'sqrt(12.0)')
        # Exp
        x = myokit.Exp(a)
        self.assertEqual(w.ex(x), 'exp(c.a)')
        # Log(a)
        x = myokit.Log(b)
        self.assertEqual(w.ex(x), 'log(12.0)')
        # Log(a, b)
        x = myokit.Log(a, b)
        self.assertEqual(w.ex(x), '(log(c.a) / log(12.0))')
        # Log10
        x = myokit.Log10(b)
        self.assertEqual(w.ex(x), 'log10(12.0)')

        # Sin
        x = myokit.Sin(b)
        self.assertEqual(w.ex(x), 'sin(12.0)')
        # Cos
        x = myokit.Cos(b)
        self.assertEqual(w.ex(x), 'cos(12.0)')
        # Tan
        x = myokit.Tan(b)
        self.assertEqual(w.ex(x), 'tan(12.0)')
        # ASin
        x = myokit.ASin(b)
        self.assertEqual(w.ex(x), 'asin(12.0)')
        # ACos
        x = myokit.ACos(b)
        self.assertEqual(w.ex(x), 'acos(12.0)')
        # ATan
        x = myokit.ATan(b)
        self.assertEqual(w.ex(x), 'atan(12.0)')

        # Floor
        x = myokit.Floor(b)
        self.assertEqual(w.ex(x), 'floor(12.0)')
        # Ceil
        x = myokit.Ceil(b)
        self.assertEqual(w.ex(x), 'ceil(12.0)')
        # Abs
        x = myokit.Abs(b)
        self.assertEqual(w.ex(x), 'fabs(12.0)')

        # Equal
        x = myokit.Equal(a, b)
        self.assertEqual(w.ex(x), '(c.a == 12.0)')
        # NotEqual
        x = myokit.NotEqual(a, b)
        self.assertEqual(w.ex(x), '(c.a != 12.0)')
        # More
        x = myokit.More(a, b)
        self.assertEqual(w.ex(x), '(c.a > 12.0)')
        # Less
        x = myokit.Less(a, b)
        self.assertEqual(w.ex(x), '(c.a < 12.0)')
        # MoreEqual
        x = myokit.MoreEqual(a, b)
        self.assertEqual(w.ex(x), '(c.a >= 12.0)')
        # LessEqual
        x = myokit.LessEqual(a, b)
        self.assertEqual(w.ex(x), '(c.a <= 12.0)')

        # Not
        cond1 = myokit.parse_expression('5 > 3')
        cond2 = myokit.parse_expression('2 < 1')
        x = myokit.Not(cond1)
        self.assertEqual(w.ex(x), '!((5.0 > 3.0))')
        # And
        x = myokit.And(cond1, cond2)
        self.assertEqual(w.ex(x), '((5.0 > 3.0) and (2.0 < 1.0))')
        # Or
        x = myokit.Or(cond1, cond2)
        self.assertEqual(w.ex(x), '((5.0 > 3.0) or (2.0 < 1.0))')

        # If
        x = myokit.If(cond1, a, b)
        self.assertEqual(w.ex(x), '((5.0 > 3.0) ? c.a : 12.0)')
        # Piecewise
        c = myokit.Number(1)
        x = myokit.Piecewise(cond1, a, cond2, b, c)
        self.assertEqual(
            w.ex(x),
            '((5.0 > 3.0) ? c.a : ((2.0 < 1.0) ? 12.0 : 1.0))')

        # Test without a Myokit expression
        self.assertRaisesRegex(
            ValueError, 'Unknown expression type', w.ex, 7)
    '''

    def test_chaste_ewriter_fetching(self):

        # Test fetching using ewriter method
        w = myokit.formats.ewriter('chaste')
        self.assertIsInstance(w, myokit.formats.chaste.ChasteExpressionWriter)


if __name__ == '__main__':
    unittest.main()