#!/usr/bin/env python3
#
# Tests Myokit's SBML support.
#
# This file is part of Myokit.
# See http://myokit.org for copyright, sharing, and licensing details.
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals

import os
import unittest

import myokit
import myokit.formats
import myokit.formats.sbml

from shared import DIR_FORMATS, WarningCollector

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


class SBMLImporterTest(unittest.TestCase):
    """
    Tests the SBMLImporter.
    """

    def test_capability_reporting(self):
        # Test if the right capabilities are reported.
        i = myokit.formats.importer('sbml')
        self.assertFalse(i.supports_component())
        self.assertTrue(i.supports_model())
        self.assertFalse(i.supports_protocol())

    def test_hh_model(self):
        # Tests importing the Hodgkin-Huxley model
        i = myokit.formats.importer('sbml')
        with WarningCollector() as w:
            model = i.model(
                os.path.join(DIR_FORMATS, 'sbml', 'HodgkinHuxley.xml'))
        self.assertIn('Unknown SBML namespace', w.text())

        v = model.get('myokit.V')
        self.assertAlmostEqual(v.rhs().eval(), -4.01765286235500341e-03)


class SBMLParserTest(unittest.TestCase):
    """
    Unit tests for the SBMLParser class.

    Further tests are provided in SBMLModelTest.
    """

    @classmethod
    def setUpClass(cls):
        cls.p = myokit.formats.sbml.SBMLParser()

    def assertBad(self, xml, message, lvl=3, v=2):
        """
        Inserts the given ``xml`` into a <model> element, parses it, and checks
        that this raises an exception matching ``message``.
        """
        self.assertRaisesRegex(
            myokit.formats.sbml.SBMLError, message, self.parse, xml, lvl, v)

    def parse(self, xml, lvl=3, v=2):
        """
        Inserts the given ``xml`` into a <model> element, parses it, and
        returns the result.
        """
        return self.p.parse_string(self.wrap(xml, lvl, v))

    def wrap(self, xml_content, level=3, version=2):
        """
        Wraps ``xml_content`` into an SBML document of the specified ``level``
        and ``version``.
        """
        lv = 'level' + str(level) + '/version' + str(version)
        return (
            '<sbml xmlns="http://www.sbml.org/sbml/' + lv + '/core"'
            ' level="' + str(level) + '"'
            ' version="' + str(version) + '">'
            + xml_content +
            '</sbml>'
        )

    def test_algebraic_assignments(self):
        # Tests parsing algebraic assignments (not supported)
        xml = (
            '<model>'
            ' <listOfRules>'
            '  <algebraicRule>'
            '   <math xmlns="http://www.w3.org/1998/Math/MathML">'
            '    <apply>'
            '    </apply>'
            '   </math>'
            '  </algebraicRule>'
            ' </listOfRules>'
            '</model>'
        )
        self.assertBad(xml, 'Algebraic assignments are not supported')

    def test_constraints(self):
        # Test parsing constraints (these are ignored)

        xml = (
            '<model>'
            ' <listOfConstraints>'
            '  <constraint>'
            '   <math xmlns="http://www.w3.org/1998/Math/MathML">'
            '    <apply> <lt/> <cn> 1 </cn> <ci> S1 </ci> </apply>'
            '   </math>'
            '   <message>'
            '     <p xmlns="http://www.w3.org/1999/xhtml">Out of range.</p>'
            '   </message>'
            '  </constraint>'
            ' </listOfConstraints>'
            '</model>')

        with WarningCollector() as w:
            self.p.parse_string(self.wrap(xml))
        self.assertIn('Ignoring SBML constraints', w.text())

    def test_events(self):
        # Test parsing constraints (these are ignored)

        xml = (
            '<model>'
            ' <listOfEvents>'
            '  <event useValuesFromTriggerTime="true">'
            '   <trigger initialValue="false" persistent="true">'
            '    <math xmlns="http://www.w3.org/1998/Math/MathML">'
            '     <apply> <leq/> <ci> P_1 </ci> <ci> P_2 </ci> </apply>'
            '    </math>'
            '   </trigger>'
            '   <listOfEventAssignments>'
            '    <eventAssignment variable="k2">'
            '     <math xmlns="http://www.w3.org/1998/Math/MathML">'
            '      <ci> k2reset </ci>'
            '     </math>'
            '    </eventAssignment>'
            '   </listOfEventAssignments>'
            '  </event>'
            ' </listOfEvents>'
            '</model>')

        with WarningCollector() as w:
            self.p.parse_string(self.wrap(xml))
        self.assertIn('Ignoring SBML events', w.text())

    def test_invalid_file(self):
        # Check whether error is thrown for invalid xml
        self.assertBad('<model>', 'Unable to parse XML: ')

    def test_parse_file(self):
        # Check whether error is thrown for invalid path
        path = 'some/path'
        message = 'Unable to parse XML: '
        self.assertRaisesRegex(
            myokit.formats.sbml.SBMLError,
            message,
            self.p.parse_file,
            path)

    def test_level_version(self):
        # Check that unsupported levels/versions trigger warnings

        # Supported level
        xml = self.wrap('<model id="a" name="a"/>', 3, 2)
        with WarningCollector() as w:
            self.p.parse_string(xml)
        self.assertNotIn('This version of SBML may not be supported', w.text())

        # Unsupported level
        xml = self.wrap('<model id="a" name="a"/>', 2, 2)
        with WarningCollector() as w:
            self.p.parse_string(xml)
        self.assertIn('This version of SBML may not be supported', w.text())

        # Unsupported version
        xml = self.wrap('<model id="a" name="a"/>', 3, 1)
        with WarningCollector() as w:
            self.p.parse_string(xml)
        self.assertIn('This version of SBML may not be supported', w.text())

    def test_mathml(self):
        """Test MathML parsing."""

        xml1 = (
            '<model name="mathml">'
            ' <listOfParameters>'
            '  <parameter id="x" constant="true" />'
            ' </listOfParameters>'
            ' <listOfInitialAssignments>'
            '  <initialAssignment symbol="x">'
            '   <math xmlns="http://www.w3.org/1998/Math/MathML">'
        )
        xml2 = (
            '   </math>'
            '  </initialAssignment>'
            ' </listOfInitialAssignments>'
            '</model>'
        )

        # Parse valid MathML
        xml = xml1 + '<cn>5</cn>' + xml2
        m = self.parse(xml)
        self.assertEqual(m.get('myokit.x').rhs().eval(), 5)

        # Check that invalid MathML raises an SBMLError
        xml = xml1 + '<apply><minus /></apply>' + xml2
        self.assertBad(xml, 'Operator needs at least one operand')

    def test_model(self):
        """Tests parsing a model."""

        # Name from name attribute
        m = self.parse('<model id="yes" name="no" />')
        self.assertEqual(m.name(), 'no')

        # Name from id attribute
        m = self.parse('<model id="yes" />')
        self.assertEqual(m.name(), 'yes')

        # Default name
        m = self.parse('<model />')
        self.assertEqual(m.name(), 'Imported SBML model')

        # No model
        self.assertBad(xml='', message='Model element not found.')
        self.assertBad(xml='<hello/>', message='Model element not found.')

    def test_function_definitions(self):
        xml = (
            '<model id="test" name="test" timeUnits="second"> '
            '<listOfFunctionDefinitions>'
            '<functionDefinition id="multiply" name="multiply">'
            '<math xmlns="http://www.w3.org/1998/Math/MathML">'
            '<lambda>'
            '<bvar>'
            '<ci> x </ci>'
            '</bvar>'
            '<bvar>'
            '<ci> y </ci>'
            '</bvar>'
            '<apply>'
            '<times/>'
            '<ci> x </ci>'
            '<ci> y </ci>'
            '</apply>'
            '</lambda>'
            '</math>'
            '</functionDefinition>'
            '</listOfFunctionDefinitions>'
            '</model>')
        self.assertBad(xml, 'Function definitions are not supported.')

    def test_missing_id(self):
        # missing unit ID
        xml = (
            '<model id="test" name="test" timeUnits="second">'
            '<listOfUnitDefinitions>'
            '<unitDefinition>'  # here is where an ID is supposed to be
            '<listOfUnits>'
            '<unit kind="litre" exponent="1" scale="0" multiplier="1"/>'
            '</listOfUnits>'
            '</unitDefinition>'
            '</listOfUnitDefinitions>'
            '</model>')
        self.assertBad(xml, 'attribute "id" missing in unitDefinition')

        # missing compartment ID
        xml = (
            '<model id="test" name="test" timeUnits="second">'
            '<listOfCompartments>'
            '<compartment/>'  # here is where the ID is missing
            '</listOfCompartments>'
            '</model>')
        self.assertBad(xml, 'attribute "id" missing in compartment')

        # missing parameter ID
        xml = (
            '<model id="test" name="test" timeUnits="second">'
            '<listOfParameters>'
            '<parameter/>'  # here is where the ID is missing
            '</listOfParameters>'
            '</model>')
        self.assertBad(xml, 'attribute "id" missing in parameter')

        # missing global conversion factor ID
        xml = (
            '<model id="test" conversionFactor="someFactor" '
            'timeUnits="second">'
            '<listOfParameters>'
            '<parameter id="someOtherFactor"/>'
            '</listOfParameters>'
            '</model>')
        self.assertBad(
            xml=xml,
            message='The model conversionFactor points to non-existent ID.')

        # missing species ID
        xml = (
            '<model id="test" name="test" timeUnits="second">'
            '<listOfSpecies>'
            '<species/>'  # here is where the ID is missing
            '</listOfSpecies>'
            '</model>')
        self.assertBad(
            xml=xml,
            message='No species ID provided.')

        # missing conversion factor ID
        xml = (
            '<model id="test" name="test" timeUnits="second">'
            '<listOfCompartments>'
            '<compartment id="someComp"/>'
            '</listOfCompartments>'
            '<listOfSpecies>'
            '<species id="someSpecies" hasOnlySubstanceUnits="true" '
            'compartment="someComp" constant="false" boundaryCondition="false"'
            ' conversionFactor="someFactor"/>'
            '</listOfSpecies>'
            '</model>')
        self.assertBad(
            xml=xml,
            message='conversionFactor refers to non-existent ID.')

        # missing reactant ID
        xml = (
            '<model id="test" name="test" timeUnits="second">'
            '<listOfReactions>'
            '<reaction>'
            '<listOfReactants>'
            '<speciesReference species="someSpecies"/>'
            '</listOfReactants>'
            '</reaction>'
            '</listOfReactions>'
            '</model>')
        self.assertBad(
            xml=xml,
            message='Species ID not existent.')

        # missing product ID
        xml = (
            '<model id="test" name="test" timeUnits="second">'
            '<listOfReactions>'
            '<reaction>'
            '<listOfProducts>'
            '<speciesReference species="someSpecies"/>'
            '</listOfProducts>'
            '</reaction>'
            '</listOfReactions>'
            '</model>')
        self.assertBad(
            xml=xml,
            message='Species ID not existent.')

        # missing modifier ID
        xml = (
            '<model id="test" name="test" timeUnits="second">'
            '<listOfCompartments>'
            '<compartment id="someComp"/>'
            '</listOfCompartments>'
            '<listOfSpecies>'
            '<species id="someSpecies" hasOnlySubstanceUnits="true" '
            'compartment="someComp" constant="false" boundaryCondition="false"'
            '/>'
            '</listOfSpecies>'
            '<listOfReactions>'
            '<reaction>'
            '<listOfReactants>'
            '<speciesReference species="someSpecies"/>'
            '</listOfReactants>'
            '<listOfModifiers>'
            '<modifierSpeciesReference species="someOtherSpecies"/>'
            '</listOfModifiers>'
            '</reaction>'
            '</listOfReactions>'
            '</model>')
        with WarningCollector() as w:
            self.assertBad(xml=xml, message='Species ID not existent.')
        self.assertEqual(w.count(), 1)
        self.assertIn('Stoichiometry has not been set', w.text())

    def test_reserved_compartment_id(self):
        # ``Myokit`` is a reserved ID that is used while importing for the
        # myokit compartment.

        xml = (
            '<model id="test" name="test" timeUnits="second">'
            '<listOfCompartments>'
            '<compartment id="myokit"/>'
            '</listOfCompartments>'
            '</model>')
        self.assertBad(xml=xml, message='The id "myokit".')

    def test_coinciding_ids(self):
        # Checks that an error is thrown when overlapping ids are used for
        # compartment, parameters or species.

        # Coinciding compartment and parameter ids
        xml = (
            '<model id="test" '
            'timeUnits="second">'
            '<listOfCompartments>'
            '<compartment id="someId"/>'
            '</listOfCompartments>'
            '<listOfParameters>'
            '<parameter id="someId"/>'
            '</listOfParameters>'
            '</model>')
        self.assertBad(xml, 'Duplicate parameter id')

        # Coinciding compartment and species ids
        xml = (
            '<model id="test" '
            'timeUnits="second">'
            '<listOfCompartments>'
            '<compartment id="someId"/>'
            '</listOfCompartments>'
            '<listOfSpecies>'
            '<species id="someId" hasOnlySubstanceUnits="true"'
            ' compartment="someId" constant="false"'
            ' boundaryCondition="true" />'
            '</listOfSpecies>'
            '</model>')
        self.assertBad(xml, 'Duplicate species id')

        # Coinciding parameter and species ids
        xml = (
            '<model id="test" '
            'timeUnits="second">'
            '<listOfCompartments>'
            '<compartment id="someComp"/>'
            '</listOfCompartments>'
            '<listOfParameters>'
            '<parameter id="someId"/>'
            '</listOfParameters>'
            '<listOfSpecies>'
            '<species id="someId" hasOnlySubstanceUnits="true"'
            ' compartment="someComp" constant="false"'
            ' boundaryCondition="true" />'
            '</listOfSpecies>'
            '</model>')
        self.assertBad(xml, 'Duplicate species id')

        # Coinciding parameter and reactant stoichiometry IDs
        stoich_id = 'someStoich'
        xml = (
            '<model id="test" name="test" timeUnits="second">'
            '<listOfCompartments>'
            '<compartment id="someComp"/>'
            '</listOfCompartments>'
            '<listOfParameters>'
            '<parameter id="' + stoich_id + '"/>'
            '</listOfParameters>'
            '<listOfSpecies>'
            '<species id="someSpecies" hasOnlySubstanceUnits="true" '
            'compartment="someComp" constant="false" boundaryCondition="false"'
            '/>'
            '</listOfSpecies>'
            '<listOfReactions>'
            '<reaction >'
            '<listOfReactants>'
            '<speciesReference species="someSpecies" '
            'id="' + stoich_id + '"/>'
            '</listOfReactants>'
            '</reaction>'
            '</listOfReactions>'
            '</model>')
        with WarningCollector() as w:
            self.assertBad(xml=xml, message='Stoichiometry ID is not unique.')
        self.assertEqual(w.count(), 1)
        self.assertIn('Stoichiometry has not been set', w.text())

        # Coinciding parameter and product stoichiometry IDs
        stoich_id = 'someStoich'
        xml = (
            '<model id="test" name="test" timeUnits="second">'
            '<listOfCompartments>'
            '<compartment id="someComp"/>'
            '</listOfCompartments>'
            '<listOfParameters>'
            '<parameter id="' + stoich_id + '"/>'
            '</listOfParameters>'
            '<listOfSpecies>'
            '<species id="someSpecies" hasOnlySubstanceUnits="true" '
            'compartment="someComp" constant="false" boundaryCondition="false"'
            '/>'
            '</listOfSpecies>'
            '<listOfReactions>'
            '<reaction >'
            '<listOfProducts>'
            '<speciesReference species="someSpecies" '
            'id="' + stoich_id + '"/>'
            '</listOfProducts>'
            '</reaction>'
            '</listOfReactions>'
            '</model>')
        with WarningCollector() as w:
            self.assertBad(xml=xml, message='Stoichiometry ID is not unique.')
        self.assertEqual(w.count(), 1)
        self.assertIn('Stoichiometry has not been set', w.text())

    def test_reserved_parameter_id(self):
        # ``globalConversionFactor`` is a reserved ID that is used while
        # importing the global conversion factor.

        xml = (
            '<model id="test" conversionFactor="someFactor" '
            'timeUnits="second">'
            '<listOfParameters>'
            '<parameter id="globalConversionFactor"/>'
            '</listOfParameters>'
            '</model>')
        self.assertBad(
            xml=xml,
            message='The ID <globalConversionFactor> is protected in a myokit'
            ' SBML import. Please rename IDs.')

    def test_missing_compartment(self):
        # Tests whether error is thrown when ``compartment``
        # attribute is not specified for a species.

        xml = (
            '<model id="test" name="test" timeUnits="second">'
            '<listOfSpecies>'
            '<species id="someSpecies"/>'
            '</listOfSpecies>'
            '</model>')
        self.assertBad(
            xml=xml,
            message='No <compartment> attribute provided.')

    def test_stoichiometry_reference(self):
        # Tests whether stoichiometry parameters are linked properly to global
        # variables.

        # Check that reactant stoichiometry is added as parameter to referenced
        # compartment
        comp_id = 'someComp'
        stoich_id = 'someStoich'
        xml = (
            '<model id="test" name="test" timeUnits="second">'
            '<listOfCompartments>'
            '<compartment id="' + comp_id + '"/>'
            '</listOfCompartments>'
            '<listOfSpecies>'
            '<species id="someSpecies" hasOnlySubstanceUnits="true" '
            'compartment="' + comp_id + '" '
            'constant="false" boundaryCondition="false"/>'
            '</listOfSpecies>'
            '<listOfReactions>'
            '<reaction compartment="' + comp_id + '">'
            '<listOfReactants>'
            '<speciesReference species="someSpecies" '
            'id="' + stoich_id + '"/>'
            '</listOfReactants>'
            '</reaction>'
            '</listOfReactions>'
            '</model>')
        with WarningCollector() as w:
            model = self.parse(xml)
        self.assertTrue(model.has_variable(comp_id + '.' + stoich_id))
        self.assertEqual(w.count(), 1)

        # Check that reactant stoichiometry is added as parameter to <myokit>
        # compartment, if no compartment is referenced
        comp_id = 'myokit'
        stoich_id = 'someStoich'
        xml = (
            '<model id="test" name="test" timeUnits="second">'
            '<listOfCompartments>'
            '<compartment id="someComp"/>'
            '</listOfCompartments>'
            '<listOfSpecies>'
            '<species id="someSpecies" hasOnlySubstanceUnits="true" '
            'compartment="someComp" constant="false" boundaryCondition="false"'
            '/>'
            '</listOfSpecies>'
            '<listOfReactions>'
            '<reaction >'
            '<listOfReactants>'
            '<speciesReference species="someSpecies" '
            'id="' + stoich_id + '"/>'
            '</listOfReactants>'
            '</reaction>'
            '</listOfReactions>'
            '</model>')
        with WarningCollector() as w:
            model = self.parse(xml)
        self.assertTrue(model.has_variable(comp_id + '.' + stoich_id))
        self.assertEqual(w.count(), 1)

        # Check that product stoichiometry is added as parameter to referenced
        # compartment
        comp_id = 'someComp'
        stoich_id = 'someStoich'
        xml = (
            '<model id="test" name="test" timeUnits="second">'
            '<listOfCompartments>'
            '<compartment id="' + comp_id + '"/>'
            '</listOfCompartments>'
            '<listOfSpecies>'
            '<species id="someSpecies" hasOnlySubstanceUnits="true" '
            'compartment="' + comp_id + '" '
            'constant="false" boundaryCondition="false"/>'
            '</listOfSpecies>'
            '<listOfReactions>'
            '<reaction compartment="' + comp_id + '">'
            '<listOfProducts>'
            '<speciesReference species="someSpecies" '
            'id="' + stoich_id + '"/>'
            '</listOfProducts>'
            '</reaction>'
            '</listOfReactions>'
            '</model>')
        with WarningCollector() as w:
            model = self.parse(xml)
        self.assertTrue(model.has_variable(comp_id + '.' + stoich_id))
        self.assertEqual(w.count(), 1)

        # Check that product stoichiometry is added as parameter to <myokit>
        # compartment, if no compartment is referenced
        comp_id = 'myokit'
        stoich_id = 'someStoich'
        xml = (
            '<model id="test" name="test" timeUnits="second">'
            '<listOfCompartments>'
            '<compartment id="someComp"/>'
            '</listOfCompartments>'
            '<listOfSpecies>'
            '<species id="someSpecies" hasOnlySubstanceUnits="true" '
            'compartment="someComp" constant="false" boundaryCondition="false"'
            '/>'
            '</listOfSpecies>'
            '<listOfReactions>'
            '<reaction >'
            '<listOfProducts>'
            '<speciesReference species="someSpecies" '
            'id="' + stoich_id + '"/>'
            '</listOfProducts>'
            '</reaction>'
            '</listOfReactions>'
            '</model>')
        with WarningCollector() as w:
            model = self.parse(xml)
        self.assertTrue(model.has_variable(comp_id + '.' + stoich_id))
        self.assertEqual(w.count(), 1)

    def test_missing_reactants_products(self):
        # Tests whether error is thrown when reaction does neither provide
        # reactants not products.

        xml = (
            '<model id="test" name="test" timeUnits="second">'
            '<listOfReactions>'
            '<reaction>'
            '</reaction>'
            '</listOfReactions>'
            '</model>')
        self.assertBad(
            xml=xml,
            message='Reaction must have at least one reactant or product.')

    def test_fast_reaction(self):
        # Tests whether error is thrown when a reaction is flagged as ``fast``.
        # Myokit treats all reactions equal, so fast reactions are not
        # supported.

        xml = (
            '<model id="test" name="test" timeUnits="second">'
            '<listOfCompartments>'
            '<compartment id="someComp"/>'
            '</listOfCompartments>'
            '<listOfSpecies>'
            '<species id="someSpecies" hasOnlySubstanceUnits="true" '
            'compartment="someComp" constant="false" boundaryCondition="false"'
            '/>'
            '</listOfSpecies>'
            '<listOfReactions>'
            '<reaction fast="true">'
            '<listOfReactants>'
            '<speciesReference species="someSpecies"/>'
            '</listOfReactants>'
            '</reaction>'
            '</listOfReactions>'
            '</model>')
        with WarningCollector() as w:
            self.assertBad(
                xml=xml,
                message='Myokit does not support the conversion of <fast>')
        self.assertEqual(w.count(), 1)

    def test_local_parameters(self):
        # Tests whether error is thrown when a reaction has
        # ``localParameters``.
        # Local parameters are currenly not supported in myokit.

        xml = (
            '<model id="test" name="test" timeUnits="second">'
            '<listOfCompartments>'
            '<compartment id="someComp"/>'
            '</listOfCompartments>'
            '<listOfSpecies>'
            '<species id="someSpecies" hasOnlySubstanceUnits="true" '
            'compartment="someComp" constant="false" boundaryCondition="false"'
            '/>'
            '</listOfSpecies>'
            '<listOfReactions>'
            '<reaction>'
            '<listOfReactants>'
            '<speciesReference species="someSpecies"/>'
            '</listOfReactants>'
            '<kineticLaw>'
            '<listOfLocalParameters>'
            '<localParameter id="someParameter"/>'
            '</listOfLocalParameters>'
            '</kineticLaw>'
            '</reaction>'
            '</listOfReactions>'
            '</model>')
        with WarningCollector() as w:
            self.assertBad(
                xml=xml,
                message='does not support the definition of local parameters')
        self.assertEqual(w.count(), 1)

    def test_bad_kinetic_law(self):
        # Tests whether an error is thrown if a kinetic law refers to
        # non-existent parameters.

        xml = (
            '<model id="test" name="test" timeUnits="second">'
            ' <listOfCompartments>'
            '  <compartment id="someComp"/>'
            ' </listOfCompartments>'
            ' <listOfSpecies>'
            '  <species id="someSpecies" hasOnlySubstanceUnits="true" '
            '            compartment="someComp" constant="false"'
            '            boundaryCondition="false" />'
            ' </listOfSpecies>'
            ' <listOfReactions>'
            '  <reaction>'
            '   <listOfReactants>'
            '    <speciesReference species="someSpecies"/>'
            '   </listOfReactants>'
            '   <kineticLaw>'
            '   <math xmlns="http://www.w3.org/1998/Math/MathML">'
            '    <apply>'
            '     <times/><ci>someParam</ci><ci>someSpecies</ci>'
            '    </apply>'
            '   </math>'
            '   </kineticLaw>'
            '  </reaction>'
            ' </listOfReactions>'
            '</model>')
        with WarningCollector() as w:
            self.assertBad(xml=xml, message='Unable to create Name:')
        self.assertEqual(w.count(), 1)


class SBMLDocumentTest(unittest.TestCase):
    """
    Tests parsing an SBML file with species, compartments, and reactions, as
    well as assignment rules.

    This tests uses a modified model (case 00004) from the SBML test suite
    http://sbml.org/Facilities/Database/.
    """
    @classmethod
    def setUpClass(cls):
        p = myokit.formats.sbml.SBMLParser()
        with WarningCollector():
            cls.model = p.parse_file(os.path.join(
                DIR_FORMATS, 'sbml', '00004-sbml-l3v2-modified.xml'))

    def test_assignment_rules(self):
        # Tests whether intermediate variables have been assigned with correct
        # expressions.

        # parameter 1
        parameter = 'S1_Concentration'
        parameter = self.model.get('compartment.' + parameter)
        expression = 'compartment.S1 / compartment.size'
        self.assertEqual(str(parameter.rhs()), expression)

        # parameter 2
        parameter = 'S2_Concentration'
        parameter = self.model.get('compartment.' + parameter)
        expression = 'compartment.S2 / compartment.size'
        self.assertEqual(str(parameter.rhs()), expression)

        # parameter 3
        parameter = 'i_Na'
        parameter = self.model.get('myokit.' + parameter)
        expression = 'myokit.g_Na * myokit.m ^ 3'
        self.assertEqual(str(parameter.rhs()), expression)

    def test_compartments(self):
        # Tests whether compartments have been imported properly. Compartments
        # should include the compartments in the SBML file, plus a myokit
        # compartment for the global parameters.

        # compartment 1
        comp = 'compartment'
        self.assertTrue(self.model.has_component(comp))

        # compartment 2
        comp = 'myokit'
        self.assertTrue(self.model.has_component(comp))

        # total number of compartments
        number = 2
        self.assertEqual(self.model.count_components(), number)

    def test_constant_parameters(self):
        # Tests whether all constant parameters in the file were properly
        # imported.

        # parameter 1
        parameter = 'k1'
        self.assertTrue(self.model.has_variable('myokit.' + parameter))
        parameter = self.model.get('myokit.' + parameter)
        self.assertTrue(parameter.is_constant())

        # parameter 2
        parameter = 'k2'
        self.assertTrue(self.model.has_variable('myokit.' + parameter))
        parameter = self.model.get('myokit.' + parameter)
        self.assertTrue(parameter.is_constant())

        # parameter 3
        parameter = 'size'
        self.assertTrue(
            self.model.has_variable('compartment.' + parameter))
        parameter = self.model.get('compartment.' + parameter)
        self.assertTrue(parameter.is_constant())

        # parameter g_Na
        parameter = 'g_Na'
        self.assertTrue(self.model.has_variable('myokit.' + parameter))
        parameter = self.model.get('myokit.' + parameter)
        self.assertTrue(parameter.is_constant())

        # parameter Cm
        parameter = 'Cm'
        self.assertTrue(self.model.has_variable('myokit.' + parameter))
        parameter = self.model.get('myokit.' + parameter)
        self.assertTrue(parameter.is_constant())

        # total number of constants
        number = 6
        self.assertEqual(self.model.count_variables(const=True), number)

    def test_initial_values(self):
        # Tests whether initial values of constant parameters and state
        # variables have been set properly.

        # State S1
        state = self.model.get('compartment.S1')
        self.assertEqual(state.state_value(), 0.15)

        # State S2
        state = self.model.get('compartment.S2')
        self.assertEqual(state.state_value(), 0)

        # State V
        state = self.model.get('myokit.V')
        self.assertEqual(state.state_value(), -80)

        # State m
        state = self.model.get('myokit.m')
        self.assertAlmostEqual(state.state_value(), 0.3)

        # parameter 1
        parameter = self.model.get('myokit.k1')
        self.assertEqual(parameter.eval(), 0.35)

        # parameter 2
        parameter = self.model.get('myokit.k2')
        self.assertEqual(parameter.eval(), 180)

        # parameter compartment.size
        parameter = self.model.get('compartment.size')
        self.assertEqual(parameter.eval(), 1)

        # parameter g_Na
        parameter = self.model.get('myokit.g_Na')
        self.assertEqual(parameter.eval(), 2)

        # parameter Cm
        parameter = self.model.get('myokit.Cm')
        self.assertEqual(parameter.eval(), 1)

    def test_intermediary_variables(self):
        # Tests whether all intermediary variables in the file were properly
        # imported.

        var = self.model.get('compartment.S1_Concentration')
        self.assertTrue(var.is_intermediary())

        var = self.model.get('compartment.S2_Concentration')
        self.assertTrue(var.is_intermediary())

        var = self.model.get('myokit.i_Na')
        self.assertTrue(var.is_intermediary())

        # total number of intermediary variables
        self.assertEqual(self.model.count_variables(inter=True), 3)

    def test_notes(self):
        # Test notes are read from model element
        notes = self.model.meta['desc']
        self.assertTrue(notes.startswith('This is an implementation of'))

    def test_model_name(self):
        # Tests whether model name is set properly.
        name = 'case00004'
        self.assertEqual(self.model.name(), name)

    def test_older_version(self):
        # Test loading the same model in an older SBML format.

        # Parse model in older format
        p = myokit.formats.sbml.SBMLParser()
        with WarningCollector():
            old_model = p.parse_file(os.path.join(
                DIR_FORMATS, 'sbml', '00004-sbml-l2v1-modified.xml'))

        # Set time units (Only introduced in level 3 version 1)
        old_model.time().set_unit(self.model.time().unit())

        # Update initial states set with initialAssignment (level 3 only)
        new_value = self.model.get('myokit.m').state_value()
        self.assertAlmostEqual(
            old_model.get('myokit.m').state_value(), new_value)
        old_model.get('myokit.m').set_state_value(new_value)

        # Update initial value set with initialAssignment (level 3 only)
        self.assertAlmostEqual(
            old_model.get('myokit.h').eval(),
            self.model.get('myokit.h').eval(),
        )
        old_model.get('myokit.h').set_rhs(
            self.model.get('myokit.h').rhs().code())

        # Now models should be equal
        #with open('new.xml', 'w') as f:
        #    f.write(self.model.code())
        #with open('old.xml', 'w') as f:
        #    f.write(old_model.code())
        self.assertEqual(self.model.code(), old_model.code())

    def test_rate_expressions(self):
        # Tests whether state variables have been assigned with the correct
        # rate expression. Those may come from a rateRule or reaction.

        # state 1
        state = 'S1'
        state = self.model.get('compartment.' + state)
        expression = (
            '-(compartment.size * myokit.k1 * compartment.S1_Concentration)'
            ' + compartment.size * myokit.k2'
            ' * compartment.S2_Concentration ^ 2')
        self.assertEqual(state.rhs().code(), expression)

        # state 2
        state = 'S2'
        state = self.model.get('compartment.' + state)
        expression = str(
            '2 * (compartment.size * myokit.k1 * '
            + 'compartment.S1_Concentration) - 2 * '
            + '(compartment.size * myokit.k2 '
            + '* compartment.S2_Concentration ^ 2)')
        self.assertEqual(str(state.rhs()), expression)

        # state 3
        state = 'V'
        state = self.model.get('myokit.' + state)
        expression = 'myokit.i_Na / myokit.Cm'
        self.assertEqual(str(state.rhs()), expression)

    def test_state_variables(self):
        # Tests whether all dynamic variables were imported properly.

        # state 1
        state = 'S1'
        self.assertTrue(self.model.has_variable('compartment.' + state))
        state = self.model.get('compartment.' + state)
        self.assertTrue(state.is_state())

        # state 2
        state = 'S2'
        self.assertTrue(self.model.has_variable('compartment.' + state))
        state = self.model.get('compartment.' + state)
        self.assertTrue(state.is_state())

        # state V
        self.assertTrue(self.model.has_variable('myokit.V'))
        state = self.model.get('myokit.V')
        self.assertTrue(state.is_state())

        # state m
        self.assertTrue(self.model.has_variable('myokit.m'))
        state = self.model.get('myokit.m')
        self.assertTrue(state.is_state())

        # total number of states
        self.assertEqual(self.model.count_variables(state=True), 4)

    def test_time(self):
        # Tests whether the time bound variable was set properly

        variable = 'time'
        self.assertTrue(self.model.has_variable('myokit.' + variable))
        variable = self.model.get('myokit.' + variable)
        self.assertTrue(variable.is_bound())

    def test_units(self):
        # Tests units parsing.

        # state 1
        state = 'S1'
        state = self.model.get('compartment.' + state)
        unit = myokit.units.mol
        self.assertEqual(state.unit(), unit)

        # state 2
        state = 'S2'
        state = self.model.get('compartment.' + state)
        unit = myokit.units.mol
        self.assertEqual(state.unit(), unit)

        # state 3
        state = 'V'
        state = self.model.get('myokit.' + state)
        unit = myokit.units.V * 10 ** (-3)
        self.assertEqual(state.unit(), unit)

        # parameter 1
        parameter = 'k1'
        parameter = self.model.get('myokit.' + parameter)
        unit = None
        self.assertEqual(parameter.unit(), unit)

        # parameter 2
        parameter = 'k2'
        parameter = self.model.get('myokit.' + parameter)
        unit = None
        self.assertEqual(parameter.unit(), unit)

        # parameter 3
        parameter = 'size'
        parameter = self.model.get('compartment.' + parameter)
        unit = myokit.units.L
        self.assertEqual(parameter.unit(), unit)

        # parameter 4
        parameter = 'S1_Concentration'
        parameter = self.model.get('compartment.' + parameter)
        unit = myokit.units.mol / myokit.units.L
        self.assertEqual(parameter.unit(), unit)

        # parameter 5
        parameter = 'S2_Concentration'
        parameter = self.model.get('compartment.' + parameter)
        unit = myokit.units.mol / myokit.units.L
        self.assertEqual(parameter.unit(), unit)

        # parameter 6
        parameter = 'i_Na'
        parameter = self.model.get('myokit.' + parameter)
        unit = None
        self.assertEqual(parameter.unit(), unit)

        # parameter 7
        parameter = 'g_Na'
        parameter = self.model.get('myokit.' + parameter)
        unit = None
        self.assertEqual(parameter.unit(), unit)

        # parameter 8
        parameter = 'm'
        parameter = self.model.get('myokit.' + parameter)
        unit = None
        self.assertEqual(parameter.unit(), unit)

        # parameter 9
        parameter = 'Cm'
        parameter = self.model.get('myokit.' + parameter)
        unit = None
        self.assertEqual(parameter.unit(), unit)


if __name__ == '__main__':
    import warnings
    warnings.simplefilter('always')
    unittest.main()
