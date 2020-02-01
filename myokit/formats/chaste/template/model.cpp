<?
# Template for Chaste model implementation.
#
# Variables:
#   class_name      A valid camel cased class name
#   header_file     The name of the accompanying header file
#   model           A model object
#   model_name      A user friendly model name (arbitrary string)
#   var_name        A method that creates variable names
#   vm              The state variable representing membrane potential
#

# Common strings
tab = '    '

?>//! @file
//!
//! This source file was generated by Myokit
//!
//! Model: <?= model_name ?>
//!
//! <autogenerated>

#include "<?= header_file ?>"
#include <cmath>
#include <cassert>
#include <memory>
#include "Exception.hpp"
#include "OdeSystemInformation.hpp"
#include "RegularStimulus.hpp"
#include "HeartConfig.hpp"
#include "IsNan.hpp"
#include "MathsCustomFunctions.hpp"

    <?= class_name ?>::<?= class_name ?>(boost::shared_ptr<AbstractIvpOdeSolver> pSolver, boost::shared_ptr<AbstractStimulusFunction> pIntracellularStimulus)
        : AbstractCardiacCell(
                pSolver,
                <?= model.count_states() ?>,
                <?= vm.indice() ?>,
                pIntracellularStimulus)
    {
        // Time units: millisecond
        //
        this->mpSystemInfo = OdeSystemInformation<<?= class_name ?>>::Instance();
        Init();
    }

    <?= class_name ?>::~<?= class_name ?>()
    {
    }

    {% endif %}double <?= class_name ?>::GetIIonic(const std::vector<double>* pStateVariables)
    {
        // For state variable interpolation (SVI) we read in interpolated state variables,
        // otherwise for ionic current interpolation (ICI) we use the state variables of this model (node).
        if (!pStateVariables) pStateVariables = &rGetStateVariables();
        const std::vector<double>& rY = *pStateVariables;
        {% for state_var in state_vars %}
        {%- if state_var.in_ionic %}double {{ state_var.var }} = {% if loop.index0 == membrane_voltage_index %}(mSetVoltageDerivativeToZero ? this->mFixedVoltage : rY[{{loop.index0}}]);{%- else %}rY[{{loop.index0}}];{%- endif %}
        // Units: {{state_var.units}}; Initial value: {{state_var.initial_value}}
        {% endif %}{%- endfor %}{% for ionic_var in ionic_vars %}
        const double {{ionic_var.lhs}} = {{ionic_var.rhs}}; // {{ionic_var.units}}
        {%- endfor %}

        const double i_ionic = var_chaste_interface__i_ionic;
        EXCEPT_IF_NOT(!std::isnan(i_ionic));
        return i_ionic;
    }

    void <?= class_name ?>::EvaluateYDerivatives(double <?= var_name(model.time()) ?>, const std::vector<double>& rY, std::vector<double>& rDY)
    {
        // Inputs:
        // Time units: millisecond
<?
for var in model.states():
    rhs = 'rY[' + str(var.indice()) + ']'
    if var is vm:
        rhs = '(mSetVoltageDerivativeToZero ? this->mFixedVoltage : ' + rhs
    print(tab*2 + 'double ' + var_name(var) + ' = ' + rhs + ';')
    print(tab*2 + '// Units: ' + str(var.unit()) + '; Initial value: ' + str(var.state_value()))
?>
        // Mathematics
        {% for deriv in y_derivative_equations %}{%- if deriv.is_voltage%}double {{deriv.lhs}};{%- endif %}{%- endfor %}
        {%- for deriv in y_derivative_equations %}{%- if not deriv.in_membrane_voltage %}
        const double {{deriv.lhs}} = {{deriv.rhs}}; // {{deriv.units}}{%- endif %}
        {%- endfor %}

        if (mSetVoltageDerivativeToZero)
        {
            {% for deriv in y_derivative_equations %}{%- if deriv.is_voltage%}{{deriv.lhs}} = 0.0;{%- endif %}{%- endfor %}
        }
        else
        {
            {%- for deriv in y_derivative_equations %}{% if deriv.in_membrane_voltage %}
            {% if not deriv.is_voltage%}const double {% endif %}{{deriv.lhs}} = {{deriv.rhs}}; // {{deriv.units}}{%- endif %}
            {%- endfor %}
        }

        // Outputs:
<?
for var in model.states():
    print(tab*2 + 'rDY[' + str(var.indice()) + '] = ' + var_name(var.lhs()) + ';')
?>    }

template<>
void OdeSystemInformation<<?= class_name ?>>::Initialise(void)
{
    this->mSystemName = "{{free_variable.system_name}}";
    this->mFreeVariableName = "{{free_variable.name}}";
    this->mFreeVariableUnits = "{{free_variable.units}}";

    {% for ode_info in ode_system_information %}// rY[{{loop.index0}}]:
    this->mVariableNames.push_back("{{ode_info.name}}");
    this->mVariableUnits.push_back("{{ode_info.units}}");
    this->mInitialConditions.push_back({{ode_info.initial_value}});

    {% endfor %}{% for param in modifiable_parameters %}// mParameters[{{loop.index0}}]:
    this->mParameterNames.push_back("{{param["name"]}}");
    this->mParameterUnits.push_back("{{param["units"]}}");

    {% endfor %}{% for attr in named_attributes %}
    this->mAttributes["{{attr["name"]}}"] = {{attr["value"]}};
    {% endfor %}this->mInitialised = true;
}


// Serialization for Boost >= 1.36
#include "SerializationExportWrapperForCpp.hpp"
CHASTE_CLASS_EXPORT(<?= class_name ?>)
