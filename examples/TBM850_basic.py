from __future__ import division
import sys
import os
import numpy as np
import openmdao.api as om

sys.path.insert(0, os.getcwd())
from openmdao.api import Problem, Group, ScipyOptimizeDriver
from openmdao.api import DirectSolver, SqliteRecorder, IndepVarComp
from openmdao.api import NewtonSolver, BoundsEnforceLS

# imports for the airplane model itself
from openconcept.analysis.aerodynamics import PolarDrag
from openconcept.utilities.math import AddSubtractComp
from openconcept.utilities.math.integrals import Integrator
from examples.methods.weights_turboprop import SingleTurboPropEmptyWeight
from examples.propulsion_layouts.simple_turboprop import TurbopropPropulsionSystem
from examples.methods.costs_commuter import OperatingCost
from openconcept.utilities.dict_indepvarcomp import DictIndepVarComp
from examples.aircraft_data.TBM850 import data as acdata
from openconcept.analysis.performance.mission_profiles import BasicMission

class TBM850AirplaneModel(Group):
    """
    A custom model specific to the TBM 850 airplane
    This class will be passed in to the mission analysis code.

    """
    def initialize(self):
        self.options.declare('num_nodes', default=1)
        self.options.declare('flight_phase', default=None)

    def setup(self):
        nn = self.options['num_nodes']
        flight_phase = self.options['flight_phase']

        # any control variables other than throttle and braking need to be defined here
        controls = self.add_subsystem('controls', IndepVarComp(), promotes_outputs=['*'])
        controls.add_output('prop1rpm', val=np.ones((nn,)) * 2000, units='rpm')

        # a propulsion system needs to be defined in order to provide thrust
        # information for the mission analysis code
        propulsion_promotes_outputs = ['fuel_flow', 'thrust']
        propulsion_promotes_inputs = ["fltcond|*", "ac|propulsion|*", "throttle"]

        self.add_subsystem('propmodel', TurbopropPropulsionSystem(num_nodes=nn),
                           promotes_inputs=propulsion_promotes_inputs,
                           promotes_outputs=propulsion_promotes_outputs)
        self.connect('prop1rpm', 'propmodel.prop1.rpm')

        # use a different drag coefficient for takeoff versus cruise
        if flight_phase not in ['climb','descent']:
            cd0_source = 'ac|aero|polar|CD0_cruise'
        else:
            cd0_source = 'ac|aero|polar|CD0_TO'
        self.add_subsystem('drag', PolarDrag(num_nodes=nn),
                           promotes_inputs=['fltcond|CL', 'ac|geom|*', ('CD0', cd0_source),
                                            'fltcond|q', ('e', 'ac|aero|polar|e')],
                           promotes_outputs=['drag'])

        # generally the weights module will be custom to each airplane
        self.add_subsystem('OEW', SingleTurboPropEmptyWeight(),
                           promotes_inputs=['*', ('P_TO', 'ac|propulsion|engine|rating')], # Is this a connection between P_TO in weights_truboprop and the engine rating in aircraft data?
                           promotes_outputs=[('OEW','ac|weights|OEW')])
        self.connect('propmodel.prop1.component_weight', 'W_propeller')
        self.connect('propmodel.eng1.component_weight', 'W_engine')

        # airplanes which consume fuel will need to integrate
        # fuel usage across the mission and subtract it from TOW
        intfuel = self.add_subsystem('intfuel', Integrator(num_nodes=nn, method='simpson', diff_units='s',
                                                              time_setup='duration'), promotes_inputs=['*'], promotes_outputs=['*'])
        intfuel.add_integrand('fuel_used', rate_name='fuel_flow', val=1.0, units='kg')
        
        self.add_subsystem('weight', AddSubtractComp(output_name='weight',
                                                     input_names=['ac|weights|OEW', 'ac|weights|payload','ac|weights|W_fuel_max', 'fuel_used'],
                                                     units='kg', vec_size=[1,1,1, nn],
                                                     scaling_factors=[1,1,1, -1]),
                           promotes_outputs=['weight'],
                           promotes_inputs=['*'])
        

class TBMAnalysisGroup(Group):
    """This is an example of a balanced field takeoff and three-phase mission analysis.
    """
    def setup(self):
        # Define number of analysis points to run pers mission segment
        nn = 11

        # Define a bunch of design varaiables and airplane-specific parameters
        dv_comp = self.add_subsystem('dv_comp',  DictIndepVarComp(acdata),
                                     promotes_outputs=["*"])
        dv_comp.add_output_from_dict('ac|aero|CLmax_TO')
        dv_comp.add_output_from_dict('ac|aero|polar|e')
        dv_comp.add_output_from_dict('ac|aero|polar|CD0_TO')
        dv_comp.add_output_from_dict('ac|aero|polar|CD0_cruise')

        dv_comp.add_output_from_dict('ac|geom|wing|S_ref')
        dv_comp.add_output_from_dict('ac|geom|wing|AR')
        dv_comp.add_output_from_dict('ac|geom|wing|c4sweep')
        dv_comp.add_output_from_dict('ac|geom|wing|taper')
        dv_comp.add_output_from_dict('ac|geom|wing|toverc')
        dv_comp.add_output_from_dict('ac|geom|hstab|S_ref')
        dv_comp.add_output_from_dict('ac|geom|hstab|c4_to_wing_c4')
        dv_comp.add_output_from_dict('ac|geom|vstab|S_ref')
        dv_comp.add_output_from_dict('ac|geom|fuselage|S_wet')
        dv_comp.add_output_from_dict('ac|geom|fuselage|width')
        dv_comp.add_output_from_dict('ac|geom|fuselage|length')
        dv_comp.add_output_from_dict('ac|geom|fuselage|height')
        dv_comp.add_output_from_dict('ac|geom|nosegear|length')
        dv_comp.add_output_from_dict('ac|geom|maingear|length')

        dv_comp.add_output_from_dict('ac|weights|MTOW')
        # dv_comp.add_output_from_dict('ac|weights|W_fuel_max')
        dv_comp.add_output_from_dict('ac|weights|MLW')
        dv_comp.add_output('ac|weights|payload', val=849, units = 'lb')

        dv_comp.add_output_from_dict('ac|propulsion|engine|rating')
        dv_comp.add_output_from_dict('ac|propulsion|propeller|diameter')

        dv_comp.add_output_from_dict('ac|num_passengers_max')
        dv_comp.add_output_from_dict('ac|q_cruise')

        # Run a full mission analysis including takeoff, climb, cruise, and descent
        analysis = self.add_subsystem('analysis',
                                      BasicMission(num_nodes=nn,
                                                          aircraft_model=TBM850AirplaneModel),
                                      promotes_inputs=['*'], promotes_outputs=['*'])
        
        self.add_subsystem('OEW', SingleTurboPropEmptyWeight(),
                           promotes_inputs=['*', ('P_TO', 'ac|propulsion|engine|rating')], # Is this a connection between P_TO in weights_truboprop and the engine rating in aircraft data?
                           promotes_outputs=[('OEW','ac|weights|OEW')])
        self.connect('propmodel.prop1.component_weight', 'W_propeller')
        self.connect('propmodel.eng1.component_weight', 'W_engine')

        self.add_subsystem('MTOW', AddSubtractComp(output_name='ac|weights|MTOW',
                                                     input_names=['ac|weights|OEW', 'ac|weights|payload','ac|weights|W_fuel_max'],
                                                     units='kg', vec_size=[1,1,1],
                                                     scaling_factors=[1,1,1]),
                           promotes_outputs=['ac|weights|MTOW'],
                           promotes_inputs=['*'])

        self.connect('descent.fuel_used_final', 'ac|weights|W_fuel_max')
        self.connect('cruise.ac|weights|OEW', 'ac|weights|OEW')
        self.set_input_defaults('ac|weights|MTOW',acdata['ac']['weights']['MTOW']['value'], units=acdata['ac']['weights']['MTOW']['units'])
        self.set_input_defaults('ac|weights|W_fuel_max',acdata['ac']['weights']['W_fuel_max']['value'], units=acdata['ac']['weights']['W_fuel_max']['units'])
        

def run_tbm_optimizer():
    # Set up OpenMDAO to analyze the airplane
    num_nodes = 11
    prob = Problem()
    prob.model = TBMAnalysisGroup()
    prob.model.nonlinear_solver = NewtonSolver(iprint=2)
    prob.model.options['assembled_jac_type'] = 'csc'
    prob.model.linear_solver = DirectSolver(assemble_jac=True)
    prob.model.nonlinear_solver.options['solve_subsystems'] = True
    prob.model.nonlinear_solver.options['maxiter'] = 10
    prob.model.nonlinear_solver.options['atol'] = 1e-6
    prob.model.nonlinear_solver.options['rtol'] = 1e-6
    prob.model.nonlinear_solver.linesearch = BoundsEnforceLS(bound_enforcement='scalar', print_bound_enforce=True)
    
    # add driver
    prob.driver=ScipyOptimizeDriver(optimizer='SLSQP')
    prob.model.add_design_var('ac|geom|wing|S_ref', lower = 10, upper=25, units='m**2')
    prob.model.add_design_var('ac|propulsion|engine|rating', lower = 500, upper=1500, units='hp', ref=800)
    prob.model.add_objective('descent.fuel_used_final')
    prob.model.add_constraint('climb.throttle', upper=1.0)
    prob.model.add_constraint('cruise.throttle', upper=1.0)
    prob.model.add_constraint('descent.throttle', upper=1.0)
    prob.driver.options['debug_print'] = ['desvars','objs','nl_cons']
    prob.driver.options['tol'] = 1e-04

    prob.setup(check=True, mode='fwd')

    # set some (required) mission parameters. Each pahse needs a vertical and air-speed
    # the entire mission needs a cruise altitude and range
    prob.set_val('climb.fltcond|vs', np.ones((num_nodes,))*1500, units='ft/min')
    prob.set_val('climb.fltcond|Ueas', np.ones((num_nodes,))*124, units='kn')
    prob.set_val('cruise.fltcond|vs', np.ones((num_nodes,))*0.01, units='ft/min')
    prob.set_val('cruise.fltcond|Ueas', np.ones((num_nodes,))*201, units='kn')
    prob.set_val('descent.fltcond|vs', np.ones((num_nodes,))*(-600), units='ft/min')
    prob.set_val('descent.fltcond|Ueas', np.ones((num_nodes,))*140, units='kn')

    prob.set_val('cruise|h0',28000.,units='ft')
    prob.set_val('mission_range',1250,units='NM')

    # (optional) guesses for takeoff speeds may help with convergence
    prob.set_val('climb.fltcond|Utrue',np.ones((num_nodes))*50,units='kn')
    prob.set_val('cruise.fltcond|Utrue',np.ones((num_nodes))*85,units='kn')
    prob.set_val('descent.fltcond|Utrue',np.ones((num_nodes))*85,units='kn')

    # set some airplane-specific values. The throttle edits are to derate the takeoff power of the PT6A
    # prob['climb.OEW.structural_fudge'] = 1.67
    prob['climb.throttle'] = np.ones((num_nodes)) / 1.21
    prob['cruise.throttle'] = np.ones((num_nodes)) / 1.21
    prob['descent.throttle'] = np.ones((num_nodes)) / 1.21

    prob.run_model()
    om.n2(prob,outfile = 'opt_test.html')
    return prob



if __name__ == "__main__":
    from openconcept.utilities.visualization import plot_trajectory
    # run the analysis
    prob = run_tbm_optimizer()

     # print some outputs
    # vars_list = ['ac|weights|MTOW','climb.OEW','climb.fuel_used_final','cruise.fuel_used_final','descent.fuel_used_final']
    # units = ['lb','lb','lb','lb','lb']
    # nice_print_names = ['MTOW', 'OEW', 'Climb fuel', 'Cruise fuel','Fuel used']
    # print("=======================================================================")
    # for i, thing in enumerate(vars_list):
    #     print(nice_print_names[i]+': '+str(prob.get_val(thing,units=units[i])[0])+' '+units[i])

    # plot some stuff
    plots = False
    if plots:
        x_var = 'range'
        x_unit = 'ft'
        y_vars = ['fltcond|Ueas', 'fltcond|h', 'fuel_used']
        y_units = ['kn', 'ft', 'lb']
        x_label = 'Distance (ft)'
        y_labels = ['Veas airspeed (knots)', 'Altitude (ft)', 'fuel used']
        phases = ['v0v1', 'v1vr', 'rotate', 'v1v0']
        plot_trajectory(prob, x_var, x_unit, y_vars, y_units, phases,
                        x_label=x_label, y_labels=y_labels,
                        plot_title='TBM850 Takeoff')

        x_var = 'range'
        x_unit = 'NM'
        y_vars = ['fltcond|h','fltcond|Ueas','fuel_used','throttle','fltcond|vs']
        y_units = ['ft','kn','lbm',None,'ft/min']
        x_label = 'Range (nmi)'
        y_labels = ['Altitude (ft)', 'Veas airspeed (knots)', 'Fuel used (lb)', 'Throttle setting', 'Vertical speed (ft/min)']
        phases = ['climb', 'cruise', 'descent']
        plot_trajectory(prob, x_var, x_unit, y_vars, y_units, phases,
                        x_label=x_label, y_labels=y_labels, marker='-',
                        plot_title='TBM850 Mission Profile')

