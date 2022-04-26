from __future__ import division
import sys
import os
import numpy as np
import openmdao.api as om

sys.path.insert(0, os.getcwd())
from openmdao.api import Problem, Group, ScipyOptimizeDriver, pyOptSparseDriver
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
from examples.sizing_functions import HStabSizing_SmallTurboprop, VStabSizing_SmallTurboprop, WingMAC_Trapezoidal, WingRoot_LinearTaper

from examples.aircraft_data.TBM850 import data as acdata
from openconcept.analysis.performance.mission_profiles import FullMissionAnalysis, FullMissionWithReserve
from openconcept.analysis.openaerostruct.drag_polar import OASDragPolar

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
        if flight_phase not in ['v0v1', 'v1v0', 'v1vr', 'rotate']:
            self.set_input_defaults('ac|aero|polar|CD0_cruise', 0.0125)
            cd0_source = 'ac|aero|polar|CD0_cruise'
        else:
            cd0_source = 'ac|aero|polar|CD0_TO'
        # self.add_subsystem('drag', PolarDrag(num_nodes=nn),
        #                    promotes_inputs=['fltcond|CL', 'ac|geom|*', ('CD0', cd0_source),
        #                                     'fltcond|q', ('e', 'ac|aero|polar|e')],
        #                    promotes_outputs=['drag'])

        self.add_subsystem('drag', OASDragPolar(num_nodes=nn, num_x=5, num_y=11, num_twist=5, alpha_train=np.linspace(-10,10,5), Mach_train=np.linspace(0.1,0.6,3), alt_train=np.linspace(0, 12e3,4)),
                           promotes_inputs=['fltcond|CL', 'ac|geom|*', ('ac|aero|CD_nonwing', cd0_source),
                                            'fltcond|q', 'fltcond|M', 
                                            'fltcond|h'],
                           promotes_outputs=['drag'])
        
        self.set_input_defaults('ac|geom|wing|twist', np.zeros(5), units='deg')

        # generally the weights module will be custom to each airplane

        # airplanes which consume fuel will need to integrate
        # fuel usage across the mission and subtract it from TOW
        intfuel = self.add_subsystem('intfuel', Integrator(num_nodes=nn, method='simpson', diff_units='s',
                                                              time_setup='duration'), promotes_inputs=['*'], promotes_outputs=['*'])
        intfuel.add_integrand('fuel_used', rate_name='fuel_flow', val=1.0, units='kg')
        
        self.add_subsystem('weight', AddSubtractComp(output_name='weight',
                                                     input_names=['ac|weights|OEW', 'ac|weights|payload','ac|weights|W_fuel_max', 'fuel_used'],
                                                     units='kg', vec_size=[1,1,1, nn],
                                                     scaling_factors=[1,1,1, -1]),
                           promotes_inputs=['*'],
                           promotes_outputs=['weight'])

def Cl_calc(weight, wing_area, Vstall):
    rho = 1.22527
    Cl_max = 2*weight/(rho*(Vstall*0.514444)**2*wing_area)
    return Cl_max


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
        # dv_comp.add_output_from_dict('ac|geom|hstab|S_ref')
        dv_comp.add_output_from_dict('ac|geom|hstab|c4_to_wing_c4')
        # dv_comp.add_output_from_dict('ac|geom|vstab|S_ref')
        dv_comp.add_output_from_dict('ac|geom|fuselage|S_wet')
        dv_comp.add_output_from_dict('ac|geom|fuselage|width')
        dv_comp.add_output_from_dict('ac|geom|fuselage|length')
        dv_comp.add_output_from_dict('ac|geom|fuselage|height')
        dv_comp.add_output_from_dict('ac|geom|nosegear|length')
        dv_comp.add_output_from_dict('ac|geom|maingear|length')

        # dv_comp.add_output_from_dict('ac|weights|MTOW')
        # dv_comp.add_output_from_dict('ac|weights|W_fuel_max')
        dv_comp.add_output_from_dict('ac|weights|MLW')
        dv_comp.add_output('ac|weights|payload', val=850, units = 'lb')

        dv_comp.add_output_from_dict('ac|propulsion|engine|rating')
        dv_comp.add_output_from_dict('ac|propulsion|propeller|diameter')

        dv_comp.add_output_from_dict('ac|num_passengers_max')
        dv_comp.add_output_from_dict('ac|q_cruise')

        # Run a full mission analysis including takeoff, climb, cruise, and descent
        
        self.connect('climb.propmodel.prop1.component_weight', 'W_propeller')
        self.connect('climb.propmodel.eng1.component_weight','W_engine')
        
        self.add_subsystem('Wing_Root', WingRoot_LinearTaper(),
                            promotes_inputs=['*'],
                            promotes_outputs=[('C_root','ac|geom|wing|root_chord')])
        
        self.add_subsystem('Wing_MAC', WingMAC_Trapezoidal(),
                            promotes_inputs=['*'],
                            promotes_outputs=[('MAC','ac|geom|wing|MAC')])
        
        self.add_subsystem('HStab', HStabSizing_SmallTurboprop(),
                            promotes_inputs=['*'],
                            promotes_outputs=[('hstab_area','ac|geom|hstab|S_ref')])
        
        self.add_subsystem('VStab', VStabSizing_SmallTurboprop(),
                            promotes_inputs=['*'],
                            promotes_outputs=[('vstab_area','ac|geom|vstab|S_ref')])
        
        self.add_subsystem('OEW', SingleTurboPropEmptyWeight(),
                           promotes_inputs=['*', ('P_TO', 'ac|propulsion|engine|rating')],
                           promotes_outputs=[('OEW','ac|weights|OEW')])
        
        self.add_subsystem('MTOW', AddSubtractComp(output_name='ac|weights|MTOW',
                                                     input_names=['ac|weights|OEW', 'ac|weights|payload','ac|weights|W_fuel_max'],
                                                     units='kg', vec_size=[1,1,1],
                                                     scaling_factors=[1,1,1]),
                           promotes_outputs=['ac|weights|MTOW'],
                           promotes_inputs=['*'])
        
        analysis = self.add_subsystem('analysis',
                                      FullMissionWithReserve(num_nodes=nn,
                                                          aircraft_model=TBM850AirplaneModel),
                                      promotes_inputs=['*'], promotes_outputs=['*'])

        

        self.connect('loiter.fuel_used_final', 'ac|weights|W_fuel_max')
        self.set_input_defaults('ac|weights|MTOW',acdata['ac']['weights']['MTOW']['value'], units=acdata['ac']['weights']['MTOW']['units'])
        self.set_input_defaults('ac|weights|W_fuel_max',acdata['ac']['weights']['W_fuel_max']['value'], units=acdata['ac']['weights']['W_fuel_max']['units'])


def run_tbm_analysis():
    # Set up OpenMDAO to analyze the airplane
    num_nodes = 11
    prob = Problem()
    prob.model = TBMAnalysisGroup()
    prob.model.nonlinear_solver = NewtonSolver(iprint=2)
    prob.model.options['assembled_jac_type'] = 'csc'
    prob.model.nonlinear_solver.options['err_on_non_converge'] = True
    prob.model.linear_solver = DirectSolver(assemble_jac=True)
    prob.model.nonlinear_solver.options['solve_subsystems'] = True
    prob.model.nonlinear_solver.options['maxiter'] = 10
    prob.model.nonlinear_solver.options['atol'] = 1e-6
    prob.model.nonlinear_solver.options['rtol'] = 1e-6
    prob.model.nonlinear_solver.linesearch = BoundsEnforceLS(bound_enforcement='scalar', print_bound_enforce=True)
    
    #add driver and objective function
    prob.driver = pyOptSparseDriver(optimizer='IPOPT') 
    prob.driver.opt_settings['limited_memory_max_history']=1000
    prob.driver.opt_settings['tol'] = 1e-5
    prob.driver.opt_settings['constr_viol_tol'] = 1e-6
    prob.model.add_design_var('ac|geom|wing|S_ref', lower = 10, upper=30, units='m**2')
    prob.model.add_design_var('ac|propulsion|engine|rating', lower = 500, upper=1500, units='hp', ref=800)
    prob.model.add_design_var('ac|geom|wing|c4sweep', lower = -2, upper=15, units='deg', ref=1)
    prob.model.add_design_var('ac|geom|wing|twist', lower = np.array([0,-5,-5,-5,-5]), upper=np.array([0,5,5,5,5]), units='deg', ref=1)
    prob.model.add_design_var('ac|geom|wing|AR', lower = 8, upper=20, ref=8.95)
    prob.model.add_design_var("ac|geom|wing|taper", lower=0, upper=1, ref=1)
    prob.model.add_objective('descent.fuel_used_final')

    # sizing throttle constraints
    prob.model.add_constraint('climb.throttle', upper=1.0) # these constraints limit throttle
    prob.model.add_constraint('cruise.throttle', upper=1.0)
    prob.model.add_constraint('descent.throttle', upper=1.0)

    # sizing CL constraint (CL<CL_max of wing)
    # prob.model.add_constraint('climb.fltcond|CL', upper=2.4476)
    prob.model.add_constraint('climb.fltcond|CL', upper=1.5) # climb is the only active constraint, need a stall condition on wing, some fxn(climb.fltcond|vs, climb.fltcond|Ueas), can we define max climb rate, stall depends on airfoil selection
    prob.model.add_constraint('cruise.fltcond|CL', upper=0.7)
    prob.model.add_constraint('descent.fltcond|CL', upper=1.4)
    prob.model.add_constraint('rotate.range_final', upper=1000)
    prob.model.add_constraint('v1v0.range_final', upper=1000)
    prob.driver.options['debug_print'] = ['desvars','objs','nl_cons']
    # prob.driver.options['tol'] = 1e-06

    prob.setup(check=True, mode='fwd')

    # set some (required) mission parameters. Each pahse needs a vertical and air-speed
    # the entire mission needs a cruise altitude and range
    prob.set_val('climb.fltcond|vs', np.ones((num_nodes,))*1500, units='ft/min')
    prob.set_val('climb.fltcond|Ueas', np.ones((num_nodes,))*124, units='kn')
    prob.set_val('cruise.fltcond|vs', np.ones((num_nodes,))*0.01, units='ft/min')
    prob.set_val('cruise.fltcond|Ueas', np.ones((num_nodes,))*201, units='kn')
    prob.set_val('descent.fltcond|vs', np.ones((num_nodes,))*(-600), units='ft/min')
    prob.set_val('descent.fltcond|Ueas', np.ones((num_nodes,))*140, units='kn')
    prob.set_val('reserve_climb.fltcond|vs', np.ones((num_nodes,))*1500, units='ft/min')
    prob.set_val('reserve_climb.fltcond|Ueas', np.ones((num_nodes,))*124, units='kn')
    prob.set_val('reserve_cruise.fltcond|vs', np.ones((num_nodes,))*0.01, units='ft/min')
    prob.set_val('reserve_cruise.fltcond|Ueas', np.ones((num_nodes,))*201, units='kn')
    prob.set_val('reserve_descent.fltcond|vs', np.ones((num_nodes,))*(-600), units='ft/min')
    prob.set_val('reserve_descent.fltcond|Ueas', np.ones((num_nodes,))*140, units='kn')
    prob.set_val('loiter.fltcond|vs', np.linspace(0.0, 0.0, num_nodes), units='ft/min')
    prob.set_val('loiter.fltcond|Ueas', np.ones((num_nodes,)) * 100, units='kn')

    prob.set_val('cruise|h0',28000.,units='ft')
    prob.set_val('mission_range',1250,units='NM')

    # (optional) guesses for takeoff speeds may help with convergence
    prob.set_val('v0v1.fltcond|Utrue',np.ones((num_nodes))*50,units='kn')
    prob.set_val('v1vr.fltcond|Utrue',np.ones((num_nodes))*85,units='kn')
    prob.set_val('v1v0.fltcond|Utrue',np.ones((num_nodes))*85,units='kn')

    # set some airplane-specific values. The throttle edits are to derate the takeoff power of the PT6A
    prob['OEW.structural_fudge'] = 1.67
    prob['v0v1.throttle'] = np.ones((num_nodes)) / 1.21
    prob['v1vr.throttle'] = np.ones((num_nodes)) / 1.21
    prob['rotate.throttle'] = np.ones((num_nodes)) / 1.21

    prob.run_driver()
    # om.n2(prob,outfile = 'full_mission_sizing.html')
    return prob

if __name__ == "__main__":
    from openconcept.utilities.visualization import plot_trajectory
    # run the analysis
    prob = run_tbm_analysis()

     # print some outputs
    vars_list = ['ac|weights|MTOW','loiter.fuel_used_final']
    units = ['lb','lb']
    nice_print_names = ['MTOW', 'Fuel used']
    print("=======================================================================")
    for i, thing in enumerate(vars_list):
        print(nice_print_names[i]+': '+str(prob.get_val(thing,units=units[i])[0])+' '+units[i])

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
