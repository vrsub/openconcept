from __future__ import division
from multiprocessing.context import ForkContext
import numpy as np
from openmdao.api import ExplicitComponent, IndepVarComp
from openmdao.api import Group
from openconcept.utilities.math import AddSubtractComp, ElementMultiplyDivideComp
import math

class WingWeight_JetTrasport(ExplicitComponent):
    """Inputs: MTOW, ac|geom|wing|S_ref, ac|geom|wing|AR, ac|geom|wing|c4sweep, ac|geom|wing|taper, ac|geom|wing|toverc, V_H (max SL speed)
    Outputs: W_wing
    Metadata: n_ult (ult load factor)

    """
    def initialize(self):
        #self.options.declare('num_nodes', default=1, desc='Number of flight/control conditions')
        #define configuration parameters
        self.options.declare('n_ult', default=3.8*1.5, desc='Ultimate load factor (dimensionless)')


    def setup(self):
        #nn = self.options['num_nodes']
        self.add_input('ac|weights|MTOW', units='lb', desc='Maximum rated takeoff weight')
        self.add_input('ac|weights|W_fuel_max', units='lb', desc='Fuel weight')
        self.add_input('ac|geom|wing|S_ref', units='ft**2', desc='Reference wing area in sq ft')
        self.add_input('ac|geom|wing|AR', desc='Wing aspect ratio')
        self.add_input('ac|geom|wing|c4sweep', units='rad', desc='Quarter-chord sweep angle')
        self.add_input('ac|geom|wing|taper', desc='Wing taper ratio')
        self.add_input('ac|geom|wing|toverc', desc='Wing max thickness to chord ratio')

        self.add_output('W_wing', units='lb', desc='Wing weight')
        self.declare_partials(['W_wing'], ['*'])

    def compute(self, inputs, outputs):
        n_ult = self.options['n_ult']
        #USAF method, Roskam PVC5pg68eq5.4
        #W_wing_USAF = 96.948*((inputs['ac|weights|MTOW']*n_ult/1e5)**0.65 * (inputs['ac|geom|wing|AR']/math.cos(inputs['ac|geom|wing|c4sweep']))**0.57 * (inputs['ac|geom|wing|S_ref']/100)**0.61 * ((1+inputs['ac|geom|wing|taper'])/2/inputs['ac|geom|wing|toverc'])**0.36 * (1+inputs['V_H']/500)**0.5)**0.993
        #Torenbeek, Roskam PVC5p68eq5.5
        #b = math.sqrt(inputs['ac|geom|wing|S_ref']*inputs['ac|geom|wing|AR'])
        #root_chord = 2*inputs['ac|geom|wing|S_ref']/b/(1+inputs['ac|geom|wing|taper'])
        #tr = root_chord * inputs['ac|geom|wing|toverc']
        #c2sweep_wing = inputs['ac|geom|wing|c4sweep'] # a hack for now
        #W_wing_Torenbeek = 0.00125*inputs['ac|weights|MTOW'] * (b/math.cos(c2sweep_wing))**0.75 * (1+ (6.3*math.cos(c2sweep_wing)/b)**0.5) * n_ult**0.55 * (b*inputs['ac|geom|wing|S_ref']/tr/inputs['ac|weights|MTOW']/math.cos(c2sweep_wing))**0.30

        W_wing_Raymer = 0.0051*(inputs['ac|weights|MTOW']*n_ult)**0.557 * (inputs['ac|geom|wing|S_ref'])**0.649*(inputs['ac|geom|wing|AR'])**0.5*(inputs['ac|geom|wing|toverc'])**-0.4*(1+inputs['ac|geom|wing|taper'])**0.1*np.cos(inputs['ac|geom|wing|c4sweep'])**-1*(0.12*inputs['ac|geom|wing|S_ref'])**0.1

        outputs['W_wing'] = W_wing_Raymer

    def compute_partials(self, inputs, J): # TO DO
        n_ult = self.options['n_ult']
        J['W_wing','ac|weights|MTOW'] = (0.0023*inputs['ac|geom|wing|AR']**(1/2)*n_ult*inputs['ac|geom|wing|S_ref']**0.7490*(inputs['ac|geom|wing|taper'] + 1)**0.1000)/(inputs['ac|geom|wing|toverc']**0.4000*np.cos(inputs['ac|geom|wing|c4sweep'])*(inputs['ac|weights|MTOW']*n_ult)**0.4430)
        J['W_wing','ac|geom|wing|S_ref'] = (0.0031*inputs['ac|geom|wing|AR']**(1/2)*(inputs['ac|weights|MTOW']*n_ult)**0.5570*(inputs['ac|geom|wing|taper'] + 1)**0.1000)/(inputs['ac|geom|wing|S_ref']**0.2510*inputs['ac|geom|wing|toverc']**0.4000*np.cos(inputs['ac|geom|wing|c4sweep']))
        J['W_wing','ac|geom|wing|AR'] = (0.0021*inputs['ac|geom|wing|S_ref']**0.7490*(inputs['ac|weights|MTOW']*n_ult)**0.5570*(inputs['ac|geom|wing|taper'] + 1)**0.1000)/(inputs['ac|geom|wing|AR']**0.5000*inputs['ac|geom|wing|toverc']**0.4000*np.cos(inputs['ac|geom|wing|c4sweep']))
        J['W_wing','ac|geom|wing|c4sweep'] = -(0.0041*inputs['ac|geom|wing|AR']**(1/2)*inputs['ac|geom|wing|S_ref']**0.7490*np.sin(inputs['ac|geom|wing|c4sweep'])*(inputs['ac|weights|MTOW']*nult)**0.5570*(inputs['ac|geom|wing|taper'] + 1)**0.1000)/(inputs['ac|geom|wing|toverc']**0.4000*(np.sin(inputs['ac|geom|wing|c4sweep'])**2 - 1))
        J['W_wing','ac|geom|wing|taper'] = (4.1256e-04*inputs['ac|geom|wing|AR']**(1/2)*inputs['ac|geom|wing|S_ref']**0.7490*(inputs['ac|weights|MTOW']*nult)**0.5570)/(inputs['ac|geom|wing|toverc']**0.4000*cos(inputs['ac|geom|wing|c4sweep'])*(inputs['ac|geom|wing|taper'] + 1)**0.9000)
        J['W_wing','ac|geom|wing|toverc'] = -(0.0017*inputs['ac|geom|wing|AR']**(1/2)*inputs['ac|geom|wing|S_ref']**0.7490*(inputs['ac|weights|MTOW']*nult)**0.5570*(inputs['ac|geom|wing|taper'] + 1)**0.1000)/(inputs['ac|geom|wing|toverc']**1.4000*np.cos(inputs['ac|geom|wing|c4sweep']))

class HstabWeight_JetTransport(ExplicitComponent):
    
    def initialize(self):
        self.options.declare('K_uht', default=1.143, desc='Scaling for all moving stabilizer')
        self.options.declare('n_ult', default=3.8*1.5, desc='Ultimate load factor (dimensionless)')

    def setup(self):
        self.add_input('ac|weights|MTOW', units='lb', desc='Maximum rated takeoff weight')
        self.add_input('ac|geom|hstab|S_ref', units='ft**2', desc='Reference wing area in sq ft')
        self.add_input('ac|geom|hstab|AR', desc='Wing aspect ratio')
        self.add_input('ac|geom|hstab|c4sweep', units='rad', desc='Quarter-chord sweep angle')
        self.add_input('ac|geom|hstab|c4_to_wing_c4', desc='Tail quarter-chord to wing quarter chord distance')
        
        self.add_output('W_hstab', units='lb', desc='Hstab weight')
        self.declare_partials(['W_hstab'],['*'])

    def compute(self, inputs, outputs):
        n_ult = self.options['n_ult']
        K_uht = self.options['K_uht']

        W_hstab_raymer = 0.0379*K_uht*(1+(0.6*13)/np.sqrt(inputs['ac|geom|hstab|AR']*inputs['ac|geom|hstab|S_ref']))**-0.25*inputs['ac|weights|MTOW']**0.639*n_ult**0.10*inputs['ac|geom|hstab|S_ref']**0.75*inputs['ac|geom|hstab|c4_to_wing_c4']**-1*(0.3*inputs['ac|geom|hstab|c4_to_wing_c4'])**0.704*np.cos(inputs['ac|geom|hstab|c4sweep'])**-1*inputs['ac|geom|hstab|AR']**0.166*(1+(0.2*inputs['ac|geom|hstab|S_ref'])/inputs['ac|geom|hstab|S_ref'])**0.1
        outputs['W_hstab'] = W_hstab_raymer


    def compute_partials(self, inputs, J):
        n_ult = self.options['n_ult']
        K_uht = self.options['K_uht']

        J['W_hstab','ac|weights|MTOW'] = (0.0247*K_uht*inputs['ac|geom|hstab|AR']**0.1660*n_ult**0.1000*inputs['ac|geom|hstab|S_ref']**0.7500*(0.3000*inputs['ac|geom|hstab|c4_to_wing_c4'])**0.7040)/(inputs['ac|geom|hstab|c4_to_wing_c4']**inputs['ac|weights|MTOW']**0.3610*np.cos(inputs['ac|geom|hstab|c4sweep'])**(7.8000/(inputs['ac|geom|hstab|AR']*inputs['ac|geom|hstab|S_ref'])**0.5000 + 1)**0.2500)
        J['W_hstab','ac|geom|hstab|S_ref'] = (0.0093*K_uht*inputs['ac|geom|hstab|AR']**0.1660*inputs['ac|weights|MTOW']**0.6390*n_ult**0.1000*(91*inputs['ac|geom|hstab|AR']*inputs['ac|geom|hstab|S_ref'] + 10*(inputs['ac|geom|hstab|AR']*inputs['ac|geom|hstab|S_ref'])**1.5000))/(inputs['ac|geom|hstab|c4_to_wing_c4']**0.2960*inputs['ac|geom|hstab|S_ref']**0.2500*np.cos(inputs['ac|geom|hstab|c4sweep'])*(inputs['ac|geom|hstab|AR']*inputs['ac|geom|hstab|S_ref'])**1.5000*((5*(inputs['ac|geom|hstab|AR']*inputs['ac|geom|hstab|S_ref'])**(1/2) + 39)/(inputs['ac|geom|hstab|AR']*inputs['ac|geom|hstab|S_ref'])**0.5000)**1.2500)
        J['W_hstab','ac|geom|hstab|AR'] = (2.0000e-08*(1.8816e+06*K_uht*inputs['ac|geom|hstab|AR']*inputs['ac|weights|MTOW']**0.6390*n_ult**0.1000*inputs['ac|geom|hstab|S_ref']**1.7500*(0.3000*inputs['ac|geom|hstab|c4_to_wing_c4'])**0.7040 + 3.2036e+05*K_uht*inputs['ac|weights|MTOW']**0.6390*n_ult**0.1000*inputs['ac|geom|hstab|S_ref']**0.7500*(0.3000*inputs['ac|geom|hstab|c4_to_wing_c4'])**0.7040*(7.8000/(inputs['ac|geom|hstab|AR']*inputs['ac|geom|hstab|S_ref'])**0.5000 + 1)*(inputs['ac|geom|hstab|AR']*inputs['ac|geom|hstab|S_ref'])**1.5000))/(inputs['ac|geom|hstab|AR']**0.8340*inputs['ac|geom|hstab|c4_to_wing_c4']*np.cos(inputs['ac|geom|hstab|c4sweep'])*(7.8000/(inputs['ac|geom|hstab|AR']*inputs['ac|geom|hstab|S_ref'])**0.5000 + 1)**1.2500*(inputs['ac|geom|hstab|AR']*inputs['ac|geom|hstab|S_ref'])**1.5000)
        J['W_hstab','ac|geom|hstab|c4sweep'] = -(0.0386*K_uht*inputs['ac|geom|hstab|AR']**0.1660*inputs['ac|weights|MTOW']**0.6390*n_ult**0.1000*inputs['ac|geom|hstab|S_ref']**0.7500*np.sin(inputs['ac|geom|hstab|c4sweep'])*(0.3000*inputs['ac|geom|hstab|c4_to_wing_c4'])**0.7040)/(inputs['ac|geom|hstab|c4_to_wing_c4']*(7.8000/(inputs['ac|geom|hstab|AR']*inputs['ac|geom|hstab|S_ref'])**0.5000 + 1)**0.2500*(np.sin(inputs['ac|geom|hstab|c4sweep'])**2 - 1))
        J['W_hstab','ac|geom|hstab|c4_to_wing_c4'] = -(0.0034*K_uht*inputs['ac|geom|hstab|AR']**0.1660*inputs['ac|weights|MTOW']**0.6390*n_ult**0.1000*inputs['ac|geom|hstab|S_ref']**0.7500)/(inputs['ac|geom|hstab|c4_to_wing_c4']*np.cos(inputs['ac|geom|hstab|c4sweep'])*(0.3000*inputs['ac|geom|hstab|c4_to_wing_c4'])**0.2960*(7.8000/(inputs['ac|geom|hstab|AR']*inputs['ac|geom|hstab|S_ref'])**0.5000 + 1)**0.2500)

class VstabWeight_JetTransport(ExplicitComponent):
    
    def initialize(self):
        self.options.declare('n_ult', default=3.8*1.5, desc='Ultimate load factor (dimensionless)')

    def setup(self):
        self.add_input('ac|weights|MTOW', units='lb', desc='Maximum rated takeoff weight')
        self.add_input('ac|geom|vstab|S_ref', units='ft**2', desc='Reference vtail area in sq ft')
        self.add_input('ac|geom|vstab|AR', desc='vtail aspect ratio')
        self.add_input('ac|geom|vstab|c4sweep', units='rad', desc='Quarter-chord sweep angle')
        self.add_input('ac|geom|hstab|c4_to_wing_c4', desc='Tail quarter-chord to wing quarter chord distance')
        self.add_input('ac|geom|vstab|toverc', desc='root t/c of v-tail, estimated same as wing')

        self.add_output('W_vstab', units='lb', desc='Vstab weight')
        self.declare_partials(['W_hstab'],['*'])

    def compute(self, inputs, outputs):
        n_ult = self.options['n_ult']

        W_vstab_raymer = 0.0026*inputs['ac|weights|MTOW']**0.556*n_ult**0.536*inputs['ac|geom|hstab|c4_to_wing_c4']**(-0.5+0.875)*inputs['ac|geom|vstab|S_ref']**0.5*np.cos(inputs['ac|geom|vstab|c4sweep'])**-1*inputs['ac|geom|vstab|AR']**0.35*inputs['ac|geom|vstab|toverc']**-0.5
        outputs['W_vstab'] = W_vstab_raymer

    def compute_partials(self, inputs, J):
        n_ult = self.options['n_ult']

        J['W_vstab','ac|weights|MTOW'] = (0.0014*inputs['ac|geom|vstab|AR']**0.3500*inputs['ac|geom|hstab|c4_to_wing_c4']**0.3750*n_ult**0.5360*inputs['ac|geom|vstab|S_ref']**(1/2))/(inputs['ac|weights|MTOW']**0.4440*inputs['ac|geom|vstab|toverc']**0.5000*np.cos(inputs['ac|geom|vstab|c4sweep']))
        J['W_vstab','ac|geom|vstab|S_ref'] = (0.0013*inputs['ac|geom|vstab|AR']**0.3500*inputs['ac|geom|hstab|c4_to_wing_c4']**0.3750*inputs['ac|weights|MTOW']**0.5560*n_ult**0.5360)/(inputs['ac|geom|vstab|S_ref']**0.5000*inputs['ac|geom|vstab|toverc']**0.5000*np.cos(inputs['ac|geom|vstab|c4sweep']))
        J['W_vstab','ac|geom|vstab|AR'] = (9.1000e-04*inputs['ac|geom|hstab|c4_to_wing_c4']**0.3750*inputs['ac|weights|MTOW']**0.5560*n_ult**0.5360*inputs['ac|geom|vstab|S_ref']**(1/2))/(inputs['ac|geom|vstab|AR']**0.6500*inputs['ac|geom|vstab|toverc']**0.5000*np.cos(inputs['ac|geom|vstab|c4sweep']))
        J['W_vstab','ac|geom|vstab|c4sweep'] = -(0.0026*inputs['ac|geom|vstab|AR']**0.3500*inputs['ac|geom|hstab|c4_to_wing_c4']**0.3750*inputs['ac|weights|MTOW']**0.5560*n_ult**0.5360*inputs['ac|geom|vstab|S_ref']**(1/2)*np.sin(inputs['ac|geom|vstab|c4sweep']))/(inputs['ac|geom|vstab|toverc']**0.5000*(np.sin(inputs['ac|geom|vstab|c4sweep'])**2 - 1))
        J['W_vstab','ac|geom|hstab|c4_to_wing_c4'] = (9.7500e-04*inputs['ac|geom|vstab|AR']**0.3500*inputs['ac|weights|MTOW']**0.5560*n_ult**0.5360*inputs['ac|geom|vstab|S_ref']**(1/2))/(inputs['ac|geom|hstab|c4_to_wing_c4']**0.6250*inputs['ac|geom|vstab|toverc']**0.5000*np.cos(inputs['ac|geom|vstab|c4sweep']))
        J['W_vstab','ac|geom|vstab|toverc'] = -(0.0013*inputs['ac|geom|vstab|AR']**0.3500*inputs['ac|geom|hstab|c4_to_wing_c4']**0.3750*inputs['ac|weights|MTOW']**0.5560*n_ult**0.5360*inputs['ac|geom|vstab|S_ref']**(1/2))/(inputs['ac|geom|vstab|toverc']**1.5000*np.cos(inputs['ac|geom|vstab|c4sweep']))

class FuselageWeight_JetTransport(ExplicitComponent):
    
    def initialize(self):
    def setup(self):
    def compute(self):
    def compute_partials(self):

class MainLandingGear_JetTransport(ExplicitComponent):
    
    def initialize(self):
    def setup(self):
    def compute(self):
    def compute_partials(self):

class NoseLandingGear_JetTransport(ExplicitComponent):
    
    def initialize(self):
    def setup(self):
    def compute(self):
    def compute_partials(self):

class Nacelle_JetTransport(ExplicitComponent):
    
    def initialize(self):
    def setup(self):
    def compute(self):
    def compute_partials(self):

class FuelSystem_JetTransport(ExplicitComponent):
    
    def initialize(self):
    def setup(self):
    def compute(self):
    def compute_partials(self):

class Equipment_JetTransport(ExplicitComponent):
    
    def initialize(self):
    def setup(self):
    def compute(self):
    def compute_partials(self):

class JetTransportEmptyWeight(Group):

    def setup(self):
        const = self.add_subsystem('const',IndepVarComp(),promotes_outputs=["*"])
        const.add_output('W_fluids', val=20, units='kg')
        const.add_output('structural_fudge', val=1.6, units='m/m')
        self.add_subsystem('wing',WingWeight_JetTrasport(),promotes_inputs=["*"],promotes_outputs=["*"])
        self.add_subsystem('hstab',HstabWeight_JetTransport(),promotes_inputs=["*"],promotes_outputs=["*"])
        self.add_subsystem('vstab',VstabWeight_JetTransport(),promotes_inputs=["*"],promotes_outputs=["*"])
        self.add_subsystem('fuselage',FuselageWeight_JetTransport(),promotes_inputs=["*"],promotes_outputs=["*"])
        self.add_subsystem('nacelle',Nacelle_JetTransport(),promotes_inputs=["*"],promotes_outputs=["*"])
        self.add_subsystem('mlg', MainLandingGear_JetTransport(), promotes_inputs=["*"],promotes_outputs=["*"])
        self.add_subsystem('flg',NoseLandingGear_JetTransport(), promotes_inputs=["*"],promotes_outputs=["*"])
        self.add_subsystem('structural',AddSubtractComp(output_name='W_structure',input_names=['W_wing','W_hstab','W_vstab','W_fuselage','W_mlg','W_mlg'], units='lb'),promotes_outputs=['*'],promotes_inputs=["*"])
        self.add_subsystem('structural_fudge',ElementMultiplyDivideComp(output_name='W_structure_adjusted',input_names=['W_structure','structural_fudge'],input_units=['lb','m/m']),promotes_inputs=["*"],promotes_outputs=["*"])
        self.add_subsystem('totalempty',AddSubtractComp(output_name='OEW',input_names=['W_structure_adjusted','W_fuelsystem','W_equipment','W_engine','W_propeller','W_fluids'], units='lb'),promotes_outputs=['*'],promotes_inputs=["*"])