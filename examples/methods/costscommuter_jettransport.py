from __future__ import division
from multiprocessing.context import ForkContext
from xml.dom.minidom import Element
from matplotlib import units
import numpy as np
from openmdao.api import ExplicitComponent, IndepVarComp
from openmdao.api import Group
from openconcept.utilities.math import AddSubtractComp, ElementMultiplyDivideComp


class AircraftCost_JetTransport(ExplicitComponent):
    def initialize(self):
        self.options.declare("CEF", default=1.73, desc="Cost Escalation Factor for aircraft price in 2022")

    def setup(self):
        self.add_input("ac|weights|MTOW", units="lb", desc="Aircraft MTOW")
        self.add_output("C_aircraft", desc="aircraft cost estimate with engines")
        self.declare_partials(["C_aircraft"], ["*"])

    def compute(self, inputs, outputs):
        CEF = self.options["CEF"]
        Caircraft_raymer = 10 ** (3.3191) * 10 ** (0.8043 * np.log10(inputs["ac|weights|MTOW"])) * CEF
        outputs["C_aircraft"] = Caircraft_raymer

    def compute_partials(self, inputs, J):
        CEF = self.options["CEF"]
        J["C_aircraft", "ac|weights|MTOW"] = CEF * 10 ** 3.3191 * 0.8043 * inputs["ac|weights|MTOW"] ** (0.8043 - 1)


class EngineCost(ExplicitComponent):
    def initialize(self):
        self.options.declare("CEF", default=1.73, desc="Cost Escalation Factor for aircraft price in 2022")

    def setup(self):
        self.add_input("ac|weights|MTOW", units="lb", desc="Aircraft MTOW")
        self.add_output("C_engine", desc="aircraft cost estimate with engines")
        self.declare_partials(["C_engine"], ["*"])

    def compute(self, inputs, outputs):
        CEF = self.options["CEF"]
        Cengine_raymer = 10 ** (2.3044) * 10 ** (0.8858 * np.log10(inputs["ac|weights|MTOW"])) * CEF
        outputs["C_engine"] = Cengine_raymer

    def compute_partials(self, inputs, J):
        CEF = self.options["CEF"]
        J["C_engine", "ac|weights|MTOW"] = CEF * 10 ** 2.3044 * 0.8858 * inputs["ac|weights|MTOW"] ** (0.8858 - 1)


class CrewCost(ExplicitComponent):
    def initialize(self):
        self.options.declare("AF", default=0.80, desc="Average airline factor")
        self.options.declare(
            "K", default=5.5, desc="Route Factor, dependent on route, default set to domestic with 2 crew"
        )
        self.options.declare("CEF", default=1.73, desc="Cost Escalation Factor for aircraft price in 2022")

    def setup(self):
        self.add_input("ac|weights|MTOW", units="lb", desc="Aircraft MTOW")
        self.add_input("block_time", units="3600*s", desc="Mission block time")
        self.add_output("C_crew", desc="Crew cost in dollars")
        self.declare_partials(["C_crew"], ["*"])

    def compute(self, inputs, outputs):
        AF = self.options["AF"]
        K = self.options["K"]
        CEF = self.options["CEF"]
        Ccrew_harris = AF * K * CEF * inputs["ac|weights|MTOW"] ** 0.4 * inputs["block_time"]
        outputs["C_crew"] = Ccrew_harris

    def compute_partials(self, inputs, J):
        AF = self.options["AF"]
        K = self.options["K"]
        CEF = self.options["CEF"]
        J["C_crew", "ac|weights|MTOW"] = (
            AF * K * CEF * 0.4 * inputs["ac|weights|MTOW"] ** (0.4 - 1) * inputs["block_time"]
        )
        J["C_crew", "block_time"] = AF * K * CEF * inputs["ac|weights|MTOW"] ** 0.4


class AttendantCost(ExplicitComponent):
    def initialize(self):
        self.options.declare("n_attendants", default=4, desc="4 attendants per flight")
        self.options.declare(
            "route_factor", default=60, desc="Route Factor, default 60 for domestic, 78 if international"
        )
        self.options.declare("CEF", default=1.73, desc="Cost Escalation Factor for aircraft price in 2022")

    def setup(self):
        self.add_input("block_time", units="3600*s", desc="Mission block time")
        self.add_output("C_attendants", desc="Crew cost in dollars")
        self.declare_partials(["C_attendants"], ["*"])

    def compute(self, inputs, outputs):
        n_attendants = self.options["n_attendants"]
        route_factor = self.options["route_factor"]
        CEF = self.options["CEF"]
        Cattendants_liebeck = route_factor * n_attendants * CEF * inputs["block_time"]
        outputs["C_attendants"] = Cattendants_liebeck

    def compute_partials(self, inputs, J):
        n_attendants = self.options["n_attendants"]
        route_factor = self.options["route_factor"]
        CEF = self.options["CEF"]
        J["C_attendants", "block_time"] = route_factor * n_attendants * CEF


class FuelCost(ExplicitComponent):
    def initialize(self):
        self.options.declare("fuel_cost", default=3.30, desc="Fuel cost per gallon, default $3.30")
        self.options.declare("fuel_rho", default=6.71)

    def setup(self):
        self.add_input("ac|weights|W_fuel_max", units="lb")
        self.add_output("C_fuel")
        self.declare_partials(["C_fuel"], ["*"])

    def compute(self, inputs, outputs):
        Pf = self.options["fuel_cost"]
        rhof = self.options["fuel_rho"]
        Cfuel_kroo = 1.02 * Pf / rhof * inputs["ac|weights|W_fuel_max"]
        outputs["C_fuel"] = Cfuel_kroo

    def compute_partials(self, inputs, J):
        Pf = self.options["fuel_cost"]
        rhof = self.options["fuel_rho"]
        J["C_fuel", "ac|weights|W_fuel_max"] = 1.02 * Pf / rhof


class OilCost(ExplicitComponent):
    def initialize(self):
        self.options.declare("oil_cost", default=90, desc="Fuel cost per gallon, default $3.30")
        self.options.declare("oil_rho", default=8)

    def setup(self):
        self.add_input("ac|weights|W_fuel_max", units="lb")
        self.add_input("block_time", units="3600*s", desc="Mission block time")
        self.add_output("C_oil")
        self.declare_partials(["C_oil"], ["*"])

    def compute(self, inputs, outputs):
        Po = self.options["oil_cost"]
        rho_o = self.options["oil_rho"]
        Coil = 1.02 * Po / rho_o * 0.0125 * inputs["ac|weights|W_fuel_max"] * 0.01 * inputs["block_time"]
        outputs["C_oil"] = Coil

    def compute_partials(self, inputs, J):
        Po = self.options["oil_cost"]
        rho_o = self.options["oil_rho"]
        J["C_oil", "ac|weights|W_fuel_max"] = 1.02 * Po / rho_o * 0.0125 * 0.01 * inputs["block_time"]
        J["C_oil", "block_time"] = 1.02 * Po / rho_o * 0.0125 * inputs["ac|weights|W_fuel_max"] * 0.01


class LandingCost(ExplicitComponent):
    def initialize(self):
        self.options.declare(
            "route_factor", default=1.5, desc="Route Factor, default 1.5 for domestic, 4.25 if international"
        )
        self.options.declare("CEF", default=1.73, desc="Cost Escalation Factor for aircraft price in 2022")

    def setup(self):
        self.add_input("ac|weights|MTOW", units="lb", desc="max takeoff weight")
        self.add_output("C_landing", desc="landing cost in dollars")
        self.declare_partials(["C_landing"], ["*"])

    def compute(self, inputs, outputs):
        route_factor = self.options["route_factor"]
        CEF = self.options["CEF"]
        Clanding = route_factor * CEF * 0.001 * inputs["ac|weights|MTOW"]
        outputs["C_landing"] = Clanding

    def compute_partials(self, inputs, J):
        route_factor = self.options["route_factor"]
        CEF = self.options["CEF"]
        J["C_landing", "ac|weights|MTOW"] = route_factor * CEF * 0.001


class NavigationCost(ExplicitComponent):
    def initialize(self):
        self.options.declare("CEF", default=1.73, desc="Cost Escalation Factor for aircraft price in 2022")

    def setup(self):
        self.add_input("ac|weights|MTOW", units="lb", desc="max takeoff weight")
        self.add_input("mission_range", units="NM", desc="Main mission range")
        self.add_input("block_time", units="3600*s", desc="Main mission block time")
        self.add_output("C_nav", desc="navigation cost in dollars")
        self.declare_partials(["C_nav"], ["*"])

    def compute(self, inputs, outputs):
        CEF = self.options["CEF"]
        Cnav = (
            0.5
            * CEF
            * 1.852
            * inputs["mission_range"]
            * inputs["block_time"] ** -1
            * (0.00045359237 / 50) ** 0.5
            * inputs["ac|weights|MTOW"] ** 0.5
        )
        outputs["C_nav"] = Cnav

    def compute_partials(self, inputs, J):
        CEF = self.options["CEF"]
        J["C_nav", "ac|weights|MTOW"] = (
            0.5
            * CEF
            * 1.852
            * inputs["mission_range"]
            * inputs["block_time"] ** -1
            * (0.00045359237 / 50) ** 0.5
            * 0.5
            * inputs["ac|weights|MTOW"] ** (0.5 - 1)
        )
        J["C_nav", "mission_range"] = (
            0.5
            * CEF
            * 1.852
            * inputs["block_time"] ** -1
            * (0.00045359237 / 50) ** 0.5
            * inputs["ac|weights|MTOW"] ** 0.5
        )
        J["C_nav", "block_time"] = (
            0.5
            * CEF
            * 1.852
            * inputs["mission_range"]
            * -1
            * inputs["block_time"] ** (-1 - 1)
            * (0.00045359237 / 50) ** 0.5
            * inputs["ac|weights|MTOW"] ** 0.5
        )


class AirframeMaintananceLaborCost(ExplicitComponent):
    def initialize(self):
        self.options.declare("RL", default=35, desc="aintanance labor rate in USD/hr")

    def setup(self):
        self.add_input("W_structure_adjusted", units="lb", desc="Airframe weight, Aircraft weight minus engine weight")
        self.add_output("C_ml")
        self.declare_partials(["C_ml"], ["W_structure_adjusted"])

    def compute(self, inputs, outputs):
        RL = self.options["RL"]
        outputs["C_ml"] = 1.03 ** RL * 3 + 1.03 * 0.067 * 0.001 * inputs["W_structure_adjusted"] * RL

    def compute_partials(self, inputs, J):
        RL = self.options["RL"]
        J["C_ml", "W_structure_adjusted"] = 1.03 * 0.067 * 0.001 * RL


class AirframeMaintananceMaterialCost(ExplicitComponent):
    def initialize(self):
        self.options.declare("CEF", default=1.73, desc="Cost escalation factor")

    def setup(self):
        self.add_input("C_airframe", desc="Airframe cost")
        self.add_output("C_mm", desc="Material maintanance cost")
        self.declare_partials(["C_mm"], ["*"])

    def compute(self, inputs, outputs):
        CEF = self.options["CEF"]
        outputs["C_mm"] = 1.03 * 30 * CEF + 0.79e-5 * inputs["C_airframe"]

    def compute_partials(self, inputs, J):
        CEF = self.options["CEF"]
        J["C_mm", "C_airframe"] = 0.79e-5


class JetEngineMaintananceLaborCost(ExplicitComponent):
    def initialize(self):
        self.options.declare("RL", default=35, desc="aintanance labor rate in USD/hr")

    def setup(self):
        self.add_input("ac|propulsion|engine|rating", units="lbf")
        self.add_input("block_time", units="3600*s")
        self.add_output("C_ml_engine", desc="engine maintanance labor cost")
        self.declare_partials(["C_ml_engine"], ["*"])

    def compute(self, inputs, outputs):
        RL = self.options["RL"]
        outputs["C_ml_engine"] = (
            RL
            * (0.645 + 0.05 * 10 ** -4 * inputs["ac|propulsion|engine|rating"])
            * (0.566 + 0.434 * inputs["block_time"] ** -1)
        )

    def compute_partials(self, inputs, J):
        RL = self.options["RL"]
        J["C_ml_engine", "ac|propulsion|engine|rating"] = (
            RL
            * (0.645 + 0.05 * 10 ** -4 * inputs["ac|propulsion|engine|rating"]) ** 0
            * 0.05
            * 10 ** -4
            * (0.566 + 0.434 * inputs["block_time"] ** -1)
        )
        J["C_ml_engine", "block_time"] = (
            RL
            * (0.645 + 0.05 * 10 ** -4 * inputs["ac|propulsion|engine|rating"])
            * (0.434 * -1 * inputs["block_time"] ** -2)
        )


class JetEngineMaintananceMaterialCost(ExplicitComponent):
    def initialize(self):
        self.options.declare("CEF", default=1.73, desc="cost escalation factor")

    def setup(self):
        self.add_input("ac|propulsion|engine|rating", units="lbf")
        self.add_input("block_time", units="3600*s")
        self.add_output("C_mm_engine", desc="engine maintanance material cost")
        self.declare_partials(["C_mm_engine"], ["*"])

    def compute(self, inputs, outputs):
        CEF = self.options["CEF"]
        outputs["C_mm_engine"] = (
            CEF
            * (25 + 18 * 10 ** -4 * inputs["ac|propulsion|engine|rating"])
            * (0.62 + 0.38 * inputs["block_time"] ** -1)
        )

    def compute_partials(self, inputs, J):
        CEF = self.options["CEF"]
        J["C_mm_engine", "ac|propulsion|engine|rating"] = (
            CEF
            * (25 + 18 * 10 ** -4 * inputs["ac|propulsion|engine|rating"]) ** 0
            * (18 * 10 ** -4)
            * (0.62 + 0.38 * inputs["block_time"] ** -1)
        )
        J["C_mm_engine", "block_time"] = (
            CEF
            * (25 + 18 * 10 ** -4 * inputs["ac|propulsion|engine|rating"])
            * (0.38 * -1 * inputs["block_time"] ** -2)
        )


class AircraftMaintananceCost(ExplicitComponent):
    def initialize(self):
        self.options.declare("n_engines", default=2, desc="number of engines")

    def setup(self):
        self.add_input("C_mm", desc="Airframe maintanance material cost")
        self.add_input("C_ml", desc="Airframe maintanance labor cost")
        self.add_input("C_mm_engine", desc="Single Engine maintanance material cost")
        self.add_input("C_ml_engine", desc="Single Engine maintanance labor cost")
        self.add_input("block_time", units="3600*s", desc="Mission block time")
        self.add_output("C_aircraft_maintanance", desc="Total aircraft maintanance cost for mission")
        self.declare_partials(["C_aircraft_maintanance"], ["*"])

    def compute(self, inputs, outputs):
        n_e = self.options["n_engines"]
        outputs["C_aircraft_maintanance"] = (
            inputs["C_ml"] * inputs["block_time"]
            + inputs["C_mm"] * inputs["block_time"]
            + n_e * inputs["C_ml_engine"] * inputs["block_time"]
            + n_e * inputs["C_mm_engine"] * inputs["block_time"]
        )

    def compute_partials(self, inputs, J):
        n_e = self.options["n_engines"]
        J["C_aircraft_maintanance", "C_mm"] = inputs["block_time"]
        J["C_aircraft_maintanance", "C_ml"] = inputs["block_time"]
        J["C_aircraft_maintanance", "C_ml_engine"] = inputs["block_time"] * n_e
        J["C_aircraft_maintanance", "C_mm_engine"] = inputs["block_time"] * n_e
        J["C_aircraft_maintanance", "block_time"] = (
            inputs["C_ml"] + inputs["C_mm"] + n_e * inputs["C_ml_engine"] + n_e * inputs["C_mm_engine"]
        )


class JetTransportCOC(Group):
    def setup(self):
        # const.add_output('W_fluids', val=20, units='kg')
        self.add_subsystem("aircraft_cost", AircraftCost_JetTransport(), promotes_inputs=["*"], promotes_outputs=["*"])
        self.add_subsystem("engine_cost", EngineCost(), promotes_inputs=["*"], promotes_outputs=["*"])
        self.add_subsystem("crew_cost", CrewCost(), promotes_inputs=["*"], promotes_outputs=["*"])
        self.add_subsystem("attendant_cost", AttendantCost(), promotes_inputs=["*"], promotes_outputs=["*"])
        self.add_subsystem("fuel_cost", FuelCost(), promotes_inputs=["*"], promotes_outputs=["*"])
        self.add_subsystem("oil_cost", OilCost(), promotes_inputs=["*"], promotes_outputs=["*"])
        self.add_subsystem("landing_cost", LandingCost(), promotes_inputs=["*"], promotes_outputs=["*"])
        self.add_subsystem("nav_cost", NavigationCost(), promotes_inputs=["*"], promotes_outputs=["*"])
        self.add_subsystem(
            "airfram_cost",
            AddSubtractComp(output_name="C_airframe", input_names=["C_aircraft", "C_engine"], scaling_factors=[1, -1]),
            promotes_outputs=["*"],
            promotes_inputs=["*"],
        )
        self.add_subsystem(
            "airframe_labor", AirframeMaintananceLaborCost(), promotes_inputs=["*"], promotes_outputs=["*"]
        )
        self.add_subsystem(
            "airframe_materials", AirframeMaintananceMaterialCost(), promotes_inputs=["*"], promotes_outputs=["*"]
        )
        self.add_subsystem(
            "engine_materials", JetEngineMaintananceMaterialCost(), promotes_inputs=["*"], promotes_outputs=["*"]
        )
        self.add_subsystem(
            "engine_labor", JetEngineMaintananceLaborCost(), promotes_inputs=["*"], promotes_outputs=["*"]
        )
        self.add_subsystem(
            "total_aircraft_maintanance", AircraftMaintananceCost(), promotes_inputs=["*"], promotes_outputs=["*"]
        )
        self.add_subsystem(
            "COC",
            AddSubtractComp(
                output_name="COC",
                input_names=[
                    "C_crew",
                    "C_attendants",
                    "C_fuel",
                    "C_oil",
                    "C_landing",
                    "C_nav",
                    "C_aircraft_maintanance",
                ],
            ),
            promotes_outputs=["*"],
            promotes_inputs=["*"],
        )


if __name__ == "__main__":
    from openmdao.api import IndepVarComp, Problem

    prob = Problem()
    prob.model = Group()
    dvs = prob.model.add_subsystem("dvs", IndepVarComp(), promotes_outputs=["*"])
    dvs.add_output("ac|weights|MTOW", 79002, units="kg")
    dvs.add_output("ac|weights|W_fuel_max", 18000, units="kg")
    dvs.add_output("mission_range", 2050, units="NM")
    dvs.add_output("block_time", 4, units="3600*s")
    dvs.add_output("W_structure_adjsuted", 87224, units="lb")
    dvs.add_output("ac|propulsion|engine|rating", 27000, units="lbf")

    prob.model.add_subsystem("COC", JetTransportCOC(), promotes_inputs=["*"])

    prob.setup(force_alloc_complex=True)
    prob.run_model()
    print("COC:")
    print(prob["COC.COC"])

    data = prob.check_partials(method="cs", compact_print=True, show_only_incorrect=False)
