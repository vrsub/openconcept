from __future__ import division
from matplotlib import units
import numpy as np
from openmdao.api import ExplicitComponent, IndepVarComp
from openmdao.api import Group
from openconcept.utilities.math import AddSubtractComp, ElementMultiplyDivideComp
import math


class HStabSizing_JetTransport(ExplicitComponent):
    """Inputs: ac|geom|wing|S_ref, ac|geom|wing|MAC, ac|geom|hstab|c4_to_wing_c4
    Outputs: hstab_area
    Metadata: C_ht, (volume coefficients for single engine general aviation aircraft)

    """

    def initialize(self):
        self.options.declare(
            "C_ht",
            default=1.0,
            desc="Tail colume coefficient for vertical stabilizer, nondimensionless, single engine airplanes",
        )

    def setup(self):
        self.add_input("ac|geom|wing|S_ref", units="ft**2", desc="Reference wing area in sq ft")
        self.add_input("ac|geom|wing|MAC", units="ft", desc="Wing mean aerodynamic chord")
        self.add_input(
            "ac|geom|hstab|c4_to_wing_c4",
            units="ft",
            desc="Distance from wing c/4 to horiz stab c/4 (tail arm distance)",
        )

        self.add_output("hstab_area", units="ft**2")
        self.declare_partials(["hstab_area"], ["*"])

    def compute(self, inputs, outputs):
        C_ht = self.options["C_ht"]
        hstabarea = (C_ht * inputs["ac|geom|wing|MAC"] * inputs["ac|geom|wing|S_ref"]) / inputs[
            "ac|geom|hstab|c4_to_wing_c4"
        ]
        outputs["hstab_area"] = hstabarea

    def compute_partials(self, inputs, J):
        C_ht = self.options["C_ht"]
        J["hstab_area", "ac|geom|wing|S_ref"] = (C_ht * inputs["ac|geom|wing|MAC"]) / inputs[
            "ac|geom|hstab|c4_to_wing_c4"
        ]
        J["hstab_area", "ac|geom|wing|MAC"] = (C_ht * inputs["ac|geom|wing|S_ref"]) / inputs[
            "ac|geom|hstab|c4_to_wing_c4"
        ]
        J["hstab_area", "ac|geom|hstab|c4_to_wing_c4"] = (
            -C_ht * inputs["ac|geom|wing|MAC"] * inputs["ac|geom|wing|S_ref"]
        ) / (inputs["ac|geom|hstab|c4_to_wing_c4"] ** 2)


class VStabSizing_JetTransport(ExplicitComponent):
    """Inputs: ac|geom|wing|S_ref, ac|geom|wing|AR, ac|geom|hstab|c4_to_wing_c4
    Outputs: vstab_area
    Metadata: C_vt (volume coefficients for single engine general aviation aircraft)

    """

    def initialize(self):
        self.options.declare(
            "C_vt",
            default=0.090,
            desc="Tail colume coefficient for vertical stabilizer, nondimensionless, single engine airplanes",
        )

    def setup(self):
        self.add_input("ac|geom|wing|S_ref", units="ft**2", desc="Reference wing area in sq ft")
        self.add_input("ac|geom|wing|AR", desc="Wing aspect ratio")
        self.add_input(
            "ac|geom|hstab|c4_to_wing_c4",
            units="ft",
            desc="Distance from wing c/4 to horiz stab c/4 (tail arm distance)",
        )

        self.add_output("vstab_area", units="ft**2")
        self.declare_partials(["vstab_area"], ["*"])

    def compute(self, inputs, outputs):
        C_vt = self.options["C_vt"]
        vstabarea = (
            C_vt
            * np.sqrt(inputs["ac|geom|wing|AR"])
            * (inputs["ac|geom|wing|S_ref"] ** (1.5))
            / inputs["ac|geom|hstab|c4_to_wing_c4"]
        )
        outputs["vstab_area"] = vstabarea

    def compute_partials(self, inputs, J):
        C_vt = self.options["C_vt"]
        J["vstab_area", "ac|geom|wing|S_ref"] = (
            1.5
            * (C_vt * np.sqrt(inputs["ac|geom|wing|AR"] * inputs["ac|geom|wing|S_ref"]))
            / inputs["ac|geom|hstab|c4_to_wing_c4"]
        )
        J["vstab_area", "ac|geom|wing|AR"] = 0.5 * (
            C_vt
            * (inputs["ac|geom|wing|AR"] ** (-0.5))
            * (inputs["ac|geom|wing|S_ref"] ** (1.5))
            / inputs["ac|geom|hstab|c4_to_wing_c4"]
        )
        J["vstab_area", "ac|geom|hstab|c4_to_wing_c4"] = (
            -C_vt * np.sqrt(inputs["ac|geom|wing|AR"]) * inputs["ac|geom|wing|S_ref"] ** (1.5)
        ) / (inputs["ac|geom|hstab|c4_to_wing_c4"] ** 2)


class HStabSizing_SmallTurboprop(ExplicitComponent):
    """Inputs: ac|geom|wing|S_ref, ac|geom|wing|MAC, ac|geom|hstab|c4_to_wing_c4
    Outputs: hstab_area
    Metadata: C_ht, (volume coefficients for single engine general aviation aircraft)

    """

    def initialize(self):
        self.options.declare(
            "C_ht",
            default=0.9,
            desc="Tail colume coefficient for vertical stabilizer, nondimensionless, single engine airplanes",
        )

    def setup(self):
        self.add_input("ac|geom|wing|S_ref", units="ft**2", desc="Reference wing area in sq ft")
        self.add_input("ac|geom|wing|MAC", units="ft", desc="Wing mean aerodynamic chord")
        self.add_input(
            "ac|geom|hstab|c4_to_wing_c4",
            units="ft",
            desc="Distance from wing c/4 to horiz stab c/4 (tail arm distance)",
        )

        self.add_output("hstab_area", units="ft**2")
        self.declare_partials(["hstab_area"], ["*"])

    def compute(self, inputs, outputs):
        C_ht = self.options["C_ht"]
        hstabarea = (C_ht * inputs["ac|geom|wing|MAC"] * inputs["ac|geom|wing|S_ref"]) / inputs[
            "ac|geom|hstab|c4_to_wing_c4"
        ]
        outputs["hstab_area"] = hstabarea

    def compute_partials(self, inputs, J):
        C_ht = self.options["C_ht"]
        J["hstab_area", "ac|geom|wing|S_ref"] = (C_ht * inputs["ac|geom|wing|MAC"]) / inputs[
            "ac|geom|hstab|c4_to_wing_c4"
        ]
        J["hstab_area", "ac|geom|wing|MAC"] = (C_ht * inputs["ac|geom|wing|S_ref"]) / inputs[
            "ac|geom|hstab|c4_to_wing_c4"
        ]
        J["hstab_area", "ac|geom|hstab|c4_to_wing_c4"] = (
            -C_ht * inputs["ac|geom|wing|MAC"] * inputs["ac|geom|wing|S_ref"]
        ) / (inputs["ac|geom|hstab|c4_to_wing_c4"] ** 2)


class VStabSizing_SmallTurboprop(ExplicitComponent):
    """Inputs: ac|geom|wing|S_ref, ac|geom|wing|AR, ac|geom|hstab|c4_to_wing_c4
    Outputs: vstab_area
    Metadata: C_vt (volume coefficients for single engine general aviation aircraft)

    """

    def initialize(self):
        self.options.declare(
            "C_vt",
            default=0.080,
            desc="Tail colume coefficient for vertical stabilizer, nondimensionless, single engine airplanes",
        )

    def setup(self):
        self.add_input("ac|geom|wing|S_ref", units="ft**2", desc="Reference wing area in sq ft")
        self.add_input("ac|geom|wing|AR", desc="Wing aspect ratio")
        self.add_input(
            "ac|geom|hstab|c4_to_wing_c4",
            units="ft",
            desc="Distance from wing c/4 to horiz stab c/4 (tail arm distance)",
        )

        self.add_output("vstab_area", units="ft**2")
        self.declare_partials(["vstab_area"], ["*"])

    def compute(self, inputs, outputs):
        C_vt = self.options["C_vt"]
        vstabarea = (
            C_vt
            * np.sqrt(inputs["ac|geom|wing|AR"])
            * (inputs["ac|geom|wing|S_ref"] ** (1.5))
            / inputs["ac|geom|hstab|c4_to_wing_c4"]
        )
        outputs["vstab_area"] = vstabarea

    def compute_partials(self, inputs, J):
        C_vt = self.options["C_vt"]
        J["vstab_area", "ac|geom|wing|S_ref"] = (
            1.5
            * (C_vt * np.sqrt(inputs["ac|geom|wing|AR"] * inputs["ac|geom|wing|S_ref"]))
            / inputs["ac|geom|hstab|c4_to_wing_c4"]
        )
        J["vstab_area", "ac|geom|wing|AR"] = 0.5 * (
            C_vt
            * (inputs["ac|geom|wing|AR"] ** (-0.5))
            * (inputs["ac|geom|wing|S_ref"] ** (1.5))
            / inputs["ac|geom|hstab|c4_to_wing_c4"]
        )
        J["vstab_area", "ac|geom|hstab|c4_to_wing_c4"] = (
            -C_vt * np.sqrt(inputs["ac|geom|wing|AR"]) * inputs["ac|geom|wing|S_ref"] ** (1.5)
        ) / (inputs["ac|geom|hstab|c4_to_wing_c4"] ** 2)


class WingRoot_LinearTaper(ExplicitComponent):
    """Inputs: ac|geom|wing|S_ref, ac|geom|wing|AR, ac|geom|wing|taper
    Outputs: C_root
    """

    def setup(self):
        self.add_input("ac|geom|wing|S_ref", units="ft**2", desc="Reference wing area in sq ft")
        self.add_input("ac|geom|wing|AR", desc="Wing Aspect Ratio")
        self.add_input("ac|geom|wing|taper", desc="Main Wing Taper Ratio")
        self.add_output("C_root", units="ft")
        self.declare_partials(["C_root"], ["*"])

    def compute(self, inputs, outputs):
        root_chord = (
            2
            * inputs["ac|geom|wing|S_ref"]
            / (np.sqrt(inputs["ac|geom|wing|S_ref"] * inputs["ac|geom|wing|AR"]) * (1 + inputs["ac|geom|wing|taper"]))
        )
        outputs["C_root"] = root_chord

    def compute_partials(self, inputs, J):
        J["C_root", "ac|geom|wing|S_ref"] = 2 / (
            (1 + inputs["ac|geom|wing|taper"]) * np.sqrt(inputs["ac|geom|wing|AR"] * inputs["ac|geom|wing|S_ref"])
        ) - (inputs["ac|geom|wing|AR"] * inputs["ac|geom|wing|S_ref"]) / (
            (1 + inputs["ac|geom|wing|taper"]) * (inputs["ac|geom|wing|AR"] * inputs["ac|geom|wing|S_ref"]) ** 1.5
        )
        J["C_root", "ac|geom|wing|AR"] = -inputs["ac|geom|wing|S_ref"] ** 2 / (
            (inputs["ac|geom|wing|taper"] + 1) * (inputs["ac|geom|wing|AR"] * inputs["ac|geom|wing|S_ref"]) ** 1.5
        )
        J["C_root", "ac|geom|wing|taper"] = (
            -2
            * inputs["ac|geom|wing|S_ref"]
            / (
                (inputs["ac|geom|wing|taper"] + 1) ** 2
                * (inputs["ac|geom|wing|AR"] * inputs["ac|geom|wing|S_ref"]) ** 0.5
            )
        )


class WingMAC_Trapezoidal(ExplicitComponent):
    """Inputs: ac|geom|wing|root_chord, ac|geom|wing|taper
    Outputs: MAC"""

    def setup(self):
        self.add_input("ac|geom|wing|root_chord", units="ft", desc="Main wing root chord")
        self.add_input("ac|geom|wing|taper", desc="Main wing taper ratio")
        self.add_output("MAC", units="ft")
        self.declare_partials(["MAC"], ["*"])

    def compute(self, inputs, outputs):
        meanAeroChord = (
            (2 / 3)
            * inputs["ac|geom|wing|root_chord"]
            * (1 + inputs["ac|geom|wing|taper"] + inputs["ac|geom|wing|taper"] ** 2)
            / (1 + inputs["ac|geom|wing|taper"])
        )
        outputs["MAC"] = meanAeroChord

    def compute_partials(self, inputs, J):
        J["MAC", "ac|geom|wing|root_chord"] = (
            (2 / 3)
            * (1 + inputs["ac|geom|wing|taper"] + inputs["ac|geom|wing|taper"] ** 2)
            / (1 + inputs["ac|geom|wing|taper"])
        )
        J["MAC", "ac|geom|wing|taper"] = (2 / 3) * inputs["ac|geom|wing|root_chord"] * (
            2 * inputs["ac|geom|wing|taper"] + 1
        ) / (inputs["ac|geom|wing|taper"] + 1) - (2 / 3) * inputs["ac|geom|wing|root_chord"] * (
            1 + inputs["ac|geom|wing|taper"] + inputs["ac|geom|wing|taper"] ** 2
        ) / (
            1 + inputs["ac|geom|wing|taper"]
        ) ** 2


class CL_MAX_cruise(ExplicitComponent):
    def setup(self):
        self.add_input("ac|aero|Cl_max")
        self.add_input("ac|geom|wing|c4sweep", units="rad")

        self.add_output("Wing_CL_max")
        self.declare_partials(["Wing_CL_max"], ["*"])

    def compute(self, inputs, outputs):
        cLmax = 0.9 * inputs["ac|aero|Cl_max"] * np.cos(inputs["ac|geom|wing|c4sweep"])
        outputs["Wing_CL_max"] = cLmax

    def compute_partials(self, inputs, J):
        J["Wing_CL_max", "ac|aero|Cl_max"] = 0.9 * np.cos(inputs["ac|geom|wing|c4sweep"])
        J["Wing_CL_max", "ac|geom|wing|c4sweep"] = (
            -0.9 * inputs["ac|aero|Cl_max"] * np.sin(inputs["ac|geom|wing|c4sweep"])
        )


class StallSpeed_wing(ExplicitComponent):
    def initialize(self):
        self.options.declare("num_nodes", default=1)

    def setup(self):
        nn = self.options["num_nodes"]
        self.add_input("ac|weights|MLW", units="kg")
        self.add_input("ac|geom|wing|S_ref", units="m**2")
        self.add_input("Wing_CL_max")

        self.add_output("Vstall_wing", units="m/s")
        self.declare_partials(["Vstall_wing"], ["*"])

    def compute(self, inputs, outputs):
        Vstall = (
            2 ** 0.5
            * inputs["ac|weights|MLW"] ** 0.5
            * inputs["ac|geom|wing|S_ref"] ** -0.5
            * 1.225 ** -0.5
            * inputs["Wing_CL_max"] ** -0.5
        )
        outputs["Vstall_wing"] = Vstall

    def compute(self, inputs, J):
        J["Vstall_wing", "ac|weights|MLW"] = (
            2 ** 0.5
            * 0.5
            * inputs["ac|weights|MLW"] ** (0.5 - 1)
            * inputs["ac|geom|wing|S_ref"] ** -0.5
            * 1.225 ** -0.5
            * inputs["Wing_CL_max"] ** -0.5
        )
        J["Vstall_wing", "ac|geom|wing|S_ref"] = (
            2 ** 0.5
            * inputs["ac|weights|MLW"] ** 0.5
            * -0.5
            * inputs["ac|geom|wing|S_ref"] ** (-0.5 - 1)
            * 1.225 ** -0.5
            * inputs["Wing_CL_max"] ** -0.5
        )
        J["Vstall_wing", "Wing_CL_max"] = (
            2 ** 0.5
            * inputs["ac|weights|MLW"] ** 0.5
            * inputs["ac|geom|wing|S_ref"] ** -0.5
            * 1.225 ** -0.5
            * -0.5
            * inputs["Wing_CL_max"] ** (-0.5 - 1)
        )


class CL_MAX(ExplicitComponent):
    def setup(self):
        self.add_input("ac|weights|MTOW", units="kg")
        self.add_input("ac|geom|wing|S_ref", units="m**2")
        self.add_input("ac|aero|Vstall_land", units="m/s")
        self.add_input("ac|geom|wing|c4sweep", units="rad")

        self.add_output("CL_max")
        self.declare_partials(["CL_max"], ["*"])

    def compute(self, inputs, outputs):
        clmax = 2 * inputs["ac|weights|MTOW"] * inputs["ac|geom|wing|S_ref"] ** -1 * 1.225 ** -1 * inputs[
            "ac|aero|Vstall_land"
        ] ** -2 - 0.9 * 1.2 * np.cos(inputs["ac|geom|wing|c4sweep"])
        outputs["CL_max"] = clmax

    def compute_partials(self, inputs, J):
        J["CL_max", "ac|weights|MTOW"] = (
            2 * inputs["ac|geom|wing|S_ref"] ** -1 * 1.225 ** -1 * inputs["ac|aero|Vstall_land"] ** -2
        )
        J["CL_max", "ac|geom|wing|S_ref"] = (
            2
            * inputs["ac|weights|MTOW"]
            * -1
            * inputs["ac|geom|wing|S_ref"] ** -2
            * 1.225 ** -1
            * inputs["ac|aero|Vstall_land"] ** -2
        )
        J["CL_max", "ac|aero|Vstall_land"] = (
            2
            * inputs["ac|weights|MTOW"]
            * inputs["ac|geom|wing|S_ref"] ** -1
            * 1.225 ** -1
            * -2
            * inputs["ac|aero|Vstall_land"] ** -3
        )
        J["CL_max", "ac|geom|wing|c4sweep"] = -0.9 * 1.2 * np.sin(inputs["ac|geom|wing|c4sweep"])


class WingSpan(ExplicitComponent):
    def setup(self):
        self.add_input("ac|geom|wing|S_ref", units="m**2")
        self.add_input("ac|geom|wing|AR")

        self.add_output("span", units="m")
        self.declare_partials(["span"], ["*"])

    def compute(self, inputs, outputs):
        b = inputs["ac|geom|wing|S_ref"] ** 0.5 * inputs["ac|geom|wing|AR"] ** 0.5
        outputs["span"] = b

    def compute_partials(self, inputs, J):
        J["span", "ac|geom|wing|S_ref"] = (
            0.5 * inputs["ac|geom|wing|S_ref"] ** (0.5 - 1) * inputs["ac|geom|wing|AR"] ** 0.5
        )
        J["span", "ac|geom|wing|AR"] = (
            inputs["ac|geom|wing|S_ref"] ** 0.5 * 0.5 * inputs["ac|geom|wing|AR"] ** (0.5 - 1)
        )


if __name__ == "__main__":
    from openmdao.api import IndepVarComp, Problem

    prob = Problem()
    prob.model = Group()
    dvs = prob.model.add_subsystem("dvs", IndepVarComp(), promotes_outputs=["*"])
    dvs.add_output("ac|geom|wing|S_ref", 124.6, units="m**2")
    dvs.add_output("ac|geom|wing|AR", 9.45)

    prob.model.add_subsystem("span", WingSpan(), promotes_inputs=["*"], promotes_outputs=["*"])

    prob.setup(force_alloc_complex=True)
    prob.run_model()
    print("Wingspan:")
    print(prob["span.span"])

    data = prob.check_partials(method="cs", compact_print=True, show_only_incorrect=False)
