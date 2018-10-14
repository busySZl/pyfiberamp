from pyfiberamp.parameters import *

class MachZehnderModulator(object):
    def __init__(
        self,
        V_pi_dc,
        V_pi_rf,
        R_load,
        optical_power_input,
        optical_loss_coeff
    ):
        self.v_pi_dc = V_pi_dc
        self.v_pi_rf = V_pi_rf
        self.r_load = R_load
        self.optical_power_input = optical_power_input
        self.optical_loss_coeff = optical_loss_coeff

    def optical_power_output(self, V_dc_bias, t=25):
        T = T0 + t
        res =



