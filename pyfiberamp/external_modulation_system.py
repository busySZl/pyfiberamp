import numpy as np
from scipy.special import jn
from pyfiberamp.parameters import *


class ExternalModulationSystem(object):
    def __init__(
        self,
        optical_wave_length,
        mzm_v_pi_dc,
        mzm_v_pi_rf,
        optical_loss_coeff,
        pd_responsivity,
        laser_optical_power_out,
        laser_RIN,
        modulator_R_internal=50,
        pd_R_internal=50,
        R_load=50,
        R_input=50,
        t=25,
        edfa_enable=False
    ):
        self.modulator = MachZehnderModulator(
            V_pi_dc=mzm_v_pi_dc,
            V_pi_rf=mzm_v_pi_rf,
            optical_loss_coeff=optical_loss_coeff,
            R_internal=modulator_R_internal
        )
        self.photonic_detector = PhotonicDetector(
            responsivity=pd_responsivity,
            R_internal=pd_R_internal
        )
        self.R_input = R_input
        self.R_load = R_load
        assert self.R_input == self.modulator.r_internal, \
            "R_input should be equal to modulator.r_internal"
        assert self.R_input == self.R_load, \
            "R_input should be equal to R_load"
        self.RIN = laser_RIN
        self.laser_optical_power_out = laser_optical_power_out
        self.T = T0 + t
        if edfa_enable:
            pass

    def get_output_info_dict(self, V_dc_bias, Power_rf_input):
        """

        :param V_dc_bias:   V
        :param Power_rf_input:    W
        :return:
        """
        system_output_info_dict = {}

        P_rf_mzm = Power_rf_input * self.R_input / (self.R_input + self.r_internal)
        optical_power_list = self.modulator.optical_power_output(
            optical_power_input=self.laser_optical_power_out,
            V_dc_bias=V_dc_bias,
            P_rf_mzm=P_rf_mzm
        )
        current_output_list = self.photonic_detector.current_output(
            optical_power_input=optical_power_list
        )
        cnt_rf = len(current_output_list)

        P_rf_ouput_list = []
        for i in range(cnt_rf):
            photonic_current = current_output_list[i] * 0.5
            if i == 0:
                P_rf_ouput = photonic_current * photonic_current * self.R_load
            else:
                P_rf_ouput = photonic_current * photonic_current * self.R_load * 0.5
            P_rf_ouput_list.append(P_rf_ouput)

        rf_gain = P_rf_ouput_list[1] / P_rf_mzm

        N_th = self.T
        N_shot = 2 * q * self.current_output_list[0] * 0.25

        system_output_info_dict["P_rf_ouput_list"] = P_rf_ouput_list
        system_output_info_dict["rf_gain"] = rf_gain

        return system_output_info_dict


class MachZehnderModulator(object):
    def __init__(
        self,
        V_pi_dc,
        V_pi_rf,
        optical_loss_coeff,
        R_internal=50
    ):
        self.v_pi_dc = V_pi_dc
        self.v_pi_rf = V_pi_rf
        self.optical_loss_coeff = optical_loss_coeff
        self.r_internal = R_internal

    def optical_power_output(self, optical_power_input, V_dc_bias, P_rf_mzm):
        """

        :param optical_power_input:
        :param V_dc_bias:
        :param P_rf_mzm:
        :return:
        """
        V_rf = np.sqrt(P_rf_mzm * 2 * self.r_internal)

        m_dc = np.pi / self.v_pi_dc * V_dc_bias
        m_rf = np.pi / self.v_pi_rf * V_rf
        k = 0.5 * self.optical_loss_coeff * optical_power_input
        p_dc = k * (1 + np.cos(m_dc)) * jn(0, m_rf)
        p_1st = - k * 2 * np.sin(m_dc) * jn(1, m_rf)
        p_2nd = - k * 2 * np.cos(m_dc) * jn(2, m_rf)
        p_3rd = k * 2 * np.sin(m_dc) * jn(3, m_rf)

        return [p_dc, p_1st, p_2nd, p_3rd]


class PhotonicDetector(object):
    def __init__(
        self,
        responsivity,
        R_internal=50,
    ):
        """

        :param responsivity:    A / W
        :param R_internal:      Omega
        :param R_load:          Omega
        """
        self.responsivity = responsivity
        self.r_internal = R_internal

    def current_output(self, optical_power_input):
        """

        :param optical_power_input: list
        :return:
        """
        assert isinstance(optical_power_input, list),\
            "Optical_power_input must be a list"

        current = []
        num_input = len(optical_power_input)
        for i in range(num_input):
            i = optical_power_input[i] * self.responsivity
            current.append(i)

        return current




