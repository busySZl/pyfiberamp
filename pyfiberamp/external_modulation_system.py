import numpy as np
from scipy.special import jn
from pyfiberamp.parameters import *


class ExternalModulationSystem(object):
    def __init__(
        self,
        mzm_v_pi_dc,
        mzm_v_pi_rf_list,
        optical_loss_coeff,
        pd_responsivity,
        laser_optical_power_out,
        laser_RIN,
        modulator_R_internal=50,
        pd_R_internal=50,
        R_load=50,
        R_input=50,
        t=25,
        signal_bandwith = 1e8,
        optical_wave_length=1550,
        edfa_enable=False
    ):
        self.modulator = MachZehnderModulator(
            V_pi_dc=mzm_v_pi_dc,
            V_pi_rf_list=mzm_v_pi_rf_list,
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
        self.signal_bandwith = signal_bandwith
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
        optical_power_array = self.modulator.optical_power_output(
            optical_power_input=self.laser_optical_power_out,
            V_dc_bias=V_dc_bias,
            P_rf_mzm=P_rf_mzm
        )
        current_output_array = self.photonic_detector.current_output(
            optical_power_input=optical_power_array
        )
        photonic_current = current_output_array / 2.0
        P_rf_ouput = pow(photonic_current, 2) * self.R_load * np.array([1, 0.5, 0.5, 0.5])

        rf_gain = P_rf_ouput[:, 1] / P_rf_mzm

        N_th_no_B = K * self.T
        N_shot_no_B = 2 * q * photonic_current[:, 0] * 0.25 * np.sqrt(2) * self.R_load
        RIN = 10 * np.log10(self.RIN / 10)
        N_rin_no_B = 0.25 * pow(photonic_current[:, 0], 2) * RIN * self.R_load

        # get thermal noise gain
        N_out_no_B = rf_gain * N_th_no_B + N_th_no_B + N_shot_no_B + N_rin_no_B

        NF = 10 * np.log10(N_out_no_B / (N_th_no_B * rf_gain))

        system_output_info_dict["P_rf_ouput_list"] = P_rf_ouput_list
        system_output_info_dict["rf_gain"] = rf_gain
        system_output_info_dict["NF"] = NF

        return system_output_info_dict


class MachZehnderModulator(object):
    def __init__(
        self,
        V_pi_dc,
        V_pi_rf_list,
        optical_loss_coeff,
        R_internal=50
    ):
        self.v_pi_dc = V_pi_dc
        self.v_pi_rf_list =  V_pi_rf_list
        self.optical_loss_coeff = optical_loss_coeff
        self.r_internal = R_internal

    def optical_power_output(self, optical_power_input, V_dc_bias, P_rf_mzm):
        """

        :param optical_power_input:
        :param V_dc_bias:
        :param P_rf_mzm:
        :return: np.array
        """
        V_rf = np.sqrt(P_rf_mzm * 2 * self.r_internal)

        m_dc = np.pi / self.v_pi_dc * V_dc_bias
        k = 0.5 * self.optical_loss_coeff * optical_power_input
        p_out = []
        for i in range(len(self.v_pi_rf_list)):
            m_rf = np.pi / self.v_pi_rf_list[i] * V_rf
            p_dc = k * (1 + np.cos(m_dc)) * jn(0, m_rf)
            p_1st = - k * 2 * np.sin(m_dc) * jn(1, m_rf)
            p_2nd = - k * 2 * np.cos(m_dc) * jn(2, m_rf)
            p_3rd = k * 2 * np.sin(m_dc) * jn(3, m_rf)
            p_out.append([p_dc, p_1st, p_2nd, p_3rd])

        res_out = np.array(p_out, dtype=np.float32)
        if len(self.v_pi_rf_list) == 1:
            res_out = res_out[np.newaxis, :]

        return res_out


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
        assert len(optical_power_input.shape) == 2 and optical_power_input.shape[1] == 4, \
            "optical_power_input.shape error: {}".format(optical_power_input.shape)
        current = optical_power_input * self.responsivity

        return current


def test_example():
    ems = ExternalModulationSystem(
        mzm_v_pi_dc=,
        mzm_v_pi_rf=,
        optical_loss_coeff=,
        pd_responsivity=,
        laser_optical_power_out=,
        laser_RIN= ,
        modulator_R_internal=50,
        pd_R_internal=50,
        R_load=50,
        R_input=50,
        t=25,
        optical_wave_length=1550,
        edfa_enable=False
    )





