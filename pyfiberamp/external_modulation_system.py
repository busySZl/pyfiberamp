import numpy as np
from scipy.special import jn
import matplotlib.pyplot as plt

from pyfiberamp.parameters import *
from pyfiberamp.helper_funcs import *
from pyfiberamp.fibers import ErDopedFiber
from pyfiberamp.steady_state import SteadyStateSimulation


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
        signal_bandwith=1e8,
        optical_wave_length=1550,
        edfa_enable=False,
        edfa_length=5.0,
        edfa_core_radius=2.2e-6,
        edfa_core_na=0.24,
        er_number_density=1e25,
        edfa_npoints=20,
        atten_loss_db=40
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
        self.modulator_R_internal = modulator_R_internal
        assert self.R_input == self.modulator.r_internal, \
            "R_input should be equal to modulator.r_internal"
        assert self.R_input == self.R_load, \
            "R_input should be equal to R_load"
        self.RIN = laser_RIN            # dB
        self.laser_optical_power_out = laser_optical_power_out
        self.T = T0 + t
        self.signal_bandwith = signal_bandwith
        self.edfa_enable = edfa_enable
        if edfa_enable:
            self.atten_loss = pow(10, - atten_loss_db / 10)
            self.edfa_fiber = ErDopedFiber(
                length=edfa_length,
                core_radius=edfa_core_radius,
                core_na=edfa_core_na,
                ion_number_density=er_number_density,
                background_loss=0
            )
            self.edfa_fiber.default_signal_mode_shape_parameters['functional_form'] = 'gaussian'
            self.edfa_fiber.default_pump_mode_shape_parameters['functional_form'] = 'gaussian'
            self.edfa_sim_points = edfa_npoints

    def get_output_info_dict(self, V_dc_bias, Power_rf_input):
        """

        :param V_dc_bias:   V
        :param Power_rf_input:    W
        :return:
        """
        system_output_info_dict = {}

        P_rf_mzm = Power_rf_input * self.R_input / (self.R_input + self.modulator_R_internal)
        optical_power_array = self.modulator.optical_power_output(
            optical_power_input=self.laser_optical_power_out,
            V_dc_bias=V_dc_bias,
            P_rf_mzm=P_rf_mzm
        )

        if self.edfa_enable:
            num_freq = optical_power_array.shape[0]
            optical_ase_array = np.zeros((num_freq, 1), dtype=np.float32)
            optical_power_array = np.append(optical_power_array, optical_ase_array, axis=1)
            for i in range(num_freq):
                edfa_input_power = optical_power_array[i, 0]
                simulation = SteadyStateSimulation()
                simulation.fiber = self.edfa_fiber
                simulation.add_cw_signal(wl=1550e-9, power=edfa_input_power)
                simulation.add_forward_pump(wl=980e-9, power=0.1)
                simulation.add_ase(wl_start=1500e-9, wl_end=1600e-9, n_bins=100)
                simulation.set_number_of_nodes(self.edfa_sim_points)
                result = simulation.run(tol=1e-4)
                _, res_optical_gain_db = result.cal_nf()
                res_optical_gain = pow(10, res_optical_gain_db / 10)
                forward_optical_ase = np.sum(result.powers.forward_ase, axis=0)
                forward_optical_ase_power = forward_optical_ase[-1]
                forward_optical_ase_power_no_B = forward_optical_ase_power / self.signal_bandwith
                optical_power_array[i, :] = optical_power_array[i, :] * res_optical_gain
                optical_power_array[i, -1] = forward_optical_ase_power_no_B

            print(">>>>>optical_power_array.shape: {}".format(optical_power_array.shape))
            print(">>>>>after edfa optical_power_array: {}".format(to_db(optical_power_array[:, 0])))

        current_output_array = self.photonic_detector.current_output(
            optical_power_input=optical_power_array
        )
        if not self.edfa_enable:
            photonic_current = current_output_array / 2.0   # impedance matching on the spectrum analyzer
            P_rf_ouput = pow(photonic_current, 2) * self.R_load * np.array([1, 0.5, 0.5, 0.5])
            P_rf_ouput_db = 10 * np.log10(P_rf_ouput)

            print("p_rf: {} dbm".format(to_db(P_rf_ouput[:, 1] * 1000)))
            print("Power_rf_input: {} dbm".format(to_db(Power_rf_input * 1000)))

            rf_gain = P_rf_ouput[:, 1] / Power_rf_input
            # rf_gain = P_rf_ouput[:, 1] / P_rf_mzm
            rf_gain_db = 10 * np.log10(rf_gain)

            N_th_no_B = K * self.T
            N_shot_no_B = 2 * q * photonic_current[:, 0] * self.R_load
            RIN = pow(10, self.RIN / 10.)
            N_rin_no_B = pow(photonic_current[:, 0], 2) * RIN * self.R_load

            # get thermal noise gain
            N_out_no_B = rf_gain * N_th_no_B + N_th_no_B + N_shot_no_B + N_rin_no_B
        else:
            photonic_current = current_output_array / 2.0  # impedance matching on the spectrum analyzer
            P_rf_ouput = pow(photonic_current, 2) * self.R_load * np.array([1, 0.5, 0.5, 0.5, 1])
            P_rf_ouput_db = 10 * np.log10(P_rf_ouput[:, :-1])

            rf_gain = P_rf_ouput[:, 1] / Power_rf_input
            rf_gain_db = 10 * np.log10(P_rf_ouput[:, 1] / Power_rf_input)

            N_th_no_B = K * self.T
            N_shot_no_B = 2 * q * photonic_current[:, 0] * self.R_load
            RIN = pow(10, self.RIN / 10.)
            N_rin_no_B = pow(photonic_current[:, 0], 2) * RIN * self.R_load
            N_ase_no_B = pow(photonic_current[:, -1], 2) * self.R_load

            N_out_no_B = rf_gain * N_th_no_B + N_th_no_B + N_shot_no_B + N_rin_no_B + N_ase_no_B

        NF_db = 10 * np.log10(N_out_no_B / (N_th_no_B * rf_gain))

        system_output_info_dict["P_rf_ouput_list"] = P_rf_ouput_db
        system_output_info_dict["rf_gain"] = rf_gain_db
        system_output_info_dict["NF"] = NF_db

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
        self.v_pi_rf_list = V_pi_rf_list
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
        print("m_dc: {}".format(m_dc))
        print("P_rf_mzm: {}".format(P_rf_mzm))
        print("v_rf: {}".format(V_rf))
        print("optical_in: {} dbm".format(to_db(optical_power_input * 1000)))
        print("optical_loss_coeff: {} dbm".format(to_db(self.optical_loss_coeff)))
        k = 0.5 * self.optical_loss_coeff * optical_power_input
        p_out = []
        for i in range(len(self.v_pi_rf_list)):
            m_rf = np.pi / self.v_pi_rf_list[i] * V_rf
            print("m_rf: {}".format(m_rf))
            p_dc = k * (1 + np.cos(m_dc) * jn(0, m_rf))
            p_1st = - k * 2 * np.sin(m_dc) * jn(1, m_rf)
            p_2nd = - k * 2 * np.cos(m_dc) * jn(2, m_rf)
            p_3rd = k * 2 * np.sin(m_dc) * jn(3, m_rf)
            # p_1st = - k * 2 * np.sin(m_dc) * jn(1, m_rf)
            # p_2nd = - k * 2 * np.cos(m_dc) * jn(2, m_rf)
            # p_3rd = k * 2 * np.sin(m_dc) * jn(3, m_rf)
            p_out.append([p_dc, p_1st, p_2nd, p_3rd])

        res_out = np.array(p_out, dtype=np.float32)
        if len(self.v_pi_rf_list) == 1:
            res_out = res_out[np.newaxis, :]

        print("mzm output p_optical :{} dbm".format(to_db(np.abs(res_out) * 1000)))

        return np.abs(res_out)


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
        assert len(optical_power_input.shape) == 2 and (optical_power_input.shape[1] == 4 or
                                                        optical_power_input.shape[1] == 5), \
            "optical_power_input.shape error: {}".format(optical_power_input.shape)
        current = optical_power_input * self.responsivity

        return current


def test_example():
    # 1 - 3 GHz
    ems = ExternalModulationSystem(
        mzm_v_pi_dc=2.5959,
        mzm_v_pi_rf_list=[1.6533,    1.5871,    1.6629],
        optical_loss_coeff=pow(10, -0.6),
        pd_responsivity=0.9,
        laser_optical_power_out=0.08,
        laser_RIN=-160,
        modulator_R_internal=50,
        pd_R_internal=50,
        R_load=50,
        R_input=50,
        t=25,
        signal_bandwith=1e8,
        optical_wave_length=1550,
        edfa_enable=False
    )
    res_info_dict = ems.get_output_info_dict(
        V_dc_bias=1.29795,
        Power_rf_input=0.001
    )
    print("ems: {}".format(res_info_dict))


def edfa_example():
    # 1 - 3 GHz
    ems = ExternalModulationSystem(
        mzm_v_pi_dc=2.5959,
        mzm_v_pi_rf_list=[1.6533, 1.5871, 1.6629],
        optical_loss_coeff=pow(10, -0.6),
        pd_responsivity=0.9,
        laser_optical_power_out=0.08,
        laser_RIN=-160,
        modulator_R_internal=50,
        pd_R_internal=50,
        R_load=50,
        R_input=50,
        t=25,
        signal_bandwith=1e8,
        optical_wave_length=1550,
        edfa_enable=False
    )
    res_info_dict = ems.get_output_info_dict(
        V_dc_bias=1.29795,
        Power_rf_input=0.000001
    )
    print("ems: {}".format(res_info_dict))


if __name__ == "__main__":
    test_example()





