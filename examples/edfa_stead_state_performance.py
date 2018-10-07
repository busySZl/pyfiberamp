import numpy as np
import matplotlib.pyplot as plt

from pyfiberamp.steady_state import SteadyStateSimulation
from pyfiberamp.fibers import ErDopedFiber

plt.rcParams["figure.figsize"] = (12, 10)

# The fiber parameters
length = 5.0
core_radius = 2.2e-6
core_na = 0.24
er_number_density = 1e25
background_loss = 0
npoints = 20

dc_fiber = ErDopedFiber(
    length=length,
    core_radius=core_radius,
    core_na=core_na,
    ion_number_density=er_number_density,
    background_loss=0
)

dc_fiber.default_signal_mode_shape_parameters['functional_form'] = 'gaussian'
dc_fiber.default_pump_mode_shape_parameters['functional_form'] = 'gaussian'

res_pts = 20
range_pump = range(res_pts)
pump_powers = (np.array(range_pump) + 1) / 100
input_powers = (np.array(range_pump) + 1) / 1000000
res_gain_dbs = np.zeros((res_pts, ))
res_nfs = np.zeros((res_pts, ))


for idx in range_pump:
    pump_power = pump_powers[idx]
    input_power = input_powers[idx]
    print("==========================================================")
    print(">>>>>>>pump_power: {}".format(pump_power))
    print(">>>>>>>input_power: {}".format(input_power))
    simulation = SteadyStateSimulation()
    simulation.fiber = dc_fiber
    # simulation.add_cw_signal(wl=1550e-9, power=0.01e-3)
    # simulation.add_forward_pump(wl=980e-9, power=pump_power)
    simulation.add_cw_signal(wl=1550e-9, power=input_power)
    simulation.add_forward_pump(wl=980e-9, power=0.1)
    simulation.add_ase(wl_start=1500e-9, wl_end=1600e-9, n_bins=100)

    # Run the simulation
    simulation.set_number_of_nodes(npoints)
    result = simulation.run(tol=1e-4)
    res_nf, res_gain_db = result.cal_nf()

    res_gain_dbs[idx] = res_gain_db
    res_nfs[idx] = res_nf


fig, ax = plt.subplots()
ax.plot(input_powers, res_gain_dbs, 'r')
ax.plot(input_powers, res_nfs, 'b')
plt.show()
