import numpy as np
import matplotlib.pyplot as plt

from pyfiberamp.steady_state import SteadyStateSimulation
from pyfiberamp.fibers import YbDopedDoubleCladFiber


# The fiber parameters
length = 5
core_radius = 5e-6
pump_cladding_radius = 50e-6
ratio_of_core_and_cladding_diameters = core_radius / pump_cladding_radius
core_na = 0.12  # This does not play a role since we define the mode field diameter later
yb_number_density = 3e25
background_loss = 0

# Create the fiber
dc_fiber = YbDopedDoubleCladFiber(
        length=length,
        core_radius=core_radius,
        core_na=core_na,
        ratio_of_core_and_cladding_diameters=ratio_of_core_and_cladding_diameters,
        ion_number_density=yb_number_density,
        background_loss=background_loss
)
dc_fiber.default_signal_mode_shape_parameters['functional_form'] = 'gaussian'

# Create the simulation and add signal and pump channels
simulation = SteadyStateSimulation()
simulation.fiber = dc_fiber
simulation.add_cw_signal(wl=1030e-9, power=0.4, mode_shape_parameters={'mode_diameter': 9.6e-6})
simulation.add_backward_pump(wl=914e-9, power=47.2)

# Run the simulation
result = simulation.run(tol=1e-4)

# Plotting the steady state
result.plot_power_evolution()

