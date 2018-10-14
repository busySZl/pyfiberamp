import os


# Physical quantities or constants
DEFAULT_RAMAN_GAIN = 1e-13
RAMAN_FREQ_SHIFT = 13.2e12
ASE_OUTPUT_POWER_GUESS = 1e-3
RAMAN_GAIN_WL_BANDWIDTH = 5e-9
REP_RATE_LOWER_LIMIT = 1e4  # Hz
NUMBER_OF_MODES_IN_SINGLE_MODE_FIBER = 2  # Two polarization modes
RAMAN_MODES_IN_PM_FIBER = 1
YB_UPPER_STATE_LIFETIME = 0.84e-3  # s
ER_UPPER_STATE_LIFETIME = 10e-3  # s /notes: according to optisystem7
c = 299792458
h = 6.62607e-34
DEFAULT_GROUP_INDEX = 1.45
q = 1.6e-19
T0 = 273.15
K = 1.38e-23

# Constants for the numerical algorithm
SIMULATION_MIN_POWER = 1e-14
SOLVER_MAX_NODES = 20000
START_NODES = 20


# Constants for active fiber
CROSS_SECTION_SMOOTHING_FACTOR = 1e-51
SPECTRUM_PLOT_NPOINTS = 1000


# Default absorption and emission cross section files
this_folder = os.path.dirname(os.path.realpath(__file__))
spectrum_folder = os.path.join(this_folder, 'spectroscopies', 'fiber_spectra')
YB_ABSORPTION_CS_FILE = os.path.join(spectrum_folder, 'ytterbium absorption cross sections.dat')
YB_EMISSION_CS_FILE = os.path.join(spectrum_folder, 'ytterbium emission cross sections.dat')

YB_ABSORPTION_CS_FILE_FROM_OPTISYSTEM = os.path.join(spectrum_folder, 'yb_absorption_parameters_curve.dat')
YB_EMISSION_CS_FILE_FROM_OPTISYSTEM = os.path.join(spectrum_folder, 'yb_emission_parameters_curve.dat')

ER_ABSORPTION_CS_FILE = os.path.join(spectrum_folder, 'erbium_absorption_cross_sections.dat')
ER_EMISSION_CS_FILE = os.path.join(spectrum_folder, 'erbium_emission_cross_sections.dat')

CHANNEL_TYPES = ['forward_signal', 'backward_signal',
                'forward_pump', 'backward_pump',
                'forward_ase', 'backward_ase',
                'forward_raman', 'backward_raman']
