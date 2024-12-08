# Algorithm constants
import scipy.constants as const

# Universal Constants
QE = const.elementary_charge
ME = const.electron_mass
MP = const.proton_mass
MN = const.neutron_mass
C  = const.speed_of_light
# KB = const.Boltzmann

# Algorithm Flags
# EM Fields
TEST      = 0 # Static uniform EM Field E = [0, 0, 0]; B = [0, 0, 0.1]
STATIC    = 1 # Static non-uniform EM Field described in Sec. 4.1
BANANA    = 2 # Static axisymmetric tokamak EM field with a banana orbit described in Sec. 4.2
TRANSIT   = 3 # Static axisymmetric tokamak EM field with a transit orbit described in Sec. 4.2
