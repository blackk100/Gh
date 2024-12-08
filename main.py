import numpy as np
import solvers
import em_field as em
import utility as util
import constants as const

# Inputs
q      = -const.QE     # Particle charge
m      = const.ME      # Particle mass
type   = const.TRANSIT # EM Field type
dx     = 500           # Number of spatial elements
order  = 2             # Order of the Gh algorithm
approx = True         # Approximate Gh schemes
qm     = 1


def main():
    # Initialize
	X0 = util.init(type)
	x0, y0, z0 = X0[0:3]
	v0 = X0[3:6]

	B = em.Bfield1(x0, y0, z0, type)
	Bmag = np.linalg.norm(B)

	w_L   = np.abs(qm) * Bmag # Larmor pulsation [rad/s]
	tau_L = 2.0 * np.pi / w_L # Larmor period [s]
	r_L   = np.linalg.norm(util.perp(v0, B)) / w_L # Larmor radius [m]

	print(f'r_L   = {r_L} [m]')
	print(f'tau_L = {tau_L} [s]')

	x = np.linspace(-1.5, 1.5, dx)
	points = (x, x, x)
	Efield = em.Efield(x, x, x, type)
	Bfield = em.Bfield(x, x, x, type)

	time = util.time(type)

	# Analytical Solution for Test case
	# X_an = solvers.analytical(X0, time, qm, points, Bfield)
	# inv_an, err_an = util.post(X_an, time, points, Bfield)
	# util.plot(X_an, time, inv_an, err_an, r_L, 'long', 'an', pdf=False)

	# RK4 method
	# X_rk = solvers.rk4(X0, time, qm, points, Efield, Bfield)
	# inv_rk, err_rk = util.post(X_rk, time, points, Bfield)
	# util.plot(X_rk, time, inv_rk, err_rk, 'transit', 'rk', pdf=False)

	# Volume preserving methods
	X_gh = solvers.Gh(X0, time, qm, points, Efield, Bfield, order, approx)
	inv_gh, err_gh = util.post(X_gh, time, points, Bfield)
	util.plot(X_gh, time, inv_gh, err_gh, 'transit', 'gh2a', pdf=False)

if __name__ == '__main__':
	main()

# Coarse = 1 period / 10
# Fine = 1 period / 25
# Compare == Fine
# Long - periods = 250
