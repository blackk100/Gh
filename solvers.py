import numpy as np
import utility as util
import em_field as em
import constants as const


def skew(b: np.ndarray) -> np.ndarray:
	"""Generate the skew-symmetric matrix defined in [1, sec. 3]

	Args:
		b (np.ndarray): Magnetic field unit vector

	Returns:
		np.ndarray: Skew-symmetric vector
	"""

	b1, b2, b3 = b

	bcap = np.array([
		[  0, -b3,  b2],
		[ b3,   0, -b1],
		[-b2,  b1,   0]
	])

	return bcap


def Gh(
		X0    : np.ndarray,
		time  : np.ndarray,
		qm    : float,
		points: tuple,
		Efield: np.ndarray,
		Bfield: np.ndarray,
		order : int,
		approx: bool
	) -> np.ndarray:
	"""Runs the Gh algorithm from He et al. Implements the 1st and 2nd order methods, in both of the exact and approximate schemes

	Args:
		X0 (np.ndarray): Initial state vector
		time (np.ndarray): Vector of time coordinates
		qm (float): Ratio of particle charge to particle mass
		points (tuple): Tuple of X-, Y- & Z-coordinates for ``Efield`` and ``Bfield``. Has a size of (N1 x N2 x N3).
		Efield (np.ndarray): Electric vector field of size 3 x N1 x N2 x N3
		Bfield (np.ndarray): Magnetic vector field of size 3 x N1 x N2 x N3
		order (int): Accuracy order of the algorithm
		approx (bool): Flag to indicate if the alternative approximation is used

	Returns:
		np.ndarray: State vectors at each time coordinate
	"""

	N = np.size(time)
	X = np.zeros((N, 6))
	X[0, :] = X0

	h  = time[1] - time[0]    # Time step size

	for n in range(N - 1):
		xk = X[n, 0:3].copy() # Position at t=n
		vk = X[n, 3:6].copy() # Velocity at t=n
		x1 = np.zeros(3)      # Position at t=n + 1
		v1 = np.zeros(3)      # Velocity at t=n + 1

		if order == 1:
			# EM Fields
			Ek = util.interp(xk, points, Efield)
			Bk = util.interp(xk, points, Bfield)

			Bk_mag = np.linalg.norm(Bk) # Magnitude of magnetic field
			bk     = Bk / Bk_mag        # Magnetic field unit vector
			# bk_hat  = skew(bk)          # Skew-symmetric matrix

			# Frequency
			wk     = -qm * Bk_mag
			if approx:
				wk = np.arctan2(-h * qm * Bk_mag, 2.0) * 2.0 / h # Approximate local frequency

			# Velocity components
			vk_perp = util.perp(vk, Bk)
			vk_ll   = vk - vk_perp

			# 1. Velocity rotation from B-field
			alpha = np.cos(h * wk)
			beta  = np.sin(h * wk)
			vm = vk_ll + alpha * vk_perp + beta * np.cross(bk, vk_perp)
			# alpha = np.sin(h * wk)
			# beta  = 1.0 - np.cos(h * wk)
			# vm = vk + alpha * np.matmul(bk_hat, vk) + beta * np.matmul(bk_hat, np.matmul(bk_hat, vk))

			# 2. Velocity translation from E-field
			v1 = vm + qm * h * Ek

			# 3. Update position
			x1 = xk + h * v1
		else: # Default to order=2
			# 1. Get staggered grid position
			x2 = xk + h * vk / 2.0

			# EM Fields
			# E2 = util.interp(x2, points, Efield)
			# B2 = util.interp(x2, points, Bfield)
			E2 = em.Efield1(x2[0], x2[1], x2[2], const.BANANA)
			B2 = em.Bfield1(x2[0], x2[1], x2[2], const.BANANA)

			B2_mag = np.linalg.norm(B2) # Magnitude of magnetic field
			b2     = B2 / B2_mag        # Magnetic field unit vector
			# b2_hat = skew(b2)           # Skew-symmetric matrix

			# Frequency
			w2     = -qm * B2_mag
			if approx:
				w2 = np.arctan2(h * w2, 2.0) * 2.0 / h # Approximate local frequency

			# 2. Velocity translation from half E-field
			ve = h * qm * E2 / 2.0
			vm = vk + ve

			# Velocity components
			vm_perp = util.perp(vm, B2)
			vm_ll   = vm - vm_perp

			# 3. Velocity rotation from B-field
			alpha = np.cos(h * w2)
			beta  = np.sin(h * w2)
			v1 = vm_ll + alpha * vm_perp + beta * np.cross(b2, vm_perp)

			# 4. Velocity translation from half E-field
			v1 += ve

			# Update position
			x1 = x2 + h * v1 / 2.0

		X[n + 1, :] = np.concatenate((x1, v1))

	return X


def analytical(
		X0    : np.ndarray,
		time  : np.ndarray,
		qm    : float,
		points: tuple,
		Bfield: np.ndarray
	) -> np.ndarray:
	"""Analytical solution for the test case (``type = constants.TEST``).

	Args:
		X0 (np.ndarray): Initial state vector
		time (np.ndarray): Vector of time coordinates
		qm (float): Ratio of particle charge to particle mass
		points (tuple): Tuple of X-, Y- & Z-coordinates for ``Efield`` and ``Bfield``. Has a size of (N1 x N2 x N3).
		Bfield (np.ndarray): Magnetic vector field of size 3 x N1 x N2 x N3

	Returns:
		np.ndarray: Array of state vectors at each time coordinate
	"""

	x = X0[0:3]
	v = X0[3:6]

	X = np.zeros((np.size(time), 6))
	B = util.interp(x, points, Bfield)
	w = qm * B[2]

	X[:, 0] = -v[1] * np.cos(w * time) / w + x[0] + v[1] / w
	X[:, 1] =  v[1] * np.sin(w * time) / w + x[1]
	X[:, 2] =  x[2]
	X[:, 3] =  v[1] * np.sin(w * time)
	X[:, 4] =  v[1] * np.cos(w * time)
	X[:, 5] = 0.0

	return X


def rk4(
		X0    : np.ndarray,
		time  : np.ndarray,
		qm    : float,
		points: tuple,
		Efield: np.ndarray,
		Bfield: np.ndarray,
	) -> np.ndarray:
	"""Performs Runge-Kutta (4th order) integration on ``dynamics``

	   Butcher Table:
	   0   | 0     0     0     0
	   1/2 | 1/2   0     0     0
	   1/2 | 0     1/2   0     0
	   1   | 0     0     1     0
	   ---------------------------
	       | 1/6   1/3   1/3   1/6

	Args:
		X0 (np.ndarray): Initial state vector
		time (np.ndarray): Vector of time coordinates
		qm (float): Ratio of particle charge to particle mass
		points (tuple): Tuple of X-, Y- & Z-coordinates for ``Efield`` and ``Bfield``. Has a size of (N1 x N2 x N3).
		Efield (np.ndarray): Electric vector field of size 3 x N1 x N2 x N3
		Bfield (np.ndarray): Magnetic vector field of size 3 x N1 x N2 x N3

	Returns:
		np.ndarray: Array of state vectors for each value of ``x``
	"""

	N = np.size(time)
	M = np.size(X0)
	X = np.zeros((N, M))

	X[0, :] = X0.copy()
	h       = time[1] - time[0]

	for n in range(0, N - 1):
		k1 = h * dynamics(X[n, :]           , qm, points, Efield, Bfield)
		k2 = h * dynamics(X[n, :] + k1 / 2.0, qm, points, Efield, Bfield)
		k3 = h * dynamics(X[n, :] + k2 / 2.0, qm, points, Efield, Bfield)
		k4 = h * dynamics(X[n, :] + k3      , qm, points, Efield, Bfield)

		X[n + 1, :] = X[n, :] + k1 / 6.0 + k2 / 3.0 + k3 / 3.0 + k4 / 6.0

	return X


def dynamics(
		X     : np.ndarray,
		qm    : float,
		points: tuple,
		Efield: np.ndarray,
		Bfield: np.ndarray
	) -> np.ndarray:
	"""Returns the state vector derivates according to the Newton-Lorentz equation

	Args:
		X (np.ndarray): State vector
		qm (float): Ratio of particle charge to particle mass
		points (tuple): Tuple of X-, Y- & Z-coordinates for ``Efield`` and ``Bfield``. Has a size of (N1 x N2 x N3).
		Efield (np.ndarray): Electric vector field of size 3 x N1 x N2 x N3
		Bfield (np.ndarray): Magnetic vector field of size 3 x N1 x N2 x N3

	Returns:
		np.ndarray: Derivates of the state vectors
	"""

	x = X[0:3] # Position at t=n
	v = X[3:6] # Velocity at t=n

	E = util.interp(x, points, Efield)
	B = util.interp(x, points, Bfield)

	Xdot = np.zeros(6)
	Xdot[0:3] = v.copy()
	Xdot[3:6] = qm * (E + np.cross(v, B)) # Newton-Lorentz

	return Xdot
