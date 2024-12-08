import numpy as np
import constants as const

def Efield1(
		x   : float,
		y   : float,
		z   : float,
		type: int
	) -> np.ndarray:
	"""Returns the electric field vector at ``(x, y, z)``

	Args:
		x (np.ndarray): X-position of size N1
		y (np.ndarray): Y-position of size N2
		z (np.ndarray): Z-position of size N3
		type (int): Flag denoting field type from ``constants``

	Returns:
		np.ndarray: Electric field vector of size 3 x N1 x N2 x N3 ``[Ex, Ey, Ez]``
	"""

	E = np.zeros(3)

	if type == const.STATIC:
		R = np.sqrt(x**2 + y**2)

		E[0] = x
		E[1] = y
		E *= 1e-2 / R**3
	elif type == const.BANANA or type == const.TRANSIT:
		pass # E[:, :, :, :] = 0.0
	else: # Default to const.TEST
		pass # E[:, :, :, :] = 0.0

	return E


def Bfield1(
		x   : float,
		y   : float,
		z   : float,
		type: int
	) -> np.ndarray:
	"""Returns the magnetic field vector at ``(x, y, z)``

	Args:
		x (np.ndarray): X-position of size N1
		y (np.ndarray): Y-position of size N2
		z (np.ndarray): Z-position of size N3
		type (int): Flag denoting field type from ``constants``

	Returns:
		np.ndarray: Magnetic field vector of size 3 x N1 x N2 x N3 ``[Bx, By, Bz]``
	"""

	B = np.zeros(3)

	if type == const.STATIC:
		B[2] = np.sqrt(x**2 + y**2)
	elif type == const.BANANA or type == const.TRANSIT:
		R = np.sqrt(x**2 + y**2)

		B[0] = -(2.0 * y + x * z) / (2.0 * R**2)
		B[1] =  (2.0 * x - y * z) / (2.0 * R**2)
		B[2] =  (R - 1)           / (2.0 * R)
	else: # Default to const.TEST
		B[2] = 0.1

	return B


def Efield(
		x   : np.ndarray,
		y   : np.ndarray,
		z   : np.ndarray,
		type: int
	) -> np.ndarray:
	"""Returns the electric field vector at ``(x, y, z)``

	Args:
		x (np.ndarray): X-position of size N1
		y (np.ndarray): Y-position of size N2
		z (np.ndarray): Z-position of size N3
		type (int): Flag denoting field type from ``constants``

	Returns:
		np.ndarray: Electric field vector of size 3 x N1 x N2 x N3 ``[Ex, Ey, Ez]``
	"""

	N  = (3, np.size(x), np.size(y), np.size(z))
	E = np.zeros(N)

	if type == const.STATIC:
		for i in range(N[0]):
			for j in range(N[1]):
				R = np.sqrt(x[i]**2 + y[j]**2)

				E[0, i, j, :] = 10e-2 * x[i] / R**3
				E[1, i, j, :] = 10e-2 * y[j] / R**3
	elif type == const.BANANA or type == const.TRANSIT:
		pass # E[:, :, :, :] = 0.0
	else: # Default to const.TEST
		pass # E[:, :, :, :] = 0.0

	return E


def Bfield(
		x   : np.ndarray,
		y   : np.ndarray,
		z   : np.ndarray,
		type: int
	) -> np.ndarray:
	"""Returns the magnetic field vector at ``(x, y, z)``

	Args:
		x (np.ndarray): X-position of size N1
		y (np.ndarray): Y-position of size N2
		z (np.ndarray): Z-position of size N3
		type (int): Flag denoting field type from ``constants``

	Returns:
		np.ndarray: Magnetic field vector of size 3 x N1 x N2 x N3 ``[Bx, By, Bz]``
	"""

	N  = (3, np.size(x), np.size(y), np.size(z))
	B = np.zeros(N)

	if type == const.STATIC:
		for i in range(N[1]):
			for j in range(N[2]):
				B[2, i, j, :] = np.sqrt(x[i]**2 + y[j]**2)
	elif type == const.BANANA or type == const.TRANSIT:
		for i in range(N[1]):
			B[1, i, :, :] = 2.0 * x[i]

			for k in range(N[3]):
				B[0, i, :, k] = -x[i] * z[k]

		for j in range(N[2]):
			B[0, :, j, :] -= 2.0 * y[j]

			for k in range(N[3]):
				B[1, :, j, k] -= y[j] * z[k]

		for i in range(N[1]):
			for j in range(N[2]):
				R = np.sqrt(x[i]**2 + y[j]**2)

				B[0, i, j, :] /= (2.0 * R**2)
				B[1, i, j, :] /= (2.0 * R**2)
				B[2, i, j, :]  = (R - 1) / (2.0 * R)
	else: # Default to const.TEST
		B[2, :, :, :] = 0.1

	return B
