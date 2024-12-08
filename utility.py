import numpy as np
import constants as const
import em_field as em
import matplotlib.pyplot as plt


def perp(
		x: np.ndarray,
	 	y: np.ndarray
	) -> np.ndarray:
	"""Returns the component of ``x`` that is perpendicular to ``y``

	Args:
		x (np.ndarray): Vector
		y (np.ndarray): Vector

	Returns:
		np.ndarray: Vector projection
	"""

	return (x - x * (np.dot(x, y) / np.dot(x, x)))


def init(
		type: int,
		qm  : float = const.QE / const.ME
	) -> np.ndarray:
	"""Returns the initial state vector ``X0`` for a given field type

	Args:
		type (int): An algorithm flag from ``constants``
		qm (float, optional): Particle mass. Only needed when ``type=constants.TEST``. Defaults to ``constants.QE / constants.ME``.

	Returns:
		tuple: Initial state vector ``[x0, y0, z0, u0, v0, w0]``
	"""

	if type == const.STATIC:
		x0 =  0.0  # m
		y0 = -1.0  # m
		z0 =  0.0  # m
		u0 =  0.1  # m / s
		v0 =  0.01 # m / s
		w0 =  0.0  # m / s
	elif type == const.BANANA or type == const.TRANSIT:
		x0 =  1.05     # m
		y0 =  0.0      # m
		z0 =  0.0      # m
		u0 =  0.0      # m / s
		v0 =  4.816e-4 # m / s
		w0 = -2.059e-3 # m / s

		if type == const.TRANSIT:
			v0 *= 2.0
	else: # Default to const.TEST
		x0 = 1.0 # m
		y0 = 0.0 # m
		z0 = 0.0 # m
		u0 = 0.0 # m / s
		v0 = 1e6 # m / s
		w0 = 0.0 # m / s

		while True:
			B = em.Bfield1(x0, y0, z0, type)
			Bmag = np.linalg.norm(B)

			w_L = np.abs(qm) * Bmag # Larmor pulsation [rad/s]
			r_L = np.linalg.norm(perp(np.array((u0, v0, w0)), B)) / w_L # Larmor radius [m]

			x0 = r_L

			if error(B, em.Bfield1(x0, y0, z0, type)) <= 1e-6:
				break

	return np.array((x0, y0, z0, u0, v0, w0))


def time(
		type   : int,
		qm     : float = const.QE / const.ME,
		periods: int   = 10
	) -> np.ndarray:
	"""Returns the time steps ``time`` for a given field type

	Args:
		type (int): An algorithm flag from ``constants``
		qm (float, optional): Particle mass. Only needed when ``type=constants.TEST``. Defaults to ``constants.QE / constants.ME``.
		periods (int, optional): Number of time periods. Only needed when ``type=constants.TEST``. Defaults to 10.

	Returns:
		np.ndarray: Vector of time steps ``[time0, time1, ..., time_n]``
	"""

	if type == const.STATIC:
		h     = np.pi / 10.0
		steps = int(5e5)
	elif type == const.BANANA or type == const.TRANSIT:
		h     = np.pi / 10.0
		steps = int(5e4)
	else: # Default to const.TEST
		x, y, z, vx, vy, vz = init(type)

		B     = em.Bfield1(x, y, z, type)
		Bmag  = np.linalg.norm(B)

		w_L   = np.abs(qm) * Bmag # Larmor pulsation [rad/s]
		tau_L = 2.0 * np.pi / w_L # Larmor period [s]

		h     = tau_L / 25.0 # has to lie in (0, tau_L / 2.0] for stability, results matche analytical for (0, tau_L / 25.0)
		steps = periods * 25

	time = np.linspace(0.0, h * steps, steps) # s

	return time


def interp(
		point : np.ndarray,
		points: tuple,
		field : np.ndarray,
		uniform: bool = True
	) -> np.ndarray:
	"""Trillinear interpolation of a vector ``field`` on a periodic cubic grid. Extrapolates when ``point`` is outside the bounds of ``points``.

	Args:
		point (np.ndarray): Point at which to interpolate to
		points (tuple): Tuple of X-, Y- & Z-Coordinates for ``field``
		field (np.ndarray): Vector field to be interpolated of size 3 x N1 x N2 x N3 ``[field_x, field_y, field_z]``
		uniform (bool): Flag indicating if the grid is unifrom. Defaults to ``True``.

	Returns:
		np.ndarray: Interpolated value
	"""

	x, y, z = point
	xx = points[0]
	yy = points[1]
	zz = points[2]

	# Locate cell that contains the given ``point``
	if uniform:
		dx = xx[1] - xx[0]
		dy = xx[1] - xx[0]
		dz = xx[1] - xx[0]

		x0 = np.clip(int((x - xx[0]) / dx), 0, np.size(xx) - 2)
		x1 = np.clip(x0 + 1, 1, np.size(xx) - 1)

		y0 = np.clip(int((y - yy[0]) / dy), 0, np.size(yy) - 2)
		y1 = np.clip(y0 + 1, 1, np.size(yy) - 1)

		z0 = np.clip(int((z - zz[0]) / dz), 0, np.size(zz) - 2)
		z1 = np.clip(z0 + 1, 1, np.size(zz) - 1)
	else:
		x0 = 0
		x1 = np.size(xx) - 1
		for i in range(1, np.size(xx)):
			if xx[i] > x:
				x0 = i - 1
				x1 = i
				break

		y0 = 0
		y1 = np.size(xx) - 1
		for i in range(1, np.size(yy)):
			if yy[i] > y:
				y0 = i - 1
				y1 = i
				break

		z0 = 0
		z1 = np.size(zz) - 1
		for i in range(1, np.size(zz)):
			if zz[i] > z:
				z0 = i - 1
				z1 = i
				break

	# Normalized distances from cell vertices
	xd = (x - xx[x0]) / (xx[x1] - xx[x0])
	yd = (y - yy[y0]) / (yy[y1] - yy[y0])
	zd = (z - zz[z0]) / (zz[z1] - zz[z0])

	# Field values at cell vertices
	f000 = field[:, x0, y0, z0]
	f001 = field[:, x0, y0, z1]
	f010 = field[:, x0, y1, z0]
	f011 = field[:, x0, y1, z1]
	f100 = field[:, x1, y0, z0]
	f101 = field[:, x1, y0, z1]
	f110 = field[:, x1, y1, z0]
	f111 = field[:, x1, y1, z1]

	# Perform successive linear interpolation along the x-, y- and z-axes.
	f00 = f000 * (1 - xd) + f100 * xd
	f01 = f001 * (1 - xd) + f101 * xd
	f10 = f010 * (1 - xd) + f110 * xd
	f11 = f011 * (1 - xd) + f111 * xd

	f0  =  f00 * (1 - yd) +  f10 * yd
	f1  =  f01 * (1 - yd) +  f11 * yd

	f   =   f0 * (1 - zd) +   f1 * zd

	return f


def error(
	 	y       : float | np.ndarray,
		y_ref   : float | np.ndarray,
	  	relative: bool = True,
		vector  : bool = True
	) -> np.floating:
	"""Calculate error

	Args:
		y (np.ndarray): Vector to check the error for
		y_ref (np.ndarray): Vector containing reference values
		relative (bool, optional): Whether to calculate relative error. Defaults to True.

	Returns:
		np.ndarray: Absolute value of maximum error (components of vector, or magnitude of vector & scalar).
	"""

	if vector:
		mag_ref = np.linalg.norm(y_ref)
		mag     = np.linalg.norm(y)
		err     = np.abs(y_ref - y)
		mag_err = np.abs(mag_ref - mag)

		if relative:
			if not np.any(np.isclose(y_ref, np.zeros(3), atol=0.0)):
				err /= y_ref
			if not np.any(np.isclose(mag_ref, np.zeros(3), atol=0.0)):
				err /= mag_ref

		err = np.max(err)
		err = np.max((err, mag_err))
	else:
		err = np.abs(y_ref - y)

		if relative and not np.isclose(y_ref, 0.0, atol=0.0):
				err /= y_ref

	return err # type: ignore


def post(
		X      : np.ndarray,
		time   : np.ndarray,
		points : tuple,
		Bfield : np.ndarray,
		test   : bool=False,
		inv_an: np.ndarray=np.zeros(0)
	) -> tuple:
	"""_summary_

	Args:
		X (np.ndarray): _description_
		time (np.ndarray): _description_
		points (tuple): _description_
		Bfield (np.ndarray): _description_
		test (bool): Flag to indicate if plotting errors for the ``constants.TEST`` case. Defaults to False.
		inv_an (np.ndarray): Invariants for the analytical solution. Needs to be supplied if ``test == True``. Defaults to an empty array.

	Returns:
		_type_: _description_
	"""

	timespan = np.size(time)

	inv = np.zeros((3, timespan)) # 0 == H, 1 == mu, 2 == p
	err = np.zeros((3, timespan)) # 0 == H, 1 == mu, 2 == p

	for t in range(timespan):
		x = X[t, 0:3]
		v = X[t, 3:6]

		R      = np.sqrt(x[0]**2 + x[1]**2)
		B      = interp(x, points, Bfield)
		v_perp = np.linalg.norm(perp(v, B))
		xi_dot = np.linalg.norm(np.cross(x, v) / np.linalg.norm(x)**2)

		inv[0, t] = np.dot(v, v) / 2.0 + 1e-2 / R
		inv[1, t] = v_perp**2 / (2 * R)
		inv[2, t] = R**2 * xi_dot + R**3 / 3

	if test: # If Test case, calculate errors against analytical solution
		for t in range(timespan):
			err[0, t] = np.abs(inv[0, t] / inv_an[0, t] - 1)
			err[1, t] = np.abs(inv[1, t] / inv_an[1, t] - 1)
			err[2, t] = np.abs(inv[2, t] / inv_an[2, t] - 1) # == 0 for t = 1
	else: # Otherwise, calculate errors against initial value (since they are all invariant with time)
		for t in range(timespan):
			err[0, t - 1] = np.abs(inv[0, t] / inv[0, 0] - 1)
			err[1, t - 1] = np.abs(inv[1, t] / inv[1, 0] - 1)
			err[2, t - 1] = np.abs(inv[2, t] / inv[2, 1] - 1) # == 0 for t = 1

		err = err[:, :timespan - 1] # Trim off the last element

	return (inv, err)


def plot(
		X      : np.ndarray,
		time   : np.ndarray,
		post   : np.ndarray,
		err    : np.ndarray,
		prepend: str="",
		append : str="",
		pdf    : bool=True,
	) -> None:
	"""Generate and save plots. Filenames will be '{prepend}_type_{append}.{png|pdf}'.

	Args:
		X (np.ndarray): State vector of the particle
		time (np.ndarray): Vector of time cooridnates of size ``N``
		post (np.ndarray): Vector of size ``3 x N`` containing the invariants of the system (Energy, Mangetic Moment, Angular Momentum)
		err (np.ndarray): Vector of size ``3 x (N - 1)`` containing the realtive errors of the invariants (relative to value at t=0)
		r_L (float): Larmor radius
		append (str): String to prepend to filename. Defaults to an empty string.
		append (str): String to append to filename. Defaults to an empty string.
		pdf (bool): Flag to indicate if a vector graphics .pdf is generated. Defaults to True.
	"""

	file_append = f'{append}.{'pdf' if pdf else 'png'}'
	tmax = time[-1]
	tnum = np.size(err[0, :])

	fig = plt.figure(1)
	fig, axes = plt.subplots()
	axes.grid(True)
	axes.axis('equal')
	axes.set_xlabel(r'$z$')
	axes.set_ylabel(r'$R$')
	axes.ticklabel_format(axis='both', style='sci', scilimits=(0, 0))
	# axes.plot(X[:, 0], X[:, 1], 'k-', linewidth=1)
	axes.plot(np.sqrt(X[:, 0]**2 + X[:, 1]**2), X[:, 2], 'k-', linewidth=1)
	fig.savefig(f'{prepend}_trajectory_{file_append}', bbox_inches='tight', dpi=300)
	fig.clear()

	fig = plt.figure(2)
	fig, axes = plt.subplots()
	axes.grid(True)
	axes.set_xlim(0, tmax)
	# axes.set_ylim(top=1)
	axes.set_xlabel('Time [s]')
	axes.set_ylabel('Error')
	axes.ticklabel_format(axis='both', style='sci', scilimits=(0, 0))
	axes.set_yscale('log')
	axes.plot(time[:tnum], err[0, :], 'k-', linewidth=1)
	fig.savefig(f'{prepend}_err_H_{file_append}', bbox_inches='tight', dpi=300)
	fig.clear()

	fig = plt.figure(3)
	fig, axes = plt.subplots()
	axes.grid(True)
	axes.set_xlim(0, tmax)
	# axes.set_ylim(top=1)
	axes.set_xlabel('Time [s]')
	axes.set_ylabel('Error')
	axes.ticklabel_format(axis='both', style='sci', scilimits=(0, 0))
	axes.set_yscale('log')
	axes.plot(time[:tnum], err[1, :], 'k-', linewidth=1)
	fig.savefig(f'{prepend}_err_mu_{file_append}', bbox_inches='tight', dpi=300)
	fig.clear()

	fig = plt.figure(4)
	fig, axes = plt.subplots()
	axes.grid(True)
	axes.set_xlim(0, tmax)
	# axes.set_ylim(top=1)
	axes.set_xlabel('Time [s]')
	axes.set_ylabel('Error')
	axes.ticklabel_format(axis='both', style='sci', scilimits=(0, 0))
	axes.set_yscale('log')
	axes.plot(time[:tnum], err[2, :], 'k-', linewidth=1)
	fig.savefig(f'{prepend}_err_p_{file_append}', bbox_inches='tight', dpi=300)
	fig.clear()
