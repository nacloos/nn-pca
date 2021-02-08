import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)


def simulate_projection(x, W, tau_fast=0.03, dt=0.01, n_iter=50, return_states=False):
	x_hat = np.zeros(W.shape[0]) 
	z = np.zeros(W.shape[1])

	if return_states:
		x_hat_states = np.zeros((n_iter, *x_hat.shape))
		z_states = np.zeros((n_iter, *z.shape))

	for k in range(n_iter):
		x_hat += dt/tau_fast*(-x_hat + W @ z)
		z += dt/tau_fast*( W.T @ (x - x_hat) )

		if return_states:
			x_hat_states[k] = x_hat; z_states[k] = z

	if return_states:
		return x_hat_states, z_states
	else:
		return x_hat, z


def compute_projection(x, W):
	z = np.linalg.solve(W.T@W, W.T @ x)
	x_hat = W @ z
	return x_hat, z


def simulate_pca(X, r, tau_fast=0.03, tau_slow=25, dt=0.01, data_update=50, n_iter=1000, return_states=False):
	# r: size of the latent variables z
	m, n = X.shape
	W = np.random.normal(size=(n, r))
	# normalize the columns of W
	W /= np.linalg.norm(W, axis=0)
	x_hat = np.zeros(n) 
	z = np.zeros(r)

	if return_states:
		W_states = np.zeros((n_iter, n, r))
		z_states = np.zeros((n_iter, r))
		x_hat_states = np.zeros((n_iter, n))
		x_states = np.zeros((n_iter, n))

	for k in range(n_iter):
		if k % data_update == 0:
			# change the input x
			x = X[np.random.randint(0, m)]
		x_hat += dt/tau_fast*(-x_hat + W @ z)
		z += dt/tau_fast*( W.T @ (x - x_hat) )
		# x_hat, z = compute_projection(x, W) # compute analytically the projection, used for tests

		W += dt/tau_slow*np.outer(x - x_hat, z)
		W /= np.linalg.norm(W, axis=0) # normalization of the columns of W, otherwise might be unstable


		if return_states:
			W_states[k] = W; z_states[k] = z; x_hat_states[k] = x_hat; x_states[k] = x

	if return_states:
		return W_states, z_states, x_hat_states, x_states
	else:
		return W

