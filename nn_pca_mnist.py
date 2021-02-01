import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from nn_pca import simulate_pca, simulate_projection


def project(V, X):
	# Compute the orthogonal projections of the columns of X onto the columns of V (not necessarily orthogonal)
	return np.linalg.solve(V.T@V, V.T@X)


def plot_projected_data(V, X):
	# plot the projection of the data on the columns of V
	Z = project(V, X.T) 	# 2xm
	
	data_lines = [matplotlib.lines.Line2D for _ in range(Z.T.shape[0])]

	for i, z in enumerate(Z.T):
		data_lines[i], = plt.plot(z[0], z[1], color="cornflowerblue" if labels[i]==0 else "yellowgreen", linestyle="", marker=".", alpha=0.2)


	plt.plot([0, 1/3], [0, 0], color="lightgray", label="$v_1$")
	plt.plot([0, 0], [0, 1/3], color="lightgray", linestyle="dotted", label="$v_2$")

	zero_handle = plt.scatter([], [], color="cornflowerblue", label="0", marker=".")
	one_handle = plt.scatter([], [], color="yellowgreen", label="1", marker=".")
	plt.legend(handles=[zero_handle, one_handle])
	plt.legend()
	plt.axis('equal')
	return data_lines


def animate_projection(fig, x, z_states, save_name=""):
	n_iter = z_states.shape[0]
	z_line, = plt.plot([], [], color="yellowgreen", marker="x", linestyle="", label="$z$")

	def animate_fun(i):
		z_line.set_data(z_states[i,0], z_states[i,1])
		return z_line,
  
	anim = animation.FuncAnimation(fig, animate_fun, frames=n_iter, interval=50, blit=True)
	plt.legend()
	plt.show()

	if save_name != "":
		anim.save("{}.gif".format(save_name), writer="imagemagick")



def animate_pca(fig, X, V, W_states, z_states, x_hat_states, x_states, save_name=""):
	# project the states on V to visualize them 
	n_iter = W_states.shape[0]
	data_lines = plot_projected_data(V[0], X)
	W_line1, = plt.plot([], [], color="coral", label="$w_1$")
	W_line2, = plt.plot([], [], color="coral", linestyle="dotted", label="$w_2$")
	x_line, = plt.plot([], [], color="black", marker=".", linestyle="", label="$x$")
	x_hat_line, = plt.plot([], [], color="black", marker="x", linestyle="", label="$\\hat{x}$")

	def animate_fun(i):
		# TODO: update scatter for zero digits and one digits, instead of for loop

		# plot the data project onto the columns of V
		proj_data = project(V[i], X.T).T
		for j, data_line in enumerate(data_lines):
			data_line.set_data(proj_data[j,0], proj_data[j,1])

		W = project(V[i], W_states[i]) / 3 # divide by 3 just to better visualize it
		W_line1.set_data([0, W[0,0]], [0, W[1,0]])
		W_line2.set_data([0, W[0,1]], [0, W[1,1]])

		x_line.set_data(project(V[i], x_states[i])[0], project(V[i], x_states[i])[1])
		x_hat_line.set_data(project(V[i], x_hat_states[i])[0], project(V[i], x_hat_states[i])[1])

		lines = data_lines + [W_line1, W_line2, x_line, x_hat_line]
		return lines
		# x_line.set_data(x_states[i,0], x_states[i,1])
		# x_hat_line.set_data(x_hat_states[i,0], x_hat_states[i,1])
		# return W_line, x_line, x_hat_line,
  
	anim = animation.FuncAnimation(fig, animate_fun, frames=n_iter, interval=20, blit=True)
	plt.legend()
	plt.axis('equal')
	plt.show()

	if save_name != "":
		# anim.save("{}.html".format(save_name))
		anim.save("{}.gif".format(save_name), writer="imagemagick")




data_path = "mnist/"
# train_data = np.loadtxt(data_path + "mnist_train.csv", delimiter=",")
# data = np.loadtxt(data_path + "mnist_test.csv", delimiter=",", max_rows=1000) 
data = np.loadtxt(data_path + "mnist_train.csv", delimiter=",", max_rows=5000) 
data = data[np.logical_or(data[:,0] == 0, data[:,0] == 1)]
labels = data[:,0] 	# mx1
X = data[:,1:]		# mxn
print(X.shape)

# center data
X = X - np.mean(X, axis=0)
# max norm of 1
X = X / np.max(np.linalg.norm(X, axis=1))




# project the 2 columns of W onto the 2 principal components V, look at the norm of the projection to know whether span<W> = span<V>
# project the reconstruction x_hat onto the 2 principal components
# if span<W> = span<V>, the projection of x_hat and x should match




eival, eivec = np.linalg.eig(1/X.shape[0] * X.T@X)
# plt.figure()
# plt.scatter(np.arange(10), eival[:10])

V = np.real(eivec[:,:2]) # nx2

fig = plt.figure()
# plot_projected_data(V, X)

# x = X[0]
# x_hat_states, z_states = simulate_projection(x, V, tau=3, dt=1, n_iter=50, return_states=True)
# plt.scatter((V.T@x)[0], (V.T@x)[1], color="cornflowerblue")
# animate_projection(fig, x, z_states)

# n_iter = 20000
n_iter = 10000
states = simulate_pca(X, 2, tau=3, tau_W=200, n_update=50, dt=1, n_iter=n_iter, return_states=True)
states = [s[-500:] for s in states]
animate_pca(fig, X, np.broadcast_to(V, (n_iter, *V.shape)), *states)

