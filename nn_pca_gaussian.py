import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


from nn_pca import simulate_pca, simulate_projection

np.random.seed(5)

n = 2
m = 100


def generate_data(m):
	v1 = 1/np.sqrt(2)*np.array([1, 1])
	v2 = 1/np.sqrt(2)*np.array([-1, 1])
	cov = 10*np.outer(v1, v1) + 1*np.outer(v2, v2)
	data = np.random.multivariate_normal(np.zeros(2), cov, m)
	return data


def animate(fig, W_states, z_states, x_hat_states, x_states, save_name=""):
	n_iter = W_states.shape[0]

	W_line, = plt.plot([], [], color="darkorange", label="$w$", linewidth=2)
	x_line, = plt.plot([], [], color="cornflowerblue", marker=".", linestyle="", label="$x$")
	x_hat_line, = plt.plot([], [], color="yellowgreen", marker="x", linestyle="", label="$\\hat{x}$")

	def animate_fun(i):
		W_line.set_data([0, W_states[i,0,0]], [0, W_states[i,1,0]])
		x_line.set_data(x_states[i,0], x_states[i,1])
		x_hat_line.set_data(x_hat_states[i,0], x_hat_states[i,1])
		return W_line, x_line, x_hat_line,
  
	anim = animation.FuncAnimation(fig, animate_fun, frames=n_iter, interval=20, blit=True)
	plt.legend()
	plt.show()

	if save_name != "":
		# anim.save("{}.html".format(save_name))
		anim.save("{}.gif".format(save_name), writer="imagemagick")


def plot_data(X):
	eival, eivec = np.linalg.eig(1/n*X.T@X)

	fig = plt.figure()
	# plt.plot([0, W[0,0]], [0,W[1,0]], color="tab:red", label="$W$")
	plt.scatter(X[:,0], X[:,1], marker=".", alpha=0.2, color="cornflowerblue")
	plt.plot([0, 2*eivec[0,0]], [0,2*eivec[1,0]], color="tab:orange", label="$v_1$")
	plt.plot([0, 2*eivec[0,1]], [0,2*eivec[1,1]], color="tab:orange", linestyle="dotted", label="$v_2$")
	plt.axis('equal')
	plt.legend()
	plt.show()


def plot_animation(X):		
	eival, eivec = np.linalg.eig(1/n*X.T@X)
	W = eivec[:,0].reshape(2, 1)

	fig = plt.figure(dpi=130)
	plt.scatter(X[:,0], X[:,1], marker=".", alpha=0.2, color="cornflowerblue")
	plt.plot([0, 2*eivec[0,0]], [0,2*eivec[1,0]], color="moccasin", label="$v_1$", linewidth=2)
	plt.plot([0, 2*eivec[0,1]], [0,2*eivec[1,1]], color="moccasin", linestyle="dotted", label="$v_2$", linewidth=2)
	plt.axis('equal')
	plt.legend()

	states = simulate_pca(X, 1, tau_W=30, return_states=True)
	W_states, z_states, x_hat_states, x_states = states
	W_states *= 2
	states = W_states, z_states, x_hat_states, x_states
	# animate(fig, *states, save_name="pca_anim")
	animate(fig, *states)




X = generate_data(m) # m x n
# plot_data(X)
plot_animation(X)