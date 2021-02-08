import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from nn_pca import simulate_pca, simulate_projection


np.random.seed(5)

save_dir = "figures/"


def generate_data(m):
	v1 = 1/np.sqrt(2)*np.array([1, 1])
	v2 = 1/np.sqrt(2)*np.array([-1, 1])
	cov = 10*np.outer(v1, v1) + 1*np.outer(v2, v2)
	data = np.random.multivariate_normal(np.zeros(2), cov, m)
	return data


def animate(fig, W_states, z_states, x_hat_states, x_states, save_name=""):
	n_iter = W_states.shape[0]

	W_line, = plt.plot([], [], color="coral", label="$w$", linewidth=2)
	x_line, = plt.plot([], [], color="black", marker=".", linestyle="", label="$x$")
	x_hat_line, = plt.plot([], [], color="black", marker="x", linestyle="", label="$\\hat{x}$")

	def animate_fun(i):
		W_line.set_data([0, W_states[i,0,0]], [0, W_states[i,1,0]])
		x_line.set_data(x_states[i,0], x_states[i,1])
		x_hat_line.set_data(x_hat_states[i,0], x_hat_states[i,1])
		return W_line, x_line, x_hat_line,
  
	anim = animation.FuncAnimation(fig, animate_fun, frames=n_iter, interval=20, blit=True)
	plt.legend()

	if save_name != "":
		anim.save("{}{}.gif".format(save_dir, save_name), writer="imagemagick")

	plt.show()


def plot_data(X, save_name=""):
	eival, eivec = np.linalg.eig(1/n*X.T@X)
	# sort in descending order
	idx = np.argsort(eival)[::-1]
	eival = eival[idx]
	eivec = eivec[:,idx]

	plt.scatter(X[:,0], X[:,1], marker=".", alpha=0.5, color="cornflowerblue")
	plt.plot([0, 2*eivec[0,0]], [0,2*eivec[1,0]], color="coral", label="$v_1$")
	plt.plot([0, 2*eivec[0,1]], [0,2*eivec[1,1]], color="coral", linestyle="dotted", label="$v_2$")
	plt.axis('equal')
	plt.axis('off')
	plt.legend()

	if save_name:
		plt.savefig("{}{}.png".format(save_dir, save_name), transparent=True)


def plot_animation(X, tau_slow, save_name=""):		
	eival, eivec = np.linalg.eig(1/n*X.T@X)
	# sort in descending order
	idx = np.argsort(eival)[::-1]
	eival = eival[idx]
	eivec = eivec[:,idx]

	W = eivec[:,0].reshape(2, 1)

	plt.scatter(X[:,0], X[:,1], marker=".", alpha=0.4, color="cornflowerblue")
	plt.plot([0, 2*eivec[0,0]], [0,2*eivec[1,0]], color="lightgray", label="$v_1$", linewidth=2)
	plt.plot([0, 2*eivec[0,1]], [0,2*eivec[1,1]], color="lightgray", linestyle="dotted", label="$v_2$", linewidth=2)
	plt.axis('equal')
	plt.axis('off')
	plt.legend()

	states = simulate_pca(X, 1, tau_slow=tau_slow, return_states=True)
	W_states, z_states, x_hat_states, x_states = states
	W_states *= 2
	states = W_states, z_states, x_hat_states, x_states
	animate(fig, *states, save_name=save_name)



n = 2
m = 100
X = generate_data(m) # m x n


fig = plt.figure(figsize=(6,4), dpi=130)
plt.title("PCA on Gaussian data")
plot_data(X, save_name="pca_gaussian")


fig = plt.figure(figsize=(6,4), dpi=110)
tau_slow = 60
plt.title("$\\tau_{{slow}}={}$".format(tau_slow))
# plot_animation(X, tau_slow, save_name="tau_slow_good")
plot_animation(X, tau_slow)


fig = plt.figure(figsize=(6,4), dpi=110)
tau_slow = 5
plt.title("$\\tau_{{slow}}={}$".format(tau_slow))
# plot_animation(X, tau_slow, save_name="tau_slow_too_fast")
plot_animation(X, tau_slow)

