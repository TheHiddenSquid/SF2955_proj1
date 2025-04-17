import matplotlib.pyplot as plt
import numpy as np

# Setup of constants
dt = 0.5
alpha = 0.6
sigma = 0.5

P = (1/20)*(15*np.eye(5)+np.ones((5,5)))
phi = np.zeros((6,6))
phi_tilde = np.array([[1,dt,dt**2/2],[0,1,dt],[0,0,alpha]])
phi[0:3,0:3]=phi_tilde
phi[3:,3:]=phi_tilde

psi_z = np.zeros((6,2))
psi_z_tilde = np.array([[dt**2/2,dt,0]]).T
psi_z[0:3,0:1] = psi_z_tilde
psi_z[3:,1:] = psi_z_tilde

psi_w = np.zeros((6,2))
psi_w_tilde = np.array([[dt**2/2,dt,1]]).T
psi_w[0:3,0:1] = psi_w_tilde
psi_w[3:,1:] = psi_w_tilde

dc_lookup = np.array([[0, 3.5, 0, 0, -3.5], [0, 0, 3.5, -3.5, 0]])


def update(oldX, oldZ):
	W = np.random.multivariate_normal([0]*2, sigma**2 * np.eye(2)).reshape(2,1)

	newX = phi @ oldX + psi_z @ dc_lookup @ oldZ + psi_w @ W

	newZ = np.zeros((5,1))
	newZ[np.random.choice(a=[0,1,2,3,4], p=(P @ oldZ).flatten())] = 1

	return newX, newZ


def main():
	np.random.seed(14)

	m = 200

	# Plot 5 random partickles
	for _ in range(5):
		X0 = np.random.multivariate_normal([0]*6, np.diag([500,5,5,200,5,5])).reshape(6,1)
		Z0 = np.zeros((5,1))
		Z0[np.random.randint(5)] = 1
		
		X = X0.copy()
		Z = Z0.copy()

		x1s = [X[0]]
		x2s = [X[3]]

		for _ in range(m):
			X, Z = update(X, Z)
			x1s.append(X[0])
			x2s.append(X[3])

		plt.scatter(X0[0], X0[3])
		plt.plot(x1s, x2s)

	plt.axis("off")
	plt.show()


if __name__ == "__main__":
	main()

