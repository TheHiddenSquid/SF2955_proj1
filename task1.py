import numpy as np



def update(oldX, oldZ, P, phi, psi_z, psi_w):
	pass


def main():
	dt = 0.5
	alpha = 0.6
	sigma = 0.5

	P = (1/20)*(15*np.eye(4)+np.ones((5,5)))
	phi = np.zeros((6,6))
	phi_tilde = np.array([[1,dt,dt**2/2],[0,1,dt],[0,0,alpha]])
	phi[0:3,0:3]=phi_tilde
	phi[3:,3:]=phi_tilde

	psi_z = np.zeros((6,2))
	psi_w = np.zeros(6,2)
	psi_z_tilde = np.array([[dt**2/2],[dt],[0]])
	psi_w_tilde = np.array([[dt**2/2],[dt],[1]])
	psi_z[0:3,0] = psi_z_tilde
	psi_z[3:,1] = psi_z_tilde
	psi_w[0:3,0] = psi_w_tilde
	psi_w[3:,1] = psi_w_tilde

	
	X0 = np.random.multivariate_normal(np.zeros((6,1)), np.diag([500,5,5,200,5,5]))
	Z0 = np.random.randrange(6)

	X, Z = update()

if "__name__" == "__main__":
	main()

