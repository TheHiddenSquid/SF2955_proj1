import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

from tkinter import Tk
from tkinter.filedialog import askopenfilename

Tk().withdraw()
file_path = askopenfilename(title="Select RSSI-measurements.txt")
Y_all = np.loadtxt(file_path, delimiter=",")
print(Y_all.shape)

# Setup constants
dt = 0.5
alpha = 0.6
sigma = 0.5
v = 90
eta = 3

# Transition matrix for discrete state Z (5x5)
P = (1/20) * (15 * np.eye(5) + np.ones((5, 5)))

# Define system matrices (phi, psi_z, psi_w)
phi = np.zeros((6, 6))
phi_tilde = np.array([[1, dt, dt**2/2], [0, 1, dt], [0, 0, alpha]])
phi[0:3, 0:3] = phi_tilde
phi[3:, 3:] = phi_tilde

psi_z = np.zeros((6, 2))
psi_z_tilde = np.array([[dt**2/2, dt, 0]]).T
psi_z[0:3, 0:1] = psi_z_tilde
psi_z[3:, 1:] = psi_z_tilde

psi_w = np.zeros((6, 2))
psi_w_tilde = np.array([[dt**2/2, dt, 1]]).T
psi_w[0:3, 0:1] = psi_w_tilde
psi_w[3:, 1:] = psi_w_tilde

# Driving command lookup (dc_lookup)
dc_lookup = np.array([[0, 3.5, 0, 0, -3.5],
                      [0, 0, 3.5, -3.5, 0]])

# Station locations
stations_x1 = np.array([0, 0, 3464.1, 3464.1, -3464.1, -3464.1])
stations_x2 = np.array([4000, -4000, 2000, -2000, -2000, 2000])

def updateX_tilde_vec(particles_X, particles_Z):

    N = particles_X.shape[1]
    
    W_matrix = np.random.multivariate_normal([0, 0], sigma**2 * np.eye(2), size=N).T # shape (2, N)
    
    new_particles_X = phi @ particles_X + psi_z @ (dc_lookup @ particles_Z) + psi_w @ W_matrix

    probabilities = P @ particles_Z  # shape (5, N)
    
    # Compute cdf (cpf)
    cum_probs = np.cumsum(probabilities, axis=0)  # shape (5, N)

    # Use cpfs for each particle to pick new Z
    u = np.random.rand(N)
    new_indices = np.argmax(cum_probs >= u, axis=0)

    new_particles_Z = np.zeros((5, N))
    new_particles_Z[new_indices, np.arange(N)] = 1

    return new_particles_X, new_particles_Z

def y_given_x_likelihood_vec(particles_X, Y_obs):
    pos1 = particles_X[0, :]
    pos2 = particles_X[3, :]
    
    dists = np.sqrt((stations_x1.reshape(6, 1) - pos1.reshape(1, -1))**2 + (stations_x2.reshape(6, 1) - pos2.reshape(1, -1))**2)  # shape (6, N)
    mu = v - 10 * eta * np.log10(dists) # shape (6, N)
    
    pdf_vals = norm.pdf(Y_obs, loc=mu, scale=1.5)  # shape (6, N)
    likelihoods = np.prod(pdf_vals, axis=0)  # shape (N)
    return likelihoods

def SIS_vec(num_particles=10000, m=200):
    particles_X = np.zeros((6, num_particles))
    particles_Z = np.zeros((5, num_particles))
    
    # prior distributions
    for i in range(num_particles):
        particles_X[:, i] = np.random.multivariate_normal(mean=[0]*6, cov=np.diag([500, 5, 5, 200, 5, 5]))
        
        state_vec = np.zeros(5)
        state_vec[np.random.randint(5)] = 1
        particles_Z[:, i] = state_vec
    
    tau1_n = np.zeros(m)
    tau2_n = np.zeros(m)
    
    # initial weights
    likelihoods = y_given_x_likelihood_vec(particles_X, Y_all[:, 0:1]) # 0:1 to not flatten
    weights = likelihoods / np.sum(likelihoods)
    
    tau1_n[0] = np.sum(weights * particles_X[0, :])
    tau2_n[0] = np.sum(weights * particles_X[3, :])
    
    for n in range(1, m):
        particles_X, particles_Z = updateX_tilde_vec(particles_X, particles_Z)
        
        # Update weights
        likelihoods = y_given_x_likelihood_vec(particles_X, Y_all[:, n:n+1]) # n:n+1 to not flatten
        weights = weights * likelihoods
        weights = weights / np.sum(weights)
        
        tau1_n[n] = np.sum(weights * particles_X[0, :])
        tau2_n[n] = np.sum(weights * particles_X[3, :])
    
    return tau1_n, tau2_n

def main():
    np.random.seed(13)
    m = 500

    num_particles = 10000
    tau1_est, tau2_est = SIS_vec(num_particles=num_particles, m=m)

    plt.figure()
    plt.scatter(stations_x1, stations_x2, marker="*", color="C1", label="Stations")
    plt.plot(tau1_est, tau2_est, label="SIS Estimated Trajectory", color="r")
    plt.xlabel("X1 (position)")
    plt.ylabel("X2 (position)")
    plt.legend()
    plt.axis("equal")
    plt.title("SIS Estimated Target Trajectory")
    plt.show()

if __name__ == "__main__":
    main()