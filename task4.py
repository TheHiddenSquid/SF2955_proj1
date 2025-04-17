import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

Ys_test = []
with open("RSSI-measurements.txt", "r") as f:
    for line in f.readlines():
        Ys_test.append([float(x) for x in line.strip().split(",")])
Ys_test = np.array(Ys_test)


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

def calcY(X_tilde):
    Ys = np.zeros((6,1))
    for i in range(6):
        V = np.random.normal(0, 1.5)
        Ys[i,0] = v - 10*eta*np.log10(np.sqrt((X_tilde[0]-stations_x1[i])**2 + (X_tilde[3]-stations_x2[i])**2)) + V
	
    return Ys

def y_given_x_likelihood_vec(particles_X, Y_obs):
    pos1 = particles_X[0, :]
    pos2 = particles_X[3, :]
    
    dists = np.sqrt((stations_x1.reshape(6, 1) - pos1.reshape(1, -1))**2 + (stations_x2.reshape(6, 1) - pos2.reshape(1, -1))**2)  # shape (6, N)
    mu = v - 10 * eta * np.log10(dists) # shape (6, N)
    
    pdf_vals = norm.pdf(Y_obs, loc=mu, scale=1.5)  # shape (6, N)
    likelihoods = np.prod(pdf_vals, axis=0)  # shape (N)
    return likelihoods

def SISR_vec(Ys, num_particles=10000):
    m = Ys.shape[1]
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
    weights = y_given_x_likelihood_vec(particles_X, Ys[:, 0:1]) # 0:1 to not flatten
    
    tau1_n[0] = np.sum(weights/np.sum(weights) * particles_X[0, :])
    tau2_n[0] = np.sum(weights/np.sum(weights) * particles_X[3, :])
    
    for n in range(1, m):
        # Redsample particles
        chosen_particles = np.random.choice([*range(num_particles)], size=num_particles, p=weights / np.sum(weights))
        new_particles_X = np.zeros((6, num_particles))
        new_particles_Z = np.zeros((5, num_particles))
        for i in range(num_particles):
            new_particles_X[:,i] = particles_X[:,chosen_particles[i]]
            new_particles_Z[:,i] = particles_Z[:,chosen_particles[i]]
        particles_X = new_particles_X
        particles_Z = new_particles_Z

        particles_X, particles_Z = updateX_tilde_vec(particles_X, particles_Z)
        
        # Update weights
        weights = y_given_x_likelihood_vec(particles_X, Ys[:, n:n+1]) # n:n+1 to not flatten
        
        tau1_n[n] = np.sum(weights/np.sum(weights) * particles_X[0, :])
        tau2_n[n] = np.sum(weights/np.sum(weights) * particles_X[3, :])
    
    return tau1_n, tau2_n

def generate_XY_pair(m):
    X0 = np.random.multivariate_normal([0]*6, np.diag([500,5,5,200,5,5])).reshape(6,1)
    Z0 = np.zeros((5,1))
    Z0[np.random.randint(5)] = 1
	
    X = X0.copy()
    Z = Z0.copy()
    
    x1s = [X[0]]
    x2s = [X[3]]

    Ys = np.zeros((6, m))
    Ys[:, 0:1] = calcY(X)
	
    for j in range(m-1):
        X, Z = updateX_tilde_vec(X,Z)
        x1s.append(X[0])
        x2s.append(X[3])
        Ys[:,j+1:j+2] = calcY(X)

    return x1s, x2s, Ys
    
def main():
    np.random.seed(11)

    m = 250
    x1s, x2s, Ys_known = generate_XY_pair(m)

    num_particles = 10_000
    tau1_est, tau2_est = SISR_vec(Ys=Ys_test, num_particles=num_particles)    # Swap between Ys_test and Ys_known

    plt.figure()
    plt.scatter(stations_x1, stations_x2, marker="*", color="C1", label="Stations")
    #plt.plot(x1s, x2s, label="True Trajectory", color="b")
    plt.plot(tau1_est, tau2_est, label="SISR Estimated Trajectory", color="r")
    plt.xlabel("X1 (position)")
    plt.ylabel("X2 (position)")
    plt.legend()
    plt.axis("equal")
    plt.title("SISR Estimated Target Trajectory")
    plt.show()

if __name__ == "__main__":
    main()