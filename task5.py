import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

Ys_test = []
with open("RSSI-measurements-unknown-sigma.txt", "r") as f:
    for line in f.readlines():
        Ys_test.append([float(x) for x in line.strip().split(",")])
Ys_test = np.array(Ys_test)


# Setup constants
dt = 0.5
alpha = 0.6
v = 90
eta = 3
sigma = 0.5

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


def y_given_x_likelihood_vec(particles_X, Y_obs, varsigma):
    pos1 = particles_X[0, :]
    pos2 = particles_X[3, :]
    
    dists = np.sqrt((stations_x1.reshape(6, 1) - pos1.reshape(1, -1))**2 + (stations_x2.reshape(6, 1) - pos2.reshape(1, -1))**2)  # shape (6, N)
    mu = v - 10 * eta * np.log10(dists) # shape (6, N)
    
    pdf_vals = norm.pdf(Y_obs, loc=mu, scale= varsigma)  # shape (6, N)
    likelihoods = np.prod(pdf_vals, axis=0)  # shape (N)
    return likelihoods

def SISR_vec(Ys, varsigma, num_particles=10000):
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
    likelihoods = y_given_x_likelihood_vec(particles_X, Ys[:, 0:1], varsigma) # 0:1 to not flatten
    weights = likelihoods
    
    tau1_n[0] = np.sum(weights * particles_X[0, :])
    tau2_n[0] = np.sum(weights * particles_X[3, :])
    log_c_n = [-np.log(num_particles)]
    
    for n in range(1, m):
        # Redsample particles
        chosen_particles = np.random.choice([*range(num_particles)], size=num_particles, p=weights/np.sum(weights))
        new_particles_X = np.zeros((6, num_particles))
        new_particles_Z = np.zeros((5, num_particles))
        for i in range(num_particles):
            new_particles_X[:,i] = particles_X[:,chosen_particles[i]]
            new_particles_Z[:,i] = particles_Z[:,chosen_particles[i]]
        particles_X = new_particles_X
        particles_Z = new_particles_Z

        particles_X, particles_Z = updateX_tilde_vec(particles_X, particles_Z)
        
        # Update weights
        likelihoods = y_given_x_likelihood_vec(particles_X, Ys[:, n:n+1], varsigma) # n:n+1 to not flatten
        weights = likelihoods
        
        tau1_n[n] = np.sum(weights / np.sum(weights) * particles_X[0, :])
        tau2_n[n] = np.sum(weights / np.sum(weights) * particles_X[3, :])
        log_c_n.append(log_c_n[-1] - np.log(num_particles) + np.log(np.sum(weights)))
    
    return tau1_n, tau2_n, (1/m) * log_c_n[-1]

    
def main():
    num_particles = 10_000
    xs = np.linspace(0.5,2.9,20)
    ys = []
    best_tau1 = None
    best_tau2 = None
    best_l = None
    for x in xs:
        tau1_est, tau2_est, likelihood = SISR_vec(Ys=Ys_test, varsigma = x, num_particles=num_particles)
        ys.append(likelihood)
        if best_l is None or likelihood > best_l:
            best_tau1 = tau1_est
            best_tau2 = tau2_est
            best_l = likelihood

    print("best:", xs[ys.index(best_l)])
    plt.plot(xs, ys)
    plt.xlabel(r"$\varsigma$")
    plt.ylabel(r"$\ell(\varsigma)$")
    plt.title("Likelihood function")
    plt.show()

    plt.figure()
    plt.scatter(stations_x1, stations_x2, marker="*", color="C1", label="Stations")
    plt.plot(best_tau1, best_tau2, label="SISR Estimated Trajectory", color="r")
    plt.xlabel("X1 (position)")
    plt.ylabel("X2 (position)")
    plt.legend()
    plt.axis("equal")
    plt.title("SISR Estimated Target Trajectory")
    plt.show()

if __name__ == "__main__":
    main()