# BASED ON Free-mode surface-wave computations
# P. W. Buchen, R. Ben-Hador 1996 FAST DELTA MATRIX method

import numpy as np
from scipy import optimize
from tqdm import tqdm
import warnings

np.seterr(invalid='ignore')
warnings.simplefilter("ignore", category=RuntimeWarning)

# Helper function for hyperbolic/trigonometric terms
def compute_terms(c, k, d, cp, cs):
    """
    Compute C and S terms for either P or S waves
    """
    if c < cp:
        r = np.sqrt(1 - c**2 / cp**2)
        C_alpha = np.cosh(k * r * d)
        S_alpha = np.sinh(k * r * d)
    else:
        r = np.sqrt(c**2 / cs**2 - 1)
        C_alpha = np.cos(k * r * d)
        S_alpha = 1j * np.sin(k * r * d)

    if c < cs:
        s = np.sqrt(1 - c**2 / cs**2)
        C_beta = np.cosh(k * s * d)
        S_beta = np.sinh(k * s * d)
    else:
        s = np.sqrt(c**2 / cs**2 - 1)
        C_beta = np.cos(k * s * d)
        S_beta = 1j * np.sin(k * s * d)

    return C_alpha, S_alpha, C_beta, S_beta, r, s



def compute_dispersion_fastdelta(c, omega, layers):

    # wavelength = (2 * np.pi * c) / omega
    k = omega / c  # wavenumber
    # k = 2 * np.pi / lambda_val

    t = np.zeros(len(layers))
    mu = np.zeros(len(layers))
    alpha = np.zeros(len(layers))
    beta = np.zeros(len(layers))
    rho = np.zeros(len(layers))
    thickness = np.zeros(len(layers))
    gamma = np.zeros(len(layers))

    for i, lay in enumerate(layers):
        thickness[i] = lay['d']
        alpha[i] = lay['alpha']
        beta[i] = lay['beta']
        rho[i] = lay['rho']
        t[i] = (c - c**2 / beta[i]**2)
        mu[i] = rho[i] * beta[i]**2
        gamma[i] = (beta[i] / c)**2

    t = 2 - c ** 2 / beta[0] ** 2

    # initialise
    X1 = mu[0] ** 2 * np.array([2 * t, -t**2, 0, 0, -4])

    # terms for half-space
    _, _, _, _, r_h, s_h = compute_terms(c, k, thickness[-1], alpha[-1], beta[-1])


    for i, lay in enumerate(layers[:-1]):

        C_alpha, S_alpha, C_beta, S_beta, r, s = compute_terms(c, k, thickness[i], alpha[i], beta[i])

        epsilon = rho[i+1] / rho[i]
        eta = 2 * (gamma[i] - epsilon * gamma[i+1])

        a = epsilon + eta
        a_prime = a - 1
        b = 1 - eta
        b_prime = b - 1

        x1 = X1[0]
        x2 = X1[1]
        x3 = X1[2]
        x4 = X1[3]
        x5 = X1[4]

        p1 = C_beta * x2 + s * S_beta * x3
        p2 = C_beta * x4 + s * S_beta * x5
        p3 = 1 / s * S_beta * x2 + C_beta * x3
        p4 = 1 / s * S_beta * x4 + C_beta * x5

        q1 = C_alpha * p1 - r * S_alpha * p2
        q2 = - 1 / r * S_alpha * p3 + C_alpha * p4
        q3 = C_alpha * p3 - r * S_alpha * p4
        q4 = - 1 / r * S_alpha * p1 + C_alpha * p2

        y1 = a_prime * x1 + a * q1
        y2 = a * x1 + a_prime * q2
        z1 = b * x1 + b_prime * q1
        z2 = b_prime * x1 + b * q2

        x_hat_1 = b_prime * y1 + b * y2
        x_hat_2 = a * y1 + a_prime * y2
        x_hat_3 = epsilon * q3
        x_hat_4 = epsilon * q4
        x_hat_5 = b_prime * z1 + b * z2
        # x_hat_6 = a * z1 + a_prime * z2 == x_hat_1

        X1 = np.array([x_hat_1, x_hat_2, x_hat_3, x_hat_4, x_hat_5])

    D = x_hat_2 + s_h * x_hat_3 - r_h * (x_hat_4 + s_h * x_hat_5)

    return D.real


def main():
    layers = [
        {'d': 5, 'alpha': 200, 'beta': 100, 'rho': 1900},
        {'d': 10, 'alpha': 400, 'beta': 200, 'rho': 1900},
        {'d': 15, 'alpha': 600, 'beta': 300, 'rho': 1900},
        {'d': np.inf, 'alpha': 800, 'beta': 400, 'rho': 1900}
    ]

    omegas = np.linspace(1, 50 * 2 * np.pi, 100)

    phase_velocity = np.zeros(len(omegas))
    new_omega = np.zeros(len(omegas))

    c_list = np.linspace(80, 500, int((500-80)*10+1))
    for j, omega in enumerate(tqdm(omegas)):
        D = [compute_dispersion_fastdelta(c, omega, layers) for c in c_list]
        # Find the first sign change
        for i in range(len(D) - 1):
            if D[i] * D[i + 1] < 0:  # Sign change detected
                root_interval = (c_list[i], c_list[i + 1])
                break

        solution = optimize.root_scalar(compute_dispersion_fastdelta,
                                            args=(omega, layers),
                                            bracket=root_interval,
                                            method='brentq'
                                            )
        # new_omega[j] = 2 * np.pi * solution.root / omega
        phase_velocity[j] = solution.root


    import matplotlib.pyplot as plt
    plt.plot(omegas / 2 / np.pi, phase_velocity, 'ro-')
    # plt.plot(new_omega / 2 / np.pi, phase_velocity, 'ro-')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Phase Velocity (m/s)')
    plt.grid()
    plt.savefig('dispersion.png')
    plt.show()

if __name__ == "__main__":
    main()