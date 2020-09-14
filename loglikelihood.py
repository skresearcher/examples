import numpy as np
from scipy.stats import rv_continuous
import matplotlib.pyplot as plt

# Draw random samples from probability density
def random_pdf(x0, x1, n, k):
    samples = []
    hwPDF = PDF()
    while len(samples) < n:
        x = np.random.uniform(low = x0, high = x1)
        prop = hwPDF.pdf(x, k)
        samples += [prop]
    return np.array(samples)

# Probability density function
class PDF(rv_continuous):
    def _pdf(self, x, k):
        return k * x ** (k - 1) * np.exp(-x ** k)

# Compare the reparameterized log-likehood functions
def main():
    # Parameters
    n = 10
    theta = np.array([0.1, 0.5, 1, 1.2, 1.5])

    # Draw the random variable from each distribution
    x1 = random_pdf(0, 2, n, np.exp(theta))
    x2 = random_pdf(0, 2, n, np.log(1 + np.exp(theta)))

    # Compute the log likelihood for each distribution
    ll1 = (n * theta) + (np.exp(theta) - 1) * np.sum(np.log(x1), axis = 0) - np.power(np.sum(x1, axis = 0), np.exp(theta))
    ll2 = (n * np.log(np.log(1 + np.exp(theta)))) + (np.log(1 + np.exp(theta)) - 1) * np.sum(np.log(x2), axis = 0)
    ll2 -= np.power(np.sum(x2, axis = 0), np.log(1 + np.exp(theta)))

    # Plot the results
    plt.figure()
    plt.plot(theta, ll1, label = '4c')
    plt.plot(theta, ll2, label = '4d')
    plt.legend()
    plt.show()

main()