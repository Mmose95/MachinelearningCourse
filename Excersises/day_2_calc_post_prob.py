import numpy as np
from scipy.stats import multivariate_normal
from sklearn.mixture import GaussianMixture


def compute_posterior_probabilities(x_test, means_individual, covariances_individual, gmm_mixed, priors=None):
    """
    Computes the posterior probabilities for multiple test points.

    Parameters:
        x_test (array): Multiple test points (shape: N x 2, LDA transformed)
        means_individual (list): List of means for individual Gaussian models
        covariances_individual (list): List of covariance matrices for individual Gaussian models
        gmm_mixed (GaussianMixture): A trained GMM on all data
        priors (list, optional): Prior probabilities for the classes (default: equal priors)

    Returns:
        dict:
            "posterior_individual" -> (N x 3) array of posterior probabilities from individual Gaussian models
            "posterior_gmm" -> (N x 3) array of posterior probabilities from the GMM
    """
    if priors is None:
        priors = [1 / 3, 1 / 3, 1 / 3]  # Assume equal priors if not provided

    num_samples = x_test.shape[0]  # Number of test points

    # ------------------------- Compute Posterior Probability for Individual Gaussians -------------------------
    posterior_individual = np.zeros((num_samples, len(means_individual)))  # Store posteriors for individual Gaussians

    for i in range(len(means_individual)):
        # Get mean and covariance of the individual Gaussian model
        mean = means_individual[i]
        cov = covariances_individual[i]

        # Compute likelihood P(x_test | C_k) for all test samples
        likelihood = multivariate_normal(mean=mean, cov=cov).pdf(x_test)

        # Compute posterior P(C_k | x_test) using Bayes’ Rule (ignoring denominator P(x))
        posterior_individual[:, i] = likelihood * priors[i]

    # Normalize posteriors row-wise to sum to 1
    posterior_individual /= posterior_individual.sum(axis=1, keepdims=True)

    # ------------------------- Compute Posterior Probability from the Single GMM -------------------------
    posterior_gmm = gmm_mixed.predict_proba(x_test)  # Shape: (N x 3)

    # ------------------------- Return Results -------------------------
    return {
        "posterior_individual": posterior_individual,  # (N x 3) Posterior probabilities from individual Gaussians
        "posterior_gmm": posterior_gmm  # (N x 3) Posterior probabilities from GMM
    }
