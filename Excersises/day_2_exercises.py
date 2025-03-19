
###################################
# Group ID : 420
# Members : Matias Mose,  Henrik Paaske Lind, Amalie Koch Andersen, Phillip Kaasgaard Sperling
# Date : 19-03-2025
# Lecture: Lecture 5: Clustering
# Dependencies: first section is an import section | day_2_calc_post_prob <-- helper function for posterior prob calc
# Python version: 3.12
# Functionality: This code uses different GMM fitted to train data, and is compared using test data. The gaussians are
#                visualized along with contours and test date points. The posterior probs are printed to terminal.
# ###################################



import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.io import loadmat
from scipy.stats import multivariate_normal
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.mixture import GaussianMixture
from sklearn.svm import SVC
from matplotlib.backends.backend_pdf import PdfPages


from day_2_calc_post_prob import compute_posterior_probabilities

data = loadmat("mnist_all.mat")

# Extract training data (digits 5, 6, and 8)
train_5, train_6, train_8 = data['train5'] / 255, data['train6'] / 255, data['train8'] / 255
test_5, test_6, test_8 = data['test5'] / 255, data['test6'] / 255, data['test8'] / 255  # Extract test data

# Assign labels: (0 for 5, 1 for 6, 2 for 8)
labels_5_train = np.full(train_5.shape[0], 0)
labels_6_train = np.full(train_6.shape[0], 1)
labels_8_train = np.full(train_8.shape[0], 2)

labels_5_test = np.full(test_5.shape[0], 0)
labels_6_test = np.full(test_6.shape[0], 1)
labels_8_test = np.full(test_8.shape[0], 2)

# Combine training and test sets
X_train = np.vstack((train_5, train_6, train_8))  # Training data
y_train = np.hstack((labels_5_train, labels_6_train, labels_8_train))  # Training labels

X_test = np.vstack((test_5, test_6, test_8))  # Test data
y_test = np.hstack((labels_5_test, labels_6_test, labels_8_test))  # Test labels

# Apply LDA (Reduce 784 → 2)
lda = LinearDiscriminantAnalysis(n_components=2)
X_train_lda = lda.fit_transform(X_train, y_train)
X_test_lda = lda.transform(X_test)  # Transform test data

# Apply the filtering condition:
filtered_indices = np.where((X_test_lda[:, 1] >= -0.5) & (X_test_lda[:, 1] <= 0.5) & (X_test_lda[:, 0] > 0))

# Select a random test point
x_test = X_test_lda[filtered_indices]  # Pick the first point from LDA-transformed testdata
x_test_label = y_test[filtered_indices]

# Gaussian Mixture Model for clustering (GMM)
# ------------------------- Train Separate GMMs for Each Digit -------------------------
gmm_individual = {}
means_individual = []
covariances_individual = []

for label in [0, 1, 2]:  # Iterate over digits 5, 6, and 8
    class_data = X_train_lda[y_train == label]

    # Train a GMM with only one component (since it's per class)
    gmm = GaussianMixture(n_components=1, covariance_type='full', random_state=42)
    gmm.fit(class_data)

    # Store the trained GMM
    gmm_individual[label] = gmm

    # Store mean and covariance
    means_individual.append(gmm.means_[0])
    covariances_individual.append(gmm.covariances_[0])

print("\nIndividual GMM Parameters:")
for i, label in enumerate([5, 6, 8]):
    print(f"Digit {label}:")
    print(f"Mean: {means_individual[i]}")
    print(f"Covariance:\n{covariances_individual[i]}\n")

# ------------------------- Train a Single GMM on Mixed Data -------------------------
# Fit Gaussian Mixture Model (GMM) with 3 components (one for each digit)
gmm_mixed = GaussianMixture(n_components=3, covariance_type='full', random_state=42)
gmm_mixed.fit(X_train_lda)

# Extract parameters from the mixed-data GMM
gmm_mixed_means = gmm_mixed.means_
gmm_mixed_covariances = gmm_mixed.covariances_

print("\nGMM (Mixed Data) Parameters:")
print("Means:\n", gmm_mixed_means)
print("Covariances:\n", gmm_mixed_covariances)

# ------------------------- Generate Grid for Visualization -------------------------
x, y = np.meshgrid(np.linspace(X_train_lda[:, 0].min(), X_train_lda[:, 0].max(), 100),
                   np.linspace(X_train_lda[:, 1].min(), X_train_lda[:, 1].max(), 100))
pos = np.dstack((x, y))

# ------------------------- Compute GMM Density for Single GMM Model -------------------------
pdf_values_gmm = np.zeros(x.shape)
for i in range(gmm_mixed.n_components):
    mean = gmm_mixed.means_[i]
    cov = gmm_mixed.covariances_[i]
    weight = gmm_mixed.weights_[i]
    component_pdf = multivariate_normal(mean=mean, cov=cov).pdf(pos)
    pdf_values_gmm += weight * component_pdf  # Sum over all components

# ------------------------- Compute Individual Gaussian Model Densities -------------------------
pdf_values_individual = [multivariate_normal(mean=means_individual[i], cov=covariances_individual[i]).pdf(pos)
                         for i in range(3)]

# ------------------------- Plot 1: Single GMM Model -------------------------

pdf_filename = "GMM_LDA_Analysis.pdf"
pdf_pages = PdfPages(pdf_filename)

plt.figure(figsize=(8, 6))
y_gmm_pred = gmm_mixed.predict(X_train_lda)  # Cluster assignments from GMM
plt.scatter(X_train_lda[:, 0], X_train_lda[:, 1], c=y_gmm_pred, cmap='viridis', alpha=0.3, label="GMM Clusters")
plt.scatter(gmm_mixed.means_[:, 0], gmm_mixed.means_[:, 1], c='red', marker='x', s=200, label="GMM Cluster Centers")
plt.contour(x, y, pdf_values_gmm, levels=10, cmap="coolwarm", alpha=0.7)
plt.scatter(x_test[:, 0], x_test[:, 1], c=x_test_label, edgecolors='black', marker='o', s=50, label="Test 6 Samples")
plt.xlabel("LD1")
plt.ylabel("LD2")
plt.title("Single GMM Model (3 Components)")
plt.legend()
pdf_pages.savefig()
plt.close()

# ------------------------- Plot 2: Three Separate GMMs (One per Class) -------------------------
plt.figure(figsize=(8, 6))
colors = ['red', 'blue', 'green']

for i, label in enumerate([5, 6, 8]):
    plt.scatter(X_train_lda[y_train == i][:, 0], X_train_lda[y_train == i][:, 1], color=colors[i], alpha=0.3,
                label=f"Digit {label}")
    plt.contour(x, y, pdf_values_individual[i], colors='black', alpha=0.7)
plt.scatter(x_test[:, 0], x_test[:, 1], c=x_test_label, edgecolors='black', marker='o', s=50, label="Test 6 Samples")
plt.xlabel("LD1")
plt.ylabel("LD2")
plt.title("Three Individual Gaussian Models (One per Class)")
plt.legend()
pdf_pages.savefig()
plt.close()

# ------------------------- Plot 3: Comparison of Both -------------------------
plt.figure(figsize=(8, 6))
y_gmm_pred = gmm_mixed.predict(X_train_lda)  # Cluster assignments from GMM
plt.scatter(X_train_lda[:, 0], X_train_lda[:, 1], c=y_gmm_pred, cmap='viridis', alpha=0.3, label="GMM Clusters")
plt.scatter(gmm_mixed.means_[:, 0], gmm_mixed.means_[:, 1], c='red', marker='x', s=200, label="GMM Cluster Centers")
plt.contour(x, y, pdf_values_gmm, levels=10, cmap="coolwarm", alpha=0.7)  # GMM Contours

# Overlay the individual Gaussian contours
for i, label in enumerate([5, 6, 8]):
    plt.contour(x, y, pdf_values_individual[i], colors=colors[i], alpha=0.7, label=f"Digit {label} Gaussian")
plt.scatter(x_test[:, 0], x_test[:, 1], c=x_test_label, edgecolors='black', marker='o', s=50, label="Test 6 Samples")
plt.xlabel("LD1")
plt.ylabel("LD2")
plt.title("Comparison: Single GMM vs. Individual Gaussian Models")
plt.legend()
pdf_pages.savefig()
plt.close()

# ---------------------------- Calculate Posterior ---------------------------


# Compute the posterior probabilities
posteriors = compute_posterior_probabilities(x_test, means_individual,
                                             covariances_individual, gmm_mixed,
                                             priors=[1/ 3, 1 / 3, 1/3])


# Print results for the first few test points
num_samples_to_print = min(5, x_test.shape[0])  # Print only the first 5 samples or fewer if available

print("\nPosterior Probabilities from Individual Gaussian Models:")
for i in range(num_samples_to_print):
    print(f"\nTest Sample {i+1} (True Label: {x_test_label[i]}):")
    for j, label in enumerate([5, 6, 8]):
        print(f"  P(Digit {label} | x_test) = {posteriors['posterior_individual'][i, j]:.4f}")

print("\nPosterior Probabilities from Single GMM:")
for i in range(num_samples_to_print):
    print(f"\nTest Sample {i+1} (True Label: {x_test_label[i]}):")
    for j in range(3):  # GMM has 3 components (not necessarily corresponding to 5, 6, 8)
        print(f"  P(Component {j} | x_test) = {posteriors['posterior_gmm'][i, j]:.4f}")


# Define the filename
output_filename = "posterior_probabilities.txt"

# Open the file in write mode
with open(output_filename, "w") as file:
    num_samples_to_print = min(5, x_test.shape[0])  # Limit to the first 5 samples

    file.write("\nPosterior Probabilities from Individual Gaussian Models:\n")
    for i in range(num_samples_to_print):
        file.write(f"\nTest Sample {i+1} (True Label: {x_test_label[i]}):\n")
        for j, label in enumerate([5, 6, 8]):
            file.write(f"  P(Digit {label} | x_test) = {posteriors['posterior_individual'][i, j]:.4f}\n")

    file.write("\nPosterior Probabilities from Single GMM:\n")
    for i in range(num_samples_to_print):
        file.write(f"\nTest Sample {i+1} (True Label: {x_test_label[i]}):\n")
        for j in range(3):  # GMM has 3 components
            file.write(f"  P(Component {j} | x_test) = {posteriors['posterior_gmm'][i, j]:.4f}\n")

# Print confirmation message
print(f"Posterior probabilities saved to {output_filename}")


# --------------------- Train SVM Classifier because reasons..  -----------------------

svm = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
svm.fit(X_train_lda, y_train)

# Predict on Test Data
y_pred = svm.predict(X_test_lda)

# Evaluate Performance
accuracy = accuracy_score(y_test, y_pred)
print(f"SVM Classification Accuracy: {accuracy:.4f}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

# Plot Confusion Matrix
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=[5, 6, 8], yticklabels=[5, 6, 8])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("SVM Confusion Matrix")
pdf_pages.savefig()
plt.close()
pdf_pages.close()
