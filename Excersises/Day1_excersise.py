# ###################################
# Group ID : 420
# Members : Matias Mose,  Henrik Paaske Lind, Amalie Koch Andersen, Phillip Kaasgaard Sperling
# Date : 17-03-2025
# Lecture: Lecture 3: Parametric methods (ML, MAP & Bayesian learning) and nonparametric methods
# Dependencies: first section is an import section.
# Python version: 3.12
# Functionality: This code trains the training data and tests on three different scenarios: 1) some test data, 2) some different test with uniform a priori data and 3) same test data as 2) but with different a priori.
# ###################################

from sklearn.metrics import confusion_matrix
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay
from scipy.io import loadmat
from scipy.stats import multivariate_normal as norm
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

#Load dataset
file = "D:\PHD\Courses\Machine Learning\Data\dataset1_G_noisy.mat"
data = loadmat(file)

 #Trainsets

#dataset1 - træning
trn_x = data["trn_x"] #input 1
trn_y = data["trn_y"] #input 2

#dataset2 - træning
trn_x_class = data["trn_x_class"] #output (class) 1
trn_y_class = data["trn_y_class"] #output (class) 2

#Testsets
tst_xy = data["tst_xy"] #input - træning
tst_xy_class = data["tst_xy_class"] #output - test
tst_xy_class = tst_xy_class-1

tst_xy_126 = data["tst_xy_126"]
tst_xy_126_class = data["tst_xy_126_class"]
tst_xy_126_class = tst_xy_126_class-1

#Uniform Prior
prior_x = 0.5
prior_y = 0.5

#Estimate mean value and covariance for X and Y
mean_x = np.mean(trn_x, axis = 0)
mean_y = np.mean(trn_y, axis = 0)
cov_x = np.cov(trn_x.T)
cov_y = np.cov(trn_y.T)

#Create multivariate Gaussian distributions for X and Y
l_x = norm(mean = mean_x, cov = cov_x)
l_y = norm(mean = mean_y, cov = cov_y)

''' EX 1  '''

#(a) classify instances in tst_xy, and use the corresponding label file tst_xy_class to calculate the accuracy;

#Posteriori probability (ignoring the normalization constant) on the test set
p1 = prior_x * l_x.pdf(tst_xy)
p2 = prior_y * l_y.pdf(tst_xy)

pdf_all = np.stack([p1, p2], axis = 1)

#Maximum a posteriori prediction
argmax_pdf_all = np.argmax(pdf_all, axis = 1)

cfm_ex1 = confusion_matrix(tst_xy_class, argmax_pdf_all)

# Create and plot confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cfm_ex1)
disp.plot(cmap='Blues')
plt.title("Confusion Matrix")
plt.show()

clf_report_ex1 = classification_report(tst_xy_class, argmax_pdf_all)
print(clf_report_ex1)

''' EX 2 '''

#(b) classify instances in tst_xy_126 by assuming a uniform prior over the space of hypotheses, and use the corresponding label file tst_xy_126_class to calculate the accuracy;
#Posteriori probability (ignoring the normalization constant) on the test set

p1 = prior_x * l_x.pdf(tst_xy_126)
p2 = prior_y * l_y.pdf(tst_xy_126)

pdf_all = np.stack([p1, p2], axis = 1)

#Maximum a posteriori prediction
argmax_pdf_all = np.argmax(pdf_all, axis = 1)

cfm_ex2 = confusion_matrix(tst_xy_126_class, argmax_pdf_all)

# Create and plot confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cfm_ex2)
disp.plot(cmap='Blues')
plt.title("Confusion Matrix")
plt.show()

clf_report_ex2 = classification_report(tst_xy_126_class, argmax_pdf_all)
print(clf_report_ex2)

''' EX 3 '''
#(c) classify instances in tst_xy_126 by assuming a prior probability of 0.9 for Class x and 0.1 for Class y, and use the corresponding label file tst_xy_126_class to calculate the accuracy; compare the results with those of (b).

#Uniform Prior
prior_x = 0.9
prior_y = 0.1

#Posteriori probability (ignoring the normalization constant) on the test set
p1 = prior_x * l_x.pdf(tst_xy_126)
p2 = prior_y * l_y.pdf(tst_xy_126)

pdf_all = np.stack([p1, p2], axis = 1)

#Maximum a posteriori prediction
argmax_pdf_all = np.argmax(pdf_all, axis = 1)

cfm_ex3 = confusion_matrix(tst_xy_126_class, argmax_pdf_all)

# Create and plot confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cfm_ex3)
disp.plot(cmap='Blues')
plt.title("Confusion Matrix")
plt.show()

clf_report_ex3 = classification_report(tst_xy_126_class, argmax_pdf_all)
print(clf_report_ex3)