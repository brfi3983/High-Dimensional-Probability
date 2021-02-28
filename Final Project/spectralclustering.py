from sklearn.cluster import SpectralClustering
from sklearn.datasets.samples_generator import make_blobs
import matplotlib.pyplot as plt
from numpy import random
from numpy import linalg as LA
import networkx as nx
from sklearn.metrics import pairwise_distances
import numpy as np
from sklearn.neighbors import kneighbors_graph

plt.style.use('ggplot')

# ========================================================
def main():
	# Creating Graph
	random.seed(10)
	X, y = make_blobs(n_samples=400, centers=5, cluster_std=0.4)
	plt.scatter(X[:,0], X[:,1])

	# Compute Similarity Matrix from Image
	A = np.exp(- 1./(2 * 1) * pairwise_distances(X, metric='sqeuclidean'))

	# Compute degree matrix
	d = np.zeros(A.shape[0])
	d = np.sum(A, axis=1)
	D = np.diag(d)

	# Simple laplacian
	L = D - A

	# Computing smallest eigenvalues to find number of clusters
	eigvals, eigvect = LA.eig(L)
	eigenvals_sorted_indices = np.argsort(eigvals)
	eigenvals_sorted = eigvals[eigenvals_sorted_indices]

	zero_eigenvals_index = np.argwhere(abs(eigvals) < 1e-3)
	fid = eigvect[eigenvals_sorted_indices[1]]
	fid_sorted = fid[np.argsort(fid)]


	# Plotting sorted eigenvalues
	plt.figure()
	plt.plot(range(1,fid.size + 1), fid_sorted)

	# Plotting first number of zeroed eigenvalues
	plt.figure()
	n = zero_eigenvals_index.size + int(zero_eigenvals_index.size*(1/2))
	plt.plot(range(1, n + 1), eigenvals_sorted[:n])

	plt.show()

# ========================================================
if __name__ == "__main__":
	main()
