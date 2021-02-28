import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh

# Global constants and pre-allocated arrays
n = 300
sparse = 1
if sparse == 1:
	a_arr = np.arange(5,70)
else:
	a_arr = np.arange(5,50)
can_w = np.ones(n)
can_w[int(n/2):] = -1
b_arr = np.arange(1,50)
model_arr = np.zeros((20, n, n))
ab_matrix = np.zeros((20, a_arr.shape[0], b_arr.shape[0]))
final_matrix = np.zeros((a_arr.shape[0], b_arr.shape[0]))

# ========================================================
def StochasticBlockModel(p, q, n):

	# Creating permutation matrix
	I = np.eye(n)
	ix = np.random.permutation(n)
	Tr = I[ix,:]

	# Creating individual entries for a binomial matrix
	n2 = int(n/2)
	P = np.random.binomial(1, p, size=(n2, n2))
	P_diag_lower = np.random.binomial(1, p, size=(n2, n2))
	Q = np.random.binomial(1, q, size=(n2, n2))

	# Slicing sections of our block model to insert diagonal
	U = np.triu(P, 1)
	L = np.tril(P, -1)
	P_diag = np.diag(P)
	P_diag_lower = np.diag(P_diag_lower)

	# Creating the four blocks to form the block model
	A0 = U + U.T + np.diag(P_diag)
	A1 = Q
	A2 = Q.T
	A3 = L + L.T + np.diag(P_diag_lower)

	# Concatonating and transforming final product
	A = np.block([[A0, A1], [A2, A3]])
	# B = Tr @ A @ Tr.T

	return A, Tr
# ========================================================
def scoring(n, w, w_tilda):

	w_tilda = 1*(w_tilda > 0)
	w_tilda = np.where(w_tilda != 0, w_tilda, -1).reshape(len(w_tilda),)
	# Comparing our eigenvector to the actual partitions

	delta_1 = (w == w_tilda)
	delta_2 = (-w == w_tilda)

	# Finding the max between the two and implementing the scoring function
	rawoverlap = np.amax([np.sum(delta_1), np.sum(delta_2)])
	overlap = (2/n)*rawoverlap - 1

	return overlap, w_tilda

def denseBoundary(b, n):
	return (b*np.log(n) + 1 - np.sqrt(4*b*np.log(n) + 1))/(np.log(n))
def sparseBoundary(b):
	return b + 1 - np.sqrt(4*b + 1)
# ========================================================
def main():

	question7 = 1
	# Question 5-6
	if question7 != 1:
		for b_idx, b in enumerate(b_arr):
			for a_idx, a in enumerate(a_arr):
				for i in range(20):

					# Values for p and q
					if sparse == 1:
						p = a/n
						q = b/n

					else:
						p = a*np.log(n)/n
						q = b*np.log(n)/n

					# Computing a model and finding its second largest eigenvalue
					model, T = StochasticBlockModel(p, q, n)
					eigval, eigvec = eigh(model, subset_by_index=[n-2, n-2])

					# Using the eigenvector more
					w = can_w
					w_tilda_eig = eigvec

					# Finding its overlap score and storing into temporary sample matrix (20 samples per combination)
					score, communities = scoring(n, w, w_tilda_eig)
					ab_matrix[i, a_idx, b_idx] = score

				# Taking the mean value of all samples
				final_matrix[a_idx, b_idx] = np.mean(ab_matrix[:, a_idx, b_idx])
				print(f'Values for b:{b} and a:{a} are finished with mean score of {final_matrix[a_idx, b_idx]}...')
		if sparse == 1:
			y = sparseBoundary(b_arr)
		else:
			y = denseBoundary(b_arr, n)
		# Showing the final matrix as a greyscale image heatmap
		plt.imshow(final_matrix, cmap=plt.cm.binary)
		plt.plot(b_arr, y, c='red', label='Boundary')
		plt.colorbar()
		plt.xlabel('beta')
		plt.ylabel('alpha')
		plt.title('Sparse Network')
		plt.legend()
		plt.show()

	# Question 7
	if question7 == 1:
		A = np.genfromtxt('A.csv', delimiter=',')
		m = A.shape[0]
		# true partitions: 1 is to the right, -1 is to the left
		w = np.array([1, 1, 1, 1, 1, 1, 1, 1, -1, -1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1])
		print(w.shape)
		# Computing a model and finding its second largest eigenvalue
		eigval, eigvec = eigh(A, subset_by_index=[m-2, m-2])

		# Using the eigenvector more
		w_tilda_eig = eigvec

		# Finding its overlap score and storing into temporary sample matrix (20 samples per combination)
		score, communities = scoring(m, w, w_tilda_eig)
		print(f'Score for Zachary\'s karate club: {score}, predictions: {communities}')

# ========================================================
if __name__ == "__main__":
	main()