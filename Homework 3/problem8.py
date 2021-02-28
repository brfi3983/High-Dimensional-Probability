import numpy as np
import matplotlib.pyplot as plt

# ============================================================
def plotRejectionPoints(n, r):
	# x-axis and plotting
	n_arr = np.arange(1, n + 1)
	plt.plot(n_arr, r)

	# Graph customization
	plt.xlabel('Dimension of n')
	plt.ylabel('Number of Rejected Points')
	plt.title('How the Position of Points Change as Dimension Grows')
	plt.show()

# ============================================================
def createRejectionPoints(N, n, r):

	# Create rejcetions arry for each dimension
	rejections = np.zeros(n)

	# Looping over the dimensions for each experiment (sampling N times for n-length points)
	for k in range(1, n + 1):
		x = np.zeros((N, k))
		# Creating N samples per dimension
		for i in range(0, N):
			# Creating points in n-dimensions
			for j in range(0, k):
				u = np.random.uniform(0, 1)
				x[i][j] = r*(1 - 2*u)
			# Rejecting points if the L-2 norm falls outside the sphere
			x_norm = np.linalg.norm(x[i, :], 2)
			if x_norm > r:
				rejections[k - 1] += 1

	return rejections

# ============================================================
def main():
	# Constants
	N = 10000
	n = 400
	r = 1

	# Generating and plotting
	rejection_arr = createRejectionPoints(N, n, r)
	plotRejectionPoints(n, rejection_arr)

# ============================================================
if __name__ == '__main__':
	main()
