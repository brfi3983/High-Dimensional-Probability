import numpy as np
import matplotlib.pyplot as plt

# ============================================================
def calcDistance(x1, x2):

	# Euclidean norm
	d = np.sqrt(np.sum((x1 - x2)**2))

	return d

# ============================================================
def comb(n, k):

	# Combination equation with failsafe
	if n > k:
		return int(np.math.factorial(n)/(np.math.factorial(n - k)*np.math.factorial(k)))
	return -1

# ============================================================
def calcAngle(x1, x2):

	# Using x1.x2 = |x1||x2|*cos(theta)
	x1_mag = np.linalg.norm(x1)
	x2_mag = np.linalg.norm(x2)
	x1_x2_dot = np.dot(x1, x2)

	# Finding final angle
	theta = np.arccos(x1_x2_dot/(x1_mag*x2_mag))

	return theta

# ============================================================
def main():

	# Samples, dimension, cube length
	N = 400
	n = 100
	l = 0.5

	# Generating and transfomring array from [0,1] to [-0.5, 0.5]
	array = np.random.rand(N, n)
	array = l*(1 - 2*array)

	# Finding number of operations to initilize arrays
	num_operations = comb(N, 2)
	theta_arr = np.zeros(num_operations)
	d_arr = np.zeros(num_operations)
	count = 0
	for i in range(0, N):
		x1 = array[i,:]
		for j in range(i, N - 1):
			x2 = array[j + 1, :]
			d_arr[count] = calcDistance(x1, x2)
			theta_arr[count] = calcAngle(x1, x2)
			count += 1

	# Plotting Results
	plt.hist(d_arr, label='Distance')
	plt.hist(theta_arr, label='Theta')
	plt.ylabel('Number of Values')
	plt.xlabel('Values')
	plt.legend(loc="upper left")
	plt.title('Orthogonality and Euclidean Norm in Higher Dimensions')
	plt.show()

# ============================================================
if __name__ == "__main__":
	main()
