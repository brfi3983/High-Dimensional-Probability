import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial import distance as dist
import time
import math

# ===========================================================================
def pdist(A):

	d, n = A.shape
	D = np.zeros((n, n))

	for i in range(0, n):
		x1 = A[:,i]
		for j in range(i, n - 1):
			x2 = A[:, j + 1]
			D[i, j + 1] = np.linalg.norm(x1 - x2)

	return D
# ===========================================================================
def JL(X, k):

	
	print('Xs Shape: {}'.format(X.shape))
	d, n = X.shape
	A = np.zeros((k, d))
	Y = np.zeros((k, n))
	
	A = np.random.normal(0, 1, size=(k, d))
	
	start = time.time()
	
	Y = np.matmul(A, X)
	Y /= np.sqrt(k)
	
	end = time.time()
	t = end - start

	return Y, t

# ===========================================================================
def FJL(X, k):


	d, n = X.shape
	A = np.zeros((k, d))
	Y = np.zeros((k, n))

	probs = [1/6, 1/6, 2/3]
	values = [1, -1, 0]
	A = np.random.choice(values, p=probs, size=(k, d))

	start = time.time()
	
	Y = np.matmul(A, X)
	Y *= np.sqrt(3/k)
	
	end = time.time()
	t = end - start

	return Y, t

# ===========================================================================
def NeighborAccuracy(A, B):
	# A and B are (# neighbor X columns) shape

	acc = np.zeros(A.shape[1])
	for i in range(A.shape[1]):
		x = A[:,i]
		y = B[:,i]

		z = np.isin(x,y)
		
		acc[i] = np.sum(z) / A.shape[0]

	return acc

# ===========================================================================
def NearestNeighbors(A, n):
	'''A is an adjacency matrix of pairwise euclidean idstances; n = number of closest neighbors'''
	
	nn_arr = np.zeros((5, A.shape[1]))
	for i in range(A.shape[1]):
		x = A[:,i]

		x_idx = np.argsort(x)
		x_idx_5 = x_idx[1:n + 1] # Skipping 0 (diagonal)
		nn_arr[:, i] = x_idx_5

	return nn_arr

# ===========================================================================
def main():
	X1 = np.genfromtxt('data1.csv', delimiter=',')
	X2 = np.genfromtxt('data2.csv', delimiter=',')
	X3 = np.genfromtxt('data3.csv', delimiter=',')
	X4 = np.genfromtxt('data4.csv', delimiter=',')
	X5 = np.genfromtxt('data5.csv', delimiter=',')
	X6 = np.genfromtxt('data6.csv', delimiter=',')
	plt.subplots_adjust(wspace=0.2, hspace=2)

	k_arr = [25, 100, 225, 400]
	X = [X1, X2, X3, X4, X5, X6]
	# JL_vec = np.vectorize(JL)
	# FJL_vec = np.vectorize(FJL)
	
	# D1 = dist.pdist(data1.T, metric='euclidean')
	# C1 = dist.pdist(Y1.T, metric='euclidean')

	acc_arr = np.zeros((6, 4, 2)) # 6x4 for 6 datasets and 4 k values
	# acc_hist_arr = np.zeros((1, 4)) # 6x4 for 6 datasets and 4 k values
	t_arr = np.zeros((6, 4, 2)) # 6x4 for 6 datasets and 4 k values
	for i, data in enumerate(X):
		for j, k in enumerate(k_arr):
			print('============================================')
			print('=> New Round!!!')
			print(f'On X{i + 1} - with {k} dimensions.')
			
			# Dataset for input
			X = data

			# Applying transformation
			Y, t = JL(X, k)
			Y_f, t_f = FJL(X, k)
			print(f'Time for JL: {t}, Time for FJL: {t_f}')
			
			# Computing distances between original and reduced
			D = pdist(X)
			C = pdist(Y)
			C_f = pdist(Y_f)

			# Flattening adjency matrix into 1D-vector for plotting
			D_flat = D[np.triu_indices(D.shape[0], k = 1)]
			C_flat = C[np.triu_indices(C.shape[0], k = 1)]
			C_f_flat = C_f[np.triu_indices(C_f.shape[0], k = 1)]

			# Distortion Ratio
			r = C_flat/D_flat
			r_f = C_f_flat/D_flat

			# Computing nearest 5 neighbors
			n = 5
			D_nn = NearestNeighbors(D, n)
			C_nn = NearestNeighbors(C, n)
			C_f_nn = NearestNeighbors(C_f, n)

			acc = NeighborAccuracy(D_nn, C_nn)
			acc_f = NeighborAccuracy(D_nn, C_f_nn)
			acc_avg = acc.mean()
			acc_avg_f = acc_f.mean()

			# Store average accuracy for (2) plot
			acc_arr[i, j, :] = np.array([acc_avg, acc_avg_f]) 
			t_arr[i, j, :] = np.array([t, t_f])
			
			fig, axs = plt.subplots(2, 2)
			fig.suptitle(f'Distortion Ratio: k = {k}, dataset = data{i + 1}')
			axs[0,0].hist2d(D_flat, C_flat, bins=100)
			axs[0,1].hist(r, ec='black')
			axs[0,1].set_xlabel('Ratio')
			axs[0,1].set_title('Distortion Ratio for JL')
			axs[0,1].set_xlim(0, 2)
			
			axs[1,0].hist2d(D_flat, C_f_flat, bins=100)
			axs[1,1].hist(r_f, ec='black')
			axs[1,1].set_xlabel('Ratio')
			axs[1,1].set_title('Distortion Ratio for FJL')
			axs[1,1].set_xlim(0, 2)
			fig.tight_layout()
			plt.subplots_adjust(top=0.85)
			fig.savefig(f'Plots/Scatter/Scatter_ratio_{k}_data{i + 1}')

			print('=> Round Complete!!!')
			if k == 25:
				fig1, (ax1, ax2) = plt.subplots(1, 2)
			ax1.hist(acc, histtype='bar', label=f'k = {k}')
			ax2.hist(acc_f, histtype='bar', label=f'k = {k}')
		# Create Histogram for that particular dataset (4 histograms of k's)
	
		ax1.set_title('JL')
		ax2.set_title('FJL')
		ax1.set_xlabel('Accuracy')
		ax2.set_xlabel('Accuracy')
		fig1.suptitle(f'data{i + 1} - Accuracy by k')
		ax1.legend()
		ax2.legend()
		
		fig1.savefig(f'Plots/Hist/Acc_hist_data{i + 1}')
	# Accuracy for JL
	fig_a, axs = plt.subplots(1, 2)
	axs[0].plot(k_arr, acc_arr[0, :, 0], label='data1', marker='s')
	axs[0].plot(k_arr, acc_arr[1, :, 0], label='data2', marker='s')
	axs[0].plot(k_arr, acc_arr[2, :, 0], label='data3', marker='s')
	axs[0].plot(k_arr, acc_arr[3, :, 0], label='data4', marker='s')
	axs[0].plot(k_arr, acc_arr[4, :, 0], label='data5', marker='s')
	axs[0].plot(k_arr, acc_arr[5, :, 0], label='data6', marker='s')
	axs[0].set_title('JL Compared to Accuracy')
	axs[0].set_xlabel('k value')
	axs[0].set_ylabel('Accuracy')
	axs[0].legend()

	# Accuracy for FJL
	axs[1].plot(k_arr, acc_arr[0, :, 1], label='data1', marker='s')
	axs[1].plot(k_arr, acc_arr[1, :, 1], label='data2', marker='s')
	axs[1].plot(k_arr, acc_arr[2, :, 1], label='data3', marker='s')
	axs[1].plot(k_arr, acc_arr[3, :, 1], label='data4', marker='s')
	axs[1].plot(k_arr, acc_arr[4, :, 1], label='data5', marker='s')
	axs[1].plot(k_arr, acc_arr[5, :, 1], label='data6', marker='s')
	axs[1].set_title('FJL Compared to Accuracy')
	axs[1].set_xlabel('k value')
	axs[1].set_ylabel('Accuracy')
	axs[1].legend()
	fig_a.tight_layout()
	fig_a.savefig(f'Plots/AccuracyPlot')
	
	# Time for JL
	fig_t, axs1 = plt.subplots(1, 2)
	axs1[0].plot(k_arr, t_arr[0, :, 0], label='data1', marker='s')
	axs1[0].plot(k_arr, t_arr[1, :, 0], label='data2', marker='s')
	axs1[0].plot(k_arr, t_arr[2, :, 0], label='data3', marker='s')
	axs1[0].plot(k_arr, t_arr[3, :, 0], label='data4', marker='s')
	axs1[0].plot(k_arr, t_arr[4, :, 0], label='data5', marker='s')
	axs1[0].plot(k_arr, t_arr[5, :, 0], label='data6', marker='s')
	axs1[0].set_title('JL Compared to Time')
	axs1[0].set_xlabel('k')
	axs1[0].set_ylabel('Time')
	axs1[0].legend()

	# Time for FJL
	axs1[1].plot(k_arr, t_arr[0, :, 1], label='data1', marker='s')
	axs1[1].plot(k_arr, t_arr[1, :, 1], label='data2', marker='s')
	axs1[1].plot(k_arr, t_arr[2, :, 1], label='data3', marker='s')
	axs1[1].plot(k_arr, t_arr[3, :, 1], label='data4', marker='s')
	axs1[1].plot(k_arr, t_arr[4, :, 1], label='data5', marker='s')
	axs1[1].plot(k_arr, t_arr[5, :, 1], label='data6', marker='s')
	axs1[1].set_title('FJL Compared to Time')
	axs1[1].set_xlabel('k')
	axs1[1].set_ylabel('Time')
	axs1[1].legend()
	fig_t.tight_layout()
	fig_t.savefig(f'Plots/TimePlot')

# ===========================================================================
if __name__ == "__main__":
	main()
