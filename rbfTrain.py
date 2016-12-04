import numpy as np
import gzip
import cPickle
import sklearn
import load_data

from scipy.spatial import distance
from sklearn.cluster import KMeans
from sklearn.preprocessing import scale

def calcSignma(centers):
	num_centroids = centers.shape[0]
	dist = np.ndarray(shape=(num_centroids, num_centroids))
	#max distance between centroids
	for i in range (0,num_centroids):
		for j in range (0, num_centroids):
			dist[i][j] = distance.euclidean(centers[i], centers[j])
	# print dist.shape, dist, np.amax(dist)
	return np.amax(dist)/np.sqrt(num_centroids)


def calcSignma_mean(centers):
	num_centroids = centers.shape[0]
	dist = np.ndarray(shape=(num_centroids, num_centroids))
	#max distance between centroids
	for i in range (0,num_centroids):
		for j in range (0, num_centroids):
			dist[i][j] = distance.euclidean(centers[i], centers[j])
	# print dist.shape, dist, np.amax(dist)
	return np.mean(dist)#/np.sqrt(num_centroids)

def calculatePhiMatrix(train_data, centers, sigma):
	phiMatrix = np.ndarray(shape=(train_data.shape[0], centers.shape[0]))
	# print 'phiMatrix shape ', phiMatrix.shape
	for (idx1,data) in enumerate(train_data):
		for (idx2,centroid) in enumerate(centers):
			dist = distance.euclidean(data, centroid)
			phiMatrix[idx1,idx2] = dist * (np.exp((-1*dist*dist)/(2*sigma*sigma)))	#dist = dist * activation
	return phiMatrix		

def createDMatrix(label,num_out):
	DMatrix = np.zeros(shape=(label.shape[0],num_out))
	# print 'Dmatrix shape ',DMatrix.shape
	for (idx,num) in enumerate(label):
		DMatrix[idx][num] = 1
	return DMatrix

def calcWeights(phi, d):
	# print phi.shape, (np.linalg.pinv(phi).shape), d.shape
	weights = np.dot(np.linalg.pinv(phi),(d))
	print "Sudo inverse of ", np.linalg.pinv(phi).shape, d.shape
	weights = np.transpose(weights)
	return weights

def trainRBF(train_data, train_label, num_centroids, num_output):
	print "Generating centroids.."
	centroids = sklearn.cluster.k_means(train_data, init='random', n_clusters=num_centroids, max_iter = 1, n_jobs=-1)
	centroids = centroids[0]
	print "calculating sigma.."
	sigma = calcSignma(centroids)
	print "calculating Phi matrix.."
	phiMatrix = calculatePhiMatrix(train_data, centroids, sigma)
	print "creating target.."
	dMatrix = createDMatrix(train_label, num_output)
	print "calculating weights.."
	weights = calcWeights(phiMatrix,dMatrix)
	print 'Train Data ', train_data.shape, 'phi ', phiMatrix.shape, 'D ',dMatrix.shape, 'Weights ', weights.shape
	print 'Saving vectors to '+ str(num_centroids)+'_weights.npz'
	np.savez(str(num_centroids)+'_vectors.npz', weights = weights, centroids = centroids, sigma = sigma)
	print '---------------- Training compele -----------------------'


if __name__ == '__main__':
	train, valid, test  = load_data.load_data('./mnist.pkl.gz')
	train_data = train[0]
	# train_data = train_data[0:200,:]
	train_label = train[1]
	# train_label = train_label[0:200]
	# print train_data.shape, train_label.shape, type(train_data)
	train_scaled = scale(train_data.astype(np.float))	
	print 'Started Training...'
	trainRBF(train_scaled, train_label, num_centroids = 40, num_output = 10)
