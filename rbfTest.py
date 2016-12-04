import numpy as np
import gzip
import cPickle
import sklearn
import load_data

from scipy.spatial import distance
from sklearn.cluster import KMeans
from sklearn.preprocessing import scale

def calculatePhiMatrix(test_data, centers, sigma):
	phiMatrix = np.ndarray(shape=(test_data.shape[0], centers.shape[0]))
	# print 'phiMatrix shape ', phiMatrix.shape
	for (idx1,data) in enumerate(test_data):
		for (idx2,centroid) in enumerate(centers):
			dist = distance.euclidean(data, centroid)
			phiMatrix[idx1,idx2] = dist * (np.exp((-1*dist*dist)/(2*sigma[idx2]*sigma[idx2])))	#dist = dist * activation
	return phiMatrix		

def calcOutput(phi, weights):
	print phi.shape, weights.shape
	output = np.dot(phi, np.transpose(weights))
	return normOutput(output)

def normOutput(output):
	normOut = np.zeros(shape = output.shape)
	target =  np.zeros(shape = output.shape[0])
	print 'normOut', normOut.shape
	for (idx,out) in enumerate(output):
		normOut[idx][out.argmax()] = 1
		target[idx] = out.argmax()
	return normOut, target

def calcAccuracy(target, test_label):
	match = 0
	for idx in range(target.shape[0]):
		if target[idx] == test_label[idx]:
			match = match+1;
	print match, target.shape[0]
	accuracy = float(match)/float(target.shape[0])
	print accuracy
	return accuracy*100

def predictRBF(test_data, test_label, centroids, weights, sigma,num_centroids, num_output):
	phiMatrix = calculatePhiMatrix(test_data, centroids, sigma)
	output , target = calcOutput(phiMatrix, weights)
	accuracy =  calcAccuracy(target, test_label)
	print 'Accuracy  : ',str(accuracy),'%'

if __name__ == '__main__':
	train, valid, test  = load_data.load_data('./mnist.pkl.gz')
	test_data = test[0]
	# test_data = test_data[0:50,:]
	test_label = test[1]
	# test_label = test_label[0:50]
	# print test_data.shape, test_label.shape
	test_scaled = scale(test_data.astype(np.float))
	data = np.load(str(40)+'_vectors.npz')
	centroids = data['centroids']
	weights = data['weights']
	sigma = data['sigma']

	# print centroids.shape, weights.shape, sigma

	print 'Started Testing'
	predictRBF(test_scaled, test_label, centroids, weights, sigma, num_centroids = 40, num_output = 10)
