#! /usr/bin/python3.4

# Copied without shame from:
# https://datasciencelab.wordpress.com/2013/12/12/clustering-with-k-means-in-python/
 
import numpy as np
import random
import os

import matplotlib.pyplot as plt

import pprint
pp = pprint.PrettyPrinter(indent=4)

import sys
import config # The path of config.py should be in the environment variable %pythonpath%
sys.path.append(config.GlobalFolder) # for global
import utilities


# Calculate distances of points from centres.
def distances(results):
	variance=[]
	for i,mu in enumerate(results[0]):
		for point in results[1][i]:
			variance.append(np.linalg.norm(mu-point))
	return sum(variance)
	
def cluster_points(X, mu):
	clusters  = {}
	for x in X:
		bestmukey = min([(i[0], np.linalg.norm(x-mu[i[0]])) \
					for i in enumerate(mu)], key=lambda t:t[1])[0]
		try:
			clusters[bestmukey].append(x)
		except KeyError:
			clusters[bestmukey] = [x]
	return clusters
 
def reevaluate_centers(mu, clusters):
	newmu = []
	keys = sorted(clusters.keys())
	for k in keys:
		newmu.append(np.mean(clusters[k], axis = 0))
	return newmu
 
def has_converged(mu, oldmu):
	return (set([tuple(a) for a in mu]) == set([tuple(a) for a in oldmu]))
 
def find_centers(X, K):
	# Initialize to K random centers
	#oldmu = random.sample(X, K)
	#mu = random.sample(X, K)
	
	# Make sure the samples start in a healthy (not equal/ converged) manner:
	mu=[np.array([0,0])]
	oldmu=[np.array([0,0])]
	while has_converged(mu, oldmu):
		oldmu = random.sample(list(X), K)
		mu = random.sample(list(X), K)	
	#pp.pprint(mu)
	#pp.pprint(oldmu)
	while not has_converged(mu, oldmu):
		oldmu = mu
		# Assign all points in X to clusters
		clusters = cluster_points(X, mu)
		# Re-evaluate centers
		mu = reevaluate_centers(oldmu, clusters)
	return(mu, clusters)

# BASED ON: https://datasciencelab.wordpress.com/2013/12/27/finding-the-k-in-k-means-clustering/
def Wk(mu, clusters):
	K = len(mu)
	return sum([np.linalg.norm(mu[i]-c)**2/(2*len(c)) \
			   for i in range(K) for c in clusters[i]])
			   
def bounding_box(X):
	xmin, xmax = min(X,key=lambda a:a[0])[0], max(X,key=lambda a:a[0])[0]
	ymin, ymax = min(X,key=lambda a:a[1])[1], max(X,key=lambda a:a[1])[1]
	return (xmin,xmax), (ymin,ymax)
 
def gap_statistic(X):
	(xmin,xmax), (ymin,ymax) = bounding_box(X)
	# Dispersion for real distribution
	ks = range(1,10)
	Wks = np.zeros(len(ks))
	Wkbs = np.zeros(len(ks))
	sk = np.zeros(len(ks))
	for indk, k in enumerate(ks):
		mu, clusters = find_centers(X,k)
		#pp.pprint(clusters)
		Wks[indk] = np.log(Wk(mu, clusters))
		# Create B reference datasets
		B = 10
		BWkbs = np.zeros(B)
		for i in range(B):
			Xb = []
			for n in range(len(X)):
				Xb.append([random.uniform(xmin,xmax),
						  random.uniform(ymin,ymax)])
			Xb = np.array(Xb)
			mu, clusters = find_centers(Xb,k)
			BWkbs[i] = np.log(Wk(mu, clusters))
		Wkbs[indk] = sum(BWkbs)/B
		sk[indk] = np.sqrt(sum((BWkbs-Wkbs[indk])**2)/B)
	sk = sk*np.sqrt(1+1/B)
	return(ks, Wks, Wkbs, sk)			   






X = []
filename = utilities.getnewestfile('*.xlsx',Path=os.path.dirname(__file__)+"\\")
for a,b in utilities.loadexceltabinmemory(file=filename,tab='DataForAnalysis')[1:]:
	X.append([float(a),float(b)])
X = np.array(X)
#pp.pprint(X)

# Calculate full variance
fullvar=[[np.array([0,0])],{0:X}]
FullVar=distances(fullvar)

# find elbow
#ks, logWks, logWkbs, sk = gap_statistic(X)
#pp.pprint(["gap stat: ",ks, logWks, logWkbs, sk])
#for i in range(1,len(logWks)-1):
	#print(i,logWks[i]>=logWks[i+1]-sk[i+1],logWks[i],logWks[i+1]-sk[i+1])
#	print("K=",i,"GapK=",logWkbs[i]-logWks[i],"GapK >= GapK+1 - Sk+1 : ",logWkbs[i]-logWks[i]>=logWkbs[i+1]-logWks[i+1]-sk[i+1])

# Monte Carlo:
simk=[]
for simulation in range(1,10):
	# find elbow
	ks, logWks, logWkbs, sk = gap_statistic(X)
	i=1
	while not logWkbs[i]-logWks[i]>=logWkbs[i+1]-logWks[i+1]-sk[i+1]:
		#print("K=",i,"GapK=",logWkbs[i]-logWks[i],"GapK >= GapK+1 - Sk+1 : ",logWkbs[i]-logWks[i]>=logWkbs[i+1]-logWks[i+1]-sk[i+1])
		i+=1
	simk.append(i)
i=int(np.median(simk))
print("K=",i,"GapK=",logWkbs[i]-logWks[i],"GapK >= GapK+1 - Sk+1 : ",logWkbs[i]-logWks[i]>=logWkbs[i+1]-logWks[i+1]-sk[i+1])

print("selected Ks: ",i)
results=find_centers(X,i)
ExplainedVar=distances(results)

# Show results

# Show constraints and clusters
# List point types
pointtypes1=["gx","gD","g*","gx","gD","g*"]

# Show all cluster centres
for i in list(range(0,len(results[0]))) :
	plt.plot(
		np.matrix(results[0][i]).transpose()[0], np.matrix(results[0][i]).transpose()[1],pointtypes1[i]
		)

pointtypes=["bx","yD","c*","bx","yD","c*"]
# Show all cluster points
for i in list(range(0,len(results[1]))) :
	plt.plot(
		np.matrix(results[1][i]).transpose()[0],np.matrix(results[1][i]).transpose()[1],pointtypes[i]
		)
plt.show()