#! /usr/bin/python3.4

# Based on:
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
    oldmu = random.sample(list(X), K)
    mu = random.sample(list(X), K)
    while not has_converged(mu, oldmu):
        oldmu = mu
        # Assign all points in X to clusters
        clusters = cluster_points(X, mu)
        # Re-evaluate centers
        mu = reevaluate_centers(oldmu, clusters)
    return(mu, clusters)


X = []
filename = utilities.getnewestfile('*.xlsx',Path=os.path.dirname(__file__)+"\\")
for a,b in utilities.loadexceltabinmemory(file=filename,tab='DataForAnalysis')[1:]:
	X.append([float(a),float(b)])
X = np.array(X)

maxK=7
elbow_search=[]

# Calculate full variance
fullvar=[[np.array([0,0])],{0:X}]
FullVar=distances(fullvar)
resultsVector=[]

# find elbow
# https://datasciencelab.wordpress.com/2013/12/27/finding-the-k-in-k-means-clustering/

for K in range(1,maxK):
	results=find_centers(X,K)
	#pp.pprint(results)
	# Calculate explained variance
	ExplainedVar=distances(results)
	resultsVector.append(results)
	elbow_search.append(np.array([K,(FullVar-ExplainedVar)/FullVar]))
elbow_search=np.array(elbow_search)
#pp.pprint(elbow_search)
elbows=[]
for i in range(2,len(elbow_search)):
	elbows.append(np.array([i,(elbow_search[i-1][1]-elbow_search[i-2][1]),(elbow_search[i][1]-elbow_search[i-1][1]),(elbow_search[i-1][1]-elbow_search[i-2][1])/(elbow_search[i][1]-elbow_search[i-1][1])]))
elbows = np.array(elbows)
pp.pprint(elbows)
el=np.argmax(elbows.transpose()[3])
print(elbows[el][1:3])

pp.pprint(np.matrix(elbow_search).transpose()[0])
pp.pprint(np.matrix(elbow_search).transpose()[0][0,el])
plt.plot(np.matrix(elbow_search).transpose()[0],np.matrix(elbow_search).transpose()[1],"gx")
plt.plot(np.matrix(elbow_search).transpose()[0][0,el],np.matrix(elbow_search).transpose()[1][0,el],"yD")
plt.show()

results=resultsVector[el]

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
