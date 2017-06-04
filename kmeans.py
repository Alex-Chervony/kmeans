#! /usr/bin/python3.6

# https://stackoverflow.com/questions/44335137/k-means-with-a-centroid-constraint

# Original code from:
# https://datasciencelab.wordpress.com/2013/12/12/clustering-with-k-means-in-python/

import matplotlib.pyplot as plt
import numpy as np
import random

# Generate possible points.
def possible_points(n=20):
	y=list(np.linspace( -1, 1, n ))
	x=[-1.2]
	X=[]
	for i in list(range(1,n)):
		x.append(x[i-1]+random.uniform(-2/n,2/n) )
	for a,b in zip(x,y):
		X.append(np.array([a,b]))
	X = np.array(X)
	return X

# Generate sample
def init_board_gauss(N, k):
    n = float(N)/k
    X = []
    for i in range(k):
        c = (random.uniform(-1, 1), random.uniform(-1, 1))
        s = random.uniform(0.05,0.5)
        x = []
        while len(x) < n:
            a, b = np.array([np.random.normal(c[0], s), np.random.normal(c[1], s)])
            # Continue drawing points from the distribution in the range [-1,1]
            if abs(a) < 1 and abs(b) < 1:
                x.append([a,b])
        X.extend(x)
    X = np.array(X)[:N]
    return X

# Identify points for each center.
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

# Get closest possible point for each cluster.
def closest_point(cluster,possiblePoints):
	closestPoints=[]
	# Check average distance for each point.
	for possible in possiblePoints:
		distances=[]
		for point in cluster:
			distances.append(np.linalg.norm(possible-point))
		closestPoints.append(np.mean(distances))
	return possiblePoints[closestPoints.index(min(closestPoints))]

# Calculate new centers.
# Here the 'coast constraint' goes.
def reevaluate_centers(clusters,possiblePoints):
    newmu = []
    keys = sorted(clusters.keys())
    for k in keys:
        newmu.append(closest_point(clusters[k],possiblePoints))
    return newmu
 
# Check whether centers converged.
def has_converged(mu, oldmu):
	return (set([tuple(a) for a in mu]) == set([tuple(a) for a in oldmu]))
 
# Meta function that runs the steps of the process in sequence.
def find_centers(X, K, possiblePoints):
    # Initialize to K random centers
    oldmu = random.sample(list(possiblePoints), K)
    mu = random.sample(list(possiblePoints), K)
    while not has_converged(mu, oldmu):
        oldmu = mu
        # Assign all points in X to clusters
        clusters = cluster_points(X, mu)
        # Re-evaluate centers
        mu = reevaluate_centers(clusters,possiblePoints)
        print("a")
    return(mu, clusters)


K=3
X = init_board_gauss(30,K)
possiblePoints=possible_points()
results=find_centers(X,K,possiblePoints)

# Show results

# Show constraints and clusters
# List point types
pointtypes1=["gx","gD","g*"]

plt.plot(
	np.matrix(possiblePoints).transpose()[0],np.matrix(possiblePoints).transpose()[1],'m.'
	)

for i in list(range(0,len(results[0]))) :
	plt.plot(
		np.matrix(results[0][i]).transpose()[0], np.matrix(results[0][i]).transpose()[1],pointtypes1[i]
		)

pointtypes=["bx","yD","c*"]
# Show all cluster points
for i in list(range(0,len(results[1]))) :
	plt.plot(
		np.matrix(results[1][i]).transpose()[0],np.matrix(results[1][i]).transpose()[1],pointtypes[i]
		)
plt.show()