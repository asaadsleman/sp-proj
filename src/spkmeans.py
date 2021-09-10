import sys
import numpy as np
from numpy.core.fromnumeric import shape
from numpy.random import rand
import pandas as pd
from enum import Enum, auto, unique
import spkmeans

np.random.seed(0)

def main():
    # Read user CMD arguments
    k = sys.argv[1]
    goal = sys.argv[2]
    filelines = pd.read_csv(sys.argv[3]).astype('float64').to_numpy()
    # Interface with C
    perform(k, goal, filelines)
    

    return None


def perform(k, objective, data):
    if(objective == "jacobi"):
        jacobi = spkmeans.BuildJacobi(data)
        ev = [data[i][i] for i in range(data.shape[0])]
        print_array(ev)
        print("\n")
        print_matrix(jacobi)
        return

    Wam = spkmeans.WAMatrix(data, data.shape[0])
    if objective == 'wam':
        print_matrix(Wam)
        return
    ddg = spkmeans.BuildDDG(Wam)
    if objective == 'ddg':
        print_matrix(ddg)
        return
    lap = spkmeans.BuildLap(ddg, Wam)
    if objective == 'lnorm':
        print_matrix(lap)
        return
    jacobi = spkmeans.BuildJacobi(lap)
    if objective == 'jacobi':
        print(jacobi)
        return
    ev = [lap[i][i] for i in range(lap.shape[0])]
    # build U and normalize it by row
    u = np.zeros((lap.shape[0], k), np.float64)
    spkmeans.BuildU(jacobi, ev, u, k)
    # Calculate Initial Centroids as in HW2
    centr_inf = centroids_init(u, k)
    centr_indx = centr_inf[1]
    centroids = centr_inf[0]
    # Run Classification
    classifc = spkmeans.fit(u, centroids)
    print_matrix(classifc)


def print_array(arr):
    for i in range(arr.shape[0]):
        num = arr[i]
        if(abs(num) < 0.0001):
            num = 0.0
        if(i == (arr.shape[0] - 1)):
            print(f"{num:.4f}")
        else:
            print(f"{num:.4f},")


def print_matrix(mat):
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            num = mat[i][j]
            if(abs(num) < 0.0001):
                num = 0.0
            if(j == (mat.shape[1] -1)):
                print(f"{num:.4f}")
            else:
                print(f"{num:.4f},")
        if(i != (mat.shape[0] - 1)):
            print("\n")


def centroids_init(data, k):
    # Create cluster centroids using the k-means++ algorithm.
    # Parameters:  
    # ds : dataset for init
    # k : clust numbers
    # Returns centroids : numpy array
    #     Collection of k centroids as a numpy array.
    
    rnd_ind = np.random.randint(0, data.shape[0])
    centroids = [data[rnd_ind]]
    cent_inds = [rnd_ind]

    for _ in range(1, k):
        dist_sq = np.array([min([np.inner(cent-x,cent-x) for cent in centroids]) for x in data])
        probs = dist_sq/dist_sq.sum()
        accum_probs = probs.cumsum()
        r = np.random.choice(accum_probs)
        for j, p in enumerate(accum_probs):
            if r < p:
                i = j
                break
        centroids.append(data[i])
        cent_inds.append(i)

    return [np.array(centroids), np.np.array(cent_inds)]


def kmeans(X, k,I, centroids, max_iter = 300):
    n_samples = X.shape[0]
    n_features = X.shape[1]
    classifications = np.zeros(n_samples, dtype = np.int64)

    loss = 0
    for m in range(0, max_iter):
        # Compute the classifications
        for i in range(0, n_samples):
            distances = np.zeros(k, dtype= np.float64)
            for j in range(0, k):
                distances[j] = np.sqrt(np.sum(np.power(X[i, :] - centroids[j], 2))) 
        classifications[i] = np.argmin(distances)
        new_centroids = np.zeros((k, n_features), dtype= np.float64)
        new_loss = 0
        for j in range(0, k):
        # compute centroids
            J = np.where(classifications == j)
            X_C = X[J]
            new_centroids[j] = X_C.mean(axis = 0)

        # Compute loss
            for i in range(0, X_C.shape[0]):
                new_loss += np.sum(np.power(X_C[i, :] - centroids[j], 2))

        # Stopping criterion            
            if np.abs(loss - new_loss) == 0:
                return new_centroids
    
        centroids = new_centroids
        loss = new_loss

    print("Failed to converge!")
    return centroids


if __name__ == '__main__':
    main()