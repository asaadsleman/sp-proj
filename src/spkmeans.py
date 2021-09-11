import sys
import numpy as np
from numpy.core.fromnumeric import shape
from numpy.random import rand
import pandas as pd
from sklearn.metrics.pairwise import pairwise_distances
import spkmeans


def main():
    # Read user CMD arguments
    k = int(sys.argv[1])
    goal = sys.argv[2]
    filelines = pd.read_csv(sys.argv[3], header=None).astype('float64').to_numpy()
    data = filelines.tolist()
    # Interface with C
    perform(k, goal, data)
    

    return None


def perform(k, objective, data):
    dim = len(data)
    if(objective == "jacobi"):
        ev = [0.0 for i in range(dim)]
        res = spkmeans.BuildJacobi(data, ev, dim)
        ev = res[0]
        jacobi = res[1]
        print_array(ev)
        print("\n", end='')
        print_matrix(jacobi)
        return
    
    Wam = spkmeans.WAMatrix(data, dim, len(data[0]))
    if objective == 'wam':
        print_matrix(Wam)
        return
    ddg = spkmeans.BuildDDG(Wam, dim)
    if objective == 'ddg':
        print_matrix(ddg)
        return
    lap = spkmeans.BuildLap(ddg, Wam, dim)
    if objective == 'lnorm':
        print_matrix(lap)
        return
    if objective != 'spk':
        print("Invalid Input!", end='')
        return
    ev = [0.0 for i in range(dim)]
    result = spkmeans.BuildJacobi(lap, ev, dim)
    ev = result[0]
    jacobi = result[1]
    # build U and normalize it by row
    u = spkmeans.BuildU(jacobi, ev, dim,  k)
    # switch to np for center initialization
    # update k first
    k = len(u[0])
    npu = np.array(u)
    # Calculate Initial Centroids as in HW2
    centr = centroids_init(npu, k)
    centr_indx = centr[0]
    centroids = centr[1]
    # Run Classification
    print_array(centr_indx)
    print("\n", end='')
    classifc = spkmeans.fit(u, centroids, dim, k)
    print_matrix(classifc)


def print_array(arr):
    for i in range(len(arr)):
        num = arr[i]
        if(num < 0.0 and abs(num) < 0.9e-4):
            num = 0.0
        if(i == (len(arr) - 1)):
            print(f"{num:8.4f}", end='')
        else:
            print(f"{num:8.4f},", end='')


def print_matrix(mat):
    for i in range(len(mat)):
        for j in range(len(mat[0])):
            num = mat[i][j]
            if(num < 0.0 and abs(num) < 0.9e-4):
                num = 0.0
            if(j == (len(mat[0]) -1)):
                print(f"{num:8.4f}", end='')
            else:
                print(f"{num:8.4f},", end='')
        if(i != (len(mat) - 1)):
            print("\n", end='')


def centroids_init(data, k):
    # Create cluster centroids using the k-means++ algorithm.
    # Parameters:  
    # ds : dataset for init
    # k : clust numbers
    # Returns centroids : numpy array
    #     Collection of k centroids as a numpy array.
    np.random.seed(0)
    indxs = [0 for i in range(k)]
    centroids = np.zeros((k, k))
    rnd_ind = np.random.choice(data.shape[0])
    centroids[0] = data[rnd_ind,:]
    indxs[0] = rnd_ind
    distances = pairwise_distances(data, [centroids[0]]).flatten()

    for i in range(1,k):
        summ = np.sum(distances)
        ps = [distances[i]/summ for i in range(distances.shape[0])]
        rnd_ind = np.random.choice(data.shape[0], size = 1, p=ps)
        centroids[i] = data[rnd_ind,:]
        indxs[i] = rnd_ind.item()
        if i == k-1:
            break
        distances_new = pairwise_distances(data, [centroids[i]]).flatten()
        distances = np.min(np.vstack((distances, distances_new)), axis = 0)
    
    return [indxs, centroids.tolist()]


if __name__ == '__main__':
    main()