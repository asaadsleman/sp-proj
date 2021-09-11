import sys
import numpy as np
from numpy.core.fromnumeric import shape
from numpy.random import rand
import pandas as pd
import spkmeans


def main():
    # Read user CMD arguments
    k = sys.argv[1]
    goal = sys.argv[2]
    filelines = pd.read_csv(sys.argv[3], header=None).astype('float64').to_numpy()
    data = filelines.tolist()
    np.random.seed(0)
    # Interface with C
    perform(k, goal, data)
    

    return None


def perform(k, objective, data):
    if(objective == "jacobi"):
        jacobi = spkmeans.BuildJacobi(data)
        ev = [data[i][i] for i in range(data.shape[0])]
        print_array(ev)
        print("\n", end='')
        print_matrix(jacobi)
        return

    Wam = spkmeans.WAMatrix(data, len(data), len(data[0]))
    if objective == 'wam':
        print_matrix(Wam)
        return
    # ddg = spkmeans.BuildDDG(Wam)
    # if objective == 'ddg':
    #     print_matrix(ddg)
    #     return
    # lap = spkmeans.BuildLap(ddg, Wam)
    # if objective == 'lnorm':
    #     print_matrix(lap)
    #     return
    # jacobi = spkmeans.BuildJacobi(lap)
    # if objective == 'jacobi':
    #     print(jacobi)
    #     return
    # ev = [lap[i][i] for i in range(lap.shape[0])]
    # # build U and normalize it by row
    # zeroes = np.zeros((lap.shape[0], k), np.float64)
    # u = zeroes.tolist()
    # spkmeans.BuildU(jacobi, ev, u, k)
    # # Calculate Initial Centroids as in HW2
    # centr_inf = centroids_init(u, k)
    # centr_indx = centr_inf[1].tolist()
    # centroids = centr_inf[0].tolist()
    # # Run Classification
    # classifc = spkmeans.fit(u, centroids)
    # print_matrix(classifc)


def print_array(arr):
    for i in range(len(arr)):
        num = arr[i]
        if(i == (len(arr) - 1)):
            print(f"{num:.4f}", end='')
        else:
            print(f"{num:.4f},", end='')


def print_matrix(mat):
    for i in range(len(mat)):
        for j in range(len(mat[0])):
            num = mat[i][j]
            if(j == (len(mat[0]) -1)):
                print(f"{num:.4f}", end='')
            else:
                print(f"{num:.4f},", end='')
        if(i != (len(mat) - 1)):
            print("\n", end='')


def centroids_init(data, k):
    # Create cluster centroids using the k-means++ algorithm.
    # Parameters:  
    # ds : dataset for init
    # k : clust numbers
    # Returns centroids : numpy array
    #     Collection of k centroids as a numpy array.
    npdata = np.array(data)
    rangeind = [i for i in range(npdata.shape[0])]
    rnd_ind = np.random.choice(rangeind)
    centroids = [npdata[rnd_ind]]
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
        centroids.append(npdata[i])
        cent_inds.append(i)

    return [np.array(centroids), np.np.array(cent_inds)]



if __name__ == '__main__':
    main()