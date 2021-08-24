import sys
import numpy as np
from numpy.random import rand
import pandas as pd
from enum import Enum, auto, unique

np.random.seed(0)

def main():
    # Read user CMD arguments
    k = sys.argv[1]
    goal = sys.argv[2]
    filelines = pd.read_csv(sys.argv[3]).astype('float64')
    # Interface with C
    dim = filelines.shape[0]
    Wam = spkmeans.WAMatrix(filelines, dim)
    if goal == 'wam':
        print(Wam)
        return
    ddg = spkmeans.BuildDDG(Wam, dim)
    if goal == 'ddg':
        print(ddg)
        return
    lap = spkmeans.BuildLap(ddg, Wam, dim)
    if goal == 'lnorm':
        print(lap)
        return
    jacobi = spkmeans.BuildJacobi(dim, lap)
    if goal == 'jacobi':
        print(jacobi)
        return
    ev = [jacobi[i][i] for i in range(len(jacobi))]
    # determine K (if 0 use heuristic)
    if k == 0:
        k = spkmeans.eigengap(dim, ev)
    # Calculate Centroids as in HW2
    centr_inf = centroids_init(filelines, k)
    centr_indx = centr_inf[1]
    centroids = centr_inf[0]
    # build U and T
    
    
    

    #Output
    if goal == 'spk':
        print(centr_indx)
        print(centroids)

    return None


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


if __name__ == '__main__':
    main()