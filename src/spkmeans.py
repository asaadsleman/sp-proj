import sys
import numpy as np
import pandas as pd
from enum import Enum, auto, unique

np.random.seed(0)

def main():
    # Read user CMD arguments
    k = sys.argv[1]
    goal = sys.argv[2]
    filelines = pd.read_csv(sys.argv[3]).astype('float64')
    # Calculate Centroids as in HW2
    centr_inf = centroids_init(filelines, k)
    centr_indx = centr_inf[1]
    centroids = centr_inf[0]
    # Interface with C

    #Output
    if goal == 'spk':
        print(centr_indx)
        print(centroids)
    elif goal == 'wam':
        return
    elif goal == 'ddg':
        return
    elif goal == 'lnorm':
        return
    elif goal == 'jacobi':
        return

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
        r = np.random.rand()
        for j, p in enumerate(accum_probs):
            if r < p:
                i = j
                break
        centroids.append(data[i])
        cent_inds.append(i)

    return [np.array(centroids), np.np.array(cent_inds)]


if __name__ == '__main__':
    main()