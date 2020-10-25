# -*- coding: utf-8 -*-
import numpy as np
import math

def _distances(values, norm=2):
    """
    Compute distance matrix for all data sets (rows of values)
    
    Parameters
    ----------
    values : 2-dimensional array
        Rows represent days and columns values
    norm : integer, optional
        Compute the distance according to this norm. 2 is the standard
        Euklidean-norm.
    
    Return
    ------
    d : 2-dimensional array
        Distances between each data set
    """
    # Initialize distance matrix
    d = np.zeros((values.shape[1], values.shape[1]))

    # Define a function that computes the distance between two days
    dist = (lambda day1, day2, r: 
            math.pow(np.sum(np.power(np.abs(day1 - day2), r)), 1/r))

    # Remember: The d matrix is symmetrical!
    for i in range(values.shape[1]): # loop over first days
        for j in range(i+1, values.shape[1]): # loop second days
            d[i, j] = dist(values[:,i], values[:,j], norm)
    
    # Fill the remaining entries
    d = d + d.T
    
    return d


###############################################################################
    

def cluster(inputs, n_cluster=12, norm=2, weights=None, cluster_method="PAM"):
    """
    Cluster a set of inputs into clusters by solving a k-medoid problem.
    
    Parameters
    ----------
    inputs : 2-dimensional array
        First dimension: Number of different input types.
        Second dimension: Values for each time step of interes.
    n_cluster : integer, optional
        How many clusters shall be computed?
    norm : integer, optional
        Compute the distance according to this norm. 2 is the standard
        Euklidean-norm.
    time_limit : integer, optional
        Time limit for the optimization in seconds
    mip_gap : float, optional
        Optimality tolerance (0: proven global optimum)
    weights : 1-dimensional array, optional
        Weight for each input. If not provided, all inputs are treated equally.
    
    Returns
    -------
    scaled_typ_days : 
        Scaled typical demand days. The scaling is based on the annual demands.
    nc : array_like
        Weighting factors of each cluster
    z : 2-dimensional array
        Mapping of each day to the clusters
    """
    # Determine time steps per day
    len_day = int(inputs.shape[1] / 365)
    
    # Set weights if not already given
    if weights == None:
        weights = np.ones(inputs.shape[0])
    elif not sum(weights) == 1: # Rescale weights
        weights = np.array(weights) / sum(weights)
    
    # Manipulate inputs
    # Initialize arrays
    inputsTransformed = []
    inputsScaled = []
    inputsScaledTransformed = []
    
    # Fill and reshape
    # Scaling to values between 0 and 1, thus all inputs shall have the same
    # weight and will be clustered equally in terms of quality 
    for i in range(inputs.shape[0]):
        vals = inputs[i,:]
        if np.max(vals) == np.min(vals):
            temp = np.zeros_like(vals)
        else:
            temp = ((vals - np.min(vals)) / (np.max(vals) - np.min(vals)) * math.sqrt(weights[i]))
            
        inputsScaled.append(temp)
        inputsScaledTransformed.append(temp.reshape((len_day, 365), order="F"))
        inputsTransformed.append(vals.reshape((len_day, 365), order="F"))

    # Put the scaled and reshaped inputs together
    L = np.concatenate(tuple(inputsScaledTransformed))

    # Compute distances
    d = _distances(L, norm)    
    
    # PAM clustering    
    if cluster_method == "PAM":
        print("Clustering design days with PAM algorithm (n_cluster=" + str(n_cluster) + ")...")      
        scaled_typ_days, nc, z = k_medoids_PAM(d, n_cluster, inputs, len_day)
    
    return scaled_typ_days, nc, z


def k_medoids_PAM(D, k, inputs, len_day, tmax=200):
    """
    Cite source of code as follows:
        
    Bauckhage C. Numpy/scipy Recipes for Data Science: k-Medoids Clustering[R]. 
    Technical Report, University of Bonn, 2015.
    
    https://github.com/letiantian/kmedoids
    """
    
    # determine dimensions of distance matrix D
    m, n = D.shape

    if k > n:
        raise Exception('too many medoids')

    # find a set of valid initial cluster medoid indices since we
    # can't seed different clusters with two points at the same location
    valid_medoid_inds = set(range(n))
    invalid_medoid_inds = set([])
    rs,cs = np.where(D==0)
    # the rows, cols must be shuffled because we will keep the first duplicate below
    index_shuf = list(range(len(rs)))
    #np.random.shuffle(index_shuf)
    rs = rs[index_shuf]
    cs = cs[index_shuf]
    for r,c in zip(rs,cs):
        # if there are two points with a distance of 0...
        # keep the first one for cluster init
        if r < c and r not in invalid_medoid_inds:
            invalid_medoid_inds.add(c)
    valid_medoid_inds = list(valid_medoid_inds - invalid_medoid_inds)

    if k > len(valid_medoid_inds):
        raise Exception('too many medoids (after removing {} duplicate points)'.format(
            len(invalid_medoid_inds)))
    # randomly initialize an array of k medoid indices
    M = np.array(valid_medoid_inds)
    #np.random.shuffle(M)
    M = np.sort(M[:k])

    # create a copy of the array of medoid indices
    Mnew = np.copy(M)

    # initialize a dictionary to represent clusters
    C = {}
    for t in range(tmax):
        # determine clusters, i. e. arrays of data indices
        J = np.argmin(D[:,M], axis=1)
        for kappa in range(k):
            C[kappa] = np.where(J==kappa)[0]
        # update cluster medoids
        for kappa in range(k):
            J = np.mean(D[np.ix_(C[kappa],C[kappa])],axis=1)
            j = np.argmin(J)
            Mnew[kappa] = C[kappa][j]
        np.sort(Mnew)
        # check for convergence
        if np.array_equal(M, Mnew):
            break
        M = np.copy(Mnew)
    else:
        # final update of cluster memberships
        J = np.argmin(D[:,M], axis=1)
        for kappa in range(k):
            C[kappa] = np.where(J==kappa)[0]

    # Sort M and accordingly C (important for further process)
    dict_sort = {}
    for j in range(len(M)):
        dict_sort.update({M[j]: C[j]})
        
    sorted_dict = {}
    for sorted_key in sorted(dict_sort.keys()):
        sorted_dict[sorted_key] = dict_sort[sorted_key]
    
    C_sorted = {}
    M_sorted = []
    counter = 0
    for j in sorted_dict.keys():
        C_sorted[counter] = sorted_dict[j]
        M_sorted.append(j)
        counter += 1        
    M_sorted = np.array(M_sorted)

    C = C_sorted
    M = M_sorted
    
    # Create weights (nc, according to tsz code)
    nc = []
    for cluster in C:
        nc.append(len(C[cluster]))
    nc = np.array(nc, dtype="int")

    # Create assignment matrix (z, according to tsz code)
    z = np.zeros((365, 365))
    for cluster in C:
        for day in C[cluster]:
            z[M[cluster],day] = 1
        
    # Create scaled type days (scaled_typ_days, according to tsz code)
    scaled_typ_days = []
    for series in range(inputs.shape[0]):
         clustered_series = np.zeros((k, len_day))
         for cluster in C:
             design_day = M[cluster]
             clustered_series[cluster] = inputs[series][0+design_day*24:24+design_day*24]
         
         sum_year = sum(sum(clustered_series[cluster][t] for t in range(24)) * nc[cluster] for cluster in C)
         if sum_year == 0:
             scaled_clustered_series = clustered_series = np.zeros((k, len_day))
         else:
             scale_factor = np.sum(inputs[series]) / sum_year
             scaled_clustered_series = clustered_series * scale_factor  
         scaled_typ_days.append(scaled_clustered_series)
    
    return scaled_typ_days, nc, z


###############################################################################

def calc_clust_error(scaled_typ_days, z, inputs):
    
    full = []
    typedays = []
    for d in range(365):
        if any(z[d]):
            typedays.append(d)            
    # Arrange time series
    for item in range(len(scaled_typ_days)):
        full_series = np.zeros(8760)    
        for d in range(365):
            match = np.where(z[:,d] == 1)[0][0]
            typeday = np.where(typedays == match)[0][0]
            full_series[24*d:24*(d+1)] = scaled_typ_days[item][typeday,:]
        full.append(full_series)
    
    err_1_norm = sum(np.sum(np.absolute(full[item]-inputs[item])) for item in range(len(scaled_typ_days)))

    return err_1_norm
