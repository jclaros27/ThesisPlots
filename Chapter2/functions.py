import collections
import matplotlib.pyplot as plt
import numpy as np
import scipy.spatial as spatial
import scipy.spatial.distance as dist
import scipy.cluster.hierarchy as hier

import numpy as np 
import matplotlib.pyplot as plt

def det_and_trace_from_jacobian_matrix(xx,yy,omega,delta,k): 

    matrix00 = -k[1,0]*np.cos(-xx-delta[1,0]) - k[0,1]*np.cos(xx-delta[0,1]) - k[2,1]*np.cos(xx-yy-delta[2,1])
    matrix01 = -k[2,0]*np.cos(-yy-delta[2,0]) + k[2,1]*np.cos(xx-yy-delta[2,1])
    matrix10 = -k[1,0]*np.cos(-xx-delta[1,0]) + k[1,2]*np.cos(yy-xx-delta[1,2])
    matrix11 = -k[2,0]*np.cos(-yy-delta[2,0]) - k[0,2]*np.cos(yy-delta[0,2]) - k[1,2]*np.cos(yy-xx-delta[1,2]) 

    det = matrix00*matrix11 - matrix01*matrix10
    trace = matrix00 + matrix11
    return det, trace

def jacobian_matrix(xx,yy,omega,delta,k):

    matrix00 = -k[1,0]*np.cos(-xx-delta[1,0]) - k[0,1]*np.cos(xx-delta[0,1]) - k[2,1]*np.cos(xx-yy-delta[2,1])
    matrix01 = -k[2,0]*np.cos(-yy-delta[2,0]) + k[2,1]*np.cos(xx-yy-delta[2,1])
    matrix10 = -k[1,0]*np.cos(-xx-delta[1,0]) + k[1,2]*np.cos(yy-xx-delta[1,2])
    matrix11 = -k[2,0]*np.cos(-yy-delta[2,0]) - k[0,2]*np.cos(yy-delta[0,2]) - k[1,2]*np.cos(yy-xx-delta[1,2]) 

    return np.array([[matrix00, matrix01],[matrix10, matrix11]])


def jacobian_2motif(xx,omega,delta,k):
    det = -k[1,0]*np.cos(-xx-delta[1,0])-k[0,1]*np.cos(xx-delta[0,1])
    return det 

def phase_locked_2motif(xx,omega,delta,k): 
    theta0 = omega[0]+k[1,0]*np.sin(-xx-delta[1,0])
    theta1 = omega[1]+k[0,1]*np.sin(xx-delta[0,1])
    dx = theta0-theta1
    return dx 

def phase_locked_states(xx,yy,omega, delta, k): 
    # define x = theta1 - theta2
    # define y = theta1 - theta3 
    # dtheta1 = omega1 + k21*np.sin(theta1-theta2) + k31*np.sin(theta1-theta3)
    # dtheta2 = omega2 + k12*np.sin(theta2-theta1) + k32*np.sin(theta2-theta3)
    # dtheta3 = omega3 + k13*np.sin(theta3-theta1) + k23*np.sin(theta3-theta2)

    dtheta0 = omega[0] + k[1,0]*np.sin(-xx-delta[1,0])  + k[2,0]*np.sin(-yy-delta[2,0])
    dtheta1 = omega[1] + k[0,1]*np.sin(xx-delta[0,1])   + k[2,1]*np.sin(xx-yy-delta[2,1])
    dtheta2 = omega[2] + k[1,2]*np.sin(yy-xx-delta[1,2]) + k[0,2]*np.sin(yy-delta[0,2])
    dx = dtheta0 - dtheta1
    dy = dtheta0 - dtheta2

    return dx, dy

def intersection(points1, points2, eps):
    tree = spatial.KDTree(points1)
    distances, indices = tree.query(points2, k=1, distance_upper_bound=eps)
    intersection_points = tree.data[indices[np.isfinite(distances)]]
    return intersection_points

def cluster(points, cluster_size):
    dists = dist.pdist(points, metric='sqeuclidean')
    linkage_matrix = hier.linkage(dists, 'average')
    groups = hier.fcluster(linkage_matrix, cluster_size, criterion='distance')
    return np.array([points[cluster].mean(axis=0)
                     for cluster in clusterlists(groups)])

def contour_points(contour, steps=1):
    return np.row_stack([path.interpolated(steps).vertices
                         for linecol in contour.collections
                         for path in linecol.get_paths()])

def contour_points_per_path(contour, steps=1):
    return list([path.interpolated(steps).vertices
                         for linecol in contour.collections
                         for path in linecol.get_paths()])

def contour_points(contour, steps=1):
    return np.row_stack([path.interpolated(steps).vertices
                         for linecol in contour.collections
                         for path in linecol.get_paths()])

def contour_points_per_path(contour, steps=1):
    return list([path.interpolated(steps).vertices
                         for linecol in contour.collections
                         for path in linecol.get_paths()])
def clusterlists(T):
    '''
    http://stackoverflow.com/a/2913071/190597 (denis)
    T = [2, 1, 1, 1, 2, 2, 2, 2, 2, 1]
    Returns [[0, 4, 5, 6, 7, 8], [1, 2, 3, 9]]
    '''
    groups = collections.defaultdict(list)
    for i, elt in enumerate(T):
        groups[elt].append(i)
    return sorted(groups.values(), key=len, reverse=True)


def get_intersection_points_2motif(X,dx,det,delt,eps=1e-2, cluster_size=100, create_plot=False):
    a = np.abs(dx)
    w = np.r_[True, a[1:] < a[:-1]] & np.r_[a[:-1] < a[1:], True]
    ww = a[w]<eps

    if len(X[w][ww])>3:
        xlist, jlist, dlist = [],[],[]
    else:
        xlist = X[w][ww]
        jlist = det[w][ww]
        dlist = delt*np.ones(len(xlist))

    return xlist, jlist, dlist

def get_intersection_points(X,Y, dx,dy,det, trace, delt,eps=1.0, cluster_size=100, create_plot=False):
    plt.figure()
    contour1 = plt.contour(dx, levels=[0],colors='k')
    contour2 = plt.contour(dy, levels=[0],colors='r')
    plt.close()

    points1 = contour_points(contour1)
    points2 = contour_points(contour2)
    intersection_points = intersection(points1, points2, eps)
    try:
        intersection_points = cluster(intersection_points, cluster_size)

        xlist = X[intersection_points[:,1].astype(int),intersection_points[:,0].astype(int)]
        ylist = Y[intersection_points[:,1].astype(int),intersection_points[:,0].astype(int)] 
        jlist = det[intersection_points[:,1].astype(int),intersection_points[:,0].astype(int)]
        tlist = trace[intersection_points[:,1].astype(int), intersection_points[:,0].astype(int)]
        dlist = delt*np.ones(len(xlist))
        if create_plot:
            fig, ax = plt.subplots(1,2,figsize=(10,5))
            delt_ = delt/np.pi
            ax[0].contourf(X,Y,det, levels=[0,np.max(det)])
            ax[0].contour(X,Y,det, levels=[0],colors='purple')
            #det_contour = ax[0].contourf(X,Y,det, levels=21)
            # fig.colorbar(det_contour, ax=ax[0])
            ax[0].contour(X,Y,dy, levels=[0],colors='k')
            ax[0].contour(X,Y,dx, levels=[0],colors='r')

            # trace
            ax[1].contourf(X,Y,trace, levels=[0,np.max(trace)])
            ax[1].contour(X,Y,trace, levels=[0],colors='purple')
            # trace_contour = ax[1].contourf(X,Y,trace, levels=21)
            # fig.colorbar(trace_contour, ax=ax[1])
            ax[1].contour(X,Y,dy, levels=[0],colors='k')
            ax[1].contour(X,Y,dx, levels=[0],colors='r')

            ax[0].scatter( xlist, ylist, s=40, color="blue" )
            ax[1].scatter( xlist, ylist, s=40,color="blue" )

            ax[0].set_xlabel(r"$\theta_{12}$",fontsize=15)
            ax[1].set_xlabel(r"$\theta_{12}$",fontsize=15)
            ax[0].set_ylabel(r"$\theta_{13}$",fontsize=15)
            ax[0].set_title(r"$\Delta = %.2f \pi$"%delt_,fontsize=15)

    except ValueError: 
        xlist, ylist, jlist, tlist, dlist = [],[],[],[],[]
    
    return xlist, ylist, jlist, tlist, dlist

def compute_stability(fixed_points, matrices):
    mask = np.zeros(np.shape(fixed_points)[0], dtype=bool)
    for i, (points, matrix) in enumerate(zip(fixed_points,matrices)):
        x, y, delt = points
        det, trace = matrix

        if (det > 1e-2) and (trace < 0.0) and (np.abs(trace) > 1e-1):
            mask[i] = True
        else:
            mask[i] = False
    return mask

def detect_outliers(matrix, matrix1, matrix2, plot=False):
    from sklearn.cluster import DBSCAN
    from scipy.ndimage import gaussian_filter
    import copy as copy

    # Get the number of rows and columns in the matrix
    rows, cols = matrix.shape
    # Create the indices for the rows and columns using meshgrid
    row_indices, col_indices = np.meshgrid(range(rows), range(cols), indexing='ij')
    ni,nj = np.shape(row_indices)[0],np.shape(row_indices)[1]
    # Reshape the matrix and the indices into a 3D dataset
    dataset = np.column_stack((row_indices.flatten(), col_indices.flatten(), matrix.flatten())).astype(int)
    # Create an instance of the DBSCAN algorithm
    dbscan = DBSCAN(eps=1, min_samples=4)
    # Fit the algorithm to the data
    dbscan.fit(dataset)
    # Obtain the outlier indices
    outlier_indices = np.where(dbscan.labels_ == -1)[0]
    # Get the outlier points
    outliers = dataset[outlier_indices]

    if plot:
        plt.figure()
        plt.pcolormesh(matrix)
        plt.colorbar()  
        plt.plot(outliers[:, 1]+0.5, outliers[:, 0]+0.5,'o',color='red')
        plt.title(f"outliers: {np.shape(outliers)[0]}")

    matrix_  = copy.deepcopy(matrix).astype(int)
    matrix1_ = copy.deepcopy(matrix1)
    matrix2_ = copy.deepcopy(matrix2)

    # for i in range(np.shape(outliers)[0]):
    #     # points around without the point itself
    #     points_around = matrix2[int(outliers[i,0])-1:int(outliers[i,0])+2,int(outliers[i,1])-1:int(outliers[i,1])+2]
    #     points_around = np.concatenate(points_around)
    #     matrix2[outliers[i,0],outliers[i,1]] = np.bincount(points_around).argmax()
    
    for point in outliers:
        i,j,k = point
        points_around = np.array( [matrix_[np.mod(i,ni),np.mod(j+1,nj)], matrix_[np.mod(i,ni),np.mod(j-1,nj)], 
                                   matrix_[np.mod(i+1,ni),np.mod(j,nj)], matrix_[np.mod(i+1,ni),np.mod(j+1,nj)], matrix_[np.mod(i+1,ni),np.mod(j-1,nj)],
                                   matrix_[np.mod(i-1,ni),np.mod(j,nj)], matrix_[np.mod(i-1,ni),np.mod(j+1,nj)], matrix_[np.mod(i-1,ni),np.mod(j-1,nj)]]).astype(int)

        matrix_[i,j] = np.bincount(points_around).argmax()

     # three scenarios:
    matrix1_[matrix_ == 0] = np.nan
    matrix1_[matrix_ == 2] = np.nan
    matrix2_[matrix_ == 0] = np.nan
    matrix2_[matrix_ == 2] = np.nan
    for point in outliers:
        i,j,_= point
        if matrix_[i,j] == 1:
            points_around = np.array( [matrix1_[np.mod(i,ni),np.mod(j+1,nj)], matrix1_[np.mod(i,ni),np.mod(j-1,nj)], 
                                       matrix1_[np.mod(i+1,ni),np.mod(j,nj)], matrix1_[np.mod(i+1,ni),np.mod(j+1,nj)], matrix1_[np.mod(i+1,ni),np.mod(j-1,nj)],
                                       matrix1_[np.mod(i-1,ni),np.mod(j,nj)], matrix1_[np.mod(i-1,ni),np.mod(j+1,nj)], matrix1_[np.mod(i-1,ni),np.mod(j-1,nj)]])

            points_around[points_around < -0.9*np.pi] += 2*np.pi
            points_mean = np.nanmean(points_around) 
            matrix1_[i,j] = points_mean


            points_around = np.array( [ matrix2_[np.mod(i,ni),np.mod(j+1,nj)], matrix2_[np.mod(i,ni),np.mod(j-1,nj)], 
                                        matrix2_[np.mod(i+1,ni),np.mod(j,nj)], matrix2_[np.mod(i+1,ni),np.mod(j+1,nj)], matrix2_[np.mod(i+1,ni),np.mod(j-1,nj)],
                                        matrix2_[np.mod(i-1,ni),np.mod(j,nj)], matrix2_[np.mod(i-1,ni),np.mod(j+1,nj)], matrix2_[np.mod(i-1,ni),np.mod(j-1,nj)]])

            matrix2_[i,j] = np.nanmean(points_around)
    # gaussian filter to smooth the image
    # import gaussian_filter
    # get lines of separation, nans
    level0 = copy.deepcopy(matrix_)
    level0[level0 > 1.5] = 0  
    level1 = copy.deepcopy(matrix_)
    level1[level1 == 2] = 1
    #matrix2_ = gaussian_filter(matrix2, sigma=1)
    #plt.pcolormesh(matrix2)
    #plt.colorbar()
    plt.figure()
    # level0 = gaussian_filter(level0, sigma=1)
    # level1 = gaussian_filter(level1, sigma=1)
    contour0 = plt.contour(level0, levels=[0])
    contour1 = plt.contour(level1, levels=[0])
    plt.close()
    level0 = contour_points_per_path(contour0)
    level1 = contour_points_per_path(contour1)

    # plt.figure()
    # plt.pcolormesh(matrix2)
    # plt.colorbar()
    # plt.title(f"outliers: {np.shape(outliers)[0]}")
    # for i in range(len(level0)):
    #     plt.plot(level0[i][:,0],level0[i][:,1],linestyle="--")
    # for i in range(len(level1)):
    #     plt.plot(level1[i][:,0],level1[i][:,1],linestyle="--")

    return matrix_, matrix1_, matrix2_, level0, level1

def determine_stability(x, y, omega, delta,k):
    from scipy.linalg import eigvals
 
    mask = np.zeros(len(x),dtype=bool)
    for ii,(x_,y_,d_) in enumerate(zip(x,y,delta)):
        # delta_matrix = np.array([ [0.0, d_, d_],[d_, 0.0, d_],[d_, d_, 0.0] ])
        # Compute the Jacobian matrix
        jacobian = jacobian_matrix(xx=x_,yy=y_,omega=omega,delta=d_,k=k)
        
        # Calculate eigenvalues of the Jacobian matrix
        eigenvalues = eigvals(jacobian)
        
        # Check the stability based on the eigenvalues
        real_parts = np.real(eigenvalues)
        imag_parts = np.imag(eigenvalues)
        # print("x:",x_,"y:",y_,"l1:",eigenvalues[0],"l2:",eigenvalues[1])

        if np.all(real_parts < 0) & np.all(np.abs(real_parts)>1e-3) & np.all(np.abs(imag_parts)< 1e-5):
            # print("x:",x_,"y:",y_,"d:",d_)
            stability = "Stable (asymptotically)"
            mask[ii] = True
            # print("eigenvalues:",eigenvalues)
            # print("Stable (asymptotically)")
        elif np.all(real_parts < 0) & np.all(np.abs(real_parts)>1e-2):
            stability = "Stable (spiral)"
            mask[ii] = True

        elif np.all(real_parts > 0):
            stability = "Unstable"
            mask[ii] = False
        else:
            stability = "Saddle Point"
            mask[ii] = False
            
    return mask

def outlier_detector(matrix):
    outliers = []
    ny,nx = np.shape(matrix)
    for i in range(1,nx-1):
        for j in range(1,ny-1):
            neighboors = np.array([matrix[j-1,i],matrix[j  ,i-1],matrix[j  ,i+1],matrix[j+1,i],])

            if len(np.unique(neighboors)) == 1:
                if neighboors[0] != matrix[j,i]:
                    outliers.append([j,i])
            # if len(np.unique(neighboors)) == 2 and any of neighboors is equal to matrix[j,i]
            if matrix[j,i] != np.nan:
                if len(np.unique(neighboors)) == 2 and len(np.where(neighboors==matrix[j,i])[0]) == 1:
                    outliers.append([j,i])
    return outliers

def correct_matrices(matrix_reference, matrix, units='radians'):
    import copy 

    matrix_ = copy.copy(matrix)

    outliers = outlier_detector(matrix_reference) 

    if units == 'radians':
        for (j,i) in outliers:
            values = np.array([matrix_[j-1,i],matrix_[j,i-1],matrix_[j,i+1],matrix_[j+1,i]])

            if np.unique(values).size == 1:
                matrix_[j,i] = values[0]
            
            elif np.unique(values).size == 2:
                matrix_[j,i] = np.mean(values)
            else: 
                values[values<-np.pi/2] += 2*np.pi
                values_mean = np.mod(np.nanmean(values),2*np.pi)
                if values_mean > np.pi:
                    values_mean -= 2*np.pi
                matrix_[j,i] = values_mean

    elif units == 'deg':
        for (j,i) in outliers:
            values = np.array([matrix_[j-1,i],matrix_[j,i-1],matrix_[j,i+1],matrix_[j+1,i]])

            if np.unique(values).size == 1:
                matrix_[j,i] = values[0]
            else:
                values[values<-np.pi/2] += 360
                values_mean = np.mod(np.nanmean(values),360)
                if values_mean > 180:
                    values_mean -= 360
                matrix_[j,i] = values_mean
    else:
        for (j,i) in outliers:
            values = np.array([matrix_[j-1,i],matrix_[j,i-1],matrix_[j,i+1],matrix_[j+1,i]])
            values_mean = np.nanmean(values)
            matrix_[j,i] = values_mean

    # plt.figure()
    # plt.pcolormesh(matrix_)
    # plt.colorbar()
    # plt.title(f"outliers: {np.shape(outliers)[0]}")
    # for (j,i) in outliers:
    #     plt.plot(i+0.5,j+0.5,'o',color='red')
    return matrix_

