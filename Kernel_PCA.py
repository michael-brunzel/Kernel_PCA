# Implementation of Kernel PCA and Tests on different Datasets #

from scipy.spatial.distance import pdist, squareform
from scipy import exp
from scipy.linalg import eigh
import numpy as np

import matplotlib.pyplot as plt
from sklearn.datasets import make_moons, make_circles, make_swiss_roll
from sklearn.metrics.pairwise import polynomial_kernel, pairwise_distances

#TODO: Implement the polynomial kernel

class Kernel_PCA():

    def __init__(self, data, n_components, kernel_type='rbf', gamma=15):
        self.data = data
        self.n_components = n_components
        self.kernel_type = kernel_type
        self.kernel_dict = {'rbf': self.rbf_kernel, 'poly': self.poly_kernel, 'linear':self.linear_kernel}
        self.gamma = gamma
        self.K_uncentered = None
        self.sel_eigvecs = None
        self.sel_eigvals = None
    
    def rbf_kernel(self, new_point=None, oos_point=False):

        self.K_uncentered = exp(-self.gamma*(pairwise_distances(X=self.data)**2))

        if oos_point:
            if len(new_point.shape) == 1:
                new_point = new_point[np.newaxis, :]
            return exp(-self.gamma*(pairwise_distances(X=new_point, Y=self.data)**2))


    def poly_kernel(self):
        pass

    def linear_kernel(self, new_point=None, oos_point=False):

        self.K_uncentered = (self.data).dot(self.data.T)

        if oos_point:
            if len(new_point.shape) == 1:
                new_point = new_point[np.newaxis, :]
            return new_point.dot(self.data.T)

    def compute_kernel_pcs(self):

        self.kernel_dict[self.kernel_type]()

        num_samples = self.K_uncentered.shape[0]
        one_n = np.ones((num_samples, num_samples)) / num_samples
        # Center the Kernel-Matrix #
        centered_K = self.K_uncentered - one_n.dot(self.K_uncentered) - self.K_uncentered.dot(one_n) + one_n.dot(self.K_uncentered).dot(one_n)

        eigvals, eigvecs = eigh(centered_K)

        self.sel_eigvecs = eigvecs[:,::-1][:,:self.n_components]
        self.sel_eigvals = eigvals[::-1][:self.n_components]

        assert all(self.sel_eigvals>0.001), "Some Eigenvalues are 0!"

        # Scale the Eigenvectors of the Kernelmatrix with the Singularvalues to obtain the projected lower-dimensional coordinates (Principal Components)
        proj_coordinates = self.sel_eigvecs.dot(np.diag(np.sqrt(self.sel_eigvals)))

        return proj_coordinates, self.sel_eigvecs, self.sel_eigvals

    def project_new_point(self, x_new):
        k = self.kernel_dict[self.kernel_type](new_point=x_new, oos_point=True)
        N = k.shape[1]
        one_n = np.ones((N,1)) / N
        one_n_n = np.ones((N,N)) / N
        k_centered = k  - (one_n.T.dot(self.K_uncentered)) - k.dot(one_n_n) + one_n.T.dot(self.K_uncentered).dot(one_n_n)

        return k_centered.dot(self.sel_eigvecs/ np.sqrt(self.sel_eigvals))

        


    