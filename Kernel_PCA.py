# Implementation of Kernel PCA and Tests on different Datasets #

from scipy.spatial.distance import pdist, squareform
from scipy import exp
from scipy.linalg import eigh
import numpy as np

import matplotlib.pyplot as plt
from sklearn.datasets import make_moons, make_circles, make_swiss_roll
from sklearn.metrics.pairwise import polynomial_kernel, pairwise_distances


class Kernel_PCA():
    #TODO: Implement the polynomial kernel

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

        

if __name__ == "__main__":
    
    #X, y = make_moons(n_samples=10, random_state=123)

    X, y = make_circles(n_samples=100, random_state=123, noise=0.1, factor=0.2)
    #X, color = make_swiss_roll(n_samples=800, random_state=123)

    plt.style.use('dark_background')
    plt.figure(figsize=(10,8))

    plt.scatter(X[y==0, 0], X[y==0, 1], color='red', alpha=0.5)
    plt.scatter(X[y==1, 0], X[y==1, 1], color='blue', alpha=0.5)
    #plt.scatter(X[:, 0], X[:, 1], c=color, cmap=plt.cm.rainbow)

    plt.title('A nonlinear 2Ddataset')
    plt.ylabel('y coordinate')
    plt.xlabel('x coordinate')

    plt.show()

    X_X_trans = X.dot(X.T)
    eigvals_new, eigvecs_new = eigh(X_X_trans) # Probe: wenn man XX^t zerlegt hat man auch nur zwei eigenwerte/eigenvektoren --> X hat hier Rang 2!

    # die eigenvektoren von K müssen noch mit den singulärwerten von K skaliert werden (nur dann sind es wirklich die projektionen!!)
    # X_pc_scaled = X_pc.dot(np.diag(np.sqrt(eigvals[::-1][:2])))

    kpca = Kernel_PCA(data=X, n_components=2, kernel_type='rbf', gamma=5)
    X_pc_scaled, X_pc, sel_eigvals = kpca.compute_kernel_pcs()

    plt.style.use('dark_background')
    plt.figure(figsize=(10,8))
    plt.scatter(X_pc_scaled[y==0, 0], X_pc_scaled[y==0, 1], color='red', alpha=0.5)
    plt.scatter(X_pc_scaled[y==1, 0], X_pc_scaled[y==1, 1], color='blue', alpha=0.5)

    #plt.scatter(X_pc_scaled[y==0, 0], np.zeros([50]), color='red', alpha=0.5)
    #plt.scatter(X_pc_scaled[y==1, 0], np.zeros([50]), color='blue', alpha=0.5)
    plt.scatter(kpca.project_new_point(x_new=X[:22])[:,0], kpca.project_new_point(x_new=X[0:22])[:,1], color="white", alpha=1, marker="+")

    #plt.scatter(X_pc_scaled[:, 0], X_pc_scaled[:, 1], c=color, cmap=plt.cm.rainbow)

    plt.title('First 2 principal components after Kernel PCA')
    #plt.text(-0.18, 0.18, 'gamma = 15', fontsize=12)
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.show()

    # projection of the "new" datapoint --> die eigenvektoren von K müssen mit 1/singulärwerte skaliert werden! 




    