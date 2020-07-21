# Main File #

from Kernel_PCA import *
import plot_utils
from plot_utils import *
import importlib
#importlib.reload(plot_utils)

def main():
    X,y = create_and_plot_data(n_samples=100, file_name='data/SpaCy_Prepped_Data.pkl', other_data=True)
    kpca = Kernel_PCA(data=X, n_components=2, kernel_type='rbf', gamma=0.5)
    X_pc_scaled, X_pc, sel_eigvals = kpca.compute_kernel_pcs()

    oos_proj_points = kpca.project_new_point(x_new=X[:20])
    plot_PCs(X_pc_scaled=X_pc_scaled, label=y, project_new_data=True, oos_proj_data=oos_proj_points)

if __name__ == "__main__":
    main()
