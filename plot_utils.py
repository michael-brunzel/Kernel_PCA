# Helper Functions to create plots # 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons, make_circles, make_swiss_roll
import pickle


def plot_points(data, label, oos_proj_data=None, project_new_data=False):
    plt.style.use('dark_background')
    plt.figure(figsize=(10,8))

    plt.scatter(data[:,0], data[:,1], c=label, cmap='Accent')

    if project_new_data:
        plt.scatter(oos_proj_data[:,0], oos_proj_data[:,1], color="white", alpha=1, marker="+")

    plt.title('Plot of the Datapoints')
    plt.ylabel('y coordinate')
    plt.xlabel('x coordinate')
    plt.show()

def create_and_plot_data(n_samples, file_name=None, circles=False, moons=False, other_data=False):
    # creates the different datasets and the respective plots
    if circles:
        data, label = make_circles(n_samples=n_samples, random_state=123, noise=0.1, factor=0.2)

    elif moons:
        data, label = make_moons(n_samples=n_samples, random_state=123)
    
    elif other_data:
        data, label = pickle.load(open(str(file_name),'rb'))

    else:
        return "Select Dataset!"

    plot_points(data=data, label=label)

    return data, label

def plot_PCs(X_pc_scaled, label, project_new_data=False, oos_proj_data=None):
    plot_points(data=X_pc_scaled, label=label, oos_proj_data=oos_proj_data, project_new_data=project_new_data)


