import os
import pyemma.coordinates as coor
import pyemma.msm as msm
import pyemma.plots as mplt
import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append("..")
from descriptors import *


# Torsion based MSM
def torsion_based_msm(
    torsions,
    use_tica=False,
    k=100,
    lag=10,
    tica_lag=5
):
    # Get the torsion features
    features = np.hstack([np.sin(torsions), np.cos(torsions)])

    # TICA: extract slow collective variables
    if use_tica:
        tica = coor.tica([features], lag=tica_lag, kinetic_map=True)
        tica_output = tica.get_output()[0]
        features = tica_output

    # Clustering directly on features
    cluster = coor.cluster_kmeans([features], k=k)

    # MSM estimation
    model = msm.estimate_markov_model(cluster.dtrajs, lag=lag)

    return model


if __name__ == "__main__":
     # Load the trajectory   
    xtc_path = os.environ.get('MD_DATA_ROOT', '') + '/output_trajectories/GEOM-QM9/4fs_HMR15_5ns_actual/train/120_results/NC_O_C_H_O_C_H_1C_C_H_1O_109/traj.xtc'
    pdb_path = os.environ.get('MD_DATA_ROOT', '') + '/output_trajectories/GEOM-QM9/4fs_HMR15_5ns_actual/train/120_results/NC_O_C_H_O_C_H_1C_C_H_1O_109/system.pdb'
    mol_path = os.environ.get('MD_DATA_ROOT', '') + '/output_trajectories/GEOM-QM9/4fs_HMR15_5ns_actual/train/120_results/NC_O_C_H_O_C_H_1C_C_H_1O_109/mol.pkl'

    # Load the trajectory
    print("Loading trajectory")
    traj = md.load(xtc_path, top=pdb_path)
    traj = traj[::100]
    coords = torch.tensor(traj.xyz, dtype=torch.float32)
    coords = coords # + torch.randn_like(coords) * 0.01  # Add small Gaussian noise

    print("Loading mol")
    mol = pickle.load(open(mol_path, 'rb'))

    # Get the torsion features
    torsions = get_torsions_in_traj(traj, mol)

    # Create the MSM
    msm = torsion_based_msm(torsions, use_tica=False, k=5, lag=2, tica_lag=1) 

    print("Count matrix:")
    print(msm.count_matrix_full)

    print("Implied timescales:")
    print(msm.timescales()[:5])
    print("Stationary distribution:")
    print(msm.stationary_distribution)
    print("Transition matrix:")
    print(msm.transition_matrix)


    # Build ITS diagnostic
    lags = [1, 2, 5, 10, 20, 50]
    its_obj = msm.its(
        [msm.discrete_trajectories_full[0]],  # Discretized trajectory
        lags=lags,
        nits=5,
        errors='bayes'
    )

    # Save the ITS plot
    plt.figure(figsize=(6, 4))
    mplt.plot_implied_timescales(its_obj, units='steps', dt=1)
    plt.axvline(x=2, color='red', linestyle='--', label='MSM lag = 2')
    plt.xlabel('Lag time (τ)')
    plt.ylabel('Implied timescales')
    plt.title('Implied Timescales vs Lag')
    plt.legend()
    plt.tight_layout()

    plot_path = 'implied_timescales_plot.png'
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Saved ITS plot to: {plot_path}")

