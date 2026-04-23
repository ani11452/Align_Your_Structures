'''
This is the main data generation script

All input / output data related functions: data_functions.py
All force field parameterization functions: force_field_calculation.py
All simulation related functions: run_simulations.py
'''

import argparse
from data_functions import *
from run_simulations import *
import os
import time


def parse_args():
    # Parse arguments
    parser = argparse.ArgumentParser()

    # Input and Output paths
    parser.add_argument('--data_pkl', type=str, help="Path to the data pickle object")
    parser.add_argument('--outdir', type=str, help='Path to the output directory to save trajectories')
    parser.add_argument('--dataset', type=str, choices=["qm9", "drugs"], help='Dataset for which the data generation is happening')
    parser.add_argument('--split', type=str, choices=["train", "test", "val"], help='Split of the dataset')
    parser.add_argument('--subsample', type=str, default=None, help='pkl file for subsampling')

    # Dataset subsampling
    # This is ultimately implemented approximately
    parser.add_argument('--percentage', type=float, default=1.0, help="Percentage of Dataset to actually use")

    # Multiprocessing and Parallel Generation
    parser.add_argument('--inference_id', type=int)
    parser.add_argument('--num_inferences', type=int)
    parser.add_argument('--num_workers', type=int)

    # Force Field / System Parameters
    parser.add_argument(
        '--cache_file', type=str, 
        default="smirnoff-molecules.json", help="Name of the Force Field Cache File"
    )
    parser.add_argument(
        '--force_field_mol', type=str, 
        default="openff-2.2.1", help="Name of the Small Molecule Force Field"
    )
    parser.add_argument(
        '--force_field_protein', type=str, 
        default="amber/protein.ff14SB.xml", help="Name of the Protein Force Field"
    )
    parser.add_argument(
        '--force_field_implicit', type=str, 
        default="amber/implicit/gbn2.xml", help="Name of the Implicit Solvent Force Field"
    )
    parser.add_argument(
        '--force_field_explicit', type=str, 
        default="amber/tip3p_standard.xml", help="Name of the Explicit Solvent Force Field"
    )
    parser.add_argument(
        '--force_field_explicit_ion', type=str, 
        default="amber/tip3p_HFE_multivalent.xml", help="Name of the Explicit Solvent Force Field for Ions"
    )
    parser.add_argument('--cutoff', type=float, default=0.9, help="Nonbonded Cutoff in Nm") # Based on the Default OpenFF Sage Requirements
    parser.add_argument('--switchDistance', type=float, default=0.8, help="Switch Distance in Nm") # Based on the Default OpenFF Sage Requirements
    parser.add_argument('--hydrogenMass', type=float, default=1.5, help="Hydrogen Mass Partitioning in Amu") # Increased for simulation efficiency
    parser.add_argument('--implicit', action='store_true', help="Use flag for implicit solvent, otherwise explicit")
    parser.add_argument('--padding', type=float, default=1.5, help="Padding used in explicit solvent in nanometers")

    # Simulation Parameters
    parser.add_argument('--sim_ns', type=float, default=5, help="Duration of the Simuation in Nanoseconds")
    parser.add_argument('--dt', type=float, default=4, help="Length of the Timestep in Femtoseconds")
    parser.add_argument('--equilibration_steps', type=int, default=5000, help="Number of NVT Equilibration Steps")
    parser.add_argument('--temperature', type=float, default=300, help="Temperature of the Simulation in Kelvin")
    parser.add_argument('--friction', type=float, default=1, help="Friction in the simulation in ps^{-1}")
    parser.add_argument('--printout_int', type=int, default=1000, help="Frame interval for printing out simulation conditions")
    parser.add_argument('--frame_interval', type=int, default=100, help="The number of integration steps between saved frames")
    parser.add_argument('--platform', type=str, default="CUDA", help="Device on which the simulation is computed")

    # Parse the args
    args = parser.parse_args()

    # Calculate and add the additional arguments
    args.frame_dt = args.frame_interval * args.dt
    args.num_steps = int((args.sim_ns * 1e6) / args.dt)
    args.outdir = os.path.join(args.outdir, f"{args.inference_id}_results")
    os.makedirs(args.outdir, exist_ok=True)
    args.cache_file = os.path.join(args.outdir, args.cache_file)

    # Print the final arguments
    print("\n=============================")
    print("      Force Field Parameters  ")
    print("=============================")
    print(f"Small Molecule Force Field      : {args.force_field_mol}")
    print(f"Protein Force Field             : {args.force_field_protein}")
    print(f"Nonbonded Cutoff                : {args.cutoff} nm")
    print(f"Switch Distance                 : {args.switchDistance} nm")
    print(f"Hydrogen Mass Repartitioning    : {args.hydrogenMass} amu")
    print(f"Explicit Solvent                : {not args.implicit}")
    if not args.implicit:
        print(f"Solvent Padding                 : {args.padding} nm")
        print(f"Explicit Solvent Force Field    : {args.force_field_explicit}")
        print(f"Explicit Solv Ion Force Field   : {args.force_field_explicit_ion}")
    else:
        print(f"Implicit Solvent Force Field    : {args.force_field_implicit}")

    print("\n=============================")
    print("      Simulation Parameters    ")
    print("=============================")
    print(f"Compute Device                  : {args.platform}")
    print(f"Simulation Length               : {args.sim_ns} ns")
    print(f"Timestep                        : {args.dt} fs")
    print(f"Total Steps                     : {args.num_steps}")

    print("\n=============================")
    print("      Output & Performance     ")
    print("=============================")
    print(f"Print Interval                  : {args.printout_int} steps")
    print(f"Frame Interval                  : {args.frame_interval} steps ({args.frame_dt} fs)")
    print(f"Temperature                     : {args.temperature} K")
    print(f"Friction                        : {args.friction} ps⁻¹")

    print("\n===================================")
    print("    Starting Molecular Simulations ")
    print("===================================\n")
    print(f"Find results in                  : {args.outdir}")

    return args


def main():
    start = time.time()
    print("Start Time: ", start)
    args = parse_args()
    run_simulations(args)
    end = time.time()
    print("End Time: ", end)
    print("Total Time Taken: ", end - start)

if __name__ == "__main__":
    main()





