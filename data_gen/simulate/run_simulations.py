from data_functions import *
from openmm import *
from openmm.app import *
from openmm import unit
import logging
from sys import stdout
import os
import re
import warnings
import gc
import pandas as pd
from tqdm import tqdm

def save_trajectory(args, load_time, mol):
    # Get the folders
    sim_folder = args.sim_folder
    output_h5 = os.path.join(sim_folder, "traj.h5")
    output_xtc = os.path.join(sim_folder, "traj.xtc")
    output_pdb = os.path.join(sim_folder, "system.pdb")
    output_csv = os.path.join(sim_folder, "scalars.csv")
    smiles_txt = os.path.join(sim_folder, "smiles.txt")
    mol_pkl = os.path.join(sim_folder, "mol.pkl")

    # Write the total time for loading and generation in the smiles.txt file
    with open(smiles_txt, 'a') as f:
        f.write(f'\nThe load time is: {load_time}\n')
        df = pd.read_csv(output_csv)
        sim_time = df["Elapsed Time (s)"].iloc[-1]
        f.write(f'\nThe simulation time is: {sim_time}\n')

    # Load the trajectory and the topology in memory
    warnings.filterwarnings("ignore", category=UserWarning, module="mdtraj")
    t = mdtraj.load(output_h5)

    # Center and align the trajectory (in memory)
    t.center_coordinates()
    t.superpose(t, frame=0)

    # Save the new trajectory and PDB for topology
    # May have to save this as xtc or something to save disk space
    t.save_xtc(output_xtc)
    t[0].save_pdb(output_pdb)  # Save the final PDB for topology
    os.remove(output_h5)
    with open(mol_pkl, 'wb') as f:
        pickle.dump(mol, f)

def run_simulation(
        modeller, system, smiles,
        args, i
):
    # Get the integrator for simulation
    integrator = LangevinMiddleIntegrator(
        args.temperature * unit.kelvin, 
        args.friction / unit.picosecond, 
        args.dt * unit.femtoseconds
    )

    # Initialize the platform
    platform = Platform.getPlatformByName(args.platform)

    # Get the simulation
    simulation = Simulation(modeller.topology, system, integrator, platform)

    # Set positions and minimize energy
    simulation.context.setPositions(modeller.positions)
    simulation.minimizeEnergy()

    # Create the data folder
    smiles = smiles.split("\n", 1)[0]
    smiles_safe = re.sub(r"[^a-zA-Z0-9_-]+", "_", smiles).strip("_") 
    sim_folder = os.path.join(args.outdir, f"{smiles_safe}_{i}")
    os.makedirs(sim_folder, exist_ok=True)
    with open(os.path.join(sim_folder, "smiles.txt"), 'w') as f:
        f.write(smiles)
    args.sim_folder = sim_folder
    output_h5 = os.path.join(sim_folder, "traj.h5")
    scalars_csv = os.path.join(sim_folder, "scalars.csv")

    # Set the listening parameters
    simulation.reporters = []

    simulation.reporters.append(
        StateDataReporter(
            scalars_csv,
            args.frame_interval,
            step=True,
            temperature=True,
            elapsedTime=True,
            volume=True,
            density=True,
            potentialEnergy=True,
            kineticEnergy=True,
            totalEnergy=True,
            speed=True,
            totalSteps=args.num_steps + args.equilibration_steps
        )
    )

    # Do the Equilibration (NVT)
    simulation.context.setVelocitiesToTemperature(args.temperature)
    simulation.step(args.equilibration_steps)

    # Set up the actual simulation listener
    top = mdtraj.Topology.from_openmm(modeller.topology)
    mask = top.select("not water")
    reporter = mdtraj.reporters.HDF5Reporter(output_h5, reportInterval=args.frame_interval,
                                             atomSubset=mask)
    simulation.reporters.append(reporter)

    # Make NPT if needed
    if not args.implicit:
        system.addForce(MonteCarloBarostat(1*unit.bar, args.temperature*unit.kelvin))
        simulation.context.reinitialize(preserveState=True)

    # Conduct the simulation
    simulation.step(args.num_steps)
    reporter.close()

    # Delete everything to ensure GPU space is made
    del simulation, integrator, modeller, system
    gc.collect()

# Runs the simulations
def run_simulations(args):
    # Set up error logging
    logger = logging.getLogger("simulation_logger")
    logger.setLevel(logging.ERROR)
    log_file = f"{args.outdir}/simulation_errors.log"
    file_handler = logging.FileHandler(log_file)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    args.log_object = logger

    # First get the data
    data = process_data_subset(args)
    
    # Stats accumulators
    sim_durations = []
    total_durations = []

    # Iterate through the data
    print('Starting Generation Process')
    for idx, out in tqdm(data):
        # Extract values
        modeller, system, smiles, load_time, mol = out

        # In case there was an issue dataloading or no smiles
        if not smiles:
            continue

        try:
            # Time the simulation
            sim_start = time.perf_counter()
            run_simulation(
                modeller, system, smiles,
                args, idx
            )
            sim_end = time.perf_counter()
            sim_time = sim_end - sim_start

            # Total time includes original load_time plus simulation
            total_time = load_time + sim_time
            total_durations.append(total_time)
            sim_durations.append(sim_time)

            save_trajectory(args, load_time, mol)
        except Exception as e:
            error_message = f"Error during simulation with {smiles} {idx}: {str(e)}"
            args.log_object.error(error_message)
            print(f"[ERROR] {error_message}")

    avg_sim = np.mean(sim_durations)
    avg_total = np.mean(total_durations)
    print(f"[INFO] Average simulation time: {avg_sim:.2f}s")
    print(f"[INFO] Average total time (load + sim): {avg_total:.2f}s")



    

