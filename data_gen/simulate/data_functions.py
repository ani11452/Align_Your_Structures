import pickle
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from force_field_calculation import *
from rdkit.Chem import AllChem
import tempfile
from openmm.app import *
import mdtraj
import rdkit
from multiprocessing import Lock
import random
import numpy as np
import time
import re
import os
import sys
sys.path.append('../utils')
from data_filter import filter_data

'''
Chooses the subsample of the dataset based on the split requested
'''
def get_subsample(smile_keys, args):
    random.seed(0) 
    if args.dataset == 'drugs':
        if args.split in ["train", "val"]:
            num_keys = max(1, int(len(smile_keys) * args.percentage))
            selected_keys = random.sample(list(smile_keys.keys()), num_keys)
            smile_keys = {key: smile_keys[key] for key in selected_keys}
        elif args.split == "test":
            smile_keys = {
                key: random.sample(smile_keys[key], min(5, len(smile_keys[key])))
                for key in smile_keys
            }    
    elif args.dataset == 'qm9':
        if args.split in ["train", "val"]:
            num_keys = max(1, int(len(smile_keys) * args.percentage))
            selected_keys = random.sample(list(smile_keys.keys()), num_keys)
            smile_keys = {key: smile_keys[key] for key in selected_keys}
        elif args.split == "test":
            smile_keys = {
                key: random.sample(smile_keys[key], min(5, len(smile_keys[key])))
                for key in smile_keys
            }

    return smile_keys


'''
Covert to SMILES dictionary
'''
def get_smiles_dict(data):
    smile_keys = {}
    for item in tqdm(data):
        if item['smiles'] not in smile_keys:
            smile_keys[item['smiles']] = []
        smile_keys[item['smiles']].append(item['rdmol'])
    return smile_keys


'''
Canonicalize the smiles
'''
def canonicalize_smiles(smiles):
    mol = Chem.RemoveHs(Chem.MolFromSmiles(smiles))
    if mol is None:
        return ''
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)

'''
Subsample
'''
def subsample_data(data, args):
    # Open the pickle file
    with open(args.subsample, 'rb') as f:
        new_keys = pickle.load(f)

    can_dict = {}
    for key in data:
        can_key = canonicalize_smiles(key)
        if can_key == '':
            continue
        can_dict[can_key] = key 

    new_data = {}
    for new_key in new_keys:
        can_key = canonicalize_smiles(new_key)
        if can_key not in can_dict:
            print(f"Key {new_key} not in data")
            continue
        assert can_dict[can_key] not in new_data
        new_data[can_dict[can_key]] = [data[can_dict[can_key]][0]]

    return new_data

'''
Input: The pickle file
Output: Subset of the molecules in OpenMM format with Force Field
'''
def process_data_subset(args):
    # Open the pickle file
    print("Opening Pickle File")
    with open(args.data_pkl, 'rb') as f:
        data = pickle.load(f)

    # Create the dictinary of Smiles to all RDKit Mols
    print("Preprocessing Data")
    data = filter_data(data)[0]

    # Get the smiles keys
    smile_keys = get_smiles_dict(data)

    # Choose the subsample
    smile_keys = get_subsample(smile_keys, args)

    # Subsample the data
    if args.subsample is not None:
        print("Subsampling the data for time eval")
        smile_keys = subsample_data(smile_keys, args)
        print(len(smile_keys))
        print(set([len(smile_keys[k]) for k in smile_keys]))

    print(f"Number of molecules: {len(smile_keys)}")

    # Get the lists of mols from this
    # This is to make sure the cache hits for force fields are maximized
    mols = list(smile_keys.values())

    # Subselect the mols for this worker
    mols = mols[args.inference_id::args.num_inferences]

    # Convert the mols into force field opjects while caching
    # Don't save the force field objects to reduce memory
    mols_flat = [mol for mol_list in mols for mol in mol_list]

    print(f'Number of Conformers: {len(mols_flat)}')

    # Create Dataset Object out of this
    dataset = ConformerData(mols_flat, args)

    # Get the associated Dataloader
    dataloader = DataLoader(
        dataset, batch_size=1, 
        shuffle=False, num_workers=args.num_workers,
        collate_fn=custom_collate_fn
    )

    print("Returning Dataloader")
    return dataloader

# Custom collate function
def custom_collate_fn(batch):
    idx, items = batch[0]
    return idx, items

# Use this to create the modeller and system
class ConformerData(Dataset):
    def __init__(self, data, args):
        self.data = data
        self.args = args
        self.lock = Lock()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Get the Mol and Force Field
        try:
            # Start time
            start = time.time()

            # Get the mol
            mol = self.data[idx]

            # Get the smiles
            smiles = Chem.MolToSmiles(Chem.RemoveHs(mol))

            # Check to see if the file already exists
            smiles_safe = re.sub(r"[^a-zA-Z0-9_-]+", "_", smiles).strip("_") 
            sim_folder = os.path.join(self.args.outdir, f"{smiles_safe}_{idx}")
            pdb_path = os.path.join(sim_folder, "mol.pkl")
            if os.path.exists(pdb_path):
                return idx, (None, None, None, None, None)

            # Convert RDKit mol to OpenFF Molecule
            molecule, flag = rdkit_to_openmm(mol)

            # Get atomic positions from RDKit conformer
            positions = rdkit_to_positions(mol)

            # Get the force field
            with self.lock:
                forcefield = generate_forcefield(molecule, self.args)

            # Get the Modeller object
            modeller = Modeller(molecule.to_topology().to_openmm(), positions)

            # Create the system
            if self.args.implicit:
                system = forcefield.createSystem(modeller.topology, constraints=HBonds)
            else:
                # Add solvent (By default uses Na+ and Cl- to Neutralize)
                modeller.addSolvent(
                    forcefield,
                    model='tip3p',
                    padding=self.args.padding*unit.nanometer
                )

                # Create the system
                if self.args.hydrogenMass:
                    system = forcefield.createSystem(
                        modeller.topology,
                        nonbondedMethod=PME,
                        nonbondedCutoff=self.args.cutoff*unit.nanometer,
                        constraints=HBonds,
                        hydrogenMass=self.args.hydrogenMass*unit.amu,
                        switchDistance=self.args.switchDistance*unit.nanometer # Per the Smirnoff Sage parameterization
                    )
                else:
                    system = forcefield.createSystem(
                        modeller.topology,
                        nonbondedMethod=PME,
                        nonbondedCutoff=self.args.cutoff*unit.nanometer,
                        constraints=HBonds,
                        switchDistance=self.args.switchDistance*unit.nanometer # Per the Smirnoff Sage parameterization
                    )

            # End time
            end = time.time()

            # Total load time
            load_time = end - start

            # Return
            return idx, (modeller, system, smiles + flag, load_time, mol)
        
        except Exception as e:
            error_message = f"Error during simulation with {smiles} {idx}: {str(e)}"
            self.args.log_object.error(error_message)
            print(f"[ERROR] {error_message}")
            return idx, (None, None, None, None, None)
