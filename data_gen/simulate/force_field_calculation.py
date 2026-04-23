# Imports
from openff.toolkit import Molecule, Topology
from openmm import *
from openmm.app import *
from openmm import unit
from openff.toolkit.topology import Topology
from rdkit import Chem
from openmmforcefields.generators import SMIRNOFFTemplateGenerator, GAFFTemplateGenerator
import argparse
import pickle
from multiprocessing import *
import functools
from tqdm import tqdm
import numpy as np
from rdkit.Chem import rdmolops

# Function for generating force field
def generate_forcefield(
        molecule,
        args
):
    smirnoff = SMIRNOFFTemplateGenerator(
        cache=args.cache_file,
        molecules=molecule,
        forcefield=args.force_field_mol
    )

    if args.implicit:
        forcefield = ForceField(
            args.force_field_protein, 
            args.force_field_implicit
        )
    else:
        forcefield = ForceField(
            args.force_field_protein, 
            args.force_field_explicit,
            args.force_field_explicit_ion
        )
    
    forcefield.registerTemplateGenerator(smirnoff.generator)

    return forcefield

# Function for converting RDKit to OpenMM Molecule
def rdkit_to_openmm(mol):
    # Assign 3D-based stereochemistry
    rdmolops.AssignStereochemistryFrom3D(mol)
    # Convert to OpenFF Mol
    flag = ""
    try:
        offmol = Molecule.from_rdkit(
            mol,
            hydrogens_are_explicit=True # Checked in the preprocessing
        )
    except Exception as e:
        offmol = Molecule.from_rdkit(
            mol, allow_undefined_stereo=True, #RDKit cannot fix Nitrogens
            hydrogens_are_explicit=True # Checked in the preprocessing
        )
        flag = "\nstereo issues"
    return offmol, flag

def rdkit_to_positions(mol):
    conformer = mol.GetConformer()
    positions = []
    for i in range(mol.GetNumAtoms()):
        pos = conformer.GetAtomPosition(i)
        positions.append((pos.x, pos.y, pos.z))  # Extract x, y, z coordinates in Angstroms
    return np.array(positions) * unit.angstrom  # Convert to OpenMM-compatible units


# # Mapping function
# def process_item(mol, args):
#     offmol = rdkit_to_openmm(mol)
#     forcefield = generate_forcefield(offmol, args)
#     return forcefield

# # Converts RDKit mol to PDB
# def mol_to_pdb(mol, filename):
#     mol_block = Chem.MolToMolBlock(mol)
#     mol = Chem.MolFromMolBlock(mol_block, removeHs=False)
#     Chem.MolToPDBFile(mol, filename)
