from rdkit import Chem
import networkx as nx
from networkx.algorithms import isomorphism
import pickle
from pathlib import Path
import re

NUM_INF = 160


class TrajFixer:
    def __init__(self, mol_path):
        with open(mol_path, 'rb') as f:
            self.mols = pickle.load(f)
    
    def pdb_conect_to_nx_graph(self, pdb_file_path):
        G = nx.Graph()
    
        with open(pdb_file_path, 'r') as f:
            lines = f.readlines()
    
        # First pass: collect atom info and add nodes
        # Convert PDB 1-based indices to 0-based by subtracting 1
        for line in lines:
            if line.startswith("ATOM") or line.startswith("HETATM"):
                serial = int(line[6:11].strip()) - 1  # Convert to 0-based
                element = line[76:78].strip() or line[12:14].strip()  # fallback if element is blank
                G.add_node(serial, 
                           element=element)
    
        # Second pass: parse CONECT records to add edges
        # Convert PDB 1-based indices to 0-based by subtracting 1 
        for line in lines:
            if line.startswith("CONECT"):
                parts = line.split()
                source = int(parts[1]) - 1  # Convert to 0-based
                for target_str in parts[2:]:
                    target = int(target_str) - 1  # Convert to 0-based
                    if G.has_node(source) and G.has_node(target):
                        G.add_edge(source, target)
    
        return G
    
    def extract_ids(self, path_str):
        path = Path(path_str)
        parts = path.parts  # tuple of path segments
        
        # Extract first number from "*_results"
        folder_with_results = next((p for p in parts if re.match(r"\d+_results", p)), None)
        first_number = int(folder_with_results.split('_')[0]) if folder_with_results else None
        
        # Extract last number from final directory (e.g. "C_C_H_..._C1_110")
        parent_dir = path.parent.name
        match = re.search(r'_(\d+)$', parent_dir)
        second_number = int(match.group(1)) if match else None
        
        return first_number, second_number
         
    def get_mol(self, pdb_path):    
        inf_id, conf_id = self.extract_ids(pdb_path)
    
        # Subselect the mols for this worker
        mols = self.mols[inf_id::NUM_INF]
        
        # Convert the mols into force field opjects while caching
        mols_flat = [mol for mol_list in mols for mol in mol_list]
        mol = mols_flat[conf_id]
        return mol
    
    def rdkit_mol_to_nx_graph(self, mol):
        G = nx.Graph()
        for atom in mol.GetAtoms():
            idx = atom.GetIdx()
            G.add_node(idx, element=atom.GetSymbol())  # <‑‑ add element label
    
        for bond in mol.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            G.add_edge(i, j)
    
        return G
    
    def are_graphs_isomorphic(self, G1: nx.Graph, G2: nx.Graph):
        node_match = isomorphism.categorical_node_match('element', None)
        matcher = isomorphism.GraphMatcher(G1, G2, node_match=node_match)

        if not matcher.is_isomorphic():
            return None

        mapping = matcher.mapping

        # --- explicit safety check (probably redundant, but cheap) ----------
        for pdb_idx, rdkit_idx in mapping.items():
            pdb_element = G1.nodes[pdb_idx]['element']
            rdkit_element = G2.nodes[rdkit_idx]['element']
            if pdb_element != rdkit_element:
                return None
                
        # Check that graphs have same number of atoms
        if G1.number_of_nodes() != G2.number_of_nodes():
            return None
        # --------------------------------------------------------------------

        return mapping
    
    def get_conf(self, pdb_path):
        mol = self.get_mol(pdb_path)
        pdb_graph   = self.pdb_conect_to_nx_graph(pdb_path)
        rdkit_graph = self.rdkit_mol_to_nx_graph(mol)

        mapping = self.are_graphs_isomorphic(pdb_graph, rdkit_graph)
        if mapping is None:
            raise ValueError(f"Graphs are not isomorphic for {pdb_path}")

        # Optionally: reorder RDKit atoms to match PDB serial numbers
        # so that subsequent indexing is guaranteed to be identical.
        # ------------------------------------------------------------------
        new_order = [mapping[pdb_idx] for pdb_idx in sorted(mapping.keys())]
        mol = Chem.RenumberAtoms(mol, new_order)
        # ------------------------------------------------------------------

        return mol