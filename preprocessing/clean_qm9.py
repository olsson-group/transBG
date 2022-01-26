from rdkit.Chem import PandasTools
import pandas as pd
import os
from rdkit import Chem
import numpy as np

cwd = os.getcwd()
dataset_path = os.path.join(cwd, "datasets/QM9/qm9/gdb9.sdf")

i_mol = 0

atom_types = []
edge_types = []
formal_charges_types = []
max_atoms = 0
max_edges = 0

total_mol = 0
unused_mol = []

for mol in Chem.SDMolSupplier(dataset_path, removeHs = False, sanitize = True):    
    total_mol += 1 
    if mol != None:
        # Check if the biggest molecule fragment has more tham 1 atom

        mol = Chem.rdmolops.GetMolFrags(mol, asMols = True) #Choose biggest fragment if several
        aux_f = lambda i: mol[i].GetNumAtoms()
        sub_mol_idx = max( range(len(mol)), key = aux_f )
        mol = mol[sub_mol_idx]

        if mol.GetNumAtoms() < 2:
            unused_mol.append(total_mol - 1)
            continue

        #Get the atoms of the molecule
        atoms = []
        formal_charges = []
        edges = []
        for atom in mol.GetAtoms():
            atoms.append(atom.GetAtomicNum())
            formal_charges.append( atom.GetFormalCharge() )

        for edge in mol.GetBonds():
                edges.append(edge.GetBondTypeAsDouble())

        new_atom_types = list(np.setdiff1d(atoms, atom_types))
        if new_atom_types != []:
            atom_types.extend(new_atom_types)

        new_formal_charges = list(np.setdiff1d(formal_charges, formal_charges_types))
        if new_formal_charges != []:
            formal_charges_types.extend(new_formal_charges)

        new_edge_types = list(np.setdiff1d(edges, edge_types))
        if new_edge_types != []:
            edge_types.extend(new_edge_types)

        if max_atoms < mol.GetNumAtoms():
            max_atoms = mol.GetNumAtoms()

    else:
        unused_mol.append(total_mol - 1)


print(atom_types, flush = True)
print(formal_charges_types, flush = True)
print(max_atoms, flush = True)
print(edge_types, flush = True)
print(len(unused_mol), flush = True)
print(total_mol, flush = True)


clean_sdf = os.path.join(cwd, "datasets/QM9/qm9/clean_gdb9.sdf")

#Cleaning
frame = PandasTools.LoadSDF(dataset_path, removeHs = False)
#frame.drop(frame.index[unused_mol], inplace = True)
indexes_to_keep = set(range(frame.shape[0])) - set(unused_mol)
clean_frame = frame.take( list(indexes_to_keep) )

PandasTools.WriteSDF(clean_frame, clean_sdf)


#Check that the procedure worked
i_mol = 0

atom_types = []
edge_types = []
formal_charges_types = []
max_atoms = 0
max_edges = 0

total_mol = 0
unused_mol = []

for mol in Chem.SDMolSupplier(clean_sdf, removeHs = False, sanitize = True):    
    total_mol += 1 
    if mol != None:
        # Check if the biggest molecule fragment has more tham 1 atom

        mol = Chem.rdmolops.GetMolFrags(mol, asMols = True) #Choose biggest fragment if several
        aux_f = lambda i: mol[i].GetNumAtoms()
        sub_mol_idx = max( range(len(mol)), key = aux_f )
        mol = mol[sub_mol_idx]

        if mol.GetNumAtoms() < 2:
            unused_mol.append(total_mol - 1)
            continue

        #Get the atoms of the molecule
        atoms = []
        formal_charges = []
        edges = []
        for atom in mol.GetAtoms():
            atoms.append(atom.GetAtomicNum())
            formal_charges.append( atom.GetFormalCharge() )

        for edge in mol.GetBonds():
                edges.append(edge.GetBondTypeAsDouble())

        new_atom_types = list(np.setdiff1d(atoms, atom_types))
        if new_atom_types != []:
            atom_types.extend(new_atom_types)

        new_formal_charges = list(np.setdiff1d(formal_charges, formal_charges_types))
        if new_formal_charges != []:
            formal_charges_types.extend(new_formal_charges)

        new_edge_types = list(np.setdiff1d(edges, edge_types))
        if new_edge_types != []:
            edge_types.extend(new_edge_types)

        if max_atoms < mol.GetNumAtoms():
            max_atoms = mol.GetNumAtoms()

    else:
        unused_mol.append(total_mol - 1)



print(atom_types, flush = True)
print(formal_charges_types, flush = True)
print(max_atoms, flush = True)
print(edge_types, flush = True)
print(len(unused_mol), flush = True)
print(total_mol, flush = True)


print()