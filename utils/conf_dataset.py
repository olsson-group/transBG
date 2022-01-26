import torch
from torch.utils.data import Dataset
import numpy as np
import os
from rdkit import Chem
import mdtraj as md
from utils.sort_atoms import breadth_first_sorting

from simtk import unit

class conformer_dataset(Dataset):
    '''
    A dataset object for this model: It can access information about the molecule, its possible 3D conformations and the corresponding energies.

    Outputs a dictionary with the following entries:
        - 'atoms' (torch.tensor, size: (n_atoms) ) : vector of atomic numbers. 
        - 'positions' (torch.tensor, size: (n_atoms,3) ) : positions of the atoms in the minimal energy conformation
        - 'edges' (torch.tensor, size: (n_edges, 3) ): Each row is: bonded atoms 1, bonded atom 2, bond type between atom 1 and 2
        - 'ref_atoms' (list, size: (n_atoms, 3) ) : index of the reference atoms of a certain atom
        - 'dataset_idx' (int): index of the molecule in the clean QM9 dataset
        - 'qm9_idx' (int): index of the molecule in the original QM9 dataset
        - 'partial_charges' (unit.Quantity, size: (n_atoms) ) : partial charges of the atoms
        - 'rdkit_mol' (rdkit.Chem.rdchem.Mol): rdkit molecule object
    '''
    def __init__(self, sdf_path, xyz_path, pdb_path = None):
        self.sdf_path = sdf_path
        self.xyz_path = xyz_path
        self.pdb_path = pdb_path # If provided, a set of non-minimal energy conformations can be found at pdb_path
        self.mol_suppl = Chem.SDMolSupplier(sdf_path, removeHs = False, sanitize = True)
        self.n_molecules = len(self.mol_suppl)

    def __len__(self):
        #Number of molecules in the dataset
        return self.n_molecules

    def __getitem__(self, idx):
        
        mol_idx = idx
        
        mol = self.mol_suppl[idx]
        qm9_index = int( mol.GetProp('_Name').split('_')[1] )
        xyz_filename = os.path.join( self.xyz_path, 'dsgdb9nsd_{:0>6d}.xyz'.format(qm9_index) )
        if self.pdb_path:
            pdb_filename = os.path.join( self.pdb_path, 'md_{:0>6d}.pdb'.format(qm9_index-1) )
            if not os.path.isfile(pdb_filename):
                pdb_filename = None
        else:
            pdb_filename = None

        # Improve in the future: ignore molecules with fragments
        fragments = Chem.rdmolops.GetMolFrags(mol, asMols = True) #Choose biggest fragment if several
        atom_indexes = list( Chem.rdmolops.GetMolFrags(mol,asMols = False)[0] )
        aux_f = lambda i: fragments[i].GetNumAtoms()
        sub_mol_idx = max( range(len(fragments)), key = aux_f )
        mol = fragments[sub_mol_idx]
        try:
            partial_charges = get_partial_charges(xyz_filename)[atom_indexes]
        except:
            partial_charges = None
            print('There was a problem reading the partial charges of molecule with qm9_idx {:0>6d}'.format(qm9_index))
        

        Chem.AssignAtomChiralTagsFromStructure(mol)  # So that openff can recover the chirality of the molecule.

        #mol.UpdatePropertyCache() 
        node_ranking = list( Chem.CanonicalRankAtoms(mol, includeChirality=True) )
        node_init = node_ranking.index(max(node_ranking))
        mol_order, ref_atoms = breadth_first_sorting(mol, node_ranking, node_init)
        
        # reference atoms and sort molecule in the new order
        new_ref_atoms = [[] for _ in range(mol.GetNumAtoms()) ]
        new_atom_idx = 0
        for i_atom in mol_order:
            for j_ref in range(3):
                if ref_atoms[i_atom][j_ref] != -1:
                    new_ref_atoms[new_atom_idx].append( mol_order.index(ref_atoms[i_atom][j_ref]) )
                else:
                    new_ref_atoms[new_atom_idx].append(-1) 
            new_atom_idx += 1

        ref_atoms = new_ref_atoms.copy()
        mol = Chem.RenumberAtoms(mol, mol_order)

        #Minimal energy conformation
        me_conformation = torch.tensor(mol.GetConformers()[0].GetPositions(), dtype = torch.float32).unsqueeze(dim = 0)

        #Non-minimal energy conformations: MIND THE UNITS!
        
        if pdb_filename:
            traj = md.load_pdb(pdb_filename)
            md_conformations = torch.tensor(traj.xyz, dtype = torch.float32)*10#Because nm #This is not the correct order in which we generate the atoms, right? Yes but keep checking
        else:
            md_conformations = None

        #Get the atoms of the molecule
        atoms = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
        edges = [ [ edge.GetBeginAtomIdx(), edge.GetEndAtomIdx(), int( edge.GetBondTypeAsDouble() )] for edge in mol.GetBonds() ]

        atoms = torch.tensor(atoms, dtype = torch.long) 
        edges = torch.tensor(edges, dtype = torch.long)       

        # Output dictionary
        sample = {'atoms': atoms, 'me_conformation': me_conformation, 'md_conformations': md_conformations, 'edges': edges, 'ref_atoms': ref_atoms, 'dataset_idx': mol_idx, 'qm9_idx': qm9_index, 'partial_charges': partial_charges, 'rdkit_mol': mol} #
        return sample

def collate_conf(mol_batch):
    '''
    Collate function that transforms a batch of molecules (X, mol_rep, at_reps) into a batch of samples for likelihhod training: (x, mol_rep, at_rep)
    Arguments:
        - 
    Returns:
        - 
    '''
    return mol_batch 

def get_partial_charges(xyz_file_name):
    '''
    Reads the partial charges from a .xyz file of QM9 dataset.
    '''
    with open(xyz_file_name) as f:
        n_atoms = int(f.readline())
        f.readline()
        partial_charges = []
        for _ in range(n_atoms):
            _, _, _, _, partial_charge = f.readline().split()
            partial_charges.append(float(partial_charge))

    return unit.Quantity(value = np.array(partial_charges), unit = unit.elementary_charge)

class analysis_dataset(Dataset):
    '''
    A dataset object for this model: It can access information about the molecule, its possible 3D conformations and the corresponding energies.

    Outputs a dictionary with the following entries:
        - 'atoms' (torch.tensor, size: (n_atoms) ) : vector of atomic numbers. 
        - 'positions' (torch.tensor, size: (n_atoms,3) ) : positions of the atoms in the minimal energy conformation
        - 'edges' (torch.tensor, size: (n_edges, 3) ): Each row is: bonded atoms 1, bonded atom 2, bond type between atom 1 and 2
        - 'ref_atoms' (list, size: (n_atoms, 3) ) : index of the reference atoms of a certain atom
        - 'dataset_idx' (int): index of the molecule in the clean QM9 dataset
        - 'qm9_idx' (int): index of the molecule in the original QM9 dataset
        - 'partial_charges' (unit.Quantity, size: (n_atoms) ) : partial charges of the atoms
        - 'rdkit_mol' (rdkit.Chem.rdchem.Mol): rdkit molecule object
    '''
    def __init__(self, sdf_path, xyz_path, pdb_path = None):
        self.sdf_path = sdf_path
        self.xyz_path = xyz_path
        self.pdb_path = pdb_path # If provided, a set of non-minimal energy conformations can be found at pdb_path
        self.mol_suppl = Chem.SDMolSupplier(sdf_path, removeHs = False, sanitize = True)
        self.n_molecules = len(self.mol_suppl)

    def __len__(self):
        #Number of molecules in the dataset
        return self.n_molecules

    def __getitem__(self, idx):
        
        mol_idx = idx
        
        mol = self.mol_suppl[idx]
        qm9_index = int( mol.GetProp('_Name').split('_')[1] )
        xyz_filename = os.path.join( self.xyz_path, 'dsgdb9nsd_{:0>6d}.xyz'.format(qm9_index) )
        if self.pdb_path:
            pdb_filename = os.path.join( self.pdb_path, 'md_{:0>6d}.pdb'.format(qm9_index-1) )
            if not os.path.isfile(pdb_filename):
                pdb_filename = None
        else:
            pdb_filename = None

        # Improve in the future: ignore molecules with fragments
        fragments = Chem.rdmolops.GetMolFrags(mol) #Choose biggest fragment if several
        if len(fragments)>1:
            sample = {'atoms': None, 'me_conformation': None, 'md_conformations': None, 'edges': None, 'dataset_idx': mol_idx, 'qm9_idx': qm9_index, 'partial_charges': None, 'rdkit_mol': None} #
            return sample

        #Get partial charges from QM9
        try:
            partial_charges = get_partial_charges(xyz_filename)
        except:
            partial_charges = None
            print('There was a problem reading the partial charges of molecule with qm9_idx {:0>6d}'.format(qm9_index))
        

        Chem.AssignAtomChiralTagsFromStructure(mol)  # So that openff can recover the chirality of the molecule.

        #mol.UpdatePropertyCache() 
        
        '''We do not renumber the atoms in this version'''
        '''
        node_ranking = list( Chem.CanonicalRankAtoms(mol, includeChirality=True) )
        node_init = node_ranking.index(max(node_ranking))

        mol_order, ref_atoms = breadth_first_sorting(mol, node_ranking, node_init)
        
        # reference atoms and sort molecule in the new order
        new_ref_atoms = [[] for _ in range(mol.GetNumAtoms()) ]
        new_atom_idx = 0
        for i_atom in mol_order:
            for j_ref in range(3):
                if ref_atoms[i_atom][j_ref] != -1:
                    new_ref_atoms[new_atom_idx].append( mol_order.index(ref_atoms[i_atom][j_ref]) )
                else:
                    new_ref_atoms[new_atom_idx].append(-1) 
            new_atom_idx += 1

        ref_atoms = new_ref_atoms.copy()
        mol = Chem.RenumberAtoms(mol, mol_order)
        '''

        #Minimal energy conformation
        me_conformation = torch.tensor(mol.GetConformers()[0].GetPositions(), dtype = torch.float32).unsqueeze(dim = 0)

        #Non-minimal energy conformations: MIND THE UNITS!
        
        if pdb_filename:
            traj = md.load_pdb(pdb_filename)
            md_conformations = torch.tensor(traj.xyz, dtype = torch.float32)*10#Because nm #This is not the correct order in which we generate the atoms, right? Yes but keep checking
        else:
            md_conformations = None

        #Get the atoms of the molecule
        atoms = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
        edges = [ [ edge.GetBeginAtomIdx(), edge.GetEndAtomIdx(), int( edge.GetBondTypeAsDouble() )] for edge in mol.GetBonds() ]

        atoms = torch.tensor(atoms, dtype = torch.long) 
        edges = torch.tensor(edges, dtype = torch.long)       

        # Output dictionary
        sample = {'atoms': atoms, 'me_conformation': me_conformation, 'md_conformations': md_conformations, 'edges': edges, 'dataset_idx': mol_idx, 'qm9_idx': qm9_index, 'partial_charges': partial_charges, 'rdkit_mol': mol} #
        return sample