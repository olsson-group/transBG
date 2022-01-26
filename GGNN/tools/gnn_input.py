from sklearn.preprocessing import OneHotEncoder
import torch

def generate_gnn_input(mol_batch, atom_types, edge_types, max_nodes):
    '''
        Generates a one-hot-encoding of the atoms in the batch based on atom-types.

        Arguments:

        - mol_batch (list of dictionaries): batch of outputs from preprocessing.
        - atom_types (list of torch.int32???): Atomic number of the atoms found in the dataset.

        Returns:

        - gnn_input (torch.tensor): One hot encding of the atom types.
    '''
    # Could this be problematic for gradients? No since we only evaluate on the inputs

    '''
    at_enc = OneHotEncoder(handle_unknown='ignore')
    at_enc.fit(atom_types)

    mol_atoms = [mol_batch[i_mol]['atoms'].unsqueeze(dim = 1).numpy() for i_mol in range(len(mol_batch))]

    nodes = [ at_enc.transform( mol_atoms[i_mol] ).toarray() for i_mol in range(len(mol_batch)) ]

    edge_enc =  OneHotEncoder(handle_unknown='ignore')
    edge_enc.fit(edge_types)
    '''

    nodes = torch.zeros(len(mol_batch), max_nodes, len(atom_types))
    for i_mol in range(len(mol_batch)):
        for i_node, atom in enumerate(mol_batch[i_mol]['atoms']):
            atom_idx = atom_types.index([atom.item()])
            nodes[i_mol][i_node][atom_idx] = 1

    edges = torch.zeros(len(mol_batch), max_nodes, max_nodes, len(edge_types))

    for i_mol in range(len(mol_batch)):
        for edge in mol_batch[i_mol]['edges']:
            bond_idx = edge_types.index([edge[2].item()])
            edges[i_mol][edge[0]][edge[1]][bond_idx] = 1
            edges[i_mol][edge[1]][edge[0]][bond_idx] = 1
            

    '''
    # Pad
    nodes = [ torch.nn.functional.pad( torch.tensor(one_hot, dtype = torch.int8), pad = (0, 0, 0, max_nodes - one_hot.shape[0]) , mode = "constant", value = 0) for one_hot in nodes ]
    nodes = torch.tensor(nodes, dtype = torch.int8 ) 
    '''

    return nodes, edges


