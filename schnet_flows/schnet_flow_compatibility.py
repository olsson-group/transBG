
'''
    Code adapted from https://github.com/atomistic-machine-learning/schnetpack
'''
import numpy as np
import torch
from ase import Atoms
from schnetpack import Properties
from schnet_flows.environment import SimpleEnvironmentProvider, collect_atom_triples #Uses a deprecated function if I import it from pip version


def flow_to_schnet(
    elements,
    positions,
    device,
    environment_provider=SimpleEnvironmentProvider(),  
    collect_triples=False,
    output=None,
):
    """
        Helper function to convert the output of the flow to SchNetPack input format.
        Args:
            elements (torch.tensor): integers codifying the elements.
            positions (torch.tensor): spatial location of atoms in the configuration.
            device(torch.device)
            environment_provider (callable): Neighbor list provider.
            collect_triples (bool, optional): Set to True if angular features are needed.
            centering_function (callable or None): Function for calculating center of
                molecule (center of mass/geometry/...). Center will be subtracted from
                positions.
            output (dict): Destination for converted atoms, if not None
    Returns:
        dict of torch.Tensor: Properties including neighbor lists and masks
            reformated into SchNetPack input format.
    """
    if output is None:
        inputs = {}
    else:
        inputs = output

    # Elemental composition
    cell = np.array( [[0,0,0],[0,0,0],[0,0,0]] , dtype=np.float32)  # get default cell array of ase.Atoms

    inputs[Properties.Z] = elements
    inputs[Properties.R] = positions
    inputs[Properties.cell] = torch.FloatTensor(cell).to(device)

    # get atom environment
    sub_mol = Atoms( elements.cpu().detach().numpy(), positions = np.array(positions.cpu().detach().numpy() ) )
    nbh_idx, offsets = environment_provider.get_environment(sub_mol)

    # Get neighbors
    inputs[Properties.neighbors] = torch.LongTensor(nbh_idx.astype(np.int))

    # Calculate masks and modify previous quantities
    inputs[Properties.atom_mask] = torch.ones_like(inputs[Properties.Z]).float()
    mask = inputs[Properties.neighbors] >= 0
    inputs[Properties.neighbor_mask] = mask.float()
    inputs[Properties.neighbors] = (
        inputs[Properties.neighbors] * inputs[Properties.neighbor_mask].long()
    )


    # Get cells
    inputs[Properties.cell] = torch.FloatTensor(cell)
    inputs[Properties.cell_offset] = torch.FloatTensor(offsets.astype(np.float32))

    # If requested get neighbor lists for triples
    if collect_triples:
        nbh_idx_j, nbh_idx_k, offset_idx_j, offset_idx_k = collect_atom_triples(nbh_idx)
        inputs[Properties.neighbor_pairs_j] = torch.LongTensor(nbh_idx_j.astype(np.int))
        inputs[Properties.neighbor_pairs_k] = torch.LongTensor(nbh_idx_k.astype(np.int))

        inputs[Properties.neighbor_offsets_j] = torch.LongTensor(
            offset_idx_j.astype(np.int)
        )
        inputs[Properties.neighbor_offsets_k] = torch.LongTensor(
            offset_idx_k.astype(np.int)
        )

    #Calculate neighbour_pairs_masks masks
    if collect_triples:
        mask_triples = torch.ones_like(inputs[Properties.neighbor_pairs_j])
        mask_triples[inputs[Properties.neighbor_pairs_j] < 0] = 0
        mask_triples[inputs[Properties.neighbor_pairs_k] < 0] = 0
        inputs[Properties.neighbor_pairs_mask] = mask_triples.float()

    # Add batch dimension and move to CPU/GPU
    for key, value in inputs.items():
        inputs[key] = value.to(device)

    return inputs


def collate_samples(examples):
    """
    Build batch from systems and properties & apply padding
    Args:
        examples (list):
    Returns:
        dict[str->torch.Tensor]: mini-batch of atomistic systems
    """
    properties = examples[0]

    # initialize maximum sizes
    max_size = {
        prop: np.array(val.size(), dtype=np.int) for prop, val in properties.items()
    }

    # get maximum sizes
    for properties in examples[1:]:
        for prop, val in properties.items():
            max_size[prop] = np.maximum(
                max_size[prop], np.array(val.size(), dtype=np.int)
            )

    # initialize batch
    batch = {
        p: torch.zeros(len(examples), *[int(ss) for ss in size]).type(
            examples[0][p].type()
        )
        for p, size in max_size.items()
    }
    has_atom_mask = Properties.atom_mask in batch.keys()
    has_neighbor_mask = Properties.neighbor_mask in batch.keys()

    if not has_neighbor_mask:
        batch[Properties.neighbor_mask] = torch.zeros_like(
            batch[Properties.neighbors]
        ).float()
    if not has_atom_mask:
        batch[Properties.atom_mask] = torch.zeros_like(batch[Properties.Z]).float()

    # If neighbor pairs are requested, construct mask placeholders
    # Since the structure of both idx_j and idx_k is identical
    # (not the values), only one cutoff mask has to be generated
    if Properties.neighbor_pairs_j in properties:
        batch[Properties.neighbor_pairs_mask] = torch.zeros_like(
            batch[Properties.neighbor_pairs_j]
        ).float()

    # build batch and pad
    for k, properties in enumerate(examples):
        for prop, val in properties.items():
            shape = val.size()
            s = (k,) + tuple([slice(0, d) for d in shape])
            batch[prop][s] = val

        # add mask
        if not has_neighbor_mask:
            nbh = properties[Properties.neighbors]
            shape = nbh.size()
            s = (k,) + tuple([slice(0, d) for d in shape])
            mask = nbh >= 0
            batch[Properties.neighbor_mask][s] = mask
            batch[Properties.neighbors][s] = nbh * mask.long()

        if not has_atom_mask:
            z = properties[Properties.Z]
            shape = z.size()
            s = (k,) + tuple([slice(0, d) for d in shape])
            batch[Properties.atom_mask][s] = z > 0

        # Check if neighbor pair indices are present
        # Since the structure of both idx_j and idx_k is identical
        # (not the values), only one cutoff mask has to be generated
        if Properties.neighbor_pairs_j in properties:
            nbh_idx_j = properties[Properties.neighbor_pairs_j]
            shape = nbh_idx_j.size()
            s = (k,) + tuple([slice(0, d) for d in shape])
            batch[Properties.neighbor_pairs_mask][s] = nbh_idx_j >= 0

    return batch


def generate_schnet_inputs(mol_batch, device, min_e = True):
    '''
    Given a batch, returns the schnet inputs that are necesary to generate the corresponding schnet representation.
    Arguments:
        - mol_batch: list of (our) dataset dictionaries.
    Returns:
        - schnet_inputs(dict[str->torch.Tensor]): : mini-batch of atomistic systems
    '''
    if min_e:
        conf_name = 'me_conformation'
    else:
        conf_name = 'md_conformations'

    schnet_inputs = []
    for sample in mol_batch:
        mol_atoms = sample['atoms']
        for i_conf in range( sample[conf_name].size()[0] ):
            mol_positions = sample[conf_name][i_conf]
            for i_atom in range(1, len(mol_atoms)):
                atoms = mol_atoms[0:i_atom]
                positions = mol_positions[0:i_atom]
                schnet_inputs.append(flow_to_schnet(atoms, positions, device))

    schnet_inputs = collate_samples(schnet_inputs)

    return schnet_inputs