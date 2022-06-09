from utils.mol_geometry import compute_torsion, compute_angle, compute_distance, ic_to_xyz
import torch
import numpy as np

def construct_z_matrix(X, ref_atoms, placing_order):
    '''
    Given a torch.tensor array, whose rows are the cartesian coordinates of a molecule, generates a Z-matrix computing internal coordinates.

    Arguments:

    - X (torch.tensor, size: (n_atoms, 3) ): Cartesian coordinates of the atoms.

    - ref_atoms (list, size: (n_atoms, 3) ): Reference atoms for internal coordinates generation. The distance will be computed with respect the first atom of this list,
                                             the angle with respect to the two first and the torsion with respect the three. 

      - placing_order (list, size: (n_atoms) ): placing order of the atoms.

    Returns: 

    - z_matrix (torch.tensor, size: (n_atoms-1, 3) ): Z-Matrix of the configuration of the molecule.  

    '''

    torch.pi = torch.tensor(np.pi)
    i3 = [triplet[0] for triplet in ref_atoms]
    i2 = [triplet[1] for triplet in ref_atoms]
    i1 = [triplet[2] for triplet in ref_atoms]

    # Mind the order!
    x4 = X[placing_order]
    x3 = X[i3[1:]]
    x2 = X[i2[2:]]
    x1 = X[i1[3:]]

    distances = compute_distance(x4[1:], x3)
    angles = torch.pi - compute_angle(x4[2:], x3[1:], x2)
    torsions = compute_torsion(x1, x2[1:], x3[2:], x4[3:])

    n_atoms = X.shape[0]
    z_matrix = torch.zeros([n_atoms -1 , 3], dtype = torch.float32)

    z_matrix[:, 0] = distances 
    z_matrix[1:, 1] = angles
    z_matrix[2:, 2] = torsions

    return z_matrix

def deconstruct_z_matrix(z_matrix, ref_atoms, loss=True):
    '''
    Generates cartesian coordinates given a z-matrix and the reference atoms. Requires the z_matrix to be correctly sorted.

    Arguments:
        - z_matrix (torch.tensor, size: (n_atoms-1, 3) )
        - ref_atoms (list, size: (n_atoms, 3) ): List of the reference atoms. Unmeaningful entries can be filled with -1s.

    Returns:
        - cartesian (torch.tensor, size: (n_atoms, 3) ): Coordinates using a nerf ref system.
        - constraints loss (torch.tensor, size (n_atoms))
    '''
    torch.pi = torch.tensor(np.pi)
    n_atoms = len(ref_atoms)
    if loss:
        def _dist_loss(dists):
            loss = torch.sum(torch.where(dists<0, dists, torch.zeros_like(dists))**2, axis=-1) 
            return loss

        def _polar_loss(angles):
            negative_loss = torch.sum(torch.where( angles < 0, - angles, torch.zeros_like(angles)) ** 2, axis=-1)
            positive_loss = torch.sum(torch.where( angles > torch.pi, angles - torch.pi, torch.zeros_like(angles)) ** 2, axis=-1)
            return  negative_loss + positive_loss

        def _torsion_loss(angles):
            negative_loss = torch.sum( torch.where(angles < - torch.pi, angles + torch.pi, torch.zeros_like(angles) )**2, axis = -1 )
            positive_loss = torch.sum( torch.where(angles > torch.pi, angles - torch.pi, torch.zeros_like(angles) )**2 )
            return  negative_loss + positive_loss 

        internal_constraints_loss = _dist_loss( z_matrix[:,0].clone() ) + _polar_loss( z_matrix[:,1].clone() ) + _torsion_loss( z_matrix[:,2].clone() )

    # Protection
    z_matrix[:,0] = torch.clamp(z_matrix[:,0].clone(), min =0)
    z_matrix[:,1] = torch.clamp(z_matrix[:,1].clone(), min =0, max = np.pi )
    z_matrix[:,2] = torch.clamp(z_matrix[:,2].clone(), min = - np.pi, max = np.pi )
    #torch.pi = torch.tensor(np.pi)

    cartesian = torch.zeros( (n_atoms, 3), dtype = torch.float32, device = z_matrix.device)
    cartesian[1] = z_matrix[0].clone()
    angle = torch.tensor( 1-ref_atoms[1][0], dtype= torch.int16 ) * torch.pi - z_matrix[1,1]
    cartesian[2,0] = cartesian[ ref_atoms[1][0] , 0] + z_matrix[1,0] * torch.cos(angle)
    cartesian[2,1] = z_matrix[1,0] * torch.sin(angle)

    for i_atom in range(3, n_atoms):
        x_ref = cartesian[ ref_atoms[i_atom] ]
        #Mind the orther!
        cartesian[i_atom] = ic_to_xyz(x_ref[2].unsqueeze(dim = 0), x_ref[1].unsqueeze(dim = 0), x_ref[0].unsqueeze(dim = 0),
                                      z_matrix[i_atom-1, 0].unsqueeze(dim = 0), z_matrix[i_atom-1, 1].unsqueeze(dim = 0), z_matrix[i_atom-1, 2].unsqueeze(dim = 0))
    if loss:
      return cartesian, internal_constraints_loss 
    else:
      return cartesian

def correct_conf_indexes(conformations):
    '''
        conformations (torch.tensor): Z-matrixes of the conformations.
    '''
    torch.pi = torch.tensor(np.pi)
    indexes = []
    for i_conf, conformation in enumerate(conformations):
        d_i = conformation[:,0] > 0
        d_a =  torch.logical_and(conformation[:,1] >= 0, conformation[:,1] <= torch.pi)
        d_t =  torch.logical_and(conformation[:,2] > - torch.pi, conformation[:,2] <= torch.pi)
        if torch.logical_not(torch.any( torch.logical_not(torch.logical_and(torch.logical_and(d_i, d_a), d_t)) )): #Simplify this
            indexes.append(i_conf)
    return indexes
