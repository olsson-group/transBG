import torch
from utils.z_matrix import construct_z_matrix, deconstruct_z_matrix

def generate_ref_frame_rep(mol_batch, add_symmetry_token = False, min_e = True):
    '''
        Generates a reference frame representation for each atom of each molecule using the reference atoms. The refererence frame correspond to the NeRF system constructed on the reference atoms. The representation consists on the coordinates of the origin and the x-y-z axes.
        Optionally (add_symmetry_token = True) add a symmetry breaking token consisting on the average of the positions of the previously placed atoms. 

        Inputs:
            - mol_batch:
            - add_symmetry_token (bool)
            - min_e (bool): If true the model learns from QM9 minimal energy conformations, otherwise the model learns from md simulated conformations
        Outputs:
            - ref_frame_rep ( list(n_mols) of torch.tensors (n_conf, n_atoms, 12)  ): reference frame representation for each atom of each molecule.
            - symmetry_tokens ( list(n_mols) of mol_ref_frame_rep (n_conf, n_atoms, 3)  ):symmetry breaking token for each atom of each molecule.

        Note: This election of variables is flexible to different number of atoms in the molecules and different number of conforamations for molecules.
    '''
    n_atoms = [ len( mol['atoms'] ) for mol in mol_batch ]
    n_mols = len(n_atoms)
    
    if min_e:
        conf_name = 'me_conformation'
    else:
        conf_name = 'md_conformations'

    ref_frame_rep = []
    if add_symmetry_token:
        symmetry_tokens = []
    for i_mol in range(n_mols):
        ref_atoms = mol_batch[i_mol]['ref_atoms']
        mol_ref_frame_rep = torch.empty(0, dtype = torch.float32)
        if add_symmetry_token:
            mol_symmetry_tokens = torch.empty(0, dtype = torch.float32)

        for i_conf in range(mol_batch[i_mol][conf_name].size()[0]):

            # This could of course be done as a preprocessing step
            z_matrix = construct_z_matrix(mol_batch[i_mol][conf_name][i_conf], ref_atoms , list(range(n_atoms[i_mol])))
            z_matrix[z_matrix != z_matrix] = 0. #Protection for nans
            positions = deconstruct_z_matrix(z_matrix, ref_atoms, loss=False).detach() # Note: Check this piece of code is not used for e-b learning

            x_ref_1 = torch.empty(0, dtype = torch.float32)
            x_ref_2 = torch.empty(0, dtype = torch.float32)
            x_ref_3 = torch.empty(0, dtype = torch.float32)
            
            if add_symmetry_token:
                #Symmetry breaking token
                conf_symmetry_tokens = torch.zeros( (n_atoms[i_mol]-1, 3), dtype = torch.float32 )
                conf_symmetry_tokens[0,:] = torch.mean( positions[0:1], dim = 0)   # 0 if the postions are in NeRF 1,2,3
                if n_atoms[i_mol] > 2: 
                    conf_symmetry_tokens[1,:] = torch.mean( positions[0:2], dim = 0)
                if n_atoms[i_mol] > 3: 
                    conf_symmetry_tokens[2,:] = torch.mean( positions[0:3], dim = 0)
                for i_atom in range(4,n_atoms[i_mol]): conf_symmetry_tokens[i_atom-1, :] =  torch.mean( positions[0:i_atom], dim = 0)

            #Positions of reference atoms 
            for i_atom in range(3,n_atoms[i_mol]):
            # Note that the index are inverted here 0 -> 2 and 2 -> 0 #Check this is correct
                x_ref = positions[ mol_batch[i_mol]['ref_atoms'][i_atom] ]
                x_ref_1 = torch.cat( (x_ref_1, x_ref[2].unsqueeze(dim = 0)) , dim = 0)
                x_ref_2 = torch.cat( (x_ref_2, x_ref[1].unsqueeze(dim = 0)) , dim = 0)
                x_ref_3 = torch.cat( (x_ref_3, x_ref[0].unsqueeze(dim = 0)) , dim = 0)
                
            #Compute representation of the ref frame
            ref_1 = torch.tensor([0,0,0,1,0,0,0,1,0,0,0,1], dtype = torch.float32).unsqueeze(dim = 0)
            if n_atoms[i_mol] > 1:
                d12  = torch.norm( positions[1] - positions[0] ).item()
                ref_2 = torch.tensor([d12,0,0,1,0,0,0,1,0,0,0,1], dtype = torch.float32).unsqueeze(dim = 0)
            else:
                ref_2 = torch.tensor([0,0,0,1,0,0,0,1,0,0,0,1], dtype = torch.float32).unsqueeze(dim = 0) # It does not make sense to use one atom molecules, so this can be omitted

            if n_atoms[i_mol] > 2:
                conf_ref_frame_rep = torch.cat( (ref_1, ref_2), dim = 0 )
            else:
                    conf_ref_frame_rep = ref_1
    
            if n_atoms[i_mol] > 3:
                x23 = x_ref_3 - x_ref_2
                x23 /=  torch.norm(x23, dim = 1, keepdim=True)
                x_axes = x23
                x12 = x_ref_2 - x_ref_1
                x12 /= torch.norm(x12, dim = 1, keepdim=True)
                z_axes = torch.cross(x12, x23, dim = 1)
                y_axes = torch.cross(z_axes, x_axes, dim = 1)

                rest_ref_frame_rep = torch.cat( (x_ref_3, x_axes, y_axes, z_axes), dim = 1)
                conf_ref_frame_rep = torch.cat( (conf_ref_frame_rep, rest_ref_frame_rep), dim = 0 )


            mol_ref_frame_rep = torch.cat( (mol_ref_frame_rep, conf_ref_frame_rep.unsqueeze(dim=0)), dim = 0)
            if add_symmetry_token:
                mol_symmetry_tokens = torch.cat( (mol_symmetry_tokens, conf_symmetry_tokens.unsqueeze(dim=0)), dim=0 )

        ref_frame_rep.append(mol_ref_frame_rep)
        if add_symmetry_token:
            symmetry_tokens.append(mol_symmetry_tokens)

    if add_symmetry_token:
        return ref_frame_rep, symmetry_tokens
    else:
        return ref_frame_rep

def generate_one_ref_frame_rep(x_ref):
    '''
        Generates a reference frame representation using the reference atoms. The refererence frame correspond to the NeRF system constrcuted on the reference atoms. The representation consists on the coordinates of the origin and the x-y-z axes.
        Optionally (add_symmetry_token = True) add a symmetry breaking token consisting on the average of the positions of the previously placed atoms. 

        Inputs:
            - mol_batch:
            - add_symmetry_token (bool)
        Outputs:
            - ref_frame_rep ( list(n_mols) of (n_atoms, 12)  ): reference frame representation for each atom of each molecule.
            - symmetry_tokens ( list(n_mols) of (n_atoms, 3)  ):symetry breaking token for each atom of each molecule.
    '''

    x_ref_1 = x_ref[2].unsqueeze(dim = 0)
    x_ref_2 = x_ref[1].unsqueeze(dim = 0)
    x_ref_3 = x_ref[0].unsqueeze(dim = 0)

    x23 = x_ref_3 - x_ref_2
    x23 /=  torch.norm(x23.clone(), dim = 1, keepdim=True)
    x_axes = x23
    x12 = x_ref_2 - x_ref_1
    x12 /= torch.norm(x12.clone(), dim = 1, keepdim=True)
    z_axes = torch.cross(x12, x23, dim = 1)
    y_axes = torch.cross(z_axes, x_axes, dim = 1)

    ref_frame_rep = torch.cat( (x_ref_3, x_axes, y_axes, z_axes), dim = 1)

    return ref_frame_rep
