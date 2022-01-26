import torch
from torch import nn
from schnet_flows.schnet_flow_compatibility import generate_schnet_inputs
from GGNN.tools import gnn_input
from utils.ref_frame import generate_ref_frame_rep
from utils.rep_concatenation import cat_reps, cat_second_atom_representation, cat_third_atom_representation, cat_atom_representation

class TransBG_model(nn.Module):
    def __init__(self, flow, schnet, gnn, params, device):
        super().__init__()
        self.flow = flow.to(device)
        self.schnet = schnet.to(device)
        self.gnn = gnn
        self.params = params
        self.device = device


    def pushback(self, mol_batch, min_e = True):
        '''
            Transforms a batch from coordinate space to latent space.
        '''
        #Compute some useful quantities
        n_atoms = [ len( mol['atoms'] ) for mol in mol_batch ]
        final_mol_idx = [n_atoms[0] - 1]
        for i_mol in range( 1, len(n_atoms) ): final_mol_idx.append(final_mol_idx[i_mol-1] + n_atoms[i_mol] )

        # Generate molecule reresentations:
        nodes, edges = gnn_input.generate_gnn_input(mol_batch, self.params.atom_types, self.params.bond_types, self.params.max_nodes)
        nodes = nodes.to(self.device)
        edges = edges.to(self.device)
        mol_rep, at_rep = self.gnn(nodes, edges)

        #Generate schNet representations: As for atoms colocated in canonical order 
        schnet_inputs = generate_schnet_inputs(mol_batch, self.device, min_e)
        schnet_rep = self.schnet(schnet_inputs)['y']
        
        #This part runs only for cartesian coordinates (add 0s representation for generating the first atom):
        if self.params.cartesian:
            raise NotImplementedError

        #Generate reference frame representation (only for IC): Note that the positions should be provided in cartesian coordinates.   
        if not self.params.cartesian:
            if self.params.add_symmerty_token:
                ref_frame_rep, symmetry_tokens = generate_ref_frame_rep(mol_batch, add_symmetry_token = self.params.add_symmerty_token, min_e = min_e)
            else:
                ref_frame_rep = generate_ref_frame_rep(mol_batch, add_symmetry_token = self.params.add_symmerty_token, min_e = min_e)
                symmetry_tokens = None

        #We concatenate the information that the flow uses and push to latent space
        if  self.params.cartesian:
            raise NotImplementedError
        else:
            if self.params.add_symmerty_token:
                d_input, da_input, dat_input = cat_reps(mol_batch = mol_batch, mol_rep = mol_rep, at_rep = at_rep, schnet_rep = schnet_rep,
                                                    ref_frame_rep = ref_frame_rep, symmetry_tokens = symmetry_tokens, cartesian = self.params.cartesian, device = self.device, min_e = min_e)
            z_d, log_det_invJ_d = self.flow.pushback(d_input, self.flow.mask_d)
            z = z_d
            log_det_invJ = log_det_invJ_d
            if not(da_input is None):
                z_da, log_det_invJ_da = self.flow.pushback(da_input, self.flow.mask_da)
                z = torch.cat( (z, z_da), dim = 0 )
                log_det_invJ = torch.cat( (log_det_invJ, log_det_invJ_da), dim = 0)
            if not(dat_input is None):
                z_dat, log_det_invJ_dat = self.flow.pushback(dat_input, self.flow.mask_dat)
                z = torch.cat( (z, z_dat), dim = 0 )
                log_det_invJ = torch.cat( (log_det_invJ, log_det_invJ_dat), dim = 0)
            # Mind the order of z for external use
            
        return z, log_det_invJ 

    def pushforward(self, mol_batch, n_conf = 1):
        '''
            Generates conformations of a given molecule transforming from latent space to coordinate space

            Arguments:
                - mol_batch: batch of molecules.
                - n_conf (int): number of conformations to generate for each molecule.
            Output:
                - conformations (list of torch.tensors): The conformations of the molecules. The size of the tensors is (n_conf, n_atoms[i_mol], 3). 
                - log_det_J (list of torch.tensors): The corresponding log_det_J contributions. The size of the tensors is (n_conf). 

        '''

        #Compute some useful quantities
        n_atoms = [ len( mol['atoms'] ) for mol in mol_batch ]
        #initial_mol_idx = [0]
        #for i_mol in range( 1, len(n_atoms) ): initial_mol_idx.append(initial_mol_idx[i_mol-1] + n_atoms[i_mol] )

        #Generate molecular and atomic representations:
        nodes, edges = gnn_input.generate_gnn_input(mol_batch, self.params.atom_types, self.params.bond_types, self.params.max_nodes)
        nodes = nodes.to(self.device)
        edges = edges.to(self.device)
        mol_rep, at_rep = self.gnn(nodes, edges)

        conformations = [ torch.zeros( (n_conf, n_atoms[i_mol] - 1, 3) , dtype = torch.float32, device = self.device ) for i_mol, _ in enumerate(mol_batch) ]
        log_det_J = [ torch.zeros( (n_conf) , dtype = torch.float32, device = self.device ) for _ in range(len(mol_batch)) ]

        if not self.params.cartesian:

            #First atom is set at the origin. For the second we just compute a distance
            cat_mol_rep, cat_at_rep, schnet_inputs, cat_ref_frame_rep, cat_symmetry_token = cat_second_atom_representation(mol_batch, mol_rep, at_rep, n_conf, self.device)
            schnet_rep = self.schnet(schnet_inputs)['y']
            z = self.flow.base_distribution.sample( (1, cat_mol_rep.size()[0]) ).squeeze(dim = 0).to(self.device)
            d_input = torch.cat( (cat_mol_rep, cat_at_rep, schnet_rep, cat_ref_frame_rep, cat_symmetry_token, z), dim = 1  ).unsqueeze( dim = 1)
            new_coord, contribution_det_J = self.flow.pushforward(d_input, self.flow.mask_d)
            new_coord = new_coord.squeeze(dim = 1)

            coord_idx = 0
            for i_mol, _ in enumerate(mol_batch):
                for i_conf in range(n_conf):
                    conformations[i_mol][i_conf, 0, 0] = new_coord[coord_idx, 0].clone() #0 because atom 0 is in the origin
                    log_det_J[i_mol][i_conf] += contribution_det_J[coord_idx].clone()
                    coord_idx += 1

            
            # Third atom we compute a distance and an angle
            cat_mol_rep, cat_at_rep, schnet_inputs, cat_ref_frame_rep, cat_symmetry_token = cat_third_atom_representation(mol_batch, mol_rep, at_rep, [mol_confs.clone().detach() for mol_confs in conformations], n_conf, self.device)
            # Note that the previous sub-conformation must be detached: autoregressive model.
            schnet_rep = self.schnet(schnet_inputs)['y']
            z = self.flow.base_distribution.sample( (1, cat_mol_rep.size()[0]) ).squeeze(dim = 0).to(self.device)
            da_input = torch.cat( (cat_mol_rep, cat_at_rep, schnet_rep, cat_ref_frame_rep, cat_symmetry_token, z), dim = 1  ).unsqueeze( dim = 1)
            new_coord, contribution_det_J = self.flow.pushforward(da_input, self.flow.mask_da)
            new_coord = new_coord.squeeze(dim = 1)

            coord_idx = 0
            for i_mol, sample in enumerate(mol_batch):
                if n_atoms[i_mol] > 2:
                    for i_conf in range(n_conf):
                        conformations[i_mol][i_conf, 1, 0:2] = new_coord[coord_idx, 0:2].clone()
                        log_det_J[i_mol][i_conf] += contribution_det_J[coord_idx].clone()
                        coord_idx += 1
            
            # Fourth and on, generate distance, angle and torsion
            
            for i_atom in range(3, max(n_atoms)):
                cat_mol_rep, cat_at_rep, schnet_inputs, cat_ref_frame_rep, cat_symmetry_token = cat_atom_representation(i_atom, mol_batch, mol_rep, at_rep, [mol_confs.clone().detach() for mol_confs in conformations], n_conf, self.device)
                # Note that the previous sub-conformation must be detached: autoregressive model.
                schnet_rep = self.schnet(schnet_inputs)['y']
                z = self.flow.base_distribution.sample( (1, cat_mol_rep.size()[0]) ).squeeze(dim = 0).to(self.device)
                dat_input = torch.cat( (cat_mol_rep, cat_at_rep, schnet_rep, cat_ref_frame_rep, cat_symmetry_token, z), dim = 1  ).unsqueeze( dim = 1)
                new_coord, contribution_det_J = self.flow.pushforward(dat_input, self.flow.mask_dat)
                new_coord = new_coord.squeeze(dim = 1)

                
                coord_idx = 0
                for i_mol, _ in enumerate(mol_batch):
                    if n_atoms[i_mol] > i_atom:
                        for i_conf in range(n_conf):
                            conformations[i_mol][i_conf, i_atom-1, :] = new_coord[coord_idx].clone()
                            log_det_J[i_mol][i_conf] += contribution_det_J[coord_idx].clone()
                            coord_idx += 1
                #print(conformations[0])
            
        return conformations, log_det_J