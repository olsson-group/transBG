import os
import torch
from torch.utils.tensorboard import SummaryWriter
from torch import nn
from torch import distributions
from torch.utils.data.sampler import SubsetRandomSampler
from flows.NVP_flow.NVP_conformer import RealNVP
import schnetpack as spk
import numpy as np
import os
import sys
from GGNN.graphinvent import models as gnn_model
from torch.utils.data import DataLoader
from transBG_model import TransBG_model
import random
import time
from statistics import mean
from openff.toolkit.topology import Molecule
from openff.toolkit.topology import Topology
from openff.toolkit.typing.engines.smirnoff import ForceField
from rdkit.Chem.rdmolfiles import PDBWriter
from rdkit.Geometry import Point3D

from utils.conf_dataset import conformer_dataset, collate_conf
from utils.z_matrix import deconstruct_z_matrix, correct_conf_indexes
from utils.conf_energy import OpenMMEnergyWrapper, get_conformations_energy

class TransBG:
    def __init__(self, params):
        self.params = params

    def build_model(self):
        p = self.params

        #Setting a fixed seed for reproducibility
        torch.manual_seed(p.random_seed)
        random.seed(p.random_seed)

        print('Preparing data...', end = '', flush = True)
        sys.stdout.flush()

        #Split data into train and validation set
        indices = [i for i in list(range(p.dataset_size)) if i not in p.energy_val_indices]
        split_1 = int(np.floor(p.val_set_size * p.dataset_size))
        split_2 =  int(np.floor( (p.val_set_size + p.test_set_size) * p.dataset_size))
        np.random.seed(p.random_seed)
        np.random.shuffle(indices)
        train_indices, val_indices, test_indices = indices[split_2:], indices[:split_1], indices[split_1:split_2]
        # Create data samplers and loaders:
        train_sampler = SubsetRandomSampler(train_indices)
        val_sampler = SubsetRandomSampler(val_indices)
        test_sampler = SubsetRandomSampler(test_indices)

        self.dataset = conformer_dataset(p.sdf_path, p.xyz_path, pdb_path = p.pdb_path)
        self.train_loader = DataLoader(self.dataset, batch_size = p.batch_size, shuffle = False, collate_fn = collate_conf, sampler = train_sampler) #Set shuffle to False as is mutually exclussive with a sampler
        self.val_loader = DataLoader(self.dataset, batch_size = p.batch_size, shuffle = False, collate_fn = collate_conf, sampler = val_sampler)
        self.test_loader = DataLoader(self.dataset, batch_size = p.batch_size, shuffle = False, collate_fn = collate_conf, sampler = test_sampler)

        #Now the same for energy based learning

        #Energy learning set
        self.energy_train_indices = p.energy_train_indices
        self.energy_val_indices = p.energy_val_indices
        self.max_energies = torch.tensor(p.max_energies, dtype = torch.float32, device = p.device)

        # Create data samplers and loaders:
        energy_train_sampler = SubsetRandomSampler(self.energy_train_indices)
        energy_val_sampler = SubsetRandomSampler(self.energy_val_indices)
        self.energy_train_loader = DataLoader(self.dataset, batch_size = p.e_batch_size, shuffle = False, collate_fn = collate_conf, sampler = energy_train_sampler) #Set shuffle to False as is mutually exclussive with a sampler
        self.energy_val_loader = DataLoader(self.dataset, batch_size = p.e_batch_size, shuffle = False, collate_fn = collate_conf, sampler = energy_val_sampler)

        print('Done!', flush = True)

        print('Building the model...', end = '', flush = True)
        sys.stdout.flush()

        #Length of the vector codifying the information that is fed into the flow: context representation
        rep_size = p.mol_rep_size + p.atom_rep_size + p.schnet_rep_size + p.ref_frame_rep_size + p.symmetry_token_size
        #Generator functions for the scaling and translation network (simple feed forward neural nets):
        nets = lambda: nn.Sequential(nn.Linear(3 + rep_size, p.intermediate_rep_size), nn.LeakyReLU(), nn.Linear(p.intermediate_rep_size, p.intermediate_rep_size), nn.LeakyReLU(), nn.Linear(p.intermediate_rep_size, 3 + rep_size), nn.Tanh())
        nett = lambda: nn.Sequential(nn.Linear(3 + rep_size, p.intermediate_rep_size), nn.LeakyReLU(), nn.Linear(p.intermediate_rep_size, p.intermediate_rep_size), nn.LeakyReLU(), nn.Linear(p.intermediate_rep_size, 3 + rep_size))
        #masking (alternate the dimensions that are unchanged and the one which is transformed): 0 means transformed. The context representation is always masked.
        mask = np.ones((3, rep_size + 3))
        mask[0, rep_size + 3 - 3] = 0
        mask_d = mask.tolist()
        masks_d = torch.from_numpy(np.array(mask_d * p.num_coupling_layers).astype(np.float32)) # Mask for transforming only distances
        # Note: this can seem not very efficient since we forward just 0s multiplications when not transforming, but this is handled at the flow model.
        mask[1, rep_size + 3 - 2] = 0
        mask_da = mask.tolist()
        masks_da = torch.from_numpy(np.array(mask_da * p.num_coupling_layers).astype(np.float32)) # Mask for transforming distances and angles
        mask[2, rep_size + 3 - 1] = 0
        mask_dat = mask.tolist()
        mask_dat = torch.from_numpy(np.array(mask_dat * p.num_coupling_layers).astype(np.float32)) 
        # define base distribution
        base_distribution = distributions.MultivariateNormal(torch.zeros(3), torch.eye(3))

        # ggnn instance
        gnn = gnn_model.initialize_GGNN(p)

        # flow instance
        flow = RealNVP(nets, nett, mask_d = masks_d, mask_da = masks_da, mask_dat = mask_dat, base_dist = base_distribution, device = p.device).to(p.device)
        # schnet instance
        atomic_reps = spk.representation.SchNet(n_atom_basis = p.schnet_rep_size, n_filters = p.schnet_rep_size, n_interactions = p.n_interactions,
                                                cutoff = p.cutoff, n_gaussians = p.n_gaussians, normalize_filter = p.normalize_filter, coupled_interactions = p.coupled_interactions, # cutoff is given in Angstrong
                                                max_z = p.max_z + 1, trainable_gaussians = p.trainable_gaussians)  # max_z needs one unit more or raises errors
        schnet_summarizer = spk.atomistic.Atomwise(n_in = p.schnet_rep_size, n_out = p.schnet_rep_size)
        schnet = spk.atomistic.AtomisticModel(atomic_reps, schnet_summarizer).to(p.device)

        self.model = TransBG_model(flow, schnet, gnn, p,  torch.device(p.device))
        #self.best_model = ConformationsGenerator(flow, schnet, gnn, p,  torch.device(p.device))    #We do not generate this for memory constrains.
        self.l_min_val_loss = float('inf')
        self.e_min_val_loss = float('inf')
        self.ft_min_val_loss = float('inf')
        self.l_training_metrics = {'train_loss' : [], 'val_loss' : [] }
        self.e_training_metrics = {'train_loss' : [], 'val_loss' : [], 'train_average_energy' : [], 'val_average_energy' : [] }


        print("Done!", flush = True)

    def train_likelihood(self):
        p = self.params

        #Setting a fixed seed for reproducibility
        torch.manual_seed(p.random_seed)
        random.seed(p.random_seed)

        #Define optimizer 
        self.l_optimizer = torch.optim.Adam(self.model.parameters(), lr = p.l_learning_rate, weight_decay = p.weight_decay)

        #Create  learning rate scheduler

        self.l_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer=self.l_optimizer,
                                                               max_lr= p.l_max_rel_lr * p.l_learning_rate,
                                                               div_factor= 1. / p.l_max_rel_lr,
                                                               final_div_factor = 1. / p.l_min_rel_lr,
                                                               pct_start = 0.01,
                                                               total_steps = p.l_epochs
                                                               )
 

        #Create a folder for the best model
        if not os.path.isdir("models"):
            os.mkdir("models/")
        
        #Create a folder for the training metrics
        if not os.path.isdir("metrics"):
            os.mkdir("metrics/")

        for epoch in range(1, p.l_epochs + 1):

            t0 = time.time()
            loss_sum = 0
            discarded_batches = 0

            for i_batch, batch in enumerate(self.train_loader ,1):

                mol_batch = batch
                self.model.train()
                z, log_det_invJ = self.model.pushback(mol_batch) 

                if not torch.any( torch.isnan(z) ):
                    z = z.squeeze( dim = 1 )
                    #nan_flag = [ i_row for i_row in range( z.size()[0] ) if not torch.any( torch.isnan(z[i_row])) or torch.any( torch.isnan(log_det_invJ[i_row])) ]
                    #z = z[nan_flag]
                    #log_det_invJ = log_det_invJ[nan_flag]

                    #Likelihood loss:
                    armonic_en = torch.tensor(0.5, dtype = torch.float32) * z * z
                    armonic_en = armonic_en.squeeze(dim = 1).sum(dim = 1)
                    loss = armonic_en - log_det_invJ

                    loss = loss[loss == loss] # Ignore nan values, in case they accidentally appear
                    loss = torch.mean( loss )

                    self.l_optimizer.zero_grad()
                    loss.backward()
                    self.l_optimizer.step()

                    train_loss = loss.item()
                    loss_sum += train_loss
                else:
                    discarded_batches += 1
                    print('Discarded batch due to the presence of nan values')

                print('.', end = '')
                sys.stdout.flush()

            #Evaluate:
            print( ' \n Evaluating: ')
            self.model.eval()
            val_loss_sum = 0
            discarded_batches = 0
            with torch.no_grad():
                for i_batch, batch in enumerate(self.val_loader ,1):
    
                    mol_batch = batch
                    z, log_det_invJ = self.model.pushback(mol_batch) 

                    if not torch.any( torch.isnan(z) ):
                        z = z.squeeze( dim = 1 )
                        nan_flag = [ i_row for i_row in range( z.size()[0] ) if not torch.any( torch.isnan(z[i_row]) ) or torch.any( torch.isnan(log_det_invJ[i_row]) ) ]
                        z = z[nan_flag]
                        log_det_invJ = log_det_invJ[nan_flag]

                        #Likelihood loss:
                        val_armonic_en = torch.tensor(0.5, dtype = torch.float32) * z * z
                        val_armonic_en = val_armonic_en.squeeze(dim = 1).sum(dim = 1)
                        val_loss = torch.mean(val_armonic_en - log_det_invJ)

                        val_loss_sum += val_loss.item()

                    else:
                        discarded_batches += 1
                        print('Discarded batch due to the presence of nan values')


                    print('.', end = '')
                    sys.stdout.flush()
                
            #Update learning rate
            self.l_scheduler.step()

            #Print some statistics:
            t1 = time.time()
            #train_loss = loss_sum/len(self.train_loader) # We print and use the train loss of the last batch (only fair to compare)
            val_loss = val_loss_sum/( len(self.val_loader) - discarded_batches )
            self.l_training_metrics['train_loss'].append(train_loss)
            np.save('metrics/train_loss_' + p.model_name + '.npy', self.l_training_metrics['train_loss'])
            self.l_training_metrics['val_loss'].append(val_loss)
            np.save('metrics/val_loss_' + p.model_name + '.npy', self.l_training_metrics['val_loss'])
            print(f' \n Epoch: {epoch}: training loss: {train_loss:.4f}, validation_loss: {val_loss:.4f}, time = {t1-t0:.4f}')
            sys.stdout.flush()

            #Keep the best model
            if val_loss < self.l_min_val_loss:
                self.l_min_val_loss = val_loss
                #self.best_model.load_state_dict( self.model.state_dict() )
                torch.save( self.model.state_dict(), 'models/' + p.model_name )

    def finetune_likelihood(self, min_e = False):
        p = self.params

        #Setting a fixed seed for reproducibility
        torch.manual_seed(p.random_seed)
        random.seed(p.random_seed)

        #Define optimizer 
        self.ft_optimizer = torch.optim.Adam(self.model.parameters(), lr = p.ft_learning_rate, weight_decay = p.weight_decay)

        #Create  learning rate scheduler

        self.ft_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer=self.ft_optimizer,
                                                            max_lr= p.ft_max_rel_lr * p.ft_learning_rate,
                                                            div_factor= 1. / p.ft_max_rel_lr,
                                                            final_div_factor = 1. / p.ft_min_rel_lr,
                                                            pct_start = 0.01,
                                                            total_steps = p.ft_epochs
                                                            )


        #Create a folder for the best model
        if not os.path.isdir("models"):
            os.mkdir("models/")
        
        #Create a folder for the training metrics
        if not os.path.isdir("metrics"):
            os.mkdir("metrics/")

        for epoch in range(1, p.ft_epochs + 1):

            t0 = time.time()
            loss_sum = 0
            discarded_batches = 0

            for i_batch, batch in enumerate(self.energy_train_loader ,1):

                mol_batch = batch
                self.model.train()
                z, log_det_invJ = self.model.pushback(mol_batch, min_e) 

                if not torch.any( torch.isnan(z) ):
                    z = z.squeeze( dim = 1 )
                    #nan_flag = [ i_row for i_row in range( z.size()[0] ) if not torch.any( torch.isnan(z[i_row])) or torch.any( torch.isnan(log_det_invJ[i_row])) ]
                    #z = z[nan_flag]
                    #log_det_invJ = log_det_invJ[nan_flag]

                    #Likelihood loss:
                    armonic_en = torch.tensor(0.5, dtype = torch.float32) * z * z
                    armonic_en = armonic_en.squeeze(dim = 1).sum(dim = 1)
                    loss = armonic_en - log_det_invJ

                    loss = loss[loss == loss] # Ignore nan values, in case they accidentally appear
                    loss = torch.mean( loss )

                    self.ft_optimizer.zero_grad()
                    loss.backward()
                    self.ft_optimizer.step()

                    train_loss = loss.item()
                    loss_sum += train_loss
                else:
                    discarded_batches += 1
                    print('Discarded batch due to the presence of nan values')

                print('.', end = '')
                sys.stdout.flush()

            #Evaluate:
            print( ' \n Evaluating: ')
            self.model.eval()
            val_loss_sum = 0
            discarded_batches = 0
            with torch.no_grad():
                for i_batch, batch in enumerate(self.energy_val_loader ,1):
    
                    mol_batch = batch
                    z, log_det_invJ = self.model.pushback(mol_batch, min_e) 

                    if not torch.any( torch.isnan(z) ):
                        z = z.squeeze( dim = 1 )
                        nan_flag = [ i_row for i_row in range( z.size()[0] ) if not torch.any( torch.isnan(z[i_row]) ) or torch.any( torch.isnan(log_det_invJ[i_row]) ) ]
                        z = z[nan_flag]
                        log_det_invJ = log_det_invJ[nan_flag]

                        #Likelihood loss:
                        val_armonic_en = torch.tensor(0.5, dtype = torch.float32) * z * z
                        val_armonic_en = val_armonic_en.squeeze(dim = 1).sum(dim = 1)
                        val_loss = torch.mean(val_armonic_en - log_det_invJ)

                        val_loss_sum += val_loss.item()

                    else:
                        discarded_batches += 1
                        print('Discarded batch due to the presence of nan values')


                    print('.', end = '')
                    sys.stdout.flush()
                
            #Update learning rate
            self.ft_scheduler.step()

            #Print some statistics:
            t1 = time.time()
            train_loss = loss_sum/len(self.energy_train_loader) 
            val_loss = val_loss_sum/( len(self.energy_val_loader) - discarded_batches )
            #self.l_training_metrics['train_loss'].append(train_loss)
            #np.save('metrics/train_loss_' + p.model_name + '.npy', self.l_training_metrics['train_loss'])
            #self.l_training_metrics['val_loss'].append(val_loss)
            #np.save('metrics/val_loss_' + p.model_name + '.npy', self.l_training_metrics['val_loss'])
            print(f' \n Epoch: {epoch}: training loss: {train_loss:.4f}, validation_loss: {val_loss:.4f}, time = {t1-t0:.4f}')
            sys.stdout.flush()

    def generate_conformations(self, mol_batch, n_conf = 1):
        '''

        '''
        conformations, log_det_J = self.model.pushforward(mol_batch, n_conf)
        return conformations, log_det_J

    def train_energy(self, min_e = False):
        '''
            The model needs to have been pretrained (i.e. using likelihood-based learning).
        ''' 

        p = self.params

        #Setting a fixed seed for reproducibility
        torch.manual_seed(p.random_seed)
        random.seed(p.random_seed)

        #Select the trainable models:
        if not p.train_GNN_ebl:
            for parameter in self.model.gnn.parameters():
                parameter.requires_grad = False

        if not p.train_schnet_ebl:
            for parameter in self.model.schnet.parameters():
                parameter.requires_grad = False

        #Define optimizer 
        self.e_optimizer = torch.optim.Adam(self.model.parameters(), lr = p.e_learning_rate, weight_decay = p.weight_decay)

        #Create  learning rate scheduler

        self.e_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer=self.e_optimizer,
                                                               max_lr= p.e_max_rel_lr * p.e_learning_rate,
                                                               div_factor= 1. / p.e_max_rel_lr,
                                                               final_div_factor = 1. / p.e_min_rel_lr,
                                                               pct_start = 0.01,
                                                               total_steps = p.e_epochs
                                                               )
 
        #Instance of the energy wrapper
        energy_wrapper = OpenMMEnergyWrapper()

        # Create the OpenMM systems
        forcefield = ForceField('openff-1.2.0.offxml')
        systems_dict = {}
        for i_mol in p.energy_train_indices + p.energy_val_indices:
            mol_dict = self.dataset[i_mol]
            rd_mol = mol_dict['rdkit_mol']
            partial_charges = mol_dict['partial_charges']

            off_mol = Molecule.from_rdkit(rd_mol)
            off_mol.partial_charges = partial_charges
            topology = Topology.from_molecules(off_mol)      
            openmm_system = forcefield.create_openmm_system(topology, charge_from_molecules = [off_mol], allow_nonintegral_charges=True)
            systems_dict[str(mol_dict['qm9_idx'])] = openmm_system

        #Create a folder for the best model
        if not os.path.isdir("models"):
            os.mkdir("models/")
        
        #Create a folder for the training metrics
        if not os.path.isdir("metrics"):
            os.mkdir("metrics/")
        
        #Tensorboard writer instance: Writer will output to ./runs/ directory by default
        if not os.path.isdir("runs/"+self.params.model_name):
            os.mkdir("runs/"+self.params.model_name)
        e_writer = SummaryWriter(log_dir="runs/"+self.params.model_name, comment='_rep')

        for epoch in range(1, p.e_epochs + 1):

            t0 = time.time()
            train_loss_sum = 0
            train_average_energy_sum = 0
            l_train_loss_sum = 0
            e_train_loss_sum = 0
            constr_train_loss_sum = 0

            pm_e_train_loss = torch.zeros((len(self.energy_train_indices)), dtype=torch.float32).to(p.device)
            
            train_energy = torch.zeros((len(self.energy_train_indices)), dtype=torch.float32).to(p.device)
            train_conf_rate = [0.0 for _ in range( len(self.energy_train_indices) )]
            train_discarded_batches = 0

            for i_batch, batch in enumerate(self.energy_train_loader ,1):

                mol_batch = batch
                self.model.train()
                 
                #Likelihood loss:
                z, log_det_invJ = self.model.pushback(mol_batch, min_e)
                z = z.squeeze( dim = 1 )
                nan_flag = [ i_row for i_row in range( z.size()[0] ) if not torch.any( torch.isnan(z[i_row]) ) or torch.any( torch.isnan(log_det_invJ[i_row]) ) ]
                z = z[nan_flag]
                log_det_invJ = log_det_invJ[nan_flag]
                armonic_en = torch.tensor(0.5, dtype = torch.float32) * z * z
                armonic_en = armonic_en.squeeze(dim = 1).sum(dim = 1)
                l_train_loss = armonic_en - log_det_invJ
                l_train_loss = l_train_loss[l_train_loss == l_train_loss] # Ignore nan values, in case they accidentally appear
                l_train_loss = torch.mean( l_train_loss )
                l_train_loss_sum += l_train_loss.item()

                #Energy loss:
                
                conformations, log_det_J = self.model.pushforward(mol_batch = mol_batch, n_conf = p.n_conformations)

                #nan protection
                for i_mol, mol_dict in enumerate(mol_batch):
                    nan_conf_indexes = torch.unique( torch.nonzero(torch.isnan(conformations[i_mol]))[:,0] ).tolist()
                    nan_det_indexes = torch.unique( torch.nonzero(torch.isnan(log_det_J[i_mol]))[:,0] ).tolist()
                    nan_conf_indexes = list(set(nan_conf_indexes) | set(nan_det_indexes))
                    nan_conf_indexes.sort()
                    ok_conf_indexes = list( set(list(range(p.n_conformations))) - set(nan_conf_indexes) )
                    if nan_conf_indexes:
                        nan_mol_index = mol_dict['dataset_idx']
                        print(f'Found nans in {len(nan_conf_indexes)} conformations of molecule {nan_mol_index}. Ignored for training.')
                        conformations[i_mol] = conformations[i_mol][ok_conf_indexes]
                        log_det_J[i_mol] = log_det_J[i_mol][ok_conf_indexes]

                mol_indexes = [ mol['dataset_idx'] for mol in mol_batch]
                mol_indexes_train_energy_set = [ self.energy_train_indices.index( mol['dataset_idx'] ) for mol in mol_batch ]
                mol_indexes_batch = []
                e_train_loss_backward = torch.zeros((len(mol_batch)), dtype=torch.float32).to(p.device)
                
                for i_mol, mol in enumerate(mol_batch):
                    mol_index_train_energy_set = self.energy_train_indices.index( mol['dataset_idx'] )
                    mol_train_energy = torch.zeros( (conformations[i_mol].size()[0] ), dtype=torch.float32).to(p.device)#len(conformations[i_mol])
                    mol_constr_loss = torch.zeros( (conformations[i_mol].size()[0] ), dtype=torch.float32).to(p.device)#
                    energy_comp_indexes = []
                    for i_conf, conformation in enumerate(conformations[i_mol]):
                        cartesian_coord, mol_constr_loss[i_conf] = deconstruct_z_matrix( conformation.clone(), mol['ref_atoms'].copy() ) #The Z-matrixes are clamped at deconstruct_z_matrix
                        try:
                            energy = energy_wrapper.apply(cartesian_coord, systems_dict[str(mol['qm9_idx'])], p.device)
                            if not torch.isnan(energy):
                                mol_train_energy[i_conf] = energy.clone()
                                energy_comp_indexes.append(i_conf) 
                        except:
                            #print('+', flush = True, end = '')
                            mol_train_energy[i_conf] = self.max_energies[ mol_index_train_energy_set ] + 1.
                            #print('\n There was a problem computing energies with conformation {idx_conf} of molecule {idx_mol}. This conformation is skipped for training.'.format(idx_mol = mol['dataset_idx'], idx_conf = i_conf), flush=True)
                    if len(energy_comp_indexes) != conformations[i_mol].size()[0]:
                        idx_mol = mol['dataset_idx']
                        energy_comp_rate = float(len(energy_comp_indexes)) / float(conformations[i_mol].size()[0])
                        print( f'OpenMM was able to compute the energy for {energy_comp_rate*100:.4f} % of the conformations of molecule {idx_mol}.' )
                    constr_train_loss_sum += torch.mean(mol_constr_loss).item()
                    mol_train_energy_backward = mol_train_energy[ mol_train_energy < self.max_energies[ mol_index_train_energy_set ] ].clone()
                    mol_log_det_J_backward = log_det_J[i_mol][mol_train_energy < self.max_energies[mol_index_train_energy_set]].clone()   
                    mol_constr_loss_backward = mol_constr_loss[mol_train_energy < self.max_energies[mol_index_train_energy_set]].clone()
                    mol_e_train_loss_backward = mol_train_energy_backward - mol_log_det_J_backward + mol_constr_loss_backward
                    if mol_e_train_loss_backward.numel():
                        e_train_loss_backward[i_mol] = torch.mean(mol_e_train_loss_backward)
                        mol_indexes_batch.append(i_mol)
                    else:
                        dataset_idx = mol['dataset_idx']
                        print(f'\n No learnable conformations generated for molecule {dataset_idx}.')
                        mol_indexes.remove(dataset_idx)
                        mol_indexes_train_energy_set.remove(mol_index_train_energy_set)

                    mol_train_energy = mol_train_energy[energy_comp_indexes]
                    mol_log_det_J = log_det_J[i_mol][energy_comp_indexes].clone()#.detach()
                    mol_e_train_loss = mol_train_energy - mol_log_det_J
                    pm_e_train_loss[ mol_index_train_energy_set ] = torch.mean(mol_e_train_loss)
                    train_energy[ mol_index_train_energy_set ] = torch.mean(mol_train_energy)

                    # Percentage of used conformations
                    mol_conf_rate = mol_train_energy_backward.size()[0]
                    if mol_conf_rate:
                        mol_conf_rate = float(mol_conf_rate)/p.n_conformations
                    else:
                        mol_conf_rate = 0.

                    train_conf_rate[ mol_index_train_energy_set ] = mol_conf_rate
                    

                e_train_loss_mean =  torch.mean(pm_e_train_loss[mol_indexes_train_energy_set])  # No track gradients # Of the batch
                e_train_loss_backward_mean =  torch.mean(e_train_loss_backward[mol_indexes_batch])
                
                if e_train_loss_backward_mean.numel():
                    total_loss_backward = p.l_weight*l_train_loss + p.e_weight*e_train_loss_backward_mean
                    self.e_optimizer.zero_grad()
                    total_loss_backward.backward()
                    self.e_optimizer.step()
                
                else:
                    train_discarded_batches += 1
                    print('Discarded batch due to non-learnable configurations.')
                
                total_loss = p.l_weight*l_train_loss + p.e_weight*e_train_loss_mean

                train_loss = total_loss.item()
                train_loss_sum += train_loss

                train_average_energy = torch.mean(train_energy).item()
                train_average_energy_sum += train_average_energy

                e_train_loss_sum += e_train_loss_mean.item()

                print('.', end = '')
                sys.stdout.flush()
            
             

            #Evaluate:
            print( ' \n Evaluating: ')
            self.model.eval()
            val_loss_sum = 0
            val_average_energy_sum = 0
            l_val_loss_sum = 0
            e_val_loss_sum = 0
            constr_val_loss_sum = 0

            pm_e_val_loss = torch.zeros((len(self.energy_val_indices)), dtype=torch.float32).to(p.device)
            
            val_energy = torch.zeros((len(self.energy_val_indices)), dtype=torch.float32).to(p.device)
            val_conf_rate = [0.0 for _ in range( len(self.energy_val_indices) )]
            val_discarded_batches = 0

            with torch.no_grad():
                for i_batch, mol_batch in enumerate(self.energy_val_loader ,1):
                    
                    #Likelihood loss:
                    z, log_det_invJ = self.model.pushback(mol_batch, min_e)
                    z = z.squeeze( dim = 1 )
                    nan_flag = [ i_row for i_row in range( z.size()[0] ) if not torch.any( torch.isnan(z[i_row]) ) or torch.any( torch.isnan(log_det_invJ[i_row]) ) ]
                    z = z[nan_flag]
                    log_det_invJ = log_det_invJ[nan_flag]
                    armonic_en = torch.tensor(0.5, dtype = torch.float32) * z * z
                    armonic_en = armonic_en.squeeze(dim = 1).sum(dim = 1)
                    l_val_loss = armonic_en - log_det_invJ
                    l_val_loss = l_val_loss[l_val_loss == l_val_loss] # Ignore nan values, in case they accidentally appear
                    l_val_loss = torch.mean( l_val_loss )
                    l_val_loss_sum += l_val_loss.item()

                    #Energy loss:
                    
                    conformations, log_det_J = self.model.pushforward(mol_batch = mol_batch, n_conf = p.n_conformations)

                    #nan protection
                    for i_mol, mol_dict in enumerate(mol_batch):
                        nan_conf_indexes = torch.unique( torch.nonzero(torch.isnan(conformations[i_mol]))[:,0] ).tolist()
                        nan_det_indexes = torch.unique( torch.nonzero(torch.isnan(log_det_J[i_mol]))[:,0] ).tolist()
                        nan_conf_indexes = list(set(nan_conf_indexes) | set(nan_det_indexes))
                        nan_conf_indexes.sort()
                        ok_conf_indexes = list( set(list(range(p.n_conformations))) - set(nan_conf_indexes) )
                        if nan_conf_indexes:
                            nan_mol_index = mol_dict['dataset_idx']
                            print(f'Found nans in {len(nan_conf_indexes)} conformations of molecule {nan_mol_index}. Ignored for training.')
                            conformations[i_mol] = conformations[i_mol][ok_conf_indexes]
                            log_det_J[i_mol] = log_det_J[i_mol][ok_conf_indexes]

                    mol_indexes = [ mol['dataset_idx'] for mol in mol_batch]
                    mol_indexes_val_energy_set = [ self.energy_val_indices.index( mol['dataset_idx'] ) for mol in mol_batch ]
                    mol_indexes_batch = []
                    #e_val_loss_backward = torch.zeros((len(mol_batch)), dtype=torch.float32).to(p.device)
                    
                    for i_mol, mol in enumerate(mol_batch):
                        mol_index_val_energy_set = self.energy_val_indices.index( mol['dataset_idx'] )
                        mol_val_energy = torch.zeros( conformations[i_mol].size()[0], dtype=torch.float32).to(p.device)
                        mol_constr_val_loss = torch.zeros( conformations[i_mol].size()[0], dtype=torch.float32).to(p.device)#p.n_conformations
                        energy_comp_indexes = []
                        for i_conf, conformation in enumerate(conformations[i_mol]):
                            cartesian_coord, mol_constr_val_loss[i_conf] = deconstruct_z_matrix( conformation.clone(), mol['ref_atoms'].copy() ) #The Z-matrixes are clamped at deconstruct_z_matrix
                            try:
                                energy = energy_wrapper.apply(cartesian_coord, systems_dict[str(mol['qm9_idx'])], p.device)
                                if not torch.isnan(energy):
                                    mol_val_energy[i_conf] = energy.clone()
                                    energy_comp_indexes.append(i_conf) 
                            except:
                                pass
                                #print('+', flush = True, end = '')
                                #print('\n There was a problem computing energies with conformation {idx_conf} of molecule {idx_mol}. This conformation is skipped for evaluation.'.format(idx_mol = mol['dataset_idx'], idx_conf = i_conf), flush=True)
                        
                        if len(energy_comp_indexes) != conformations[i_mol].size()[0]:
                            idx_mol = mol['dataset_idx']
                            energy_comp_rate = float( len(energy_comp_indexes) ) / float(conformations[i_mol].size()[0])
                            print( f'OpenMM was able to compute the energy for {energy_comp_rate*100:.4f} % of the conformations of molecule {idx_mol}.' )

                        mol_val_energy = mol_val_energy[energy_comp_indexes]
                        constr_val_loss_sum += torch.mean(mol_constr_val_loss).item() 
                        mol_constr_val_loss = mol_constr_val_loss[energy_comp_indexes]
                        mol_log_det_J = log_det_J[i_mol][energy_comp_indexes].clone().detach()
                        mol_e_val_loss = mol_val_energy - mol_log_det_J + mol_constr_val_loss
                        pm_e_val_loss[ mol_index_val_energy_set ] = torch.mean(mol_e_val_loss)

                        val_energy[ mol_index_val_energy_set ] = torch.mean(mol_val_energy)

                    # Compute statistics
                    e_val_loss_mean =  torch.mean(pm_e_val_loss[mol_indexes_val_energy_set])
                    total_loss = p.l_weight*l_val_loss + p.e_weight*e_val_loss_mean

                    val_loss = total_loss.item()
                    val_loss_sum += val_loss

                    val_average_energy = torch.mean(val_energy).item()
                    val_average_energy_sum += val_average_energy

                    e_val_loss_sum += e_val_loss_mean.item()

                    print('.', end = '')
                    sys.stdout.flush()


                
            #Update learning rate
            self.e_scheduler.step()

            #Print some (a lot of :) )  statistics:
            t1 = time.time()
            train_loss = train_loss_sum/ ( len(self.energy_train_loader) -  train_discarded_batches )
            val_loss = val_loss_sum / ( len(self.energy_val_loader) -  val_discarded_batches )
            train_average_energy = train_average_energy_sum / len(self.energy_train_loader)
            val_average_energy = val_average_energy_sum / len(self.energy_val_loader)
            l_train_loss = l_train_loss_sum / len(self.energy_train_loader)
            e_train_loss = e_train_loss_sum / len(self.energy_train_loader)
            l_val_loss = l_val_loss_sum / len(self.energy_val_loader)
            e_val_loss = e_val_loss_sum / len(self.energy_val_loader)
            constr_train_loss = constr_train_loss_sum / len(self.energy_train_indices) 
            constr_val_loss = constr_val_loss_sum / len(self.energy_val_indices) 

            self.e_training_metrics['train_loss'].append(train_loss)
            np.save('metrics/e_train_loss_' + p.model_name + '.npy', self.e_training_metrics['train_loss'])
            self.e_training_metrics['val_loss'].append(val_loss)
            np.save('metrics/e_val_loss_' + p.model_name + '.npy', self.e_training_metrics['val_loss'])
            self.e_training_metrics['train_average_energy'].append(train_average_energy)
            np.save('metrics/e_train_average_energy_' + p.model_name + '.npy', self.e_training_metrics['train_average_energy'])
            self.e_training_metrics['val_average_energy'].append(val_average_energy)
            np.save('metrics/e_val_average_energy_' + p.model_name + '.npy', self.e_training_metrics['val_average_energy'])
            print(f' \n Epoch: {epoch}: training loss: {train_loss:.4f}, validation_loss: {val_loss:.4f}, training average energy: {train_average_energy:.4f}, validation average energy: {val_average_energy:.4f}, \n likelihood train loss: {l_train_loss:.4f}, likelihood validation loss: {l_val_loss:.4f}, energy train loss: {e_train_loss:.4f}, energy validation loss: {e_val_loss:.4f}, \n train conformations rate: {mean(train_conf_rate):4f}, training constraint loss: {constr_train_loss:.4f}, validation constraint loss: {constr_val_loss:.4f}, time = {t1-t0:.4f}')
            sys.stdout.flush()

            e_writer.add_scalars('Loss/Total loss', {'Total train loss':train_loss, 'Total val loss':val_loss}, epoch)
            e_writer.add_scalars('Loss/Likelihood loss', {'Likelihood train loss':l_train_loss, 'Likelihood val loss':l_val_loss}, epoch)
            e_writer.add_scalars('Loss/Energy loss', {'Energy train loss':e_train_loss, 'Energy val loss':e_val_loss}, epoch)
            


            # Per molecule statistics
            print('\n Metrics for molecules in the training set:')

            for mol_index_dataset, mol_index in enumerate(self.energy_train_indices):
                print(f' \n Molecule: {mol_index}: average energy: {train_energy[mol_index_dataset].item():.4f}, energy (train) loss: {pm_e_train_loss[mol_index_dataset].item():.4f} \n conformations rate: {train_conf_rate[mol_index_dataset]:4f}') 

            print('\n Metrics for molecules in the validation set:')
            for mol_index_dataset, mol_index in enumerate(self.energy_val_indices):
                print(f' \n Molecule: {mol_index}: average energy: {val_energy[mol_index_dataset].item():.4f}, energy (validation) loss: {pm_e_val_loss[mol_index_dataset].item():.4f}') 

            #Save metrics so that they can be plotted 
            sys.stdout.flush()

            #Compute the energy histograms at this epoch

            if self.params.hist_evo:
                print('Computing training set histograms...', flush=True)
                for mol_index in self.params.energy_train_indices:
                    conformations, _ = self.generate_histogram(mol_index=mol_index, n_conf=self.params.hist_conf, step_conf= 5000, visualize = False, save_files = False, verbose = False, ignore_constraints = False)
                    e_writer.add_histogram('Train histograms/Molecule '+str(mol_index),np.array(conformations), epoch) 
                print('Done!', flush=True)
                print('Computing validation set histograms...', flush=True)
                for mol_index in self.params.energy_val_indices:
                    conformations, _ = self.generate_histogram(mol_index=mol_index, n_conf=self.params.hist_conf, step_conf= 5000, visualize = False, save_files = False, verbose = False, ignore_constraints = False)
                    e_writer.add_histogram('Val histograms/Molecule '+str(mol_index),np.array(conformations), epoch)
                print('Done!', flush=True)

            #Keep the best model
            if val_loss < self.e_min_val_loss:
                self.e_min_val_loss = val_loss
                #self.best_model.load_state_dict( self.model.state_dict() )
                torch.save( self.model.state_dict(), 'models/e_tuned_' + p.model_name )

        e_writer.close()

    def generate_correct_conformations(self, mol_index, n_conf, step_conf):
        '''

        '''
        mol_dict = self.dataset[mol_index]
        mol_batch = [mol_dict]
        n_ok_conf = 0
        conformations = torch.empty(0, dtype=torch.float32)
        log_det_J = torch.empty(0, dtype=torch.float32)
        while n_ok_conf < n_conf:
            if n_conf - n_ok_conf > step_conf:
                now_conf = step_conf
            else:
                now_conf = n_conf - n_ok_conf
            with torch.no_grad():
                new_conformations, new_log_det_J = self.generate_conformations(mol_batch, n_conf = now_conf)
                new_conformations = new_conformations[0]
                new_log_det_J = new_log_det_J[0]
            ok_conf_indexes = correct_conf_indexes(new_conformations) 
            new_conformations = new_conformations[ok_conf_indexes]
            new_log_det_J = new_log_det_J[ok_conf_indexes]
            conformations = torch.cat( (conformations, new_conformations.cpu()), dim=0)
            log_det_J = torch.cat( (log_det_J, new_log_det_J.cpu()), dim=0 ) 
            n_ok_conf += len(ok_conf_indexes)
        return conformations, log_det_J

    def generate_histogram(self, mol_index, n_conf = 100, step_conf = 50, visualize = False, file_name = None, save_files = True, verbose = True, ignore_constraints = False):
        '''

        '''
        if save_files:
            if not file_name:
                file_name =  'mol_'+ str(mol_index) + '_' + self.params.model_name
            energy_file_name = 'histograms_data/' + file_name + '_energies.npy'
            log_det_J_file_name = 'histograms_data/' + file_name + '_log_det_J.npy'
            if visualize:
                pdb_writer = PDBWriter('histograms_data/' + file_name + '.pdb', flavor = 4)

        mol_dict = self.dataset[mol_index]
        self.model.eval()
        with torch.no_grad():
            if verbose: print(f'Generating conformations of molecule {mol_index}...', flush=True)
            if ignore_constraints:
                conformations, log_det_J = self.generate_conformations([mol_dict], n_conf)
                conformations = conformations[0]
                log_det_J = conformations[0]
            else:
                conformations, log_det_J = self.generate_correct_conformations(mol_index, n_conf, step_conf)
            if verbose: print(f'Conformations of molecule {mol_index} generated, computing energies...', flush=True)
        
        forcefield = ForceField('openff-1.2.0.offxml')
        rdkit_mol = mol_dict['rdkit_mol']
        partial_charges = mol_dict['partial_charges']
        ref_atoms = mol_dict['ref_atoms']

        off_mol = Molecule.from_rdkit(rdkit_mol)
        off_mol.partial_charges = partial_charges
        topology = Topology.from_molecules(off_mol)      
        openmm_system = forcefield.create_openmm_system(topology, charge_from_molecules = [off_mol], allow_nonintegral_charges=True)

        energies = []

        for i_conf, conformation in enumerate(conformations):
            cartesian_coord = deconstruct_z_matrix(conformation, ref_atoms, loss = False)
            try:
                energy = get_conformations_energy(cartesian_coord, openmm_system)
            except:
                energy = np.nan
                if verbose: print(f'Failed to compute the energy of coformation {i_conf}.')
            if visualize:
                #Set the rdkit molecule to the conformation coordinates
                conformer = rdkit_mol.GetConformer()
                for i_atom in range(rdkit_mol.GetNumAtoms()):
                    x = cartesian_coord[i_atom, 0].item()
                    y = cartesian_coord[i_atom, 1].item()
                    z = cartesian_coord[i_atom, 2].item()
                    conformer.SetAtomPosition(i_atom, Point3D(x,y,z) )
                pdb_writer.write(rdkit_mol)
                
            energies.append(energy)
        log_det_J = log_det_J.cpu().numpy()
        if save_files:
            np.save(energy_file_name, energies)
            np.save(log_det_J_file_name, log_det_J)

        if verbose: print('Done!', flush=True)

        return energies, log_det_J