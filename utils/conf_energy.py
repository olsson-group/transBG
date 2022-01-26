import torch
from openff.toolkit.topology import Molecule
from openff.toolkit.topology import Topology
from openff.toolkit.typing.engines.smirnoff import ForceField


from simtk import openmm, unit

from rdkit import Chem
from rdkit.Geometry import Point3D

def get_conformations_energy(positions, system):
    '''
    Arguments:
        - system: openmm system instance.
        - positions (simtk.unit.quantity.Quantity, size: (n_atoms, 3) ): position of the atoms of the molecule in Ångström. 
        - integrator (openmm.integrator)
    Retunrs:
        - energy (torch.tensor, size: (1) ): Energy in dimless units
    '''
    integrator = openmm.LangevinIntegrator(300.0 * unit.kelvin, 1.0 / unit.picosecond, 1.0 * unit.femtosecond)
    context = openmm.Context(system, integrator)
    context.setPositions(positions.cpu().numpy()/10.) #Positions must be given in nm!
    state = context.getState(getEnergy=True)
    kB_NA = unit.BOLTZMANN_CONSTANT_kB * unit.AVOGADRO_CONSTANT_NA
    energy_factor = 1. / (integrator.getTemperature() * kB_NA) #0.4 mol/kj
    energy = state.getPotentialEnergy() * energy_factor
    return energy

def get_energy_and_foces(positions, openmm_system, device):
    '''
    Arguments:
        - positions (torch.tensor, size: ( (n_atoms,3) ):position of the atoms of a molecule. Assumed to be in Å.
        - mol (dictionary): mol of conf_dataset
    Retunrs:
        - energy ( torch.tensor, size: (1)  )
        - forces (torch.tensor, size: (n_atoms,3) )
    '''

    integrator = openmm.LangevinIntegrator(300.0 * unit.kelvin, 1.0 / unit.picosecond, 1.0 * unit.femtosecond)
    context = openmm.Context(openmm_system, integrator)
    context.setPositions(positions.cpu().numpy()/10.) #Positions must be given in nm!
    state = context.getState(getEnergy=True, getForces = True)
    kB_NA = unit.BOLTZMANN_CONSTANT_kB * unit.AVOGADRO_CONSTANT_NA
    energy_factor = 1. / (integrator.getTemperature() * kB_NA) #0.4 mol/kj
    energy = torch.tensor(state.getPotentialEnergy() * energy_factor, dtype = torch.float32, device = device) # .value_in_unit() #units are handled by the package
    forces = state.getForces(asNumpy=True).in_units_of(unit.kilojoule/(unit.angstrom*unit.mole)) #Note: here the length unit has to be the one the model generates. 
    #This quantity has dims 1/distance dim of the model
    forces = forces * energy_factor
    forces = torch.tensor( forces._value, dtype = torch.float32, device = device) #Check that the units are correct here!

    return energy, forces


class OpenMMEnergyWrapper(torch.autograd.Function):  
    @staticmethod
    def forward(ctx, positions, openmm_system, device):
        energy, forces = get_energy_and_foces(positions, openmm_system, device)
        ctx.save_for_backward(-forces)
        if torch.any(torch.isnan(forces)) or torch.isnan(energy):
            #print('\n Found infinite values computing nergy and forces.', flush=True)
            raise 
        else:
            return energy
    @staticmethod
    def backward(ctx, grad_output):
        neg_force, = ctx.saved_tensors
        grad_input = grad_output * neg_force
        return grad_input, None, None


def old_get_energy_and_foces(positions, mol, device):
    '''
    Arguments:
        - positions (torch.tensor, size: ( (n_atoms,3) ):position of the atoms of a molecule.
        - mol (dictionary): mol of conf_dataset
    Retunrs:
        - energy ( torch.tensor, size: (1)  )
        - forces (torch.tensor, size: (n_atoms,3) )
    '''
    rd_mol = mol['rdkit_mol']
    partial_charges = mol['partial_charges']
    ref_atoms = mol['ref_atoms']

    conformer = rd_mol.GetConformer()
                        
    #Set the rdkit molecule to the conformation coordinates
    for i_atom in range(rd_mol.GetNumAtoms()):
        x = positions[i_atom, 0].item()
        y = positions[i_atom, 1].item()
        z = positions[i_atom, 2].item()
        conformer.SetAtomPosition(i_atom, Point3D(x,y,z) )
    
    rd_mol.UpdatePropertyCache()
    Chem.AssignAtomChiralTagsFromStructure(rd_mol) 
    off_mol = Molecule.from_rdkit(rd_mol)
    off_mol.partial_charges = partial_charges
    topology = Topology.from_molecules(off_mol)
    forcefield = ForceField('openff-1.2.0.offxml')
    openmm_system = forcefield.create_openmm_system(topology, charge_from_molecules = [off_mol], allow_nonintegral_charges=True)

    positions = off_mol.conformers[0]

    integrator = openmm.LangevinIntegrator(300.0 * unit.kelvin, 1.0 / unit.picosecond, 1.0 * unit.femtosecond)
    context = openmm.Context(openmm_system, integrator)
    context.setPositions(positions)
    state = context.getState(getEnergy=True, getForces = True)
    kB_NA = unit.BOLTZMANN_CONSTANT_kB * unit.AVOGADRO_CONSTANT_NA
    energy_factor = 1. / (integrator.getTemperature() * kB_NA) #0.4 mol/kj
    energy = torch.tensor(state.getPotentialEnergy() * energy_factor, dtype = torch.float32, device = device) # .value_in_unit() #units are handled by the package
    forces = state.getForces(asNumpy=True)
    length_unit = unit.nanometer 
    forces = forces * energy_factor * length_unit
    forces = torch.tensor( forces, dtype = torch.float32, device = device)

    return energy, forces


    