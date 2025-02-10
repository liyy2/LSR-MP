import torch
from schnetpack.datasets import MD17
from torch_scatter import scatter
# __all__ = ["build_mse_loss", "build_mse_loss_with_forces"]


class LossFnError(Exception):
    pass


def build_mse_loss(properties, loss_tradeoff=None):
    """
    Build the mean squared error loss function.

    Args:
        properties (list): mapping between the model properties and the
            dataset properties
        loss_tradeoff (list or None): multiply loss value of property with tradeoff
            factor

    Returns:
        mean squared error loss function

    """
    if loss_tradeoff is None:
        loss_tradeoff = [1] * len(properties)
    if len(properties) != len(loss_tradeoff):
        raise LossFnError("loss_tradeoff must have same length as properties!")

    def loss_fn(batch, result):
        loss = 0.0
        for prop, factor in zip(properties, loss_tradeoff):
            #diff = batch[prop] - torch.sum(batch['group_energy'], dim=1) - result[prop]
            if prop == 'group_energy':
                diff = batch[prop] - result[prop]
                diff = diff ** 2
                err_sq = factor * torch.mean(diff)
                loss += err_sq
            elif prop == 'total_rho':
                mesh_size = result[prop].shape[-1]
                norm = mesh_size**3
                diff = batch[prop][:,:norm] - result[prop].reshape(-1,norm)
                diff = diff **2
                err_sq = factor * torch.sum(diff)
                loss += err_sq
            elif prop == 'diff_U0_group':
                diff = batch[prop] - result[prop]
                diff = diff ** 2
                err_sq = factor * torch.mean(diff)
                loss += err_sq
        return loss, diff

    return loss_fn


def mse_loss(pred,target):
    diff = pred-target
    diff = diff ** 2
    loss = torch.mean(diff)
    return loss,diff

def build_mse_loss_with_forces(energy_weight,force_weight,with_forces):
    """
    Build the mean squared error loss function.

    Args:
        loss_tradeoff (list or None): multiply loss value of property with tradeoff
            factor

    Returns:
        mean squared error loss function

    """

    def loss_with_forces(data, result,mean,std):
        # compute the mean squared error on the energies
        diff_energy = ((data["energy"]-mean)/std-result["energy"]).reshape(-1,1)
        err_sq_energy = torch.mean(diff_energy ** 2)
        # compute the mean squared error on the forces
        diff_forces = data["forces"]/std-result["forces"]
        err_sq_forces = torch.mean(diff_forces ** 2)
        # print(data["energy"],result["energy"])
        # print(err_sq_energy.item(),err_sq_forces.item())
        # build the combined loss function
        consistency_loss = result["consistency_loss"] if "consistency_loss" in result else 0.0
        err_sq = energy_weight*err_sq_energy + force_weight*err_sq_forces + 0.01*consistency_loss

        return err_sq
    
    def loss_for_energy(data, result,mean,std):
        # compute the mean squared error on the energies
        diff_energy = ((data["energy"]-mean)/std-result["energy"]).reshape(-1,1)
        err_sq_energy = torch.mean(diff_energy ** 2)

        return err_sq_energy
    if with_forces:
        return loss_with_forces
    else:
        return loss_for_energy



def build_l2maeloss(energy_weight,force_weight,with_forces):
    def loss_with_forces(data, result,mean,std, atom_ref=None):
        # should also minus atom_ref
        if atom_ref is not None:
            ref_energy = scatter(atom_ref[data.atomic_numbers.long()], data.batch, dim=0, reduce='sum')
            diff_energy = ((data["energy"] - mean - ref_energy)/std-result["energy"]).reshape(-1,1)
        else:
            diff_energy = ((data["energy"]-mean)/std-result["energy"]).reshape(-1,1)
        diff_energy = torch.norm(diff_energy, p=2, dim=-1)
        diff_energy = torch.mean(diff_energy)
        
        diff_forces = data["forces"]/std-result["forces"]
        diff_forces = torch.norm(diff_forces, p=2, dim=-1)
        diff_forces = torch.mean(diff_forces)
        # print(data["energy"],result["energy"])
        # print(err_sq_energy.item(),err_sq_forces.item())
        # build the combined loss function
        return energy_weight*diff_energy + force_weight*diff_forces 
    
    def loss_for_energy(data, result,mean,std, atom_ref=None):
        # compute the mean squared error on the energies
        diff_energy = ((data["energy"]-mean)/std-result["energy"]).reshape(-1,1)
        diff_energy = torch.norm(diff_energy, p=2, dim=-1)
        diff_energy = torch.mean(diff_energy)
        return diff_energy
    
    if with_forces:
        return loss_with_forces
    else:
        return loss_for_energy
