import torch
import torch.nn as nn
from torch_geometric.nn import global_mean_pool, global_max_pool

from e3nn.o3 import Irreps
from torch_scatter import scatter


# From "Geometric and Physical Quantities improve E(3) Equivariant Message Passing"
class IrrepsScatter(nn.Module):
    '''Scatter for irreps
    ----------
    irreps : `Irreps`
        representation
    '''

    def __init__(self, irreps, reduce= 'mean'):
        super().__init__()

        self.irreps = Irreps(irreps)
        self.reduce = reduce


    def __repr__(self):
        return f"{self.__class__.__name__} ({self.irreps}"


    #@torch.autocast(device_type='cuda', enabled=False)
    def forward(self, node_input, labels, **kwargs):
        '''evaluate
        Parameters
        ----------
        node_input : `torch.Tensor`
            tensor of shape ``(batch, ..., irreps.dim)``
        Returns
        -------
        `torch.Tensor`
            tensor of shape ``(batch, ..., irreps.dim)``
        '''
        # batch, *size, dim = node_input.shape  # TODO: deal with batch
        # node_input = node_input.reshape(batch, -1, dim)  # [batch, sample, stacked features]
        # node_input has shape [batch * nodes, dim], but with variable nr of nodes.
        # the node_input batch slices this into separate graphs
        dim = node_input.shape[-1]

        fields = []
        ix = 0

        for mul, ir in self.irreps:  # mul is the multiplicity (number of copies) of some irrep type (ir)
            d = ir.dim
            #field = node_input[:, ix: ix + mul * d]  # [batch * sample, mul * repr]
            field = node_input.narrow(1, ix, mul*d)
            ix += mul * d

            # [batch * sample, mul, repr]
            field = field.reshape(-1, mul, d)
            
            group_field = scatter(field, labels, dim=0, reduce = 'mean')
            fields.append(group_field.reshape(-1, mul * d))  # [batch * sample, mul * repr]

        if ix != dim:
            fmt = "`ix` should have reached node_input.size(-1) ({}), but it ended at {}"
            msg = fmt.format(dim, ix)
            raise AssertionError(msg)

        output = torch.cat(fields, dim=-1)  # [batch * sample, stacked features]
        return output
    
if __name__ == '__main__':
    from e3nn.o3 import Irreps
    from torch_scatter import scatter
    from torch_geometric.data import Batch
    
    irreps = Irreps("128x0e + 128x1o")
    node_input = torch.randn(10, 128 * 4)
    labels = torch.randint(3, (10,))
    
    irr_scatter = IrrepsScatter(irreps)
    output = irr_scatter(node_input, labels)
    print(output.shape)
    
    
    