
import torch.distributed as dist
import torch



def reduce_cat(tensor,world_size):
    ############## Engergy Colletive Function ##############
    # preds_list = [torch.zeros_like(energy).to(self.device) for _ in range(self.world_size)] # [[B, ...], [B, ...], ...]
    # out = torch.zeros([world_size]+list(tensor.shape),device = tensor.device,dtype = torch.float64)
    # dist.all_gather(out, tensor) reduce tensor with same shape

    out = [None for _ in range(world_size)]
    dist.all_gather_object(out, tensor)
    out = [p.to(tensor.device) for p in out]
    return torch.cat(out, dim=0).contiguous()

# reduce the information from children progress.
# for each children, the shape must be the same!!!!
def reduce_mean(tensor,world_size):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= world_size
    return rt

# reduce the information from children progress.
def reduce_sum(tensor, world_size=1):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    return rt