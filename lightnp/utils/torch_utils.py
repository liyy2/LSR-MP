import torch
import numpy as np

def torchify_dict(data):
    """
    data = {
        name:np.array()
    }
    
    Transform np.ndarrays to torch.tensors.
    """
    torch_properties = {}
    for pname, prop in data.items():

        if prop.dtype in [np.int, np.int32, np.int64]:
            torch_properties[pname] = torch.LongTensor(prop.copy())
        elif prop.dtype in [np.float, np.float32, np.float64]:
            torch_properties[pname] = torch.FloatTensor(prop.copy())
        else:
            raise AtomsDataError(
                "Invalid datatype {} for property {}!".format(type(prop), pname)
            )
    return torch_properties


def numpyfy_dict(data):
    """
    Transform floats, ints and dimensionless numpy in a dict to arrays to numpy arrays with dimenison.

    """
    for k, v in data.items():
        if type(v) in [int, float]:
            v = np.array([v])
        if v.shape == ():
            v = v[np.newaxis]
        data[k] = v
    return data