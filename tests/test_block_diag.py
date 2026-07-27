import torch

def block_diag(*args):
    assert len(args) > 0, "At least one tensor must be provided."
    
    assert all(tensor.dim() == 2 for tensor in args), "All tensors must be 2D."
    
    sizes = [tensor.shape for tensor in args]
    
    rows = sum(size[0] for size in sizes)
    cols = sum(size[1] for size in sizes)
    
    result = torch.zeros(rows, cols)
    
    row_start = 0
    col_start = 0
    for tensor in args:
        row_end = row_start + tensor.shape[0]
        col_end = col_start + tensor.shape[1]
        result[row_start:row_end, col_start:col_end] = tensor
        row_start = row_end
        col_start = col_end
        
    return result
