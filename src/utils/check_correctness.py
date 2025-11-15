import torch

# |ref - out| <= atol + rtol * |ref| 
def check_correctness(ref, out, rtol=1e-3, atol=1e-3, name="kernel"):
    """
    Compare two tensors.
    """
    if not torch.allclose(ref, out, rtol=rtol, atol=atol):
        max_err = (ref - out).abs().max().item()
        print(f"[ERROR] {name}: correctness failed! max_err = {max_err}")
        return False

    print(f"[OK] {name}: correctness passed")
    return True
