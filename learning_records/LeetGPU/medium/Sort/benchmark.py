import ctypes
from typing import Any, List, Dict
import torch

from LazyGPU import LazyGPUBase

class LazyGPUBench(LazyGPUBase):
    def __init__(self):
        super().__init__(
            name="Sorting",
            atol=1e-05,
            rtol=1e-05,
        )
        
    def reference_impl(self, data: torch.Tensor, N: int):
        assert data.shape == (N,)
        data.copy_(data.sort()[0])

    def get_solve_signature(self) -> Dict[str, tuple]:
        return {
            "data": (ctypes.POINTER(ctypes.c_float), "inout"),
            "N": (ctypes.c_int, "in"),
        }

    def generate_example_test(self) -> Dict[str, Any]:
        dtype = torch.float32
        data = torch.tensor([5.0, 2.0, 8.0, 1.0, 9.0, 4.0], device="cuda", dtype=dtype)
        return {
            "data": data,
            "N": 6,
        }

    def generate_functional_test(self) -> List[Dict[str, Any]]:
        dtype = torch.float32
        tests = []
        # already_sorted
        tests.append(
            {"data": torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], device="cuda", dtype=dtype), "N": 5}
        )
        # reverse_sorted
        tests.append(
            {"data": torch.tensor([5.0, 4.0, 3.0, 2.0, 1.0], device="cuda", dtype=dtype), "N": 5}
        )
        # all_same
        tests.append({"data": torch.tensor([5.0] * 10, device="cuda", dtype=dtype), "N": 10})
        # single_element
        tests.append({"data": torch.tensor([7.0], device="cuda", dtype=dtype), "N": 1})
        # power_of_two
        tests.append(
            {
                "data": torch.empty(1024, device="cuda", dtype=dtype).uniform_(-100.0, 100.0),
                "N": 1024,
            }
        )
        # non_power_of_two
        tests.append(
            {
                "data": torch.empty(1000, device="cuda", dtype=dtype).uniform_(-100.0, 100.0),
                "N": 1000,
            }
        )
        # large_array
        tests.append(
            {
                "data": torch.empty(32768, device="cuda", dtype=dtype).uniform_(-1000.0, 1000.0),
                "N": 32768,
            }
        )
        return tests

    def generate_performance_test(self) -> Dict[str, Any]:
        dtype = torch.float32
        N = 1000000
        data = torch.empty(N, device="cuda", dtype=dtype).uniform_(-1000.0, 1000.0)
        return {
            "data": data,
            "N": N,
        }
    

if __name__ == "__main__":
    benchmark = LazyGPUBench()
    benchmark.check_env()
    
    from sort_triton import solve as solve_triton
    
    benchmark.verify_and_bench(solve_fn=solve_triton)
    