import torch
import ctypes
from typing import Any, List, Dict
from abc import ABC, abstractmethod

from .check_env import check_env, print_env

class LazyGPUBase(ABC):
    def __init__(self, name, atol=1e-5, rtol=1e-5):
        self.name = name
        self.atol = atol
        self.rtol = rtol
        self.iters = 10

    @abstractmethod
    def reference_impl(self, **kwargs): 
        pass

    @abstractmethod
    def get_solve_signature(self) -> Dict[str, tuple]:
        pass
    
    @abstractmethod
    def generate_example_test(self) -> Dict[str, Any]:
        pass

    @abstractmethod
    def generate_functional_test(self) -> List[Dict[str, Any]]:
        pass

    @abstractmethod
    def generate_performance_test(self) -> Dict[str, Any]:
        pass

    def check_correctness(self, ref, out, verbose=False):
        # |ref - out| <= atol + rtol * |ref| 
        
        if not torch.allclose(ref, out, rtol=self.rtol, atol=self.atol):
            max_err = (ref - out).abs().max().item()
            print(f"[ERROR] {self.name}: correctness failed! max_err = {max_err}")
            return False

        if verbose:
            print(f"[OK] {self.name}: correctness passed")

        return True

    def check_env(self):
        ## check env 
        env = check_env()
        print_env(env)

    def verify_and_bench(self, solve_fn):
        print(f"🔥 Grilling LazyGPU: {self.name} by {solve_fn.__module__}.{solve_fn.__name__}")

        ## auto extract output keys from signature
        sig = self.get_solve_signature()
        out_keys = [k for k, v in sig.items() if v[1] == "out"]

        def run_test(test_case, label="Test"):
            ref_case = {k: v.clone() if k in out_keys else v for k, v in test_case.items()}
            
            self.reference_impl(**ref_case)
            
            solve_fn(**test_case)
            
            for k in out_keys:
                if not self.check_correctness(ref_case[k], test_case[k]):
                    print(f"❌ {label} failed on parameter: {k}")
                    return False
            return True
        
        ## Check Correctness (Example & Functional)
        print(">>> Running Example Test...")
        if not run_test(self.generate_example_test(), "Example Test"): 
            return
        print(f"✅ [PASS] {self.name}: All Example tests passed!")
        
        print(">>> Running Functional Tests...")
        for i, case in enumerate(self.generate_functional_test()):
            if not run_test(case, f"Functional Case {i}"): 
                return
        print(f"✅ [PASS] {self.name}: All functional tests passed!")

        ## Performance Test (Benchmark)
        print(f">>> Running Performance Test...")
        perf_test = self.generate_performance_test()
        
        # Warm up
        for _ in range(10):
            solve_fn(**perf_test)
        torch.cuda.synchronize()

        # loop benchmark
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        start_event.record()
        for _ in range(self.iters):
            solve_fn(**perf_test)
        end_event.record()
        
        torch.cuda.synchronize()
        avg_time_ms = start_event.elapsed_time(end_event) / self.iters
        print(f"🚀 [BENCH] {self.name}: Average execution time = {avg_time_ms:.3f} ms ({self.iters} iterations)")
        return avg_time_ms
    