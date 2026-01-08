#!/usr/bin/env python3

import subprocess
import shutil
import re
import colorama


def check_env():

    env = {}

    # --- NVIDIA GPU 和驱动 ---
    try:
        # 1. 运行 nvidia-smi (完整命令) 来获取驱动版本
        driver_result = subprocess.run(
            ["nvidia-smi"],
            capture_output=True, text=True, timeout=5
        )
        
        if driver_result.returncode == 0:
            # 解析 "Driver Version: 535.104.05" 这样的字符串
            match = re.search(r"Driver Version:\s*([\d\.]+)", driver_result.stdout)
            if match:
                env["cuda_driver"] = match.group(1)
            else:
                env["cuda_driver"] = "Error (Can't parse driver from nvidia-smi)"
        else:
            error_msg = driver_result.stderr.strip() or f"exit code {driver_result.returncode}"
            env["cuda_driver"] = f"Error ({error_msg})"

        
        # 2. 运行 nvidia-smi 获取 GPU 型号
        gpu_result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5
        )
        
        if gpu_result.returncode == 0:
            env["gpu_device"] = gpu_result.stdout.strip().split('\n')[0]
        else:
            error_msg = gpu_result.stderr.strip() or f"exit code {gpu_result.returncode}"
            env["gpu_device"] = f"Error ({error_msg})"
            
    except FileNotFoundError:
        env["cuda_driver"] = "Not Found (nvidia-smi)"
        env["gpu_device"] = "Not Found (nvidia-smi)"
    except subprocess.TimeoutExpired:
        env["cuda_driver"] = "Error (nvidia-smi timed out)"
        env["gpu_device"] = "Error (nvidia-smi timed out)"
    except Exception as e:
        env["cuda_driver"] = f"Error (Python: {e})"
        env["gpu_device"] = f"Error (Python: {e})"

    # --- NVCC 编译器 ---
    nvcc_path = shutil.which("nvcc")
    if nvcc_path:
        env["nvcc_path"] = nvcc_path
        try:
            # 运行 nvcc --version
            nvcc_result = subprocess.run(
                [nvcc_path, "--version"],
                capture_output=True, text=True, timeout=5, check=True
            )
            # 解析版本号, e.g., "release 12.2, V12.2.140"
            match = re.search(r"release (\d+\.\d+)", nvcc_result.stdout)
            if match:
                env["nvcc_version"] = match.group(1)
            else:
                env["nvcc_version"] = "Installed (Parsing failed)"
        except Exception as e:
            env["nvcc_version"] = f"Error ({e})"
    else:
        env["nvcc_path"] = "Not Found in PATH"
        env["nvcc_version"] = "Not Found"

    # --- Python 库 ---

    # PyTorch
    try:
        import torch
        env["pytorch_version"] = torch.__version__
        env["pytorch_cuda"] = torch.cuda.is_available()
    except ImportError:
        env["pytorch_version"] = "Not Installed"
        env["pytorch_cuda"] = False
    except Exception as e:
        env["pytorch_version"] = f"Error ({e})"
        env["pytorch_cuda"] = False

    # Triton
    try:
        import triton
        env["triton_version"] = triton.__version__
    except ImportError:
        env["triton_version"] = "Not Installed"
    except AttributeError:
        env["triton_version"] = "Installed (Version unknown)"
    except Exception as e:
        env["triton_version"] = f"Error ({e})"

    # CUTE DSL (via cutlass package)
    try:
        import cutlass
        try:
            env["cute_dsl_cutlass"] = cutlass.__version__
        except AttributeError:
            env["cute_dsl_cutlass"] = "Installed (Version unknown)"
    except ImportError:
        env["cute_dsl_cutlass"] = "Not Installed"
    except Exception as e:
        env["cute_dsl_cutlass"] = f"Error ({e})"
        
    # Tile-lang
    try:
        import tilelang
        try:
            env["tilelang_version"] = tilelang.__version__
        except AttributeError:
            env["tilelang_version"] = "Installed (Version unknown)"
    except ImportError:
        env["tilelang_version"] = "Not Installed"
    except Exception as e:
        env["tilelang_version"] = f"Error ({e})"

    # # CUDA extension placeholder
    # try:
    #     import cuda_ext  # 你自己的 CUDA kernel module
    #     try:
    #         env["cuda_ext_version"] = cuda_ext.__version__
    #     except AttributeError:
    #         env["cuda_ext_version"] = "Installed (Version unknown)"
    # except ImportError:
    #     env["cuda_ext_version"] = "Not Installed"
    # except Exception as e:
    #     env["cuda_ext_version"] = f"Error ({e})"

    return env


def print_env(env: dict):

    ## output define
    colorama.init(autoreset=True)
    CHECK = colorama.Fore.GREEN + '✔'
    CROSS = colorama.Fore.RED + '✘'

    print(colorama.Style.BRIGHT + "=" * 50 + " Environment Check " + "=" * 50)
    
    key_width = max(len(k) for k in env.keys()) if env else 20
    
    for k, v in env.items():
        status_icon = ""
        status_text = ""
        
        if isinstance(v, bool):
            if v:
                status_icon = CHECK
                status_text = "Available"
            else:
                status_icon = CROSS
                status_text = "Not Available"
        else:
            v_str = str(v)
            if "Not Found" in v_str or "Not Installed" in v_str or "Error" in v_str:
                status_icon = CROSS
                status_text = v_str
            else:
                status_icon = CHECK
                status_text = v_str
                
        print(f"{k:<{key_width}} : {status_icon} {status_text}")
        
    print(colorama.Style.BRIGHT + "=" * 119 + "\n")


if __name__ == "__main__":
    env = check_env()
    print_env(env)