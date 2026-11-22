# -*- coding: utf-8 -*-
"""H3-FP4 环境自检。用法: python check_env.py

检查项: Python / torch / CUDA 可用性 / GPU 架构 / _scaled_mm_v2 / cuda-python / NVRTC / 磁盘余量。
退出码 0 = 可以跑; 1 = 有阻断项。
"""
import importlib.util
import shutil
import sys


def main():
    fails = []

    print("python:", sys.version.split()[0])
    try:
        import torch
    except Exception as e:
        print("[FAIL] import torch:", e)
        return 1
    print("torch:", torch.__version__)
    print("scaled_mm_v2:", hasattr(torch, "_scaled_mm_v2"))

    if not torch.cuda.is_available():
        print("[FAIL] CUDA 不可用: 需要 CUDA 12.8 版 torch (>=2.11)，CPU 版无法运行本引擎")
        print("       注意: AMD ROCm / 魔搭 CPU Notebook 同样不可运行（FP4 是 NVIDIA Blackwell 专属）")
        fails.append("cuda")
    else:
        p = torch.cuda.get_device_properties(0)
        print("gpu:", p.name)
        print("compute capability:", "%d.%d" % (p.major, p.minor))
        print("vram: %.1f GiB" % (p.total_memory / 2 ** 30))
        if p.major < 10:
            print("[FAIL] 非 Blackwell (sm100/sm120): 无 FP4 张量核，本引擎不可用")
            fails.append("arch")
        if not hasattr(torch, "_scaled_mm_v2"):
            print("[FAIL] torch 缺 _scaled_mm_v2: 需要 torch>=2.11 cu128 (或更新)")
            fails.append("scaled_mm_v2")

    def _has(mod):
        # find_spec 在父包不存在时会抛 ModuleNotFoundError 而不是返回 None
        try:
            return importlib.util.find_spec(mod) is not None
        except Exception:
            return False

    for mod, hint in [("cuda.bindings.driver", "pip install cuda-python"),
                      ("nvidia.cuda_nvrtc", "pip install nvidia-cuda-nvrtc-cu12"),
                      ("nvidia.cuda_runtime", "pip install nvidia-cuda-runtime-cu12")]:
        if not _has(mod):
            print("[WARN] 缺 %s (%s): 会回退慢速 Python 量化, 实测视频单步 11.88s -> 32.55s (2.7x)。两包共约 50MB, 纯加法安装" % (mod, hint))
        else:
            print("%s: OK" % mod)

    free = shutil.disk_usage(".").free / 2 ** 30
    print("当前盘余量: %.1f GiB" % free)
    if free < 110:
        print("[WARN] 低于 110GiB: BF16 原件 66GB + FP4 37.5GB ≈ 104GB，单模态至少留 110GiB")

    if fails:
        print("==> 不可运行:", fails)
        return 1
    print("==> 环境检查通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
