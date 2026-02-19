"""
Hardware detection for EvalKit.

The recommendations engine needs to know what machine it's running on —
Apple Silicon vs NVIDIA vs CPU-only changes what advice is relevant.

We detect:
  - OS (macOS / Linux / Windows)
  - CPU architecture (arm64 = Apple Silicon, x86_64 = Intel/AMD)
  - Whether Apple Silicon Metal GPU is available
  - Whether NVIDIA CUDA is available
  - Total RAM

We don't error out if detection fails — we just return None for that field
and skip any rules that require that information.
"""

import platform
import subprocess
import shutil
from dataclasses import dataclass
from typing import Optional


@dataclass
class HardwareProfile:
    os: str                             # "macOS", "Linux", "Windows"
    arch: str                           # "arm64", "x86_64"
    cpu_name: Optional[str]             # "Apple M4", "Intel Core i9", etc.
    total_ram_gb: Optional[float]       # total system RAM in GB
    is_apple_silicon: bool              # True if arm64 macOS
    has_metal: bool                     # True if Apple Silicon (Metal = Apple's GPU API)
    has_cuda: bool                      # True if NVIDIA GPU with CUDA drivers found
    gpu_name: Optional[str]             # "Apple M4 GPU", "NVIDIA RTX 4090", etc.

    def summary(self) -> str:
        """One-line human readable summary."""
        parts = [self.cpu_name or self.arch]
        if self.has_metal:
            parts.append("Metal GPU")
        elif self.has_cuda:
            parts.append(f"CUDA ({self.gpu_name})")
        else:
            parts.append("CPU only")
        if self.total_ram_gb:
            parts.append(f"{self.total_ram_gb:.0f}GB RAM")
        return " · ".join(parts)


def detect_hardware() -> HardwareProfile:
    """
    Detect the current machine's hardware profile.
    Safe to call — never raises, returns None for fields it can't detect.
    """
    os_name = _detect_os()
    arch = platform.machine()           # "arm64" or "x86_64"
    is_apple_silicon = (os_name == "macOS" and arch == "arm64")

    return HardwareProfile(
        os=os_name,
        arch=arch,
        cpu_name=_detect_cpu_name(os_name, is_apple_silicon),
        total_ram_gb=_detect_ram_gb(os_name),
        is_apple_silicon=is_apple_silicon,
        has_metal=is_apple_silicon,     # Metal is available on all Apple Silicon Macs
        has_cuda=_detect_cuda(),
        gpu_name=_detect_gpu_name(os_name, is_apple_silicon),
    )


# ── Private helpers ────────────────────────────────────────────────────────────

def _detect_os() -> str:
    system = platform.system()
    if system == "Darwin":
        return "macOS"
    elif system == "Linux":
        return "Linux"
    elif system == "Windows":
        return "Windows"
    return system


def _detect_cpu_name(os_name: str, is_apple_silicon: bool) -> Optional[str]:
    try:
        if os_name == "macOS":
            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True, text=True, timeout=3
            )
            name = result.stdout.strip()
            # Apple Silicon reports "Apple M1" / "Apple M4" etc here
            return name if name else None

        elif os_name == "Linux":
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.startswith("model name"):
                        return line.split(":")[1].strip()

    except Exception:
        pass
    return None


def _detect_ram_gb(os_name: str) -> Optional[float]:
    try:
        if os_name == "macOS":
            result = subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                capture_output=True, text=True, timeout=3
            )
            bytes_val = int(result.stdout.strip())
            return round(bytes_val / (1024 ** 3), 1)

        elif os_name == "Linux":
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal"):
                        kb = int(line.split()[1])
                        return round(kb / (1024 ** 2), 1)

    except Exception:
        pass
    return None


def _detect_cuda() -> bool:
    """Check if nvidia-smi is available — rough proxy for CUDA availability."""
    return shutil.which("nvidia-smi") is not None


def _detect_gpu_name(os_name: str, is_apple_silicon: bool) -> Optional[str]:
    try:
        if is_apple_silicon:
            # Extract chip name from CPU brand string for GPU label
            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True, text=True, timeout=3
            )
            chip = result.stdout.strip()  # e.g. "Apple M4"
            return f"{chip} GPU (Metal)" if chip else "Apple Silicon GPU (Metal)"

        if _detect_cuda():
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=5
            )
            name = result.stdout.strip().split("\n")[0]
            return name if name else "NVIDIA GPU"

    except Exception:
        pass
    return None
