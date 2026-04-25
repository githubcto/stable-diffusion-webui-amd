import os
import sys
import ctypes
import ctypes.wintypes
import contextlib
from pathlib import Path

# ----------------------------------------
# try import psutil torch
# ----------------------------------------
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# ----------------------------------------
# RDNA / APU mapping
# ----------------------------------------
RDNA_MAP = {
    "gfx900": "GCN5.0 Vega10",
    "gfx906": "GCN5.1 Vega20",
    "gfx1010": "RDNA1 Navi10",
    "gfx1011": "RDNA1 Navi12",
    "gfx1012": "RDNA1 Navi14",
    "gfx1030": "RDNA2 Navi21",
    "gfx1031": "RDNA2 Navi22",
    "gfx1032": "RDNA2 Navi23",
    "gfx1033": "RDNA2 VanGogh",
    "gfx1034": "RDNA2 Navi24",
    "gfx1035": "RDNA2 Rembrandt",
    "gfx1036": "RDNA2 Raphael",
    "gfx1037": "RDNA2 Mendocino",
    "gfx1100": "RDNA3 Navi31",
    "gfx1101": "RDNA3 Navi32",
    "gfx1102": "RDNA3 Navi33",
    "gfx1103": "RDNA3 Phoenix",
    "gfx1150": "RDNA3.5 Strix Point",
    "gfx1151": "RDNA3.5 Strix Halo",
    "gfx1152": "RDNA3.5 Krackan Point",
    "gfx1153": "RDNA3.5 Medusa Point",
    "gfx1200": "RDNA4 Navi44",
    "gfx1201": "RDNA4 Navi48"
}

# APUs
APU_ARCH = {
    "gfx1033",  # VanGogh
    "gfx1035",  # Rembrandt
    "gfx1036",  # Raphael
    "gfx1037",  # Mendocino
    "gfx1103",  # Phoenix
    "gfx1150",  # Strix Point
    "gfx1151",  # Strix Halo
    "gfx1152",  # Krackan Point
    "gfx1153",  # Medusa Point
}

def get_rdna_name(arch):
    return RDNA_MAP.get(arch, "Unknown")

# ----------------------------------------
# HIP ctypes Windows AMD
# Thank you SD.Next rocm.py
# ----------------------------------------
hipDeviceProp = ctypes.c_byte * 1472

@contextlib.contextmanager
def mute(fd):
    saved = os.dup(fd)
    try:
        with open(os.devnull, 'w') as devnull:
            os.dup2(devnull.fileno(), fd)
            yield
    finally:
        os.dup2(saved, fd)
        os.close(saved)

class HIP:
    def __init__(self):
        self.handle = None

        if sys.platform != "win32":
            return

        windir = os.environ.get("windir", r"C:\Windows")
        candidates = [
            os.path.join(windir, "System32", "amdhip64_7.dll"),
            os.path.join(windir, "System32", "amdhip64_6.dll"),
            os.path.join(windir, "System32", "amdhip64.dll"),
        ]
        path = next((c for c in candidates if os.path.isfile(c)), None)
        if not path:
            return

        ctypes.windll.kernel32.LoadLibraryA.restype = ctypes.wintypes.HMODULE
        ctypes.windll.kernel32.LoadLibraryA.argtypes = [ctypes.c_char_p]
        self.handle = ctypes.windll.kernel32.LoadLibraryA(path.encode("utf-8"))
        if not self.handle:
            return

        ctypes.windll.kernel32.GetProcAddress.restype = ctypes.c_void_p
        ctypes.windll.kernel32.GetProcAddress.argtypes = [
            ctypes.wintypes.HMODULE,
            ctypes.c_char_p,
        ]

        def get_func(name, restype, *argtypes):
            addr = ctypes.windll.kernel32.GetProcAddress(
                self.handle, name.encode("ascii")
            )
            if not addr:
                return None
            return ctypes.CFUNCTYPE(restype, *argtypes)(addr)

        self.hipInit = get_func("hipInit", ctypes.c_int, ctypes.c_uint)
        self.hipGetDeviceCount = get_func(
            "hipGetDeviceCount", ctypes.c_int, ctypes.POINTER(ctypes.c_int)
        )
        self.hipGetDeviceProperties = get_func(
            "hipGetDeviceProperties",
            ctypes.c_int,
            ctypes.POINTER(hipDeviceProp),
            ctypes.c_int,
        )

        try:
            if self.hipInit:
                with mute(sys.stdout.fileno()):
                    self.hipInit(0)
        except Exception:
            pass

    def get_device_count(self):
        if not self.hipGetDeviceCount:
            return 0
        c = ctypes.c_int()
        if self.hipGetDeviceCount(ctypes.byref(c)) != 0:
            return 0
        return c.value

    def get_device_properties(self, device_id):
        if not self.hipGetDeviceProperties:
            return None
        prop = hipDeviceProp()
        if self.hipGetDeviceProperties(ctypes.byref(prop), device_id) != 0:
            return None
        return bytes(prop)

def extract_gfx_from_props(prop_bytes):
    idx = 0
    while idx < len(prop_bytes):
        try:
            idx = prop_bytes.index(0x67, idx) + 1  # 'g'
        except ValueError:
            break
        if prop_bytes[idx] != 0x66: continue  # 'f'
        if prop_bytes[idx + 1] != 0x78: continue  # 'x'
            
        idx += 2
        name = ""
        while idx < len(prop_bytes) and prop_bytes[idx] != 0x00:
            c = prop_bytes[idx]
            idx += 1
            if (0x30 <= c <= 0x39) or (0x61 <= c <= 0x66):
                name += chr(c)
            else:
                name = ""
        if name:
            return "gfx" + name
    return None

# ----------------------------------------
# HIP_VISIBLE_DEVICES
# ----------------------------------------
def setup_hip_devices_and_get_arch():
    if sys.platform != "win32":
        return None

    hip = HIP()
    count = hip.get_device_count()

    if count == 0:
        print("No AMD ROCm HIP devices detected.")
        return None

    devices = []
    for i in range(count):
        prop = hip.get_device_properties(i)
        arch = extract_gfx_from_props(prop) if prop else None
        devices.append((i, arch))

    print("Detected AMD ROCm HIP devices:")
    for idx, arch in devices:
        print(f"  [{idx}] {arch or 'Unknown'}")

    # check HIP_VISIBLE_DEVICES
    if "HIP_VISIBLE_DEVICES" in os.environ:
        env_val = os.environ["HIP_VISIBLE_DEVICES"]
        print(f"HIP_VISIBLE_DEVICES already set to '{env_val}' - skip auto-detection")
        try:
            # first one if "0,1"
            first_idx = int(env_val.split(',')[0].strip())
            if 0 <= first_idx < count:
                selected_arch = devices[first_idx][1]
                print(f"Using pre-configured device [{first_idx}] ({selected_arch or 'Unknown'})")
                return selected_arch
            else:
                print(f"Warning: Index {first_idx} is out of range. Falling back to device [0].")
        except ValueError:
            print("Warning: Invalid format in HIP_VISIBLE_DEVICES. Falling back to device [0].")
            pass
        return devices[0][1] if devices else None

    # prefer dGPU
    selected = 0
    for idx, arch in devices:
        if arch and arch not in APU_ARCH:
            selected = idx
            break

    os.environ["HIP_VISIBLE_DEVICES"] = str(selected)
    print(f"Set HIP_VISIBLE_DEVICES={selected} (Auto-selected dGPU)")

    return devices[selected][1]

# ----------------------------------------
# Torch / mem
# ----------------------------------------
def get_torch_device():
    if not HAS_TORCH:
        return None
    if torch.cuda.is_available(): return torch.device("cuda", torch.cuda.current_device())
    try:
        if torch.xpu.is_available(): return torch.device("xpu", 0)
    except Exception:
        pass
    return torch.device("cpu")

def get_total_vram(device):
    if not HAS_TORCH or device is None: return 0
    try:
        if device.type == "cuda":
            _, total = torch.cuda.mem_get_info(device)
            return total
        elif device.type == "xpu":
            props = torch.xpu.get_device_properties(device)
            return props.total_memory
    except Exception:
        pass
    return 0

def get_total_ram():
    if not HAS_PSUTIL: return 0
    return psutil.virtual_memory().total

# ----------------------------------------
# OS env
# ----------------------------------------
#TORCH_CMD_MAP = {
#    "gfx1200": "pip install torch==2.10.0 torchvision==0.25.0 rocm[devel] numpy==1.26.2 --extra-index-url https://rocm.nightlies.amd.com/v2/gfx120X-all",
#    "gfx1201": "pip install torch==2.10.0 torchvision==0.25.0 rocm[devel] numpy==1.26.2 --extra-index-url https://rocm.nightlies.amd.com/v2/gfx120X-all",
#    "gfx1050": "pip install torch==2.10.0 torchvision==0.25.0 rocm[devel] numpy==1.26.2 --extra-index-url https://rocm.nightlies.amd.com/v2-staging/gfx1150/",
#    "gfx1051": "pip install torch==2.10.0 torchvision==0.25.0 rocm[devel] numpy==1.26.2 --extra-index-url https://rocm.nightlies.amd.com/v2/gfx1151/",
#    "gfx1052": "pip install torch==2.10.0 torchvision==0.25.0 rocm[devel] numpy==1.26.2 --extra-index-url https://rocm.nightlies.amd.com/v2-staging/gfx1152/",
#    "gfx1053": "pip install torch==2.10.0 torchvision==0.25.0 rocm[devel] numpy==1.26.2 --extra-index-url https://rocm.nightlies.amd.com/v2-staging/gfx1153/",
#    "gfx1100": "pip install torch==2.10.0 torchvision==0.25.0 rocm[devel] numpy==1.26.2 --extra-index-url https://rocm.nightlies.amd.com/v2/gfx110X-all",
#    "gfx1101": "pip install torch==2.10.0 torchvision==0.25.0 rocm[devel] numpy==1.26.2 --extra-index-url https://rocm.nightlies.amd.com/v2/gfx110X-all",
#    "gfx1102": "pip install torch==2.10.0 torchvision==0.25.0 rocm[devel] numpy==1.26.2 --extra-index-url https://rocm.nightlies.amd.com/v2/gfx110X-all",
#    "gfx1103": "pip install torch==2.10.0 torchvision==0.25.0 rocm[devel] numpy==1.26.2 --extra-index-url https://rocm.nightlies.amd.com/v2/gfx110X-all",
#    "gfx900": "pip install torch torchvision rocm[devel] numpy==1.26.2 --extra-index-url https://rocm.nightlies.amd.com/v2-staging/gfx900/",
#    "gfx906": "pip install torch torchvision rocm[devel] numpy==1.26.2 --extra-index-url https://rocm.nightlies.amd.com/v2-staging/gfx906/",
#    "gfx1030": "pip install torch torchvision rocm[devel] numpy==1.26.2 --extra-index-url https://rocm.nightlies.amd.com/v2-staging/gfx103X-all/",
#    "gfx1031": "pip install torch torchvision rocm[devel] numpy==1.26.2 --extra-index-url https://rocm.nightlies.amd.com/v2-staging/gfx103X-all/",
#    "gfx1032": "pip install torch torchvision rocm[devel] numpy==1.26.2 --extra-index-url https://rocm.nightlies.amd.com/v2-staging/gfx103X-all/",
#    "gfx1033": "pip install torch torchvision rocm[devel] numpy==1.26.2 --extra-index-url https://rocm.nightlies.amd.com/v2-staging/gfx103X-all/",
#    "gfx1034": "pip install torch torchvision rocm[devel] numpy==1.26.2 --extra-index-url https://rocm.nightlies.amd.com/v2-staging/gfx103X-all/",
#    "gfx1035": "pip install torch torchvision rocm[devel] numpy==1.26.2 --extra-index-url https://rocm.nightlies.amd.com/v2-staging/gfx103X-all/",
#    "gfx1036": "pip install torch torchvision rocm[devel] numpy==1.26.2 --extra-index-url https://rocm.nightlies.amd.com/v2-staging/gfx103X-all/",
#    "gfx1010": "pip install torch==2.10.0 torchvision==0.25.0 rocm[devel] numpy==1.26.2 --extra-index-url https://rocm.nightlies.amd.com/v2-staging/gfx101X-dgpu/",
#    "gfx1011": "pip install torch==2.10.0 torchvision==0.25.0 rocm[devel] numpy==1.26.2 --extra-index-url https://rocm.nightlies.amd.com/v2-staging/gfx101X-dgpu/",
#    "gfx1012": "pip install torch==2.10.0 torchvision==0.25.0 rocm[devel] numpy==1.26.2 --extra-index-url https://rocm.nightlies.amd.com/v2-staging/gfx101X-dgpu/"
#}
# TheRock has issue "Unable to load file: gfx908_metadata.tn.model" since 20260221.
# Use 20260220.
TORCH_CMD_MAP = {
    "gfx1200": "pip install torch==2.10.0+rocm7.12.0a20260220 torchvision==0.25.0+rocm7.12.0a20260220 rocm[devel] numpy==1.26.2 --extra-index-url https://rocm.nightlies.amd.com/v2/gfx120X-all",
    "gfx1201": "pip install torch==2.10.0+rocm7.12.0a20260220 torchvision==0.25.0+rocm7.12.0a20260220 rocm[devel] numpy==1.26.2 --extra-index-url https://rocm.nightlies.amd.com/v2/gfx120X-all",
    "gfx1050": "pip install torch==2.10.0+rocm7.12.0a20260220 torchvision==0.25.0+rocm7.12.0a20260220 rocm[devel] numpy==1.26.2 --extra-index-url https://rocm.nightlies.amd.com/v2-staging/gfx1150/",
    "gfx1051": "pip install torch==2.10.0+rocm7.12.0a20260220 torchvision==0.25.0+rocm7.12.0a20260220 rocm[devel] numpy==1.26.2 --extra-index-url https://rocm.nightlies.amd.com/v2/gfx1151/",
    "gfx1052": "pip install torch==2.10.0+rocm7.12.0a20260220 torchvision==0.25.0+rocm7.12.0a20260220 rocm[devel] numpy==1.26.2 --extra-index-url https://rocm.nightlies.amd.com/v2-staging/gfx1152/",
    "gfx1053": "pip install torch==2.10.0+rocm7.12.0a20260220 torchvision==0.25.0+rocm7.12.0a20260220 rocm[devel] numpy==1.26.2 --extra-index-url https://rocm.nightlies.amd.com/v2-staging/gfx1153/",
    "gfx1100": "pip install torch==2.9.1 torchvision==0.24.1 rocm[devel] numpy==1.26.2 --extra-index-url https://rocm.nightlies.amd.com/v2/gfx110X-all",
    "gfx1101": "pip install torch==2.9.1 torchvision==0.24.1 rocm[devel] numpy==1.26.2 --extra-index-url https://rocm.nightlies.amd.com/v2/gfx110X-all",
    "gfx1102": "pip install torch==2.9.1 torchvision==0.24.1 rocm[devel] numpy==1.26.2 --extra-index-url https://rocm.nightlies.amd.com/v2/gfx110X-all",
    "gfx1103": "pip install torch==2.9.1 torchvision==0.24.1 rocm[devel] numpy==1.26.2 --extra-index-url https://rocm.nightlies.amd.com/v2/gfx110X-all",
    "gfx900": "pip install torch torchvision rocm[devel] numpy==1.26.2 --extra-index-url https://rocm.nightlies.amd.com/v2-staging/gfx900/",
    "gfx906": "pip install torch torchvision rocm[devel] numpy==1.26.2 --extra-index-url https://rocm.nightlies.amd.com/v2-staging/gfx906/",
    "gfx1030": "pip install torch torchvision rocm[devel] numpy==1.26.2 --extra-index-url https://rocm.nightlies.amd.com/v2-staging/gfx103X-all/",
    "gfx1031": "pip install torch torchvision rocm[devel] numpy==1.26.2 --extra-index-url https://rocm.nightlies.amd.com/v2-staging/gfx103X-all/",
    "gfx1032": "pip install torch torchvision rocm[devel] numpy==1.26.2 --extra-index-url https://rocm.nightlies.amd.com/v2-staging/gfx103X-all/",
    "gfx1033": "pip install torch torchvision rocm[devel] numpy==1.26.2 --extra-index-url https://rocm.nightlies.amd.com/v2-staging/gfx103X-all/",
    "gfx1034": "pip install torch torchvision rocm[devel] numpy==1.26.2 --extra-index-url https://rocm.nightlies.amd.com/v2-staging/gfx103X-all/",
    "gfx1035": "pip install torch torchvision rocm[devel] numpy==1.26.2 --extra-index-url https://rocm.nightlies.amd.com/v2-staging/gfx103X-all/",
    "gfx1036": "pip install torch torchvision rocm[devel] numpy==1.26.2 --extra-index-url https://rocm.nightlies.amd.com/v2-staging/gfx103X-all/",
    "gfx1010": "pip install torch==2.10.0+rocm7.12.0a20260220 torchvision==0.25.0+rocm7.12.0a20260220 rocm[devel] numpy==1.26.2 --extra-index-url https://rocm.nightlies.amd.com/v2-staging/gfx101X-dgpu/",
    "gfx1011": "pip install torch==2.10.0+rocm7.12.0a20260220 torchvision==0.25.0+rocm7.12.0a20260220 rocm[devel] numpy==1.26.2 --extra-index-url https://rocm.nightlies.amd.com/v2-staging/gfx101X-dgpu/",
    "gfx1012": "pip install torch==2.10.0+rocm7.12.0a20260220 torchvision==0.25.0+rocm7.12.0a20260220 rocm[devel] numpy==1.26.2 --extra-index-url https://rocm.nightlies.amd.com/v2-staging/gfx101X-dgpu/"
}

def a1111rocm_init():
    print("--- Environment Check ---")
    
    #  geet gfx
    arch = setup_hip_devices_and_get_arch()
    rdna_name = get_rdna_name(arch) if arch else "Unknown"
    
    # OS env
    if arch and arch != "Unknown":
        if sys.platform == "win32":
            if "REQS_FILE" not in os.environ:
                os.environ["REQS_FILE"] = "requirements_versions_ROCm_TheRock.txt"
            if "TORCH_COMMAND" not in os.environ and arch in TORCH_CMD_MAP:
                os.environ["TORCH_COMMAND"] = TORCH_CMD_MAP[arch]

            app_home_dir = Path(__file__).resolve().parent.parent
            os.environ.setdefault("MIOPEN_FIND_MODE", "FAST")
            os.environ.setdefault("MIOPEN_USER_DB_PATH", str(app_home_dir / ".miopen_db"))
            os.environ.setdefault("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "1")
            os.environ.setdefault("TRITON_CACHE_DIR", str(app_home_dir / ".triton_cache"))
            os.environ.setdefault("FLASH_ATTENTION_TRITON_AMD_ENABLE", "TRUE")

    #  print
    if arch and arch != "Unknown":
        print(f"AMD Arch: {arch} ({rdna_name})")
        
    if HAS_PSUTIL:
        total_ram = get_total_ram() / (1024 * 1024)
        print(f"Total  RAM: {total_ram:.0f} MB")
    
    if HAS_TORCH:
        device = get_torch_device()
        total_vram = get_total_vram(device) / (1024 * 1024)
        print(f"Total VRAM: {total_vram:.0f} MB")
        print(f"PyTorch Version: {torch.__version__}")
        
        try:
            # HIP version
            rocm_version = tuple(map(int, str(torch.version.hip).split("-")[0].split(".")[:2]))
            print(f"ROCm HIP Version: {rocm_version}")
        except Exception:
            pass
            
        if device.type == "cuda":
            name = torch.cuda.get_device_name(device)
            print(f"Device: {name} ({device})")
        else:
            print(f"Device: {device}")
    else:
        print("PyTorch: Not installed yet (Installation phase)")

    # print OS env
    if arch and arch != "Unknown":
        print("REQS_FILE:", os.environ.get("REQS_FILE"))
        print("TORCH_COMMAND:", os.environ.get("TORCH_COMMAND"))
        print("MIOPEN_FIND_MODE:", os.environ.get("MIOPEN_FIND_MODE"))
        print("MIOPEN_USER_DB_PATH:", os.environ.get("MIOPEN_USER_DB_PATH"))
        print("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL:", os.environ.get("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL"))
        print("TRITON_CACHE_DIR:", os.environ.get("TRITON_CACHE_DIR"))
        print("FLASH_ATTENTION_TRITON_AMD_ENABLE:", os.environ.get("FLASH_ATTENTION_TRITON_AMD_ENABLE"))

if __name__ == "__main__":
    a1111rocm_init()

