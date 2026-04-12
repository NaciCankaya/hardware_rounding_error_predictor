# Installing flash-attn from source: hard-won lessons

## The problem
flash-attn has no pre-built wheels for recent PyTorch. You must build from source.

## What goes wrong if you're naive
1. `TORCH_CUDA_ARCH_LIST` is **IGNORED**. flash-attn uses `FLASH_ATTN_CUDA_ARCHS` (line ~70 in setup.py).
2. Default builds ALL architectures (sm_80, sm_90, sm_100, sm_120) × ALL head dims × fwd + bwd + split = ~300 nvcc invocations. Takes forever, eats disk.
3. `pip install flash-attn --no-build-isolation` with a `timeout` WILL kill the build mid-compile, wasting everything. **Never use timeouts.**
4. Stripping .cu files from the sources list causes **undefined symbol** errors at import if you miss any template instantiation that `flash_api.cpp` references.
5. Nested Python-in-bash-in-Python causes escaping nightmares with backslashes in C macros. **Write patches to separate .py files**, don't inline them.

## What you CANNOT strip
- **Head dims**: `flash_api.cpp` instantiates ALL head dims (32, 64, 96, 128, 192, 256). Missing any → undefined symbol at import.
- **fp16 if you only patch sources**: `FP16_SWITCH` macro instantiates BOTH `half_t` and `bfloat16_t` at compile time. Removing fp16 .cu files → undefined symbol.

## What you CAN strip (with corresponding patches)
- **fp16 kernels** — but you MUST also patch `FP16_SWITCH` in `static_switch.h`
- **bwd kernels** — but you MUST also stub `run_mha_bwd` in `flash_api.cpp`
- **split kernels** — risky, some API paths may call them

## Correct procedure

### 1. Clone source
```bash
cd /tmp
git clone --depth 1 --branch v<VERSION> https://github.com/Dao-AILab/flash-attention.git fa2_src_build
```

### 2. Clean before every build attempt
```bash
cd /tmp/fa2_src_build
rm -rf build *.so
rm -rf /tmp/tmpxft_* /tmp/pip-ephem-wheel-cache-*
```

### 3. Patch (three independent patches, apply any combination)

**Patch A: Force single architecture** (always do this)
```bash
sed -i 's|"80;90;100;120"|"80"|' setup.py
```

**Patch B: Remove backward kernels** (saves ~50% compile time)

Remove bwd .cu lines from setup.py:
```python
with open('setup.py') as f:
    lines = f.readlines()
with open('setup.py', 'w') as f:
    for line in lines:
        if 'flash_bwd' in line:
            continue
        f.write(line)
```

Stub `run_mha_bwd` in `flash_api.cpp`:
```python
with open('csrc/flash_attn/flash_api.cpp') as f:
    content = f.read()
old = '''void run_mha_bwd(Flash_bwd_params &params, cudaStream_t stream) {
    FP16_SWITCH(!params.is_bf16, [&] {
        HEADDIM_SWITCH(params.d, [&] {
            BOOL_SWITCH(params.is_causal, Is_causal, [&] {
                run_mha_bwd_<elem_type, kHeadDim, Is_causal>(params, stream);
            });
        });
    });
}'''
new = '''void run_mha_bwd(Flash_bwd_params &params, cudaStream_t stream) {
    TORCH_CHECK(false, "Backward not compiled");
}'''
content = content.replace(old, new)
with open('csrc/flash_attn/flash_api.cpp', 'w') as f:
    f.write(content)
```

**Patch C: Remove fp16 kernels** (saves another ~50%)

Remove fp16 .cu lines from setup.py:
```python
with open('setup.py') as f:
    lines = f.readlines()
with open('setup.py', 'w') as f:
    for line in lines:
        if '_fp16_' in line and '.cu' in line:
            continue
        f.write(line)
```

Patch `FP16_SWITCH` in `static_switch.h` — **write to a .py file to avoid escaping issues**:

Save this as `/tmp/patch_fp16.py` and run it:
```python
import os
os.chdir('/tmp/fa2_src_build')
with open('csrc/flash_attn/src/static_switch.h') as f:
    lines = f.readlines()
out = []
i = 0
while i < len(lines):
    if '#define FP16_SWITCH' in lines[i]:
        out.append('#define FP16_SWITCH(COND, ...)               \\\n')
        out.append('  [&] {                                      \\\n')
        out.append('      using elem_type = cutlass::bfloat16_t; \\\n')
        out.append('      return __VA_ARGS__();                  \\\n')
        out.append('  }()\n')
        while i < len(lines) and '}()' not in lines[i]:
            i += 1
        i += 1
    else:
        out.append(lines[i])
        i += 1
with open('csrc/flash_attn/src/static_switch.h', 'w') as f:
    f.writelines(out)
print('Patched FP16_SWITCH')
```

### 4. Verify before building
```bash
cd /tmp/fa2_src_build
echo "Arch:"
grep FLASH_ATTN_CUDA_ARCHS setup.py
echo "bwd in sources: $(grep -c flash_bwd setup.py)"
echo "fp16 in sources: $(grep -c '_fp16_' setup.py)"
echo "Total .cu: $(grep -c '.cu' setup.py)"
echo "run_mha_bwd_ refs: $(grep -c 'run_mha_bwd_' csrc/flash_attn/flash_api.cpp)"
echo "half_t refs: $(grep -c half_t csrc/flash_attn/flash_api.cpp)"
grep -A4 'define FP16_SWITCH' csrc/flash_attn/src/static_switch.h
```

### 5. Build with Popen (NO timeout)
```python
import subprocess, os
env = os.environ.copy()
env['MAX_JOBS'] = '24'  # scale to available RAM
proc = subprocess.Popen(
    ["/venv/main/bin/pip", "install", "-e", "/tmp/fa2_src_build", "--no-build-isolation"],
    env=env
)
print(f"PID: {proc.pid}")
```

### 6. Monitor in separate cell
```python
import subprocess
r = subprocess.run(["bash", "-c", """
echo "nvcc: $(ps aux | grep nvcc | grep -v grep | wc -l)"
echo "ninja: $(ps aux | grep ninja | grep -v grep | wc -l)"
echo "objects: $(find /tmp/fa2_src_build/build -name '*.o' 2>/dev/null | wc -l)"
df -h / | tail -1
"""], capture_output=True, text=True)
print(r.stdout)
```

### 7. If pip dies but ninja/nvcc survive
Ninja may finish on its own. Once .so exists, install manually:
```bash
cd /tmp/fa2_src_build && pip install -e . --no-build-isolation
```

## Build size estimates (sm_80 only)
| Configuration | .cu files | Time (~24 jobs) |
|---|---|---|
| All patches (A+B+C) | ~24 | ~5 min |
| A+B only (keep fp16) | ~48 | ~10 min |
| A only (safest) | ~72 | ~15 min |
| No patches (default) | ~300 | hours |

## Key facts
- Each nvcc for flash-attn kernels takes 2-5 minutes and ~2GB RAM
- Disk: ~30MB per .o, ~120MB per .cu in /tmp nvcc intermediates. Watch df.
- `--threads 4` in nvcc flags means each .cu spawns sub-processes
- .o files only appear when nvcc FINISHES (not during)
- The source tree includes its own CUTLASS headers in csrc/cutlass/
- NEVER use `kill -9 -1` to clean up — it kills Jupyter
- Use `pkill -9 -f nvcc; pkill -9 -f ninja` for targeted cleanup
- If 0% CPU + 0 objects for several minutes, build is stuck — kill and retry
- Clean `/tmp/tmpxft_*` between attempts (stale nvcc intermediates eat disk)
- Always `rm -rf build *.so` before rebuilding after a patch
- pip `uninstall` deletes .o files from editable installs — don't uninstall if you want incremental builds
