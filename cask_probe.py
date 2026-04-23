#!/usr/bin/env python3
"""
cask_probe.py — quick scan of libcublasLt's .cask_resource section to check
whether nvjet kernel cubins are embedded there.

Run on any pod that has libcublasLt.so installed:
    python3 cask_probe.py                      # auto-locates the library
    python3 cask_probe.py /path/to/libcublasLt.so.13.1.0.3

Reports:
  - section size
  - count of embedded ELF headers (and how many look like CUDA ELF e_machine=0xBE)
  - hits for 'nvjet_' and related name templates in the raw bytes
  - attempts to extract the first CUDA ELF and run cuobjdump on it
"""
import subprocess
import sys
import os
import glob


def find_libcublasLt():
    candidates = []
    for root in ["/usr/local/cuda/lib64",
                 "/usr/local/cuda/targets/x86_64-linux/lib",
                 "/usr/lib/x86_64-linux-gnu",
                 "/opt/conda/lib"]:
        candidates.extend(glob.glob(f"{root}/libcublasLt.so*"))
    for c in sorted(candidates, key=len, reverse=True):
        if not os.path.islink(c):
            return c
    return candidates[0] if candidates else None


def get_section_info(lib, section_name):
    out = subprocess.check_output(["readelf", "-WS", lib], text=True)
    for line in out.splitlines():
        if section_name not in line:
            continue
        parts = line.split()
        for i, p in enumerate(parts):
            if p == section_name:
                off = int(parts[i + 3], 16)
                size = int(parts[i + 4], 16)
                return off, size
    return None, None


def scan_elf_magic(data):
    pattern = b"\x7fELF"
    offsets = []
    off = 0
    while True:
        i = data.find(pattern, off)
        if i == -1:
            break
        if i + 20 < len(data):
            ei_class = data[i + 4]
            ei_data  = data[i + 5]
            ei_ver   = data[i + 6]
            if ei_class in (1, 2) and ei_data in (1, 2) and ei_ver == 1:
                offsets.append(i)
        off = i + 4
    return offsets


def elf_e_machine(data, off):
    if data[off + 4] == 2:  # 64-bit
        m_off = off + 18
    else:
        m_off = off + 18
    if m_off + 2 > len(data):
        return None
    return data[m_off] | (data[m_off + 1] << 8)


def elf_total_size(data, off):
    """Estimate ELF total size from header (ELF64)."""
    if data[off + 4] != 2:
        return None
    # ELF64: e_shoff at off+40, e_shnum at off+60, e_shentsize at off+58
    import struct
    try:
        e_shoff    = struct.unpack_from("<Q", data, off + 40)[0]
        e_shentsz  = struct.unpack_from("<H", data, off + 58)[0]
        e_shnum    = struct.unpack_from("<H", data, off + 60)[0]
        return e_shoff + e_shentsz * e_shnum
    except Exception:
        return None


def scan_strings(data, patterns):
    results = {}
    for p in patterns:
        pb = p.encode()
        hits = []
        off = 0
        while True:
            i = data.find(pb, off)
            if i == -1:
                break
            start, end = max(0, i - 8), min(len(data), i + 96)
            ctx = data[start:end]
            printable = "".join(chr(b) if 32 <= b < 127 else "." for b in ctx)
            hits.append((i, printable))
            off = i + len(pb)
        results[p] = hits
    return results


def main():
    lib = sys.argv[1] if len(sys.argv) > 1 else find_libcublasLt()
    if not lib:
        print("ERROR: libcublasLt.so not found. Pass path as arg.")
        sys.exit(1)
    print(f"Library: {lib}")
    print(f"Size:    {os.path.getsize(lib):,} bytes")

    print("\n=== Section table (cask / fatbin / large rodata) ===")
    out = subprocess.check_output(["readelf", "-WS", lib], text=True)
    for line in out.splitlines():
        ll = line.lower()
        if any(t in ll for t in ["cask", "nv_fatbin", ".rodata"]):
            print(line.rstrip())

    off, size = get_section_info(lib, ".cask_resource")
    if off is None:
        print("No .cask_resource section found — stop.")
        return
    print(f"\n.cask_resource: file offset=0x{off:x}, size={size:,} bytes "
          f"({size/1e6:.1f} MB)")

    out_path = "/tmp/cask_resource.bin"
    print(f"Extracting to {out_path} ...")
    with open(lib, "rb") as f:
        f.seek(off)
        data = f.read(size)
    with open(out_path, "wb") as f:
        f.write(data)
    print(f"Wrote {len(data):,} bytes")

    print("\n=== ELF magic scan inside .cask_resource ===")
    elf_offsets = scan_elf_magic(data)
    cuda_elves = []
    for o in elf_offsets:
        m = elf_e_machine(data, o)
        if m == 0xbe:
            cuda_elves.append(o)
    print(f"Total ELF headers:       {len(elf_offsets)}")
    print(f"CUDA ELF (e_machine=BE): {len(cuda_elves)}")
    for i, o in enumerate(cuda_elves[:10]):
        tot = elf_total_size(data, o)
        print(f"  [{i}] offset=0x{o:x} (+{o:,} in section)  estimated_size={tot}")
    if len(cuda_elves) > 10:
        print(f"  ... and {len(cuda_elves) - 10} more")

    print("\n=== nvjet / CASK name scan ===")
    patterns = ["nvjet_", "nvjet_sm90", "nvjet_sm100", "nvjet_hsh",
                "CASK", "cask_", "sm_90a", "cublasLt"]
    results = scan_strings(data, patterns)
    for p, hits in results.items():
        print(f"\n  '{p}': {len(hits)} hits")
        for offset, ctx in hits[:3]:
            print(f"    0x{offset:x}: {ctx!r}")
        if len(hits) > 3:
            print(f"    ... and {len(hits) - 3} more")

    # If we found a CUDA ELF, try to extract it and disassemble
    if cuda_elves:
        first = cuda_elves[0]
        tot = elf_total_size(data, first)
        if tot and tot < 50_000_000:
            cubin_path = "/tmp/cask_cubin_0.cubin"
            with open(cubin_path, "wb") as f:
                f.write(data[first:first + tot])
            print(f"\n=== First CUDA ELF extracted to {cubin_path} ({tot:,} B) ===")
            for cmd in (["cuobjdump", "--list-text", cubin_path],
                        ["cuobjdump", "--dump-elf-symbols", cubin_path]):
                print(f"\n$ {' '.join(cmd)}")
                try:
                    r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
                    head = (r.stdout or r.stderr).splitlines()[:30]
                    for ln in head:
                        print(f"  {ln}")
                    if r.returncode != 0:
                        print(f"  (exit {r.returncode})")
                except Exception as e:
                    print(f"  FAILED: {e}")

    print("\n=== Verdict heuristic ===")
    nvjet_hits = len(results.get("nvjet_", []))
    if cuda_elves and nvjet_hits > 0:
        print("LIKELY: .cask_resource contains CUDA cubins AND nvjet strings.")
        print("Recommend: extract all CUDA ELF spans, cuobjdump each, map to nvjet names.")
    elif cuda_elves:
        print("PARTIAL: CUDA ELFs present but no nvjet strings visible.")
        print("nvjet name templates may be stored elsewhere or format-generated.")
    elif nvjet_hits > 0:
        print("PARTIAL: nvjet strings present but no raw ELF headers.")
        print("Kernels likely stored in an opaque CASK blob format (compressed/encoded).")
    else:
        print("UNLIKELY: No CUDA ELFs and no nvjet strings in .cask_resource.")
        print("nvjet storage is somewhere else entirely.")


if __name__ == "__main__":
    main()
