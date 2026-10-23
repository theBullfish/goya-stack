#!/usr/bin/env python3
"""List and extract the prebuilt TPC kernels inside Intel's libtpc_kernels.so.1.

Each kernel is a standalone TPC ELF embedded in the library's .data section behind the symbols
_binary_<name>_o_start / _binary_<name>_o_size. Extracting one gives the <name>.o file that
libgoyamme's gmme_tpc_fire_vendor() loads.

  goya-kernels.py list                 # gen-1 Goya kernels (names ending _goya2 are Greco, left out)
  goya-kernels.py list --all           # every kernel in the library
  goya-kernels.py extract softmax_dim0_f32 relu_fwd_f32 -o ./kernels
  goya-kernels.py extract --all-gen1 -o ./kernels

Install the library first: sudo scripts/install-vendor-kernels.sh
"""
import argparse, os, struct, sys

DEFAULT_LIB = "/usr/local/lib/goya-stack/vendor/libtpc_kernels.so.1"
PRE, START, SIZE = "_binary_", "_o_start", "_o_size"


def read_kernels(path):
    """Return ({name: (file_offset, size)}, data_section) from an ELF64 shared library."""
    with open(path, "rb") as f:
        blob = f.read()
    if blob[:4] != b"\x7fELF" or blob[4] != 2:
        sys.exit(path + " is not a 64-bit ELF file")
    shoff, = struct.unpack_from("<Q", blob, 0x28)
    shentsize, shnum, shstrndx = struct.unpack_from("<HHH", blob, 0x3A)
    secs = []
    for i in range(shnum):
        name, typ, flags, addr, off, size, link, info, align, entsize = \
            struct.unpack_from("<IIQQQQIIQQ", blob, shoff + i * shentsize)
        secs.append((name, typ, addr, off, size, link, entsize))
    shstr_off = secs[shstrndx][3]
    def secname(s):
        end = blob.index(b"\0", shstr_off + s[0])
        return blob[shstr_off + s[0]:end].decode()
    by_name = {secname(s): s for s in secs}
    for need in (".dynsym", ".dynstr", ".data"):
        if need not in by_name:
            sys.exit("%s has no %s section; not the expected library" % (path, need))
    _, _, _, sym_off, sym_size, _, sym_ent = by_name[".dynsym"]
    str_off = by_name[".dynstr"][3]
    data = by_name[".data"]
    starts, sizes = {}, {}
    for i in range(sym_size // sym_ent):
        st_name, st_info, st_other, st_shndx, st_value, st_size = \
            struct.unpack_from("<IBBHQQ", blob, sym_off + i * sym_ent)
        end = blob.index(b"\0", str_off + st_name)
        n = blob[str_off + st_name:end].decode(errors="replace")
        if not n.startswith(PRE):
            continue
        if n.endswith(START):
            starts[n[len(PRE):-len(START)]] = st_value
        elif n.endswith(SIZE):
            sizes[n[len(PRE):-len(SIZE)]] = st_value
    kernels = {}
    for k, vaddr in starts.items():
        if k in sizes:
            kernels[k] = (vaddr - data[2] + data[3], sizes[k])
    return kernels, blob


def main():
    ap = argparse.ArgumentParser(description="List and extract TPC kernels from libtpc_kernels.so.1")
    ap.add_argument("--lib", default=os.environ.get("GOYA_TPC_KERNEL_LIB", DEFAULT_LIB))
    sub = ap.add_subparsers(dest="cmd", required=True)
    pl = sub.add_parser("list"); pl.add_argument("--all", action="store_true")
    pe = sub.add_parser("extract"); pe.add_argument("names", nargs="*")
    pe.add_argument("--all-gen1", action="store_true"); pe.add_argument("-o", "--out", default=".")
    a = ap.parse_args()
    if not os.path.exists(a.lib):
        sys.exit("%s not found. Run: sudo scripts/install-vendor-kernels.sh" % a.lib)
    kernels, blob = read_kernels(a.lib)
    gen1 = sorted(k for k in kernels if not k.endswith("_goya2"))
    if a.cmd == "list":
        for k in (sorted(kernels) if a.all else gen1):
            print(k)
        print("# %d listed (%d in the library, %d gen-1 Goya)" %
              (len(kernels) if a.all else len(gen1), len(kernels), len(gen1)), file=sys.stderr)
        return
    names = gen1 if a.all_gen1 else a.names
    if not names:
        sys.exit("name one or more kernels, or use --all-gen1")
    os.makedirs(a.out, exist_ok=True)
    for k in names:
        if k not in kernels:
            sys.exit("no kernel named %r (try: goya-kernels.py list)" % k)
        off, size = kernels[k]
        obj = blob[off:off + size]
        if obj[:4] != b"\x7fELF":
            sys.exit("%s: extracted bytes are not an ELF object" % k)
        with open(os.path.join(a.out, k + ".o"), "wb") as f:
            f.write(obj)
    print("extracted %d kernel%s to %s" % (len(names), "" if len(names) == 1 else "s", a.out))


if __name__ == "__main__":
    main()
