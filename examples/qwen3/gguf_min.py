"""Minimal GGUF v2/v3 reader for generate.py: metadata and tensor locations, no dependencies beyond numpy."""
import collections, struct, sys
import numpy as np


class GGUF:
    """Minimal GGUF v2/v3 reader: metadata and tensor locations. Tensor data is read on demand."""
    SCALAR = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}
    BLOCK = {0: (1, 4), 8: (32, 34)}   # F32, Q8_0: (elements, bytes) per block

    Tensor = collections.namedtuple("Tensor", "dtype shape offset byte_size")

    def __init__(self, path):
        self.f = f = open(path, "rb")
        if f.read(4) != b"GGUF":
            sys.exit(path + " is not a GGUF file")
        version, n_tensors, n_kv = struct.unpack("<IQQ", f.read(20))
        if version < 2:
            sys.exit("GGUF version %d is too old" % version)
        self.metadata = {}
        for _ in range(n_kv):
            key = self._str()
            self.metadata[key] = self._val(struct.unpack("<I", f.read(4))[0])
        infos = []
        for _ in range(n_tensors):
            name = self._str()
            nd, = struct.unpack("<I", f.read(4))
            shape = struct.unpack("<%dQ" % nd, f.read(8 * nd))
            dtype, off = struct.unpack("<IQ", f.read(12))
            infos.append((name, dtype, shape, off))
        align = self.metadata.get("general.alignment", 32)
        self.data_offset = (f.tell() + align - 1) // align * align
        self.tensors = {}
        for name, dtype, shape, off in infos:
            bs, bb = self.BLOCK.get(dtype, (0, 0))
            n = int(np.prod(shape))
            self.tensors[name] = self.Tensor(dtype, shape, off, n // bs * bb if bs else 0)

    def _str(self):
        n, = struct.unpack("<Q", self.f.read(8))
        return self.f.read(n).decode("utf-8", errors="replace")

    def _val(self, t):
        if t == 8:
            return self._str()
        if t == 9:
            et, n = struct.unpack("<IQ", self.f.read(12))
            if et in self.SCALAR:
                c = self.SCALAR[et]
                return list(struct.unpack("<%d%s" % (n, c), self.f.read(struct.calcsize(c) * n)))
            return [self._val(et) for _ in range(n)]
        c = self.SCALAR[t]
        return struct.unpack("<" + c, self.f.read(struct.calcsize(c)))[0]
