#!/usr/bin/env python3
"""Qwen3-30B-A3B text generation on one Goya card: text in, text out.

  embed -> [48x: attn_norm, QKV, q/k_norm, RoPE, GQA attention + KV cache,
            attn_out, residual, ffn_norm, route top-8 of 128 experts, residual]
        -> output_norm -> lm_head -> argmax -> next token

Every projection runs as int8 GEMM on the MME (per-column weight scale, per-row
activation scale). Norms, RoPE, softmax pieces and SwiGLU run as f32 kernels on
the TPC. Needs numpy, the libgoyamme install from scripts/build-runtime.sh, and
the Q8_0 GGUF named in README.md.

  sudo python3 generate.py --model Qwen3-30B-A3B-Instruct-2507-Q8_0.gguf
  sudo python3 generate.py --model ... --check      # the reference test, 4 tokens
"""
import argparse, collections, ctypes, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gguf_min import GGUF

ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
ap.add_argument("--model", default=os.environ.get("GOYA_QWEN3_GGUF"),
                help="path to Qwen3-30B-A3B-Instruct-2507-Q8_0.gguf (or set GOYA_QWEN3_GGUF)")
ap.add_argument("--card", default="auto", help='PCI address like 0000:03:00.0, or "auto" (default)')
ap.add_argument("--prompt", default="The capital of France is")
ap.add_argument("--tokens", type=int, default=8, help="new tokens to generate (default 8)")
ap.add_argument("--check", action="store_true",
                help="run the reference test: default prompt, 4 tokens, compare with the recorded run")
ap.add_argument("--expert-cache-gb", type=float, default=float(os.environ.get("GOYA_EXPERT_CACHE_GB", "8")),
                help="host RAM for quantised expert weights (default 8)")
args = ap.parse_args()
if not args.model:
    ap.error("--model is required (or set GOYA_QWEN3_GGUF)")
if args.check:
    args.prompt, args.tokens = "The capital of France is", 4
REFERENCE = [12095, 13, 576, 6722]   # " Paris", ".", " The", " capital" -- recorded 2026-09-02

LIB = os.environ.get("GOYA_LIB", "/usr/local/lib/goya-stack/libgoyamme.so")
TPC_DIR = os.environ.get("GOYA_TPC_DIR", "/usr/local/share/goya-stack/tpc-kernels")
os.environ.setdefault("GMME_ARENA_MB", "7168")


gf = GGUF(args.model)
fh = gf.f
HID, D, QH, KVH, NE, TOPK, FF = 2048, 128, 32, 4, 128, 8, 768
NL = gf.metadata["qwen3moe.block_count"]
EPS = gf.metadata["qwen3moe.attention.layer_norm_rms_epsilon"]
EOS = gf.metadata["tokenizer.ggml.eos_token_id"]
VOCAB = gf.metadata["tokenizer.ggml.tokens"]
f = np.float32
def M(a, b): return (a*b).astype(np.float32)

# ---- byte-level BPE alphabet (GPT-2 style, which Qwen uses) --------------
def bytes_to_unicode():
    bs = list(range(33,127))+list(range(161,173))+list(range(174,256))
    cs = bs[:]; n = 0
    for b in range(256):
        if b not in bs: bs.append(b); cs.append(256+n); n += 1
    return dict(zip(bs, [chr(c) for c in cs]))
B2U = bytes_to_unicode(); U2B = {v:k for k,v in B2U.items()}
VIDX = {t:i for i,t in enumerate(VOCAB)}
def encode(text):
    s = "".join(B2U[b] for b in text.encode("utf-8"))
    ids, i = [], 0
    while i < len(s):                       # greedy longest match
        for j in range(min(len(s), i+32), i, -1):
            if s[i:j] in VIDX: ids.append(VIDX[s[i:j]]); i = j; break
        else:
            ids.append(VIDX.get(s[i], 0)); i += 1
    return ids
def decode(ids):
    s = "".join(VOCAB[i] for i in ids)
    return bytes([U2B[c] for c in s if c in U2B]).decode("utf-8", errors="replace")

# ---- library ------------------------------------------------------------
L = ctypes.CDLL(LIB)
L.gmme_open.restype = ctypes.c_void_p;  L.gmme_open.argtypes = [ctypes.c_char_p]
L.gmme_error.restype = ctypes.c_char_p; L.gmme_error.argtypes = [ctypes.c_void_p]
L.gmme_close.argtypes = [ctypes.c_void_p]
I8 = ctypes.POINTER(ctypes.c_int8); I32 = ctypes.POINTER(ctypes.c_int32); U64 = ctypes.c_uint64
L.gmme_gemm_i8.argtypes = [ctypes.c_void_p, I8, I8, I32] + [ctypes.c_int]*3
L.gmme_put_c.argtypes = [ctypes.c_void_p, U64, ctypes.c_void_p, ctypes.c_uint32]
L.gmme_get_c.argtypes = [ctypes.c_void_p, U64, ctypes.c_void_p, ctypes.c_uint32]
L.gmme_tpc_op_resident.argtypes = [ctypes.c_void_p, ctypes.c_char_p, U64, U64, U64, ctypes.c_int, ctypes.c_int]
L.gmme_tpc_op3_resident.argtypes = [ctypes.c_void_p, ctypes.c_char_p, U64, U64, U64, U64, ctypes.c_int, ctypes.c_int]
L.gmme_tpc_dotrows_f32.argtypes = [ctypes.c_void_p, U64, U64, U64] + [ctypes.c_int]*3
L.gmme_tpc_sumsq_f32_rows.argtypes = [ctypes.c_void_p, U64, U64] + [ctypes.c_int]*3
L.gmme_tpc_reduce_f32_rows.argtypes = [ctypes.c_void_p, U64, U64] + [ctypes.c_int]*3
L.gmme_tpc_max_f32_rows.argtypes = [ctypes.c_void_p, U64, U64, U64] + [ctypes.c_int]*3
g = L.gmme_open(None if args.card == "auto" else args.card.encode())
err = L.gmme_error(g)
if not g or (err and err.decode()):
    sys.exit("open: " + (err.decode() if err else "no free Goya card"))
A_, B_, C_, D_, O_, S_ = (i << 20 for i in (0,1,2,3,4,5))
def put(o,a):
    a=np.ascontiguousarray(a,np.float32); L.gmme_put_c(g,o,a.ctypes.data_as(ctypes.c_void_p),a.size*4)
def get(o,n):
    b=np.empty(n,np.float32); L.gmme_get_c(g,o,b.ctypes.data_as(ctypes.c_void_p),n*4); return b
def kpath(k): return os.path.join(TPC_DIR, k + ".o").encode()
def op(k,a,b,o,n):
    if L.gmme_tpc_op_resident(g,kpath(k),a,b,o,n,0):
        raise RuntimeError(k+": "+L.gmme_error(g).decode())
def op3(k,a,b,c,o,n):
    if L.gmme_tpc_op3_resident(g,kpath(k),a,b,c,o,n,0):
        raise RuntimeError(k+": "+L.gmme_error(g).decode())

# ---- weights ------------------------------------------------------------
# f32 weights are dequantised only when their int8 copy does not exist yet, then dropped.
# Small vectors (norm weights) stay cached. Expert int8 copies live in a bounded LRU cache.
_vec = {}
def raw(name):
    t = gf.tensors[name]
    if t.dtype == 0:
        if name not in _vec:
            fh.seek(gf.data_offset+t.offset)
            _vec[name] = np.frombuffer(fh.read(t.byte_size),dtype=np.float32).reshape(t.shape[::-1]).copy()
        return _vec[name]
    bs,bb = GGUF.BLOCK[t.dtype]
    n = int(np.prod(t.shape)); nb = n//bs
    fh.seek(gf.data_offset+t.offset)
    blk = np.frombuffer(fh.read(nb*bb),dtype=np.uint8).reshape(nb,bb)
    sc = blk[:,:2].copy().view(np.float16).reshape(nb,1).astype(np.float32)
    return (blk[:,2:].copy().view(np.int8).astype(np.float32)*sc).reshape(t.shape[::-1])

def slab(name,e,K_,N_):
    t = gf.tensors[name]
    bs,bb = GGUF.BLOCK[t.dtype]
    nb=(K_*N_)//bs; per=nb*bb
    fh.seek(gf.data_offset+t.offset+e*per)
    blk = np.frombuffer(fh.read(per),dtype=np.uint8).reshape(nb,bb)
    sc = blk[:,:2].copy().view(np.float16).reshape(nb,1).astype(np.float32)
    return (blk[:,2:].copy().view(np.int8).astype(np.float32)*sc).reshape(N_,K_)

_q8 = {}                                   # attention, router and lm_head: kept for the run
_q8e = collections.OrderedDict()           # experts: least recently used goes first
_q8e_bytes = [0]
_q8e_cap = int(args.expert_cache_gb * (1 << 30))
def _quant(W):
    Win = np.ascontiguousarray(W.T)
    sc = (np.abs(Win).max(axis=0)/127.0).astype(np.float32); sc[sc==0]=f(1)
    return (np.rint(Win/sc[None,:]).clip(-127,127).astype(np.int8), sc)

def quant_w(key, load, expert=False):
    """[out,in] -> (int8 [in,out], per-column scale). Done once per weight while it stays cached."""
    if not expert:
        if key not in _q8: _q8[key] = _quant(load())
        return _q8[key]
    if key in _q8e:
        _q8e.move_to_end(key); return _q8e[key]
    v = _quant(load())
    _q8e[key] = v; _q8e_bytes[0] += v[0].nbytes + v[1].nbytes
    while _q8e_bytes[0] > _q8e_cap and len(_q8e) > 1:
        _, old = _q8e.popitem(last=False); _q8e_bytes[0] -= old[0].nbytes + old[1].nbytes
    return v

def gemm(key, Hf, load, expert=False):
    Wq8, s_col = quant_w(key, load, expert)
    s_row = (np.abs(Hf).max(axis=1)/127.0).astype(np.float32); s_row[s_row==0]=f(1)
    Hq8 = np.rint(Hf/s_row[:,None]).clip(-127,127).astype(np.int8)
    Mr, Kd = Hq8.shape; N = Wq8.shape[1]
    out = np.empty((Mr,N),np.int32)
    ca = (ctypes.c_int8*(Mr*Kd)).from_buffer_copy(Hq8.tobytes())
    for j0 in range(0,N,256):
        w = min(256,N-j0)
        cb = (ctypes.c_int8*(Kd*w)).from_buffer_copy(np.ascontiguousarray(Wq8[:,j0:j0+w]).tobytes())
        cc = (ctypes.c_int32*(Mr*w))()
        if L.gmme_gemm_i8(g,ca,cb,cc,Mr,Kd,w):
            raise RuntimeError("gemm %s: %s"%(key,L.gmme_error(g).decode()))
        out[:,j0:j0+w] = np.frombuffer(bytes(cc),dtype=np.int32).reshape(Mr,w)
    return (out.astype(np.float32)*s_row[:,None]*s_col[None,:]).astype(np.float32)

def rmsnorm(x, w, eps):
    rows,hid = x.shape
    put(A_, x)
    L.gmme_tpc_sumsq_f32_rows(g,A_,S_,rows,hid,0)
    ss = get(S_,rows*64)[::64][:rows]
    put(B_, np.concatenate([np.full(hid,v,np.float32) for v in (M(ss,f(1.0/hid))+f(eps))]))
    op("k_rsqrt_f32",B_,B_,B_,rows*hid)
    put(C_, np.tile(w,(rows,1)))
    op3("k_mul3_f32",A_,B_,C_,D_,rows*hid)
    return get(D_,rows*hid).reshape(rows,hid)

INV = (1.0/(1e7**(np.arange(0,D,2,dtype=np.float64)/D))).astype(np.float32)
def rope(z, pos):                      # z [n, D], pos [n]
    ang = (pos[:,None].astype(np.float32)*INV[None,:]).astype(np.float32)
    put(A_, z); put(B_, np.cos(ang).astype(np.float32)); put(C_, np.sin(ang).astype(np.float32))
    op3("k_rope_f32",A_,B_,C_,D_, z.size)
    return get(D_, z.size).reshape(z.shape)

Kc = [np.zeros((0,KVH,D),np.float32) for _ in range(NL)]
Vc = [np.zeros((0,KVH,D),np.float32) for _ in range(NL)]

def layer(li, x, pos):
    P = "blk.%d." % li
    h = rmsnorm(x, raw(P+"attn_norm.weight").ravel(), EPS)
    q = gemm(P+"q", h, lambda: raw(P+"attn_q.weight")).reshape(-1,QH,D)
    k = gemm(P+"k", h, lambda: raw(P+"attn_k.weight")).reshape(-1,KVH,D)
    v = gemm(P+"v", h, lambda: raw(P+"attn_v.weight")).reshape(-1,KVH,D)
    qn = raw(P+"attn_q_norm.weight").ravel(); kn = raw(P+"attn_k_norm.weight").ravel()
    T = q.shape[0]
    q = rmsnorm(q.reshape(-1,D), qn, EPS).reshape(T,QH,D)
    k = rmsnorm(k.reshape(-1,D), kn, EPS).reshape(T,KVH,D)
    rp = np.repeat(pos, QH); q = rope(q.reshape(-1,D), rp).reshape(T,QH,D)
    rp = np.repeat(pos, KVH); k = rope(k.reshape(-1,D), rp).reshape(T,KVH,D)
    Kc[li] = np.concatenate([Kc[li], k]); Vc[li] = np.concatenate([Vc[li], v])
    S = Kc[li].shape[0]
    out = np.zeros((T,QH,D),np.float32)
    for hh in range(QH):
        kv = hh//(QH//KVH)
        qh = np.ascontiguousarray(q[:,hh,:]); kh = np.ascontiguousarray(Kc[li][:,kv,:])
        put(A_, np.repeat(qh,S,axis=0)); put(B_, np.tile(kh,(T,1)))
        L.gmme_tpc_dotrows_f32(g,A_,B_,O_,T*S,D,0)
        sc = (get(O_,T*S*64)[::64][:T*S].reshape(T,S)/np.sqrt(np.float32(D))).astype(np.float32)
        base = S-T
        keep = np.array([[1.0 if j<=base+i else 0.0 for j in range(S)] for i in range(T)],np.float32)
        sc = np.where(keep>0, sc, -np.inf).astype(np.float32)
        W = ((S+63)//64)*64
        pad = np.full((T,W),-np.inf,np.float32); pad[:,:S]=sc
        put(C_,pad); L.gmme_tpc_max_f32_rows(g,C_,S_,O_,T,W,0)
        rmax = get(O_,T*64)[::64][:T]
        put(A_, np.maximum(sc-rmax[:,None],f(-80)).astype(np.float32))
        op("k_exp_f32",A_,A_,C_,T*S)
        ex = (get(C_,T*S).reshape(T,S)*keep).astype(np.float32)
        pad = np.zeros((T,W),np.float32); pad[:,:S]=ex
        put(A_,pad); L.gmme_tpc_reduce_f32_rows(g,A_,O_,T,W,0)
        den = get(O_,T*64)[::64][:T]
        put(A_,np.repeat(den,64)); op("k_recip_f32",A_,A_,C_,T*64)
        rcp = get(C_,T*64)[::64][:T]
        out[:,hh,:] = (M(ex,rcp[:,None]) @ Vc[li][:,kv,:]).astype(np.float32)
    x = (x + gemm(P+"o", out.reshape(T,QH*D), lambda: raw(P+"attn_output.weight"))).astype(np.float32)
    hf = rmsnorm(x, raw(P+"ffn_norm.weight").ravel(), EPS)
    lg = gemm(P+"gi", hf, lambda: raw(P+"ffn_gate_inp.weight"))
    top = np.argsort(-lg,axis=1)[:,:TOPK]
    tw = np.take_along_axis(lg,top,axis=1); tw = np.exp(tw-tw.max(axis=1,keepdims=True))
    tw = (tw/tw.sum(axis=1,keepdims=True)).astype(np.float32)
    ff = np.zeros((T,HID),np.float32)
    for i in range(T):
        for s in range(TOPK):
            e = int(top[i,s])
            gp = gemm("%sg%d"%(P,e), hf[i:i+1], lambda: slab(P+"ffn_gate_exps.weight",e,HID,FF), True)
            up = gemm("%su%d"%(P,e), hf[i:i+1], lambda: slab(P+"ffn_up_exps.weight",e,HID,FF), True)
            put(A_,gp.ravel()); put(B_,up.ravel()); op("k_swiglu_f32",A_,B_,C_,FF)
            act = get(C_,FF).reshape(1,FF)
            dn = gemm("%sd%d"%(P,e), act, lambda: slab(P+"ffn_down_exps.weight",e,FF,HID), True)
            ff[i] = (ff[i] + M(dn.ravel(), tw[i,s])).astype(np.float32)
    return (x + ff).astype(np.float32)

def embed(ids):
    t = gf.tensors["token_embd.weight"]
    bs,bb = GGUF.BLOCK[t.dtype]
    nbrow = HID//bs
    out = np.zeros((len(ids),HID),np.float32)
    for i,tid in enumerate(ids):
        fh.seek(gf.data_offset+t.offset+tid*nbrow*bb)
        blk = np.frombuffer(fh.read(nbrow*bb),dtype=np.uint8).reshape(nbrow,bb)
        sc = blk[:,:2].copy().view(np.float16).reshape(nbrow,1).astype(np.float32)
        out[i] = (blk[:,2:].copy().view(np.int8).astype(np.float32)*sc).ravel()
    return out

ids = encode(args.prompt)
n_prompt = len(ids)
print("prompt %r -> %d tokens -> %r" % (args.prompt, n_prompt, decode(ids)), flush=True)
t_start = time.perf_counter()
for step in range(args.tokens):
    feed = ids if step == 0 else ids[-1:]
    pos  = np.arange(len(ids)-len(feed), len(ids))
    x = embed(feed)
    t0 = time.perf_counter()
    for li in range(NL):
        x = layer(li, x, pos)
    x = rmsnorm(x, raw("output_norm.weight").ravel(), EPS)
    logits = gemm("lm_head", x[-1:], lambda: raw("output.weight"))
    nxt = int(np.argmax(logits[0]))
    ids.append(nxt)
    print("  [%d] %.1f s  -> %-6d %r" % (step, time.perf_counter()-t0, nxt,
                                         decode([nxt])), flush=True)
    if nxt == EOS: break
print("\n=== OUTPUT ===")
print(decode(ids))
print("=== %.1f s for %d tokens ===" % (time.perf_counter()-t_start, len(ids)-n_prompt))
L.gmme_close(g)
if args.check:
    got = ids[n_prompt:n_prompt+4]
    if got == REFERENCE:
        print("CHECK PASS: the 4 tokens match the reference run exactly.")
    else:
        print("CHECK FAIL: got %s, reference %s." % (got, REFERENCE)); sys.exit(1)
