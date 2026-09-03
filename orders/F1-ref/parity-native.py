import sys, torch; sys.path.insert(0, "/port/src")
import vllm_exl3_c as C
from vllm_exl3.exl3 import make_linear_exl3
torch.manual_seed(0); dev="cuda"
H, I = 4096, 2048
MCG = torch.tensor([-877912083], dtype=torch.int32, device=dev)
def mk(in_f, out_f, K):
    t = torch.randint(-32768, 32767, (in_f//16, out_f//16, 16*K), dtype=torch.int16, device=dev)
    suh = (torch.randn(in_f, device=dev)/64).half(); svh = torch.randn(out_f, device=dev).half()
    return t, suh, svh
for K in (2, 3):
    for E in (1, 6):
        gate=[mk(H,I,K) for _ in range(E)]; up=[mk(H,I,K) for _ in range(E)]; down=[mk(I,H,K) for _ in range(E)]
        x = (torch.randn(1, H, device=dev)*0.1).half()
        w = torch.softmax(torch.randn(1, E, device=dev), -1).half()
        ref = torch.zeros(1, H, device=dev)
        for e in range(E):
            lg = make_linear_exl3(gate[e][0], gate[e][1], gate[e][2], MCG, None, out_dtype=torch.float16)
            lu = make_linear_exl3(up[e][0], up[e][1], up[e][2], MCG, None, out_dtype=torch.float16)
            ld = make_linear_exl3(down[e][0], down[e][1], down[e][2], MCG, None, out_dtype=torch.float16)
            g = lg.forward(x, {}, out_dtype=torch.float32); u = lu.forward(x, {}, out_dtype=torch.float32)
            a = torch.nn.functional.silu(g) * u
            d = ld.forward(a.half().contiguous(), {}, out_dtype=torch.float32)
            ref += w[0, e].float() * d
        ptr = lambda L, i: torch.tensor([int(t[i].data_ptr()) for t in L], dtype=torch.int64, device=dev)
        out = torch.zeros(1, H, device=dev).half()
        C.p2b_fused_moe(x, out, ptr(gate,0), ptr(gate,1), ptr(gate,2), ptr(up,0), ptr(up,1), ptr(up,2), ptr(down,0), ptr(down,1), ptr(down,2), torch.arange(E, dtype=torch.int32, device=dev), w, K, K, K, True, 0.0)
        torch.cuda.synchronize()
        cos = torch.nn.functional.cosine_similarity(out.float().view(-1), ref.view(-1), dim=0).item()
        rel = ((out.float()-ref).norm()/ref.norm()).item()
        print(f"K={K} E={E}: cos={cos:.5f} rel_err={rel:.4f}  ref_norm={ref.norm().item():.3f} out_norm={out.float().norm().item():.3f}", flush=True)
