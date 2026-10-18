import flashinfer, torch, triton, triton.language as tl

@triton.jit
def _k(P, LO, HI, N: tl.constexpr):
    o = tl.arange(0, N)
    p = tl.load(P + o)
    lo, hi = tl.inline_asm_elementwise(
        asm="""{
        .reg .b8 b0, b1, b2, b3;
        .reg .b32 r0, r1, r2, r3;
        mov.b32 {b0, b1, b2, b3}, $4;
        cvt.rn.f16x2.e2m1x2 r0, b0;
        cvt.rn.f16x2.e2m1x2 r1, b1;
        cvt.rn.f16x2.e2m1x2 r2, b2;
        cvt.rn.f16x2.e2m1x2 r3, b3;
        prmt.b32 $0, r0, r1, 0x5410;
        prmt.b32 $1, r2, r3, 0x5410;
        prmt.b32 $2, r0, r1, 0x7632;
        prmt.b32 $3, r2, r3, 0x7632;
        }""",
        constraints="=r,=r,=r,=r,r", args=[p], dtype=(tl.float16, tl.float16), is_pure=True, pack=4)
    tl.store(LO + o, lo)
    tl.store(HI + o, hi)

p = torch.arange(256, dtype=torch.int32).to(torch.uint8).cuda()
lo = torch.empty(256, dtype=torch.float16, device="cuda"); hi = torch.empty_like(lo)
_k[(1,)](p, lo, hi, N=256)
vals = torch.tensor([0,.5,1,1.5,2,3,4,6,-0.,-.5,-1,-1.5,-2,-3,-4,-6])
pl = torch.arange(256) & 15; ph = torch.arange(256) >> 4
print("lo ok", torch.equal(lo.cpu().float(), vals[pl]), "hi ok", torch.equal(hi.cpu().float(), vals[ph]))
print(lo[:20].tolist(), hi[16:40:4].tolist())
