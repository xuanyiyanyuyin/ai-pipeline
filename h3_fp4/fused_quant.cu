#include <cuda_bf16.h>
#include <cuda_fp8.h>

// one thread per 16-elem block: absmax -> fp8 scale -> E2M1 round -> pack 8 bytes
extern "C" __global__ void fused_quant(const __nv_bfloat16* __restrict__ x,
                                       float g, unsigned char* __restrict__ packed,
                                       __nv_fp8_storage_t* __restrict__ scale,
                                       long nblocks, int K) {
  long b = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (b >= nblocks) return;
  const __nv_bfloat16* xp = x + b * 16;
  float v[16]; float am = 0.f;
  #pragma unroll
  for (int i = 0; i < 16; i++) { v[i] = __bfloat162float(xp[i]); float a = fabsf(v[i]); am = fmaxf(am, a); }
  am = fmaxf(am, 1e-12f);
  float s = (am / 6.0f) * g;
  __nv_fp8_storage_t s8 = __nv_cvt_float_to_fp8(s, __NV_SATFINITE, __NV_E4M3);
  scale[b] = s8;
  float se = __half2float(__nv_cvt_fp8_to_halfraw(s8, __NV_E4M3));
  se = fmaxf(se, 1e-12f);
  unsigned char bytes[8];
  #pragma unroll
  for (int i = 0; i < 8; i++) {
    unsigned char c0 = 0, c1 = 0;
    for (int j = 0; j < 2; j++) {
      float n = v[i*2+j] * g / se; n = fminf(fmaxf(n, -6.f), 6.f);
      float a = fabsf(n);
      int code = (a>=0.25f)+(a>=0.75f)+(a>=1.25f)+(a>=1.75f)+(a>=2.5f)+(a>=3.5f)+(a>=5.0f);
      unsigned char c = (unsigned char)code | ((n < 0.f) ? 8 : 0);
      if (j == 0) c0 = c; else c1 = c;
    }
    bytes[i] = c0 | (c1 << 4);
  }
  unsigned char* pp = packed + b * 8;
  #pragma unroll
  for (int i = 0; i < 8; i++) pp[i] = bytes[i];
}