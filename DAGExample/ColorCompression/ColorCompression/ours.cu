#include "ours.h"
#include "ours_varbit.h"
#include <cuda.h>
#include <device_launch_parameters.h>
#include <device_atomic_functions.h>
#include <sm_35_intrinsics.h>
#include <vector_functions.h>
#include <vector>
#include "BlockBuild.h"

float __device__ rand_lut[64] = {
  0.9486982, 0.972871958, 0.248168957, 0.493126931, 0.738212088, 0.653544012, 0.67056634, 0.204192427,
  0.972804412, 0.991614247, 0.907730512, 0.826491797, 0.79865054, 0.94179941, 0.867766025, 0.280207877,
  0.757674479, 0.184792714, 0.894972863, 0.700680464, 0.397279507, 0.827494222, 0.0977131338, 0.108998389,
  0.503181245, 0.207843145, 0.828793763, 0.973948237, 0.791490856, 0.913345491, 0.859345573, 0.333875761,
  0.250367264, 0.68947019, 0.08739105, 0.95076748, 0.934732119, 0.928425798, 0.199394464, 0.549738232,
  0.0507203067, 0.0957588106, 0.0232692207, 0.611455875, 0.0713980492, 0.485231558, 0.162556085, 0.944551351,
  0.0615243969, 0.0616311938, 0.0927325418, 0.450354735, 0.0980233341, 0.962107109, 0.411898038, 0.560993149,
  0.997294696, 0.845310842, 0.522109665, 0.293706246, 0.542670523, 0.79422221, 0.0684990289, 0.410180829 };

using std::vector;

//#include "colorspace.h"
namespace colorspace {
  struct Lab {
    float L;
    float a;
    float b;
  };

  struct sRGB {
    float R;
    float G;
    float B;
  };

  // Whitepoints (D65)
#define CSPC_XR 0.95047f
#define CSPC_YR 1.00000f
#define CSPC_ZR 1.08883f
  // Constants
#define CSPC_EPS 0.008856f
#define CSPC_KAPPA 903.3f

  __device__ inline Lab as_Lab(sRGB c) {
    // For reference http://www.brucelindbloom.com/
    auto sRGB_compand = [](float val)
    {
      return (val > 0.04045f ? powf((val + 0.055f) / 1.055f, 2.4f) : val / 12.92f);
    };

    c.R = sRGB_compand(c.R);
    c.G = sRGB_compand(c.G);
    c.B = sRGB_compand(c.B);

    // sRGB -> XYZ
    float X = 0.4124564f * c.R + 0.3575761f * c.G + 0.1804375f * c.B;
    float Y = 0.2126729f * c.R + 0.7151522f * c.G + 0.0721750f * c.B;
    float Z = 0.0193339f * c.R + 0.1191920f * c.G + 0.9503041f * c.B;

    // XYZ -> Lab
    float xr = X / CSPC_XR;
    float yr = Y / CSPC_YR;
    float zr = Z / CSPC_ZR;

    float fx = xr > CSPC_EPS ? pow(xr, 1.0f / 3.0f) : (CSPC_KAPPA * xr + 16.0f) / 116.0f;
    float fy = yr > CSPC_EPS ? pow(yr, 1.0f / 3.0f) : (CSPC_KAPPA * yr + 16.0f) / 116.0f;
    float fz = zr > CSPC_EPS ? pow(zr, 1.0f / 3.0f) : (CSPC_KAPPA * zr + 16.0f) / 116.0f;

    float L = 116.0f * fy - 16.0f;
    float a = 500.0f * (fx - fy);
    float b = 200.0f * (fy - fz);
    return Lab{ L, a, b };
  };

  __device__ inline sRGB as_sRGB(const Lab &c) {
    // For reference http://www.brucelindbloom.com/

    // Lab -> XYZ
    float fy = (c.L + 16.0f) / 116.0f;
    float fz = (fy - c.b / 200.0f);
    float fx = (fy + c.a / 500.0f);

    float fx3 = fx * fx * fx;
    float fy3 = fy * fy * fy;
    float fz3 = fz * fz * fz;

    float xr = fx3 > CSPC_EPS ? fx3 : (116.0f * fx - 16.0f) / CSPC_KAPPA;
    float yr = c.L > (CSPC_KAPPA * CSPC_EPS) ? fy3 : c.L / CSPC_KAPPA;
    float zr = fz3 > CSPC_EPS ? fz3 : (116.0f * fz - 16.0f) / CSPC_KAPPA;

    float X = xr * CSPC_XR;
    float Y = yr * CSPC_YR;
    float Z = zr * CSPC_ZR;

    // XYZ -> sRGB
    float R = 3.2404542f * X - 1.5371385f * Y - 0.4985314f * Z;
    float G = -0.9692660f * X + 1.8760108f * Y + 0.0415560f * Z;
    float B = 0.0556434f * X - 0.2040259f * Y + 1.0572252f * Z;

    auto sRGB_compand = [](float val)
    {
      return (val > 0.0031308f ? 1.055f*pow(val, 1.0f / 2.4f) - 0.055f : val * 12.92f);
    };

    R = sRGB_compand(R);
    G = sRGB_compand(G);
    B = sRGB_compand(B);

    return sRGB{ R, G, B };
  };
}  // namespace colorspace

// clang-format off
__host__ __device__ float3 inline operator * (const float a, const float3 &b) { return make_float3(a * b.x, a * b.y, a * b.z); };
__host__ __device__ float3 inline operator * (const float3 &b, const float a) { return a * b; };
__host__ __device__ float3 inline operator - (const float3 &a, const float3 &b) { return make_float3(a.x - b.x, a.y - b.y, a.z - b.z); };
__host__ __device__ float3 inline operator + (const float3 &a, const float3 &b) { return make_float3(a.x + b.x, a.y + b.y, a.z + b.z); };
__host__ __device__ float3 inline operator * (const float3 &a, const float3 &b) { return make_float3(a.x * b.x, a.y * b.y, a.z * b.z); };
__host__ __device__ float3 inline operator / (const float3 &a, const float3 &b) { return make_float3(a.x / b.x, a.y / b.y, a.z / b.z); };
__host__ __device__ float3 inline operator - (float3 a) { return make_float3(-a.x, -a.y, -a.z); };
__host__ __device__ inline float dot(const float3 &a, const float3 &b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
__host__ __device__ inline float length(float3 a) { return sqrt(dot(a, a)); }
__host__ __device__ inline float3 normalize(float3 a) { return (1.0f / length(a)) * a; }
// clang-format on

template<class T> __host__ __device__ T min(T a, T b) { return a < b ? a : b; }
template<class T> __host__ __device__ T max(T a, T b) { return a > b ? a : b; }
template<class T> __host__ __device__ T clamp(T val, T minVal, T maxVal) { return min(max(minVal, val), maxVal); }

struct float3x3 {
  float3 c1, c2, c3;

  __host__ __device__ float3x3(const float3 &v1, const float3 &v2, const float3 &v3)
    : c1(v1), c2(v2), c3(v3) {};

  __host__ __device__  inline const float3 operator * (const float3& v) const
  {
    return v.x * c1 + v.y * c2 + v.z * c3;
  }

  __host__ __device__  inline const float3x3 operator + (const float3x3& m) const
  {
    return float3x3(c1 + m.c1, c2 + m.c2, c3 + m.c3);
  }

  __host__ __device__  inline const float3x3 operator - (const float3x3& m) const
  {
    return float3x3(c1 - m.c1, c2 - m.c2, c3 - m.c3);
  }

};

__host__ __device__ inline float trace(const float3x3 &m) { return m.c1.x + m.c2.y + m.c3.z; }

__host__ __device__  inline const float3x3 operator * (const float s, const float3x3& m)
{
  return float3x3(s * m.c1, s * m.c2, s * m.c3);
}

__host__ __device__ float3x3 make_float3x3(const float3 &v1, const float3 &v2, const float3 &v3) {
  return float3x3(v1, v2, v3);
}

__host__ __device__ float3x3 make_float3x3(float s) {
  return float3x3(make_float3(s, 0.0f, 0.0f),
                  make_float3(0.0, s, 0.0f),
                  make_float3(0.0f, 0.0f, s));

}

template<class T> __host__ __device__ inline T compensatedSum(T val, T &sum, T &error) {
  T y = val - error;
  T t = sum + y;
  error = (t - sum) - y;
  sum = t;
  return sum;
}

// clang-format off
///////////////____R____///////////////
__device__ float3 r4_to_float3(uint32_t rgb) {
  return make_float3(
    ((rgb >> 0) & 0xF) / 15.0f,
    0.0f,
    0.0f
  );
}
__device__ float3 r8_to_float3(uint32_t rgb) {
  return make_float3(
    ((rgb >> 0) & 0xFF) / 255.0f,
    0.0f,
    0.0f
  );
}
__device__ float3 r16_to_float3(uint32_t rgb) {
  return make_float3(
    ((rgb >> 0) & 0xFFFF) / 65535.0f,
    0.0f,
    0.0f
  );
}
///////////////____RG____///////////////
__device__ float3 rg88_to_float3(uint32_t rgb) {
  return make_float3(
    ((rgb >> 0) & 0xFF) / 255.0f,
    ((rgb >> 8) & 0xFF) / 255.0f,
    0.0f
  );
}
__device__ float3 rg1616_to_float3(uint32_t rgb) {
  return make_float3(
    ((rgb >> 0) & 0xFFFF) / 65535.0f,
    ((rgb >> 16) & 0xFFFF) / 65535.0f,
    0.0f
  );
}
///////////////____RGB____///////////////
__device__ float3 rgb888_to_float3(uint32_t rgb) {
  return make_float3(
    ((rgb >> 0) & 0xFF) / 255.0f,
    ((rgb >> 8) & 0xFF) / 255.0f,
    ((rgb >> 16) & 0xFF) / 255.0f
  );
}

__device__  float3 rgb101210_to_float3(uint32_t rgb) {
  return make_float3(
    ((rgb >> 0) & 0x3FF) / 1023.0f,
    ((rgb >> 10) & 0xFFF) / 4095.0f,
    ((rgb >> 22) & 0x3FF) / 1023.0f
  );
}

__device__  float3 rgb565_to_float3(uint32_t rgb) {
  return make_float3(
    ((rgb >> 0) & 0x1F) / 31.0f,
    ((rgb >> 5) & 0x3F) / 63.0f,
    ((rgb >> 11) & 0x1F) / 31.0f
  );
}

///////////////____R____///////////////
__device__  uint32_t float3_to_r4(float3 c) {
  float R = min(1.0f, max(0.0f, c.x));
  return
    (uint32_t(round(R * 15.0f)) << 0);
}

__device__  uint32_t float3_to_r8(float3 c) {
  float R = min(1.0f, max(0.0f, c.x));
  return
    (uint32_t(round(R * 255.0f)) << 0);
}

__device__  uint32_t float3_to_r16(float3 c) {
  float R = min(1.0f, max(0.0f, c.x));
  return
    (uint32_t(round(R * 65535.0f)) << 0);
}
///////////////____RG____///////////////
__device__  uint32_t float3_to_rg88(float3 c) {
  float R = min(1.0f, max(0.0f, c.x));
  float G = min(1.0f, max(0.0f, c.y));
  return
    (uint32_t(round(R * 255.0f)) << 0) |
    (uint32_t(round(G * 255.0f)) << 8);
}

__device__  uint32_t float3_to_rg1616(float3 c) {
  float R = min(1.0f, max(0.0f, c.x));
  float G = min(1.0f, max(0.0f, c.y));
  return
    (uint32_t(round(R * 65535.0f)) << 0) |
    (uint32_t(round(G * 65535.0f)) << 16);
}
///////////////____RGB____///////////////
__device__  uint32_t float3_to_rgb888(float3 c) {
  float R = min(1.0f, max(0.0f, c.x));
  float G = min(1.0f, max(0.0f, c.y));
  float B = min(1.0f, max(0.0f, c.z));
  return
    (uint32_t(round(R * 255.0f)) << 0) |
    (uint32_t(round(G * 255.0f)) << 8) |
    (uint32_t(round(B * 255.0f)) << 16);
}

__device__  uint32_t float3_to_rgb101210(float3 c) {
  float R = min(1.0f, max(0.0f, c.x));
  float G = min(1.0f, max(0.0f, c.y));
  float B = min(1.0f, max(0.0f, c.z));
  return
    (uint32_t(round(R * 1023.0f)) << 0) |
    (uint32_t(round(G * 4095.0f)) << 10) |
    (uint32_t(round(B * 1023.0f)) << 22);
}

__device__  uint32_t float3_to_rgb565(float3 c) {
  float R = min(1.0f, max(0.0f, c.x));
  float G = min(1.0f, max(0.0f, c.y));
  float B = min(1.0f, max(0.0f, c.z));
  return
    (uint32_t(round(R * 31.0f)) << 0) |
    (uint32_t(round(G * 63.0f)) << 5) |
    (uint32_t(round(B * 31.0f)) << 11);
}

__device__ uint32_t float3_to_rgbxxx(float3 c, ColorLayout layout) {
  switch (layout)
  {
  case R_4:          return float3_to_r4(c);
  case R_8:          return float3_to_r8(c);
  case R_16:         return float3_to_r16(c);
  case RG_8_8:       return float3_to_rg88(c);
  case RG_16_16:     return float3_to_rg1616(c);
  case RGB_8_8_8:    return float3_to_rgb888(c);
  case RGB_10_12_10: return float3_to_rgb101210(c);
  case RGB_5_6_5:    return float3_to_rgb565(c);
  default: break;
  }
  return 0;
}

__device__ float3 rgbxxx_to_float3(uint32_t rgb, ColorLayout layout) {
  switch (layout)
  {
  case R_4:          return r4_to_float3(rgb);
  case R_8:          return r8_to_float3(rgb);
  case R_16:         return r16_to_float3(rgb);
  case RG_8_8:       return rg88_to_float3(rgb);
  case RG_16_16:     return rg1616_to_float3(rgb);
  case RGB_8_8_8:    return rgb888_to_float3(rgb);
  case RGB_10_12_10: return rgb101210_to_float3(rgb);
  case RGB_5_6_5:    return rgb565_to_float3(rgb);
  default: break;
  }
  return make_float3(0.f, 0.f, 0.f);
}

__device__ float3 minmaxCorrectedColor(const float3 &c, ColorLayout layout) {
  return rgbxxx_to_float3(float3_to_rgbxxx(c, layout), layout);
}

__device__ float3 minmaxSingleCorrectedColor(const float3 &c, ColorLayout layout) {
  ColorLayout single_color_layout;
  switch (layout)
  {
  case R_4:       single_color_layout = R_8;          break;
  case R_8:       single_color_layout = R_16;         break;
  case RG_8_8:    single_color_layout = RG_16_16;     break;
  case RGB_5_6_5: single_color_layout = RGB_10_12_10; break;
  default:        single_color_layout = NONE;         break;
  }
  return minmaxCorrectedColor(c, single_color_layout);
}
// clang-format on

///////////////////////////////////////////////////////////////////////
// Get the "error" between two colors. Should be perceptually sane. 
///////////////////////////////////////////////////////////////////////
__device__ __forceinline
float3 minmax_correctred(const float3 &c)
{
  return rgb888_to_float3(float3_to_rgb888(c));
}

__device__ __forceinline
float getErrorSquared(const float3 & c1, const float3 & c2, bool minmax_correction)
{
  const float3 err_vec = minmax_correction ?
    minmax_correctred(c1) - minmax_correctred(c2) :
    c1 - c2;
  return
    err_vec.x * err_vec.x +
    err_vec.y * err_vec.y +
    err_vec.z * err_vec.z;
};

__device__ __forceinline
float getError(const float3 & c1, const float3 & c2, bool minmax_correction)
{
  return sqrt(getErrorSquared(c1, c2, minmax_correction));
};



#define FULL_MASK 0xffffffff

__device__ inline float warpMin(float x) {
  x = min(x, __shfl_down_sync(FULL_MASK, x, 16));
  x = min(x, __shfl_down_sync(FULL_MASK, x, 8));
  x = min(x, __shfl_down_sync(FULL_MASK, x, 4));
  x = min(x, __shfl_down_sync(FULL_MASK, x, 2));
  x = min(x, __shfl_down_sync(FULL_MASK, x, 1));
  return __shfl_sync(FULL_MASK, x, 0);
}

__device__ inline float warpMax(float x) {
  x = max(x, __shfl_down_sync(FULL_MASK, x, 16));
  x = max(x, __shfl_down_sync(FULL_MASK, x, 8));
  x = max(x, __shfl_down_sync(FULL_MASK, x, 4));
  x = max(x, __shfl_down_sync(FULL_MASK, x, 2));
  x = max(x, __shfl_down_sync(FULL_MASK, x, 1));
  return __shfl_sync(FULL_MASK, x, 0);
}

__device__ inline float warpSum(float x) {
  x += __shfl_down_sync(FULL_MASK, x, 16);
  x += __shfl_down_sync(FULL_MASK, x, 8);
  x += __shfl_down_sync(FULL_MASK, x, 4);
  x += __shfl_down_sync(FULL_MASK, x, 2);
  x += __shfl_down_sync(FULL_MASK, x, 1);
  return __shfl_sync(FULL_MASK, x, 0);
}

__device__ inline float3 warpSum(float3 v) {
  v.x = warpSum(v.x);
  v.y = warpSum(v.y);
  v.z = warpSum(v.z);
  return v;
}

__device__ inline float3x3 warpSum(float3x3 m) {
  m.c1 = warpSum(m.c1);
  m.c2 = warpSum(m.c2);
  m.c3 = warpSum(m.c3);
  return m;
}

template<bool minmaxcorrection, bool laberr>
__global__ void scorefunction_gpu_warp(
  int numColors,
  int numBlocks,
  const float3 * colors,
  const BlockBuild * blocks,
  float * scores,
  uint8_t * weights,
  float3 * colorRanges,
  float error_treshold,
  ColorLayout layout,
  int K,
  int * globalJobQueue,
  bool finalEval = false
)
{
  int laneId = threadIdx.x & 31;

  int jobId = INT32_MAX;
  if (laneId == 0)
  {
    jobId = atomicSub(globalJobQueue, 1);
  }

  jobId = __shfl_sync(FULL_MASK, jobId, 0);

  while (__any_sync(FULL_MASK, jobId >= 0))
  {
    BlockBuild currblock = blocks[jobId];
    BlockBuild nextblock = blocks[jobId + 1];

    int start = currblock.blockStart;
    int range = currblock.blockLength + (finalEval ? 0 : nextblock.blockLength);
    // start + range <= numColors
    range = min(range, numColors - start);

    float3 minpoint, maxpoint;

    if (K > 1)
    {
      // fit
      if (range == 1)
      {
        minpoint = colors[start];
        maxpoint = colors[start];
      }
      else if (range == 2)
      {
        minpoint = colors[start];
        maxpoint = colors[start + 1];
      }
      else
      {
        float3 o, d;
        // least square fit
        {
          o = make_float3(0.0f, 0.0f, 0.0f);
          float3 error = o;
          for (int i = start + laneId; i < start + range; i += 32)
          {
            o = compensatedSum(colors[i], o, error);
          }
          o = warpSum(o);

          o = (1.0f / range) * o;

          float3 zeros = make_float3(0.0f, 0.0f, 0.0f);
          float3x3 scatterMatrix = make_float3x3(zeros, zeros, zeros);

          for (int i = start + laneId; i < start + range; i += 32)
          {
            float3 relpos = colors[i] - o;
            float3x3 outerProd =
              make_float3x3(
                relpos.x * relpos,
                relpos.y * relpos,
                relpos.z * relpos
              );
            scatterMatrix = scatterMatrix + outerProd;
          }

          scatterMatrix = warpSum(scatterMatrix);

          // force dead-code elimination since matrix is symmetric
          scatterMatrix.c1.y = scatterMatrix.c2.x;
          scatterMatrix.c1.z = scatterMatrix.c3.x;
          scatterMatrix.c2.z = scatterMatrix.c3.y;
          if (trace(scatterMatrix) == 0.f)
          {
            d = normalize(o - minmaxCorrectedColor(o, layout));
            minpoint = o;
            maxpoint = o;
          }
          else
          {

            // power method to find eigenvector of largest eigenvalue
            unsigned randidx = 0;
            float3 v = make_float3(
              rand_lut[randidx++ % 64],
              rand_lut[randidx++ % 64],
              rand_lut[randidx++ % 64]
            );
            for (int i = 0; i < 20; i++)
            {
              if (length(v) == 0.f)
              {
                v = make_float3(
                  rand_lut[randidx++ % 64],
                  rand_lut[randidx++ % 64],
                  rand_lut[randidx++ % 64]
                );
                i = 0;
              }
              v = scatterMatrix * normalize(v);
            }
            float3 eigenvector = normalize(v);
            d = eigenvector;
          }
        }

        float mindist = FLT_MAX;
        float maxdist = -FLT_MAX;
        for (int i = start + laneId; i < start + range; i += 32)
        {
          float distance = dot(colors[i] - o, d);
          mindist = min(mindist, distance);
          maxdist = max(maxdist, distance);
        }
        mindist = warpMin(mindist);
        maxdist = warpMax(maxdist);

        minpoint = o + mindist * d;
        maxpoint = o + maxdist * d;
      }

      if (minmaxcorrection)
      {
        minpoint = minmaxCorrectedColor(minpoint, layout);
        maxpoint = minmaxCorrectedColor(maxpoint, layout);
      }
    }
    else
    {
      float3 o = make_float3(0.0f, 0.0f, 0.0f);
      float3 error = o;
      for (int i = start + laneId; i < start + range; i += 32)
      {
        o = compensatedSum(colors[i], o, error);
      }
      o = warpSum(o);

      o = (1.0f / range) * o;
      if (minmaxcorrection)
      {
        o = minmaxSingleCorrectedColor(o, layout);
      }
      minpoint = o;
      maxpoint = o;
    }


    if (finalEval && (laneId == 0))
    {
      colorRanges[2 * jobId + 0] = minpoint;
      colorRanges[2 * jobId + 1] = maxpoint;
    }
    // ~fit

    // evaluate
    bool bEval = true;
    float mse = 0.0f;
    if (K > 1)
    {
      if (range == 1)
      {
        if (getError(minpoint, colors[start], minmaxcorrection) > error_treshold ||
            getError(maxpoint, colors[start], minmaxcorrection) > error_treshold)
        {
          bEval = false;
        }
      }
      else if (range == 2)
      {
        if (getError(minpoint, colors[start], minmaxcorrection) > error_treshold ||
            getError(maxpoint, colors[start + 1], minmaxcorrection) > error_treshold)
        {
          bEval = false;
        }
      }
      else
      {
        float msesum = 0.0f;

        const float3 & A = minpoint;
        const float3 & B = maxpoint;
        float colorRangeInvSq = 1.0f / dot(B - A, B - A);

        for (int i = start + laneId; i < start + range; i += 32)
        {
          const float3 & p = colors[i];
          float distance = 0.0f;
          if (length(B - A) > (1e-4f))
          {
            distance = colorRangeInvSq * dot(p - A, B - A);
          }

          float w = clamp(round(distance * float(K - 1)), 0.0f, float(K - 1));
          float3 interpolated_color = A + w / float(K - 1) * (B - A);

          float error = getError(p, interpolated_color, minmaxcorrection);
          msesum += getErrorSquared(p, interpolated_color, minmaxcorrection);

          if (error > error_treshold)
          {
            bEval = false;
          }
          if (finalEval)
          {
            weights[i] = w;
          }
        }

        msesum = warpSum(msesum);

        // true iff all bEval are true
        bEval = __all_sync(FULL_MASK, bEval);

        mse = msesum / float(range * 3);
      }
    }
    else
    {
      float msesum = 0.0f;
      for (int i = start + laneId; i < start + range; i += 32)
      {
        const float3 & p = colors[i];
        float3 interpolated_color = minpoint;

        float error = getError(p, interpolated_color, minmaxcorrection);
        msesum += getErrorSquared(p, interpolated_color, minmaxcorrection);
        if (error > error_treshold)
        {
          bEval = false;
        }
      }

      msesum = warpSum(msesum);

      // true iff all bEval are true
      bEval = __all_sync(FULL_MASK, bEval);
      mse = msesum / float(range * 3);
    }
    // ~evaluate

    float score = bEval ? 1.0f / (mse + 1.0f) : -1.0f;

    if (K > 1 && range == 2 && bEval)
    {
      score = 1.0f / (length(colors[start] - colors[start + 1]) + 1.0f);
    }

    if (!isfinite(score))
    {
      score = -1.0f;
    }

    if (laneId == 0)
    {
      scores[jobId] = score;
    }

    // fetch next job
    if (laneId == 0)
      jobId = atomicSub(globalJobQueue, 1);

    jobId = __shfl_sync(FULL_MASK, jobId, 0);
  }// ~while(jobId >= 0)
}// ~scorefunction_gpu_warp

float3 * g_dev_colors = nullptr;
uint8_t * g_dev_weights = nullptr;
size_t g_numColors = 0;

void uploadColors(const vector<float3> &colors)
{
  if (g_dev_weights) cudaFree(g_dev_weights);
  if (g_dev_colors) cudaFree(g_dev_colors);

  g_numColors = colors.size();

  // alloc per-color memory
  cudaMalloc(&g_dev_weights, colors.size() * sizeof(uint8_t));
  cudaMalloc(&g_dev_colors, colors.size() * sizeof(float3));
  cudaMemcpy(g_dev_colors, &colors[0], colors.size() * sizeof(float3), cudaMemcpyHostToDevice);
}

void scores_gpu(
  const vector<BlockBuild> &blocks,
  vector<float> &scores,
  vector<uint8_t> &weights,
  vector<float3> &colorRanges,
  float error_treshold,
  bool minmaxcorrection,
  bool laberr,
  ColorLayout layout,
  int K,
  bool finalEval
)
{
  if (g_numColors == 0)
  {
    // no colors? nothing more to do.
    return;
  }

  static BlockBuild * pBlocks = nullptr;
  static float * pScores = nullptr;
  static float3 * pColorRanges = nullptr;

  // alloc per-block memory
  static size_t blockAllocSize = 0;
  if (blockAllocSize < blocks.size())
  {
    if (pBlocks) cudaFree(pBlocks);
    if (pScores) cudaFree(pScores);
    if (pColorRanges) cudaFree(pColorRanges);

    blockAllocSize = blocks.size();

    cudaMalloc(&pBlocks, blockAllocSize * sizeof(BlockBuild));
    cudaMalloc(&pScores, blockAllocSize * sizeof(float));
    cudaMalloc(&pColorRanges, blockAllocSize * 2 * sizeof(float3));
  }

  // upload blocks
  cudaMemcpy(pBlocks, &blocks[0], blocks.size() * sizeof(BlockBuild), cudaMemcpyHostToDevice);

  static int *jobQueue = nullptr;
  if (!jobQueue)
  {
    cudaMalloc(&jobQueue, sizeof(int));
  }
  int jobs = int(blocks.size()) - 1; // first job is the last valid idx.
  cudaMemcpy(jobQueue, &jobs, sizeof(int), cudaMemcpyHostToDevice);

  {
    dim3 blockDim(128);
    dim3 gridDim(20 * 16);
    // reduce register preassure via templates
    if (minmaxcorrection)
    {
      scorefunction_gpu_warp<true, false> << <gridDim, blockDim >> > (
        int(g_numColors),
        int(blocks.size()),
        g_dev_colors,
        pBlocks,
        pScores,
        g_dev_weights,
        pColorRanges,
        error_treshold,
        layout,
        K,
        jobQueue,
        finalEval
        );
    }
  }
  scores.resize(blocks.size());
  cudaMemcpy(&scores[0], pScores, blocks.size() * sizeof(float), cudaMemcpyDeviceToHost);
  if (finalEval)
  {
    colorRanges.resize(2 * blocks.size());
    cudaMemcpy(&colorRanges[0], pColorRanges, blocks.size() * 2 * sizeof(float3), cudaMemcpyDeviceToHost);

    weights.resize(g_numColors);
    cudaMemcpy(&weights[0], g_dev_weights, g_numColors * sizeof(uint8_t), cudaMemcpyDeviceToHost);
  }
}

