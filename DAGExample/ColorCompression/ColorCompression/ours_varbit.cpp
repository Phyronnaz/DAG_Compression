#include "ours_varbit.h"
#include "../bits_in_uint_array.h"
//#include "colorspace.h"
#include "svd.h"

#include <algorithm>
#include <array>
#include <inttypes.h>
#include <sstream>
#include <iostream>
#include <fstream>
#include <tuple>
#include <chrono>

#include <glm/glm.hpp>

#define _USE_MATH_DEFINES
#include <math.h>

#include <csignal>

#define DEBUG_ERROR false

using namespace std;
using glm::vec3;

namespace ours_varbit {
  // clang-format off
  ///////////////____R____///////////////
  vec3 r4_to_float3(uint32_t rgb)
  {
    return vec3(
      float((rgb >> 0) & 0xF) / 15.0f,
      0.0f,
      0.0f
    );
  }

  vec3 r8_to_float3(uint32_t rgb)
  {
    return vec3(
      float((rgb >> 0) & 0xFF) / 255.0f,
      0.0f,
      0.0f
    );
  }

  vec3 r16_to_float3(uint32_t rgb)
  {
    return vec3(
      float((rgb >> 0) & 0xFFFF) / 65535.0f,
      0.0f,
      0.0f
    );
  }
  ///////////////____RG____///////////////
  vec3 rg88_to_float3(uint32_t rgb)
  {
    return vec3(
      ((rgb >> 0) & 0xFF) / 255.0f,
      ((rgb >> 8) & 0xFF) / 255.0f,
      0.0f
    );
  }

  vec3 rg1616_to_float3(uint32_t rgb)
  {
    return vec3(
      ((rgb >> 0) & 0xFFFF) / 65535.0f,
      ((rgb >> 16) & 0xFFFF) / 65535.0f,
      0.0f
    );
  }
  ///////////////____RGB____///////////////
  vec3 rgb888_to_float3(uint32_t rgb)
  {
    return vec3(
      ((rgb >> 0) & 0xFF) / 255.0f,
      ((rgb >> 8) & 0xFF) / 255.0f,
      ((rgb >> 16) & 0xFF) / 255.0f
    );
  }

  vec3 rgb101210_to_float3(uint32_t rgb)
  {
    return vec3(
      ((rgb >> 0) & 0x3FF) / 1023.0f,
      ((rgb >> 10) & 0xFFF) / 4095.0f,
      ((rgb >> 22) & 0x3FF) / 1023.0f
    );
  }

  vec3 rgb565_to_float3(uint32_t rgb)
  {
    return vec3(
      ((rgb >> 0) & 0x1F) / 31.0f,
      ((rgb >> 5) & 0x3F) / 63.0f,
      ((rgb >> 11) & 0x1F) / 31.0f
    );
  }

  ///////////////____R____///////////////
  uint32_t float3_to_r4(vec3 c)
  {
    float R = min(1.0f, max(0.0f, c.x));
    return
      (uint32_t(round(R * 15.0f)) << 0);
  }

  uint32_t float3_to_r8(vec3 c)
  {
    float R = min(1.0f, max(0.0f, c.x));
    return
      (uint32_t(round(R * 255.0f)) << 0);
  }

  uint32_t float3_to_r16(vec3 c)
  {
    float R = min(1.0f, max(0.0f, c.x));
    return
      (uint32_t(round(R * 65535.0f)) << 0);
  }
  ///////////////____RG____///////////////
  uint32_t float3_to_rg88(vec3 c)
  {
    float R = min(1.0f, max(0.0f, c.x));
    float G = min(1.0f, max(0.0f, c.y));
    return
      (uint32_t(round(R * 255.0f)) << 0) |
      (uint32_t(round(G * 255.0f)) << 8);
  }

  uint32_t float3_to_rg1616(vec3 c)
  {
    float R = min(1.0f, max(0.0f, c.x));
    float G = min(1.0f, max(0.0f, c.y));
    return
      (uint32_t(round(R * 65535.0f)) << 0) |
      (uint32_t(round(G * 65535.0f)) << 16);
  }
  ///////////////____RGB____///////////////
  uint32_t float3_to_rgb888(vec3 c)
  {
    float R = min(1.0f, max(0.0f, c.x));
    float G = min(1.0f, max(0.0f, c.y));
    float B = min(1.0f, max(0.0f, c.z));
    return
      (uint32_t(round(R * 255.0f)) << 0) |
      (uint32_t(round(G * 255.0f)) << 8) |
      (uint32_t(round(B * 255.0f)) << 16);
  }

  uint32_t float3_to_rgb101210(vec3 c)
  {
    float R = min(1.0f, max(0.0f, c.x));
    float G = min(1.0f, max(0.0f, c.y));
    float B = min(1.0f, max(0.0f, c.z));
    return
      (uint32_t(round(R * 1023.0f)) << 0) |
      (uint32_t(round(G * 4095.0f)) << 10) |
      (uint32_t(round(B * 1023.0f)) << 22);
  }

  uint32_t float3_to_rgb565(vec3 c)
  {
    float R = min(1.0f, max(0.0f, c.x));
    float G = min(1.0f, max(0.0f, c.y));
    float B = min(1.0f, max(0.0f, c.z));
    return
      (uint32_t(round(R * 31.0f)) << 0) |
      (uint32_t(round(G * 63.0f)) << 5) |
      (uint32_t(round(B * 31.0f)) << 11);
  }

  uint32_t float3_to_rgbxxx(vec3 c, ColorLayout layout)
  {
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

  vec3 rgbxxx_to_float3(uint32_t rgb, ColorLayout layout)
  {
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
    return vec3(0.f, 0.f, 0.f);
  }

  vec3 minmaxCorrectedColor(const vec3 &c, ColorLayout layout)
  {
    return rgbxxx_to_float3(float3_to_rgbxxx(c, layout), layout);
  }

  vec3 minmaxSingleCorrectedColor(const vec3 &c, ColorLayout layout)
  {
    ColorLayout single_color_layout;
    switch (layout)
    {
    case R_4: single_color_layout = R_8; break;
    case R_8: single_color_layout = R_16; break;
    case RG_8_8: single_color_layout = RG_16_16;  break;
    case RGB_5_6_5: single_color_layout = RGB_10_12_10; break;
    default: single_color_layout = NONE; break;
    }
    return minmaxCorrectedColor(c, single_color_layout);
  }
  // clang-format on

  struct end_block
  {
    using float3 = vec3;
    float3 minpoint;
    float3 maxpoint;
    uint32_t bpw;
    size_t start_node;
    size_t range;
  };

  struct CompressionInfo
  {
    vector<int> wrong_colors;
    vector<int> ok_colors;
    std::size_t total_bits = 0;
    std::size_t nof_blocks = 0;
    std::size_t weights_size = 0;
    std::size_t macro_header_size;
    std::size_t headers_size;
    std::size_t colors_size;
    double max_error = 0.0;
  };

  class CompressionState
  {
  public:
    const unsigned bits_per_weight = 4;
    const int K = 1 << bits_per_weight;

    const int COLOR_COST = 16 + 16;
    const int START_IDX_COST = 14;
    const int WEIGHT_IDX_COST = 16;
    const int BPW_ID_COST = 2;
    const int HEADER_COST = 32;

    explicit CompressionState(disc_vector<uint32_t>&& original_colors, const float error_treshold_, const ColorLayout layout)
      : original_colors_ref{ std::move(original_colors) }
      , error_treshold{ error_treshold_ }
      , compression_layout{ layout }
    {
      const size_t n_colors = original_colors.size();
      const uint64_t bits_required = n_colors * bits_per_weight;

      switch (layout)
      {
      case ColorLayout::RGB_10_12_10:
      case ColorLayout::RG_16_16:
        const_cast<int&>(COLOR_COST) = 32 + 32;
        break;
      case ColorLayout::RGB_8_8_8:
        const_cast<int&>(COLOR_COST) = 24 + 24;
        break;
      case ColorLayout::RG_8_8:
      case ColorLayout::RGB_5_6_5:
      case ColorLayout::R_16:
        const_cast<int&>(COLOR_COST) = 16 + 16;
        break;
      case ColorLayout::R_8:
        const_cast<int&>(COLOR_COST) = 8 + 8;
      case ColorLayout::R_4:
        const_cast<int&>(COLOR_COST) = 4 + 4;
        break;
      }

      m_weights.resize(n_colors);

      h_weights.resize(((bits_required - 1ull) / 32ull) + 1);
      h_block_headers.reserve(10000000);
      h_macro_block_headers.reserve(100000);
    }

    tuple<CompressionInfo, OursData> compress();

    vector<end_block> compress_range(size_t part_start, size_t part_size);

    double add_to_final(
      const vector<end_block>& solution,
      size_t& global_bptr,
      size_t& macro_w_bptr,
      vector<int>& wrong_colors,
      vector<int>& ok_colors
    );

    bool assign_weights(
      size_t start,
      size_t range,
      const vec3& A,
      const vec3& B,
      const float error_treshold,
      int vals_per_weight,
      double* max_error = nullptr,
      double* mse = nullptr
    );

    float getError(const vec3& a_, const vec3& b_);
    float getErrorSquared(const vec3& a_, const vec3& b_);
    vec3 ref_color(std::size_t start)
    {
      switch (compression_layout)
      {
      case R_8:
      case R_4:
        return r8_to_float3(original_colors_ref[start]);
      case RG_8_8:
        return rg88_to_float3(original_colors_ref[start]);
      case RGB_5_6_5:
      default:
        return rgb888_to_float3(original_colors_ref[start]);
      }
    };

    ColorLayout compression_layout;
    const float error_treshold;

    bool use_LAB_error = false;
    bool use_minmax_correction = true;

    disc_vector<uint32_t> original_colors_ref;
    vector<uint32_t> m_weights;

    vector<uint32_t> h_weights;
    vector<uint32_t> h_block_headers;
    vector<uint8_t> h_block_colors;
    vector<uint64_t> h_macro_block_headers;
  };

  bool
    getErrInfo(
      const vector<uint32_t>& colors,
      const string filename,
      const ColorLayout layout,
      float* mse,
      float* maxR,
      float* maxG,
      float* maxB,
      float* maxLength
    )
  {
    /////////////////////////////////////////////////////////////////////////////
    //// Read the data from disk...
    /////////////////////////////////////////////////////////////////////////////
    //ifstream is(filename, ios::binary);
    //if (!is.good())
    //  return false;

    //CacheHeader nfo;
    //is.read(reinterpret_cast<char*>(&nfo), sizeof(CacheHeader));

    //const uint32_t colors_per_macro_block = 16 * 1024;
    //const unsigned required_macro_blocks =
    //  (nfo.nof_colors + colors_per_macro_block - 1) / colors_per_macro_block;
    //streamsize macro_offset_size =
    //  sizeof(uint64_t) * 2 * required_macro_blocks;

    //vector<uint64_t> macro_block_offset(macro_offset_size / sizeof(uint64_t));
    //uint32_t macro_block_count = macro_block_offset.size() / 2;
    //is.read(reinterpret_cast<char*>(macro_block_offset.data()),
    //        macro_offset_size);

    //cout << "No block header, rewrite this code." << __LINE__ << " " << __FILE__ << '\n';
    //vector<uint32_t> block_headers(nfo.headers_size / sizeof(uint32_t));
    //uint32_t ours_nof_block_headers = block_headers.size() / 2;
    //is.read(reinterpret_cast<char*>(block_headers.data()), nfo.headers_size);

    //vector<uint32_t> weights(nfo.weights_size / sizeof(uint32_t));
    //is.read(reinterpret_cast<char*>(weights.data()), nfo.weights_size);
    //is.close();

    /////////////////////////////////////////////////////////////////////////////
    //// Get next color...
    /////////////////////////////////////////////////////////////////////////////
    //size_t position = 0;
    //uint32_t color_idx = 0;
    //uint64_t ours_nof_colors = colors.size();
    //auto getNextColor = [&]() {
    //  const uint32_t colors_per_macro_block = 16 * 1024;
    //  size_t macro_block_idx = color_idx / colors_per_macro_block;
    //  size_t local_color_idx = color_idx % colors_per_macro_block;

    //  uint32_t block_idx_macro = macro_block_offset[2 * macro_block_idx + 0];
    //  uint64_t w_bptr_macro = macro_block_offset[2 * macro_block_idx + 1];
    //  uint32_t block_idx_macro_upper = ours_nof_block_headers - 1;
    //  uint32_t macro_block_count =
    //    (ours_nof_colors + colors_per_macro_block - 1) / colors_per_macro_block;

    //  if (macro_block_idx < macro_block_count - 1)
    //    block_idx_macro_upper =
    //      macro_block_offset[2ull * macro_block_idx + 2] - 1;

    //  ///////////////////////////////////////////////////////////////////////////
    //  // Binary search through headers to find the block containing my node
    //  ///////////////////////////////////////////////////////////////////////////
    //  const int header_size = 2;
    //  int position, lowerbound = block_idx_macro,
    //                upperbound = block_idx_macro_upper;
    //  position = (lowerbound + upperbound) / 2;

    //  uint32_t header0 = block_headers[(size_t)position * header_size];
    //  uint32_t block_start_color = (header0 & (colors_per_macro_block - 1));
    //  while (block_start_color != local_color_idx && (lowerbound <= upperbound)) {
    //    if (block_start_color > local_color_idx) {
    //      upperbound = position - 1;
    //    } else {
    //      lowerbound = position + 1;
    //    }

    //    position = (lowerbound + upperbound) / 2;
    //    header0 = block_headers[(size_t)position * header_size];
    //    block_start_color = (header0 & (colors_per_macro_block - 1));
    //  }

    //  ///////////////////////////////////////////////////////////////////////////
    //  // Fetch min/max and weight and interpolate out the color
    //  ///////////////////////////////////////////////////////////////////////////

    //  int w_local = header0 >> 16;
    //  int bpw = 0;
    //  if (w_local < UINT16_MAX) {
    //    bpw = ((header0 >> 14) & 0x3) + 1;
    //  }

    //  uint32_t header1 = block_headers[(size_t)position * header_size + 1];

    //  vec3 mincolor, maxcolor;
    //  if (bpw > 0) {
    //    mincolor = rgb565_to_float3(header1 & 0xFFFF);
    //    maxcolor = rgb565_to_float3((header1 >> 16) & 0xFFFF);
    //  } else {
    //    mincolor = rgb101210_to_float3(
    //      header1); // to be replaced by higher precision color
    //  }

    //  uint32_t block_weight_offset = (local_color_idx - block_start_color) * bpw;
    //  uint64_t weight_idx = w_bptr_macro + w_local + block_weight_offset;
    //  int weight = extract_bits(bpw, weights.data(), weight_idx);

    //  bool is_single_color_block = (bpw == 0);
    //  vec3 decompressed_color =
    //    is_single_color_block
    //      ? mincolor
    //      : mincolor + (weight / float((1 << bpw) - 1)) * (maxcolor - mincolor);
    //  ++color_idx;
    //  return float3_to_rgb888(decompressed_color);
    //};

    /////////////////////////////////////////////////////////////////////////////
    //// Actually compute MSE...
    /////////////////////////////////////////////////////////////////////////////
    //auto to_float3 = [](uint32_t rgb) {
    //  return vec3(((rgb >> 0) & 0xFF), ((rgb >> 8) & 0xFF), ((rgb >> 16) & 0xFF));
    //};
    //size_t N = colors.size();
    //double errsq = 0.f;
    //// float errsq = 0.f; // original submission. Precision problems.

    //double max_errR = -numeric_limits<double>::max();
    //double max_errG = -numeric_limits<double>::max();
    //double max_errB = -numeric_limits<double>::max();
    //double max_length = -numeric_limits<double>::max();

    //for (size_t i = 0; i < N; ++i) {
    //  auto a3 = to_float3(colors[i]);
    //  auto b3 = to_float3(getNextColor());
    //  double errR = double(abs(a3.x - b3.x));
    //  double errG = double(abs(a3.y - b3.y));
    //  double errB = double(abs(a3.z - b3.z));

    //  errsq += errR * errR + errG * errG + errB * errB;

    //  // errR /= 255.f;
    //  // errG /= 255.f;
    //  // errB /= 255.f;
    //  // a3 /= 255.f;
    //  // b3 /= 255.f;

    //  max_errR = max(errR, max_errR);
    //  max_errG = max(errG, max_errG);
    //  max_errB = max(errB, max_errB);
    //  max_length = max(double(glm::length(a3 - b3)), max_length);
    //}

    //*mse = errsq / double(N * 3);
    //*maxR = max_errR;
    //*maxG = max_errG;
    //*maxB = max_errB;
    //*maxLength = max_length;
    //return true;
    return false;
  }

  float getPSNR(float mse)
  {
    return mse > 0.f ? 20.f * log10(255.f) - 10.f * log10(mse) : -1.f;
  }

  ///////////////////////////////////////////////////////////////////////
  // Get the "error" between two colors. Should be perceptually sane.
  ///////////////////////////////////////////////////////////////////////
  vec3 minmax_correctred(const vec3 &c)
  {
    return rgb888_to_float3(float3_to_rgb888(c));
  }

  float CompressionState::getError(const vec3& c1, const vec3& c2)
  {
    return sqrt(getErrorSquared(c1, c2));
  };

  float CompressionState::getErrorSquared(const vec3& c1, const vec3& c2)
  {
    const vec3 err_vec = use_minmax_correction ?
      minmax_correctred(c1) - minmax_correctred(c2) :
      c1 - c2;
    return
      err_vec.x * err_vec.x +
      err_vec.y * err_vec.y +
      err_vec.z * err_vec.z;
  };

  ///////////////////////////////////////////////////////////////////////////
  // Evaluate if a range of colors can be represented as interpolations of
  // two given min and max points, and update the corresponding weights.
  ///////////////////////////////////////////////////////////////////////////
  bool CompressionState::assign_weights(
    size_t start,
    size_t range,
    const vec3& A,
    const vec3& B,
    const float error_treshold,
    int vals_per_weight,
    double* max_error,
    double* mse
  )
  {
    int K = vals_per_weight;
    // Trivial block of length 1.
    if (range == 1)
    {
      m_weights[start] = 0;
      if (max_error != nullptr)
      {
        *max_error = 0.0;
      }
      if (mse != nullptr)
      {
        *mse = 0.0f;
      }

      if (DEBUG_ERROR)
      {
        const vec3 p = ref_color(start);
        if (getError(A, p) > error_treshold || getError(B, p) > error_treshold)
        {
          cout << "1:" << K << '\n';
          return false;
        }
      }
      return true;
    }
    // Trivial block of length 2.
    else if (range == 2 && K > 1)
    {
      m_weights[start] = 0;
      m_weights[start + 1] = K - 1;
      if (max_error != nullptr)
      {
        *max_error = 0.0;
      }
      if (mse != nullptr)
      {
        *mse = 0.0f;
      }
      if (DEBUG_ERROR)
      {
        const vec3 p1 = ref_color(start);
        const vec3 p2 = ref_color(start + 1);
        if (getError(A, p1) > error_treshold || getError(B, p2) > error_treshold)
        {
          cout << "2\n";
          return false;
        }
      }
      return true;
    }

    // Blocks of non-trivial configurations.
    if (max_error != nullptr)
    {
      *max_error = numeric_limits<double>::lowest();
    }

    double msesum = 0.0;
    bool bEval = true;
    // Multi color blocks.
    if (K > 1)
    {
      for (std::size_t i = start; i < start + range; i++)
      {
        if (i >= original_colors_ref.size())
        {
          cout << "YOU HAVE MESSED UP YOU SILLY GOOSE!\n";
        }
        const vec3 p = ref_color(i);

        // Since A and B can be extremely close, we need to bail out
        // This is safe since we are talking about colors that will be equal when
        // truncated.
        const float distance =
          length(B - A) < (1e-4f) ?
          0.0f :
          dot(p - A, B - A) / dot(B - A, B - A);

        auto calc_w = [&](const float distance)
        {
          const int max_w = K - 1;
          const int w = static_cast<int>(round(distance * float(max_w)));
          return min(max(w, 0), max_w);
        };

        auto error_w = [&](const int w)
        {
          const int max_w = K - 1;
          const float t = float(w) / float(max_w);
          const vec3 interpolated_color = A + t * (B - A);
          return getError(p, interpolated_color);
        };

        auto best_w = [&](float distance)
        {
          const int w = calc_w(distance);
          float min_error = error_w(w);
          int result = w;
          // Check against one lower weight.
          {
            const int w_lo = w - 1;
            if (w_lo >= 0)
            {
              const float this_error = error_w(w_lo);
              if (this_error < min_error)
              {
                min_error = this_error;
                result = w_lo;
              }
            }
          }
          // Check against one higher weight.
          {
            const int w_hi = w + 1;
            if (w_hi <= K - 1)
            {
              const float this_error = error_w(w_hi);
              if (this_error < min_error)
              {
                result = w_hi;
              }
            }
          }
          return result;
        };

        const int w = best_w(distance);
        const float t = float(w) / float(K - 1);
        vec3 interpolated_color = A + t * (B - A);
        double error = getError(p, interpolated_color);

        if (max_error != nullptr)
        {
          *max_error = std::max(*max_error, error);
        }
        msesum += getErrorSquared(p, interpolated_color);

        if (DEBUG_ERROR)
        {
          if (error > error_treshold)
          {
            bEval = false;
            cout << "i: " << i << '\n';
            cout << "distance: " << distance << '\n';
            cout << "K-1: " << K - 1 << '\n';
            cout << "p: " << p.x << " " << p.y << " " << p.z << '\n';
            cout << "A: " << A.x << " " << A.y << " " << A.z << '\n';
            cout << "B: " << B.x << " " << B.y << " " << B.z << '\n';
            cout << "w: " << w << '\n';
            cout
              << "interpolated_color: "
              << interpolated_color.x << " "
              << interpolated_color.y << " "
              << interpolated_color.z << '\n';
            cout << "error(corrected): " << error << '\n';
            cout << "error(plain): " << length(p - interpolated_color) << '\n';
            cout << "-----------" << '\n';
          }
        }
        m_weights[i] = w;
      }
    }
    // Single color blocks.
    else
    {
	  for (std::size_t i = start; i < start + range; i++)
      {
        if (i >= original_colors_ref.size())
        {
          cout << "YOU HAVE MESSED UP YOU SILLY GOOSE!\n";
        }
        const vec3 p = ref_color(i);
        vec3 interpolated_color = A;
        double error = getError(p, interpolated_color);
        if (max_error != nullptr)
        {
          *max_error = std::max(*max_error, error);
        }

        msesum += getErrorSquared(p, interpolated_color);
        if (DEBUG_ERROR)
        {
          if (error > error_treshold)
          {
            bEval = false;
          }
        }
        m_weights[i] = 0;
      }
    }

    if (mse != nullptr)
    {
      *mse = static_cast<float>(msesum / double(range * 3));
    }

    if (!bEval)
    {
      cout << "3: " << start << " " << range << '\n';
    }
    return bEval;
  };

  // Recursively prunes the score tree, and return the best cut.
  template<typename end_block, typename block, typename block_score>
  vector<end_block> find_best_nodes(
    vector<vector<block>>& block_tree,
    vector<vector<block_score>>& score_tree,
    vector<vector<vector<uint32_t>>>& children,
    size_t bits_per_weight,
    size_t block_idx
  )
  {
    vector<end_block> result;
    block_score this_score = score_tree[bits_per_weight][block_idx];
    if (this_score.best.total_bit_cost == this_score.my.total_bit_cost)
    {
      // This blocks score is the best score, i.e., the memory overhead
      // of this block is less than the sum of it's children.
      // So keep this block, and prune the children.
      block this_block = block_tree[bits_per_weight][block_idx];
      end_block eb;
      eb.minpoint = this_block.minpoint;
      eb.maxpoint = this_block.maxpoint;
      eb.bpw = uint32_t(bits_per_weight);
      eb.start_node = this_block.start_node;
      eb.range = this_block.range;
      result.push_back(eb);
    }
    else
    {
      // There exists some configuration of children which has less overhead.
      // So traverse each child until we find the cheapest block and append that 
      // to our solution.
      const auto& childs = children[bits_per_weight][block_idx];
      for (const auto& current_child : childs)
      {
        vector<end_block> best_blocks =
          find_best_nodes<end_block>(
            block_tree,
            score_tree,
            children,
            bits_per_weight - 1,
            current_child
            );
        result.insert(result.end(), best_blocks.begin(), best_blocks.end());
      }
    }
    return result;
  };

  // Compress colors using CUDA.
  std::tuple<CompressionInfo, OursData> CompressionState::compress()
  {
    CompressionInfo nfo;
    OursData ours_dat;
    nfo.wrong_colors.resize((size_t)bits_per_weight + 1, 0);
    nfo.ok_colors.resize((size_t)bits_per_weight + 1, 0);

    const size_t n_colors = original_colors_ref.size();
	auto result = (n_colors + macro_block_size - 1) / macro_block_size;
	assert(result < std::numeric_limits<int>::max());
	const int nof_parts = int(result);

    size_t global_bptr = 0;
    size_t macro_w_bptr = 0;

    const auto startTime = std::chrono::high_resolution_clock::now();

    const auto printSeconds = [](uint64_t input_seconds)
    {
        size_t minutes = input_seconds / 60;
        size_t seconds = input_seconds % 60;

        cout << minutes << ":" << seconds << " ";
    };

    for (int part = 0; part < nof_parts; part++)
    {
        if (part % 100 == 0)
        {
            const double elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::high_resolution_clock::now() - startTime).count();
            cout << "Part: "
                 << part
                 << " of "
                 << nof_parts
                 << " header_size: "
                 << h_block_headers.size()
                 << " Elapsed: ";
            printSeconds(uint64_t(elapsed));
            cout << "Remaining: ";
            printSeconds(uint64_t(elapsed * nof_parts / part - elapsed));
            cout << '\n';
        }

      const size_t part_size =
        (part == nof_parts - 1) ?
        (n_colors % macro_block_size) : macro_block_size;

      const size_t part_start = part * macro_block_size;
      const vector<end_block> solution = compress_range(part_start, part_size);
      double max_error_eval =
        add_to_final(
          solution,
          global_bptr,
          macro_w_bptr,
          nfo.wrong_colors,
          nfo.ok_colors
        );

      // Info
      for (const auto& b : solution)
      {
        nfo.total_bits += b.range * b.bpw + HEADER_COST + COLOR_COST;
      }
      nfo.nof_blocks += solution.size();
      nfo.max_error = max(max_error_eval, nfo.max_error);
    }

    // uint32_t elements
    const size_t weight_container_size = (global_bptr + 31) / 32;
    h_weights.resize(weight_container_size);
    nfo.weights_size = weight_container_size * sizeof(uint32_t);
    nfo.macro_header_size = h_macro_block_headers.size() * sizeof(uint64_t);
    nfo.headers_size = h_block_headers.size() * sizeof(uint32_t);
    nfo.colors_size = h_block_colors.size() * sizeof(uint8_t);

    ours_dat.nof_blocks = nfo.nof_blocks;
    ours_dat.nof_colors = n_colors;
    ours_dat.bits_per_weight = bits_per_weight;
    ours_dat.use_single_color_blocks = false;

    ours_dat.h_block_headers = h_block_headers;
    ours_dat.h_block_colors = h_block_colors;

    if (nfo.weights_size != 0)
    {
      ours_dat.h_weights = h_weights;
    }

    ours_dat.h_macro_w_offset = h_macro_block_headers;
    ours_dat.color_layout = compression_layout;

    return { nfo, ours_dat };
  }

  vector<end_block> CompressionState::compress_range(size_t part_start, size_t part_size)
  {
    const int max_bits_per_weight = bits_per_weight;
    const int min_bits_per_weight = 0;

    vector<float3> workingColorSet(part_size);
    for (size_t i = part_start, j = 0;
         i < (part_start + part_size);
         i++, j++)
    {
      const vec3 c = ref_color(i);
      workingColorSet[j] = { c.x, c.y, c.z };
    }

    uploadColors(workingColorSet);
    struct block
    {
      block()
        : start_node(0xBADC0DE) {};
      block(size_t start, size_t rng)
        : start_node(start)
        , range(rng)
        , dirty(true) {};
      size_t start_node;
      size_t range;
      vec3 minpoint;
      vec3 maxpoint;
      bool dirty;
    };

    vector<vector<block>> block_tree((size_t)max_bits_per_weight + 1);
    for (
      int bits_per_weight = min_bits_per_weight;
      bits_per_weight <= max_bits_per_weight;
      bits_per_weight++
      )
    {
      int vals_per_weight = 1 << bits_per_weight;

      size_t block_index = 0;
      vector<BlockBuild> buildBlocks(workingColorSet.size(), BlockBuild(-1));

      // Start with one block per color.
      if (bits_per_weight == min_bits_per_weight)
      {
        for (size_t colorIdx = 0; colorIdx < workingColorSet.size(); ++colorIdx)
        {
          buildBlocks[colorIdx] = BlockBuild(colorIdx);
        }
      }
      // Use the previous bit-rate blocks as a starting point.
      else
      {
        buildBlocks.clear();
        auto& prev_blocks = block_tree[(size_t)bits_per_weight - 1];
        while (block_index < prev_blocks.size())
        {
          block& candidateBlock = prev_blocks[block_index];
		  assert(candidateBlock.start_node >= part_start);
          buildBlocks.push_back(
            BlockBuild(
              candidateBlock.start_node - part_start,
              candidateBlock.range
            )
          );

          block_index++;
        }
      }

      size_t nof_blocks_merged = 1;
      while (nof_blocks_merged > 0)
      {
        nof_blocks_merged = 0;

        // compute scores
        vector<float> scores(buildBlocks.size(), 0.0f);
        vector<float3> colorRanges(buildBlocks.size());
        static int pass = 0;
        vector<uint8_t> weights;
        scores_gpu(
          buildBlocks,
          scores,
          weights,
          colorRanges,
          error_treshold,
          use_minmax_correction,
          use_LAB_error,
          compression_layout,
          vals_per_weight
        );

        cudaError_t err = cudaGetLastError();
		if( cudaSuccess != err ) {
			std::fprintf( stderr, "ERROR %s:%d: cuda error \"%s\"\n", __FILE__, __LINE__, cudaGetErrorString(err) );
		}

        // Loop through all scores and merge with the best one.
        // We start at the second block and jump two (or three) blocks
        // forward each iteration.
        // Example:
        //
        // Original state:
        //  _l_ _i_ _r_ ___ ___ ___
        // |_0_|_1_|_2_|_3_|_4_|_5_|
        //
        // No merge:
        // Two steps.
        // (if 1 can't merge with 2, then 2 can't merge with 1)
        //  ___ ___ _l_ _i_ _r_ ___
        // |_0_|_1_|_2_|_3_|_4_|_5_|
        //
        // Merged left:
        // Two steps.
        //  _______ _l_ _i_ _r_ ___
        // |_0_,_1_|_2_|_3_|_4_|_5_|
        //
        // Merged right:
        // Three steps.
        //  ___ ___ ___ _l_ _i_ _r_
        // |_0_|_1_,_2_|_3_|_4_|_5_|
        //
        for (size_t idx = 1; idx < buildBlocks.size(); idx += 2)
        {
          const bool seqDirty =
            buildBlocks[idx - 1].dirty ||
            buildBlocks[idx].dirty ||
            (idx + 1 < buildBlocks.size() && buildBlocks[idx + 1].dirty);

          if (seqDirty)
          {
            const float left_score = scores[idx - 1];
            const float right_score =
              idx + 1 < buildBlocks.size() ?
              scores[idx] :
              -1.0f;

            if (left_score >= 0.0f || right_score >= 0.0f)
            {
              // At least one merge is possible.
              // Do the best merge, and invalidate
              // the one we "merged from" by setting the
              // block length to zero.
              if (left_score > right_score)
              {
                buildBlocks[idx - 1].blockLength +=
                  buildBlocks[idx].blockLength;
                buildBlocks[idx - 1].dirty = true;
                buildBlocks[idx].blockLength = 0;
              }
              else
              {
                buildBlocks[idx].blockLength +=
                  buildBlocks[idx + 1].blockLength;
                buildBlocks[idx].dirty = true;
                buildBlocks[idx + 1].blockLength = 0;
                // Take 3 steps instead of 2 next time.
                idx++;
              }
              nof_blocks_merged += 1;
            }
            else
            {
              // Can't merge, so don't bother with this block any more.
              buildBlocks[idx].dirty = false;
            }
          }
        }

        // Add the merged blocks to a new array.
        vector<BlockBuild> newblocks;
        newblocks.reserve(buildBlocks.size() - nof_blocks_merged);
        for (size_t idx = 0; idx < buildBlocks.size(); idx += 1)
        {
          if (buildBlocks[idx].blockLength > 0)
          {
            newblocks.push_back(buildBlocks[idx]);
          }
        }
        buildBlocks = newblocks;
      }

      // We are now done with merging, so we perform a last pass,
      // computing the final evaluation of the blocks we got.
      vector<block> blocks;
      {
        vector<float> scores;
        vector<uint8_t> weights;
        vector<float3> colorRanges;
        scores_gpu(
          buildBlocks,
          scores,
          weights,
          colorRanges,
          error_treshold,
          use_minmax_correction,
          use_LAB_error,
          compression_layout,
          vals_per_weight,
          true
        );

        // Insert the blocks into our block tree.
        for (size_t i = 0; i < buildBlocks.size(); i++)
        {
          block tmp(buildBlocks[i].blockStart + part_start, buildBlocks[i].blockLength);
          tmp.minpoint = vec3(
            colorRanges[2 * i + 0].x,
            colorRanges[2 * i + 0].y,
            colorRanges[2 * i + 0].z
          );
          tmp.maxpoint = vec3(
            colorRanges[2 * i + 1].x,
            colorRanges[2 * i + 1].y,
            colorRanges[2 * i + 1].z
          );
          blocks.push_back(tmp);
        }
      }
      block_tree[bits_per_weight] = blocks;
    } // ~weights

    /////////////////////////////////////////////////////////////////////////
    // Building the scores
    /////////////////////////////////////////////////////////////////////////
    // The three dimensional vector "children" lets us keep track of which
    // blocks wchich was used to construct the parent,
    // given a bit width and the parent block.
    // Bits -- block_id -- children (by block in level below)
    vector<vector<vector<uint32_t>>> children(block_tree.size());

    for (int parent_bits = min_bits_per_weight + 1;
         parent_bits <= max_bits_per_weight;
         parent_bits++)
    {
      int children_bits = parent_bits - 1;
      children[parent_bits].resize(block_tree[parent_bits].size());
      size_t child_idx = 0;
      for (size_t parent_idx = 0;
           parent_idx < block_tree[parent_bits].size();
           parent_idx++)
      {
        const block& parent = block_tree[parent_bits][parent_idx];
        size_t parent_stop = parent.start_node + parent.range;
        size_t children_start = block_tree[children_bits][child_idx].start_node;
        while (children_start < parent_stop)
        {
            assert(child_idx < std::numeric_limits<uint32_t>::max());
          children[parent_bits][parent_idx].push_back(child_idx);
          child_idx++;
          if (child_idx >= block_tree[children_bits].size())
          {
            break;
          }
          children_start = block_tree[children_bits][child_idx].start_node;
        }
      }
    }

    // It's time to compute the scores for all blocks.
    // We place this in a separate tree.
    struct block_score
    {
      struct
      {
        size_t total_bit_cost;
      } my, best;
    };

    using ScoreVector = vector<block_score>;
    vector<ScoreVector> score_tree(block_tree.size());
    for (size_t i = 0; i < score_tree.size(); i++)
    {
      score_tree[i].resize(block_tree[i].size());
    }

    // First calculate per block costs for all.
    for (size_t bits_per_weight = min_bits_per_weight;
         bits_per_weight < block_tree.size();
         bits_per_weight++)
    {
      for (size_t block_idx = 0;
           block_idx < block_tree[bits_per_weight].size();
           block_idx++)
      {
        const block& b = block_tree[bits_per_weight][block_idx];
        block_score& s = score_tree[bits_per_weight][block_idx];
        const size_t total_bits = HEADER_COST + COLOR_COST + bits_per_weight * b.range;
        s.my.total_bit_cost = total_bits;
        // Need to initialize the "best" cost for the leaves,
        // which will be used later to compute the parents costs.
        if (bits_per_weight == min_bits_per_weight)
        {
          s.best.total_bit_cost = s.my.total_bit_cost;
        }
      }
    }

    // Then calculate best cost, start at bpw 1 as we look
    // down one level.
    // The best cost is the minimum of the sum of the childrens total cost, and their best cost.
    for (size_t bits_per_weight = min_bits_per_weight + 1;
         bits_per_weight < block_tree.size();
         bits_per_weight++)
    {
      for (size_t block_idx = 0;
           block_idx < block_tree[bits_per_weight].size();
           block_idx++)
      {
        const block& b = block_tree[bits_per_weight][block_idx];
        block_score& s = score_tree[bits_per_weight][block_idx];
        const auto& childs = children[bits_per_weight][block_idx];
        size_t total_cost = 0;
        size_t best_cost = 0;
        for (const auto& current_child : childs)
        {
          best_cost +=
            score_tree[bits_per_weight - 1][current_child].best.total_bit_cost;
          total_cost +=
            score_tree[bits_per_weight - 1][current_child].my.total_bit_cost;
        }
        s.best.total_bit_cost = min(best_cost, s.my.total_bit_cost);
      }
    }

    // When the score tree is calculated we find the best cut, which is our solution.
    vector<end_block> solution;
    for (size_t parent_block = 0;
         parent_block < block_tree[max_bits_per_weight].size();
         parent_block++)
    {
      vector<end_block> tmp =
        find_best_nodes<end_block, block, block_score>(
          block_tree,
          score_tree,
          children,
          max_bits_per_weight,
          parent_block
          );
      solution.insert(solution.end(), tmp.begin(), tmp.end());
    }

    return solution;
  }

  double CompressionState::add_to_final(
    const vector<end_block>& solution,
    size_t& global_bptr,
    size_t& macro_w_bptr,
    vector<int>& wrong_colors,
    vector<int>& ok_colors
  )
  {
    const int max_bits_per_weigth = (int)log2(K);
    const int min_bits_per_weigth = 0;

    auto header_compressor =
      [](size_t local_start_color, size_t local_w_bptr, int bitrate)
    {
      uint32_t header = 0;
      header |= local_start_color; // bit: 0 - 13

      if (bitrate > 0)
      {
        // assuming bitrate in [1, 4]
        header |= (bitrate - 1) << 14; // bit: 14-15
        header |= local_w_bptr << 16;  // bit: 16-31
      }
      else
      {
        // assuming bitrate == 0
        header |= 0xffff0000ul; // bit: 16-31
      }
      return header;
    };

    double max_error_eval = numeric_limits<double>::lowest();
    for (auto b : solution)
    {
      int vals_per_weight = 1 << b.bpw;
      double error;
      bool should_be_true =
        assign_weights(
          b.start_node,
          b.range,
          b.minpoint,
          b.maxpoint,
          error_treshold,
          vals_per_weight,
          &error
        );

      max_error_eval = max(max_error_eval, error);
      if (!should_be_true)
      {
        wrong_colors[b.bpw] += 1;
      }
      else
      {
        ok_colors[b.bpw] += 1;
      }

      if ((b.start_node % macro_block_size) == 0)
      {
        // New macro block.
        // Index to first block
        h_macro_block_headers.push_back(h_block_headers.size());
        // Bit index to first weight
        h_macro_block_headers.push_back(global_bptr);
        macro_w_bptr = global_bptr;
      }

      // Add block header.
      uint32_t bitrate = b.bpw;
      h_block_headers.push_back(
        header_compressor(
          b.start_node % macro_block_size,
          global_bptr - macro_w_bptr,
          bitrate
        )
      );

      // Add block colors.
      if (b.bpw > 0)
      {
        switch (compression_layout)
        {
        case R_4: {
          uint32_t minC = float3_to_r4(b.minpoint);
          uint32_t maxC = float3_to_r4(b.maxpoint);
          h_block_colors.push_back((minC & 0xF) | ((maxC & 0xF) << 4));
          break;
        }
        case R_8: {
          uint32_t minC = float3_to_r8(b.minpoint);
          uint32_t maxC = float3_to_r8(b.maxpoint);
          h_block_colors.push_back((minC & (0xFF << 0)) >> 0);
          h_block_colors.push_back((maxC & (0xFF << 0)) >> 0);
          break;
        }
        case RG_8_8: {
          uint32_t minC = float3_to_rg88(b.minpoint);
          uint32_t maxC = float3_to_rg88(b.maxpoint);
          h_block_colors.push_back((minC & (0xFF << 0)) >> 0);
          h_block_colors.push_back((minC & (0xFF << 8)) >> 8);
          h_block_colors.push_back((maxC & (0xFF << 0)) >> 0);
          h_block_colors.push_back((maxC & (0xFF << 8)) >> 8);
          break;
        }
        case RGB_5_6_5: {
          uint32_t minC = float3_to_rgb565(b.minpoint);
          uint32_t maxC = float3_to_rgb565(b.maxpoint);
          h_block_colors.push_back((minC & (0xFF << 0)) >> 0);
          h_block_colors.push_back((minC & (0xFF << 8)) >> 8);
          h_block_colors.push_back((maxC & (0xFF << 0)) >> 0);
          h_block_colors.push_back((maxC & (0xFF << 8)) >> 8);
          break;
        }
        default:
          cout << "Missing compression layout!\n";
          break;
        }
      }
      else
      {
        switch (compression_layout)
        {
        case R_4: {
          uint32_t theC = float3_to_r8(b.maxpoint);
          h_block_colors.push_back((theC & (0xFF << 0)) >> 0);
          break;
        }
        case R_8: {
          uint32_t theC = float3_to_r16(b.maxpoint);
          h_block_colors.push_back((theC & (0xFF << 0)) >> 0);
          h_block_colors.push_back((theC & (0xFF << 8)) >> 8);
          break;
        }
        case RG_8_8: {
          uint32_t theC = float3_to_rg1616(b.maxpoint);
          h_block_colors.push_back(uint8_t((theC & (0xFF <<  0)) >>  0));
          h_block_colors.push_back(uint8_t((theC & (0xFF <<  8)) >>  8));
          h_block_colors.push_back(uint8_t((theC & (0xFF << 16)) >> 16));
          h_block_colors.push_back(uint8_t((theC & (0xFF << 24)) >> 24));
          break;
        }
        case RGB_5_6_5: {
          uint32_t theC = float3_to_rgb101210(b.maxpoint);
          h_block_colors.push_back(uint8_t((theC & (0xFF <<  0)) >>  0));
          h_block_colors.push_back(uint8_t((theC & (0xFF <<  8)) >>  8));
          h_block_colors.push_back(uint8_t((theC & (0xFF << 16)) >> 16));
          h_block_colors.push_back(uint8_t((theC & (0xFF << 24)) >> 24));
          break;
        }
        default:
          cout << "Missing compression layout!\n";
          break;
        }
      }

      // Assign weight bit offset and update global bit pointer (bit index).
      for (size_t i = b.start_node; i < b.start_node + b.range; i++)
      {
        global_bptr = insert_bits(m_weights[i], bitrate, &h_weights[0], global_bptr);
      }
    }
    return max_error_eval;
  }

  OursData
    compressColors_alternative_par(
      disc_vector<uint32_t>&& original_colors,
      const float error_treshold,
      const ColorLayout layout
    )
  {
    //ColorLayout layout = R_8;
    auto[nfo, result] =
      CompressionState{ std::move(original_colors), error_treshold, layout }.compress();
    int n_channels =
      layout == RGB_5_6_5 ? 3 :
      layout == RG_8_8 ? 2 :
      layout == R_8 ? 1 : 1;

    result.error_threshold = error_treshold;

    cout << "Uncompressed color size: " << original_colors.size() * n_channels << " bytes\n";
    result.bytes_raw = original_colors.size() * n_channels;
    cout
      << "Size of variable bitrate colors: "
      << nfo.total_bits
      << "bits ("
      << nfo.total_bits / 8
      << "bytes) with compression at "
      << 100.f * float(nfo.total_bits) / float(original_colors.size() * n_channels * 8)
      << "%\n";

    cout << "Nof blocks: " << nfo.nof_blocks << '\n';
    cout << "Average nof colors/block: " << original_colors.size() / float(nfo.nof_blocks) << '\n';

    for (size_t i = 0; i < nfo.wrong_colors.size(); i++)
    {
      cout
        << nfo.wrong_colors[i]
        << " wrong evals at "
        << i
        << "bpw... "
        << nfo.ok_colors[i]
        << " was correct.. ("
        << 100.f * float(nfo.wrong_colors[i]) / float(nfo.wrong_colors[i] + nfo.ok_colors[i])
        << " %)\n";
    }
    cout << "Max error is: " << nfo.max_error << '\n';

    ///////////////////////////////////////////////////////////////////////
    // Put in final data structure
    ///////////////////////////////////////////////////////////////////////

    {
      cout << "Headers size: " << nfo.headers_size << " bytes.\n";
      cout << "Colors size: " << nfo.colors_size << " bytes.\n";
      cout << "Weights size: " << nfo.weights_size << " bytes.\n";
      cout << "Macro header size: " << nfo.macro_header_size << " bytes.\n";
      float compression = float(
        double((uint64_t)nfo.headers_size + (uint64_t)nfo.colors_size + nfo.weights_size + (uint64_t)nfo.macro_header_size)
        / double(original_colors.size() * n_channels));
      cout
        << "Total: "
        << (uint64_t)nfo.headers_size + (uint64_t)nfo.colors_size + nfo.weights_size + (uint64_t)nfo.macro_header_size
        << " bytes ("
        << compression * 100.0f << "%).\n";
      result.compression = compression;
	  result.bytes_compressed = nfo.headers_size + nfo.colors_size + nfo.weights_size + nfo.macro_header_size;
    }
    return result;
  }

  void upload_to_gpu(OursData &ours_dat)
  {
    // NOTE: "auto*& d_vec" - We do **not** want a copy of the pointer.
    auto upload_vector = [](const auto& h_vec, auto*& d_vec)
    {
      using T = typename decay<decltype(h_vec)>::type::value_type;
      if (d_vec)
      {
        cudaFree(d_vec);
        d_vec = nullptr;
      }
      const size_t count = h_vec.size() * sizeof(T);
      cudaMallocManaged((void**)&d_vec, count);

    printf("Allocating %zu things (%fMB)\n", h_vec.size(), h_vec.size()  * sizeof(T) / double(1 << 20));
      cudaMemcpy(d_vec, h_vec.data(), count, cudaMemcpyHostToDevice);
    };
    upload_vector(ours_dat.h_block_headers, ours_dat.d_block_headers);
    upload_vector(ours_dat.h_block_colors, ours_dat.d_block_colors);
    upload_vector(ours_dat.h_weights, ours_dat.d_weights);
    upload_vector(ours_dat.h_macro_w_offset, ours_dat.d_macro_w_offset);
  }
} // namespace ours_varbit
