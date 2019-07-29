#pragma once
#include "BlockBuild.h"
#include <cstdint>
#include <utility>  // std::pair, std::make_pair
#include <vector>

#include <fstream>
template<typename T>
static void write_to_disc(const std::string file, const std::vector<T> &vec)
{
  std::ofstream ofs{ file, std::ofstream::binary | std::ofstream::out };
  ofs.write(
    reinterpret_cast<const char*>(vec.data()),
    vec.size() * sizeof(T)
  );
}

template<typename T>
class disc_vector {
public:
  disc_vector(const disc_vector&) = delete;
  disc_vector(disc_vector&&) = default;
  disc_vector& operator=(const disc_vector&) = delete;
  disc_vector& operator=(disc_vector&&) = default;
  virtual ~disc_vector() = default;
  disc_vector(const std::string file, std::size_t cache_size) : ifs{ file, std::ifstream::binary | std::ifstream::in | std::ifstream::ate }
  {
    const_cast<std::size_t&>(_size) = ifs.tellg() / sizeof(T);
    ifs.seekg(std::ifstream::beg);
    _cache.resize(cache_size);
  };

  const T operator [] (std::size_t i) {
    if (!is_in_cache(i))
    {
      //std::cout << "Not in cache\n";
      read_block(i);
    }
    return _cache[i % _cache.size()];
  }
  std::size_t size() const { return _size; }
private:

  bool is_in_cache(const std::size_t i) const {
    //std::cout
    //  << "i: " << i << '\n'
    //  << " _cache.size() * _block:" << _cache.size() * _block << '\n'
    //  << " _cache.size() * (_block + 1): " << _cache.size() * (_block + 1) << '\n';
    return
      _has_cache &&
      _cache.size() * _block <= i &&
      _cache.size() * (_block + 1) > i;
  }

  void read_block(const std::size_t i) {
    //std::cout << "Read\n";
    _block = i / _cache.size();
    const std::size_t read_start  = _block * _cache.size();
    std::size_t num_to_read       = _cache.size();
    const std::size_t read_end    = read_start + num_to_read;
    const std::size_t max_read_to = _size;
    if (read_end > max_read_to)
    {
      num_to_read = max_read_to - read_start;
    }
    ifs.seekg(read_start * sizeof(T), std::ifstream::beg);
    ifs.read(
      reinterpret_cast<char*>(_cache.data()),
      num_to_read * sizeof(T)
    );
    _has_cache = true;
  }

  std::ifstream ifs;
  std::vector<T> _cache;
  bool _has_cache{ false };
  std::size_t _block{ 0 };
  const std::size_t _size{ 0 };
};

constexpr uint64_t macro_block_size = 16ull * 1024ull;
namespace ours_varbit {
  using ColorLayout = ColorLayout;
  struct OursData {
    uint32_t *d_block_headers = nullptr;
    uint8_t *d_block_colors = nullptr;
    uint32_t *d_weights = nullptr;
    uint64_t *d_macro_w_offset = nullptr;
    uint64_t nof_blocks;
    uint64_t nof_colors;
    uint32_t bits_per_weight;
    ColorLayout color_layout;
    bool use_single_color_blocks;
    std::vector<uint32_t> h_block_headers;
    std::vector<uint8_t> h_block_colors;
    std::vector<uint32_t> h_weights;
    std::vector<uint64_t> h_macro_w_offset;
    float compression;
    float error_threshold;
    uint64_t bytes_raw;
    uint64_t bytes_compressed;
  };

  OursData compressColors_alternative_par(
    //std::vector<uint32_t> &original_colors,
    disc_vector<uint32_t> &&original_colors,
    const float error_threshold,
    const ColorLayout layout
  );

  bool getErrInfo(
    const std::vector<uint32_t> &colors,
    const std::string filename,
    const ColorLayout layout,
    float *mse,
    float *maxR,
    float *maxG,
    float *maxB,
    float *maxLength
  );

  float getPSNR(float mse);

  void upload_to_gpu(OursData &data);
};  // namespace ours_varbit
