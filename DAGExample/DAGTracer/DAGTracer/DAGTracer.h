#pragma once
#include <stdint.h>
#include "../CudaHelpers.h" //FIXME: Proper search paths
#include <DAG/DAG.h>
#include <fstream>
#include <iosfwd>

class FileWriter
{
public:
	explicit FileWriter(const std::string& path)
		: os(path, std::ios::binary)
	{
	}
	~FileWriter()
	{
		assert(os.good());
		os.close();
	}
	
	inline void write(const void* data, size_t num)
	{
		os.write(reinterpret_cast<const char*>(data), std::streamsize(num));
	}
	inline void write(uint32_t data)
	{
		write(&data, sizeof(uint32_t));
	}
	inline void write(uint64_t data)
	{
		write(&data, sizeof(uint64_t));
	}
	inline void write(double data)
	{
		write(&data, sizeof(double));
	}
	template<typename T>
	inline void write(const std::vector<T>& array)
	{
		write(uint64_t(array.size()));
		write(array.data(), array.size() * sizeof(T));
	}

private:
	std::ofstream os;
};

#define MAX_LEVELS 18
#define IN_ORDER_TRAVERSAL 1
#define NEXT_CHILD_LUT 1

///////////////////////////////////////////////////////////////////////////
// Only need this right now. Dag should be on GPU when built. 
///////////////////////////////////////////////////////////////////////////
void upload_to_gpu(dag::DAG &dag);

struct ColorData {
  uint64_t *d_macro_w_offset;
  uint32_t *d_block_headers;
  uint8_t *d_block_colors;
  uint32_t *d_weights;
  uint64_t nof_blocks;
  uint32_t bits_per_weight;
  uint64_t nof_colors;
};


namespace chag{
		class view;
}
class DAGTracer
{
public:
	DAGTracer(); 
	~DAGTracer();
	///////////////////////////////////////////////////////////////////////////
	// Only actual state of the DAGTracer is the output buffers and dimensions
	///////////////////////////////////////////////////////////////////////////
	CUDAGLInteropSurface m_color_buffer; 
	CUDAGLInteropSurface m_path_buffer;
	CUDAGLInteropSurface m_depth_buffer;

  ColorData m_compressed_colors;

	uint32_t m_width;
	uint32_t m_height; 
	void resize(uint32_t width, uint32_t height);
#if NEXT_CHILD_LUT
	///////////////////////////////////////////////////////////////////////////
	// This lookup table is for calculating which is the next child to traverse
	// into to follow ray-order. 
	///////////////////////////////////////////////////////////////////////////
	uint8_t * calculateNextChildLookupTable();
	uint8_t * d_next_child_lookup_table; 
#endif
	///////////////////////////////////////////////////////////////////////////
	// Primary raytracing of DAG
	///////////////////////////////////////////////////////////////////////////
	void resolve_paths(const dag::DAG &dag, const chag::view & camera, int color_lookup_level);
	void resolve_colors(const dag::DAG &dag, int color_lookup_level);
};


