#pragma once
#include <stdint.h>
#include "../CudaHelpers.h" //FIXME: Proper search paths
#include <DAG/DAG.h>

#define MAX_LEVELS 17
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


