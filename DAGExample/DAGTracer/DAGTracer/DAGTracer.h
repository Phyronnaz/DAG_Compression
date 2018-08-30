#pragma once
#include <stdint.h>
#include "../CudaHelpers.h" //FIXME: Proper search paths
#include <DAG/DAG.h>
//#include "elmar.h"
//#include "ours.h"
//#include "ours_varbit.h"

#define MAX_LEVELS 17
#define IN_ORDER_TRAVERSAL 1
#define NEXT_CHILD_LUT 1

///////////////////////////////////////////////////////////////////////////
// Only need this right now. Dag should be on GPU when built. 
///////////////////////////////////////////////////////////////////////////
void temp_copy_dag_to_device(dag::DAG &dag);

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
	void resolve_colors(const dag::DAG &dag, /*elmar::ElmarData & elmar_data, ours::OursData & ours_data, ours_varbit::OursData*,*/ bool all_colors, int color_lookup_level);
};


