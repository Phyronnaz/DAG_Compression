#include "DAGTracer.h"
#include <GL/glew.h>
#include <cuda_gl_interop.h>
#define _USE_MATH_DEFINES
#include <math.h>
#include <algorithm>
#include "../CudaHelpers.h"        //FIXME: Proper search paths
#include "../bits_in_uint_array.h" //FIXME: Proper search paths
#include <utils/view.h>

#include <bitset>
#include <limits>

template <typename T>
unsigned popcnt_safe(T v) {
	return static_cast<unsigned>(std::bitset<std::numeric_limits<T>::digits>(v).count());
}

std::vector<uint32_t> upload_to_gpu(dag::DAG &dag)
{
	std::vector<uint32_t> dag_array;
	std::vector<uint32_t> lvl_offsets;
	uint32_t ctr{0};
	for (const auto &lvl : dag.m_data) 
	{
		lvl_offsets.push_back(ctr);
		ctr += (uint32_t)lvl.size();
	}
	dag_array.reserve(ctr);
	auto tmp_dag = dag.m_data;
	// For each lvl (we will not update the final lvl, hence the -1)
	std::size_t levels_to_process = lvl_offsets.size() - 1;
	for (std::size_t lvl{0}; lvl < levels_to_process; ++lvl) 
	{
		// For each node
		for (std::size_t node_start{0}; node_start < tmp_dag[lvl].size(); ++node_start) 
		{
			uint32_t mask       = tmp_dag[lvl][node_start];
			unsigned n_children = popcnt_safe(mask & 0xFF);
			for (std::size_t child{0}; child < n_children; ++child) {
				tmp_dag[lvl][node_start + child + 1] += lvl_offsets[lvl + 1];
			}
			node_start += n_children;
		}
	}

	// Copy DAG to array
	for (const auto &lvl : tmp_dag) { dag_array.insert(dag_array.end(), lvl.begin(), lvl.end()); }


	if (dag.d_data)            { cudaFree(dag.d_data);            dag.d_data            = nullptr;}
	if (dag.d_color_data)      { cudaFree(dag.d_color_data);      dag.d_color_data      = nullptr;}
	if (dag.d_enclosed_leaves) { cudaFree(dag.d_enclosed_leaves); dag.d_enclosed_leaves = nullptr;}

	cudaMalloc(&dag.d_data, dag_array.size() * sizeof(uint32_t));
	cudaMemcpy(dag.d_data,  dag_array.data(), dag_array.size() * sizeof(uint32_t), cudaMemcpyHostToDevice);

	cudaMalloc(&dag.d_color_data, dag.m_base_colors.size() * sizeof(uint32_t));
	cudaMemcpy(dag.d_color_data, dag.m_base_colors.data(), dag.m_base_colors.size() * sizeof(uint32_t), cudaMemcpyHostToDevice);

	if (dag.m_enclosed_leaves.size() != 0){
		cudaMalloc(&dag.d_enclosed_leaves, dag.m_enclosed_leaves.size() * sizeof(uint32_t));
		cudaMemcpy(dag.d_enclosed_leaves,  dag.m_enclosed_leaves.data(), dag.m_enclosed_leaves.size() * sizeof(uint32_t), cudaMemcpyHostToDevice);
	}
	return dag_array;
}

///////////////////////////////////////////////////////////////////////////////
// Constructor
// ============================================================================
// Avoiding GL functions (e.g. glGenTextures) here to allow the class to be
// instantiated before glew initialization. 
///////////////////////////////////////////////////////////////////////////////
DAGTracer::DAGTracer()
{
#if NEXT_CHILD_LUT
	uint8_t *h_next_child_lookup_table = calculateNextChildLookupTable();
	const int lookup_table_size = 8 * 256 * 256 * sizeof(uint8_t); 
	cudaMalloc(&d_next_child_lookup_table, lookup_table_size);
	cudaMemcpy(d_next_child_lookup_table, h_next_child_lookup_table, lookup_table_size, cudaMemcpyHostToDevice);
	delete[] h_next_child_lookup_table;
#endif
}

///////////////////////////////////////////////////////////////////////////////
// Destructor
///////////////////////////////////////////////////////////////////////////////
DAGTracer::~DAGTracer()
{
#if NEXT_CHILD_LUT
	cudaFree(d_next_child_lookup_table); 
#endif
	m_color_buffer.unregisterResource(); 
	m_path_buffer.unregisterResource(); 
	m_depth_buffer.unregisterResource();
	if (m_color_buffer.m_gl_idx != 0) glDeleteTextures(1, &m_color_buffer.m_gl_idx);
	if (m_path_buffer.m_gl_idx != 0) glDeleteTextures(1, &m_path_buffer.m_gl_idx);
	if (m_depth_buffer.m_gl_idx != 0) glDeleteTextures(1, &m_depth_buffer.m_gl_idx);
}

///////////////////////////////////////////////////////////////////////////////
// Resize output buffer (on init and when window changes size)
///////////////////////////////////////////////////////////////////////////////
void DAGTracer::resize(uint32_t width, uint32_t height)
{
	m_width = width; 
	m_height = height; 

	///////////////////////////////////////////////////////////////////////////
	// Create path output texture (this might not really need to be an OpenGL
	// texture...)
	///////////////////////////////////////////////////////////////////////////
	if (m_path_buffer.m_gl_idx == 0) glGenTextures(1, &m_path_buffer.m_gl_idx);
	m_path_buffer.unregisterResource();
	glBindTexture(GL_TEXTURE_2D, m_path_buffer.m_gl_idx);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32UI, m_width, m_height, 0, GL_RGBA_INTEGER, GL_UNSIGNED_INT, nullptr);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	{
		float floatmax = std::numeric_limits<float>::max();
		uint32_t ufloatmax = reinterpret_cast<uint32_t&>(floatmax);
		auto start_path = make_uint4(0,0,0, ufloatmax);
		glClearTexImage(m_path_buffer.m_gl_idx, 0, GL_RGBA_INTEGER, GL_UNSIGNED_INT, &start_path);
	}
	glBindTexture(GL_TEXTURE_2D, 0);
	m_path_buffer.registerResource();

	///////////////////////////////////////////////////////////////////////////
	// Create color output texture
	///////////////////////////////////////////////////////////////////////////
	if (m_color_buffer.m_gl_idx == 0) glGenTextures(1, &m_color_buffer.m_gl_idx); 
	m_color_buffer.unregisterResource(); 
	glBindTexture(GL_TEXTURE_2D, m_color_buffer.m_gl_idx);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, m_width, m_height, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	{
		uint32_t clear_val = 0xFFFFFF;
		glClearTexImage(m_color_buffer.m_gl_idx, 0, GL_RGBA, GL_UNSIGNED_BYTE, &clear_val);
	}
	glBindTexture(GL_TEXTURE_2D, 0);
	m_color_buffer.registerResource();

	///////////////////////////////////////////////////////////////////////////
	// Create depth output texture
	///////////////////////////////////////////////////////////////////////////
	if (m_depth_buffer.m_gl_idx == 0) glGenTextures(1, &m_depth_buffer.m_gl_idx); 
	m_depth_buffer.unregisterResource();
	glBindTexture(GL_TEXTURE_2D, m_depth_buffer.m_gl_idx);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, m_width, m_height, 0, GL_RED, GL_FLOAT, nullptr);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	{
		float max_depth = std::numeric_limits<float>::max();
		glClearTexImage(m_depth_buffer.m_gl_idx, 0, GL_RED, GL_FLOAT, &max_depth);
	}
	glBindTexture(GL_TEXTURE_2D, 0);
	m_depth_buffer.registerResource();
}

///////////////////////////////////////////////////////////////////////////////
// Having a different child order than %000 means that we don't  process the 
// children in order 0,2,3,4,5,6,7. For example, child_order == %001 means that 
// we should do high z-values before low z values. 
// It all boils down to that the order is obtained as 
// child_numnber[child_order] = child_number[%000] ^ child_order
///////////////////////////////////////////////////////////////////////////////
__host__ __device__ uint8_t getNextChild(uint8_t child_order, uint8_t testmask, uint8_t childmask)
{
	for (int i = 0; i < 8; i++) {
		uint8_t child_in_order = i ^ child_order;
		bool child_exists = (childmask & (1 << child_in_order)) != 0;
		bool child_is_taken = (~testmask & (1 << child_in_order)) != 0;
		if (child_exists && !child_is_taken) return child_in_order;
	}
	return uint8_t(9); // Should not happen!
};

#if NEXT_CHILD_LUT
///////////////////////////////////////////////////////////////////////////////
// In the lookup table, we want to maintain locality as much as we can. 
// The "child_order" is very likely to be the same (at least for primary rays)
// between pixels. The "testmask" is probably more volatile than the 
// "childmask", so do that last.
///////////////////////////////////////////////////////////////////////////////
uint8_t *  DAGTracer::calculateNextChildLookupTable()
{
	uint8_t *lut = new uint8_t[8 * 256 * 256];
	for (uint32_t child_order = 0; child_order < 8; child_order++) {
		for (uint32_t childmask = 0; childmask < 256; childmask++) {
			for (uint32_t testmask = 0; testmask < 256; testmask++) {
				lut[child_order * (256 * 256) + childmask * 256 + testmask] =
					getNextChild(child_order, testmask, childmask); 
			}
		}
	}
	return lut; 
}
#endif

///////////////////////////////////////////////////////////////////////////////
// Function for intersecting a node, returning a bitmask with the 
// childnodes that are intersected (ignoring whether they exist).
///////////////////////////////////////////////////////////////////////////////
inline __device__ uint8_t getIntersectionMask(const uint3 & stack_path, const uint32_t nof_levels,
												const uint32_t stack_level, const float3 & ray_o,
												const float3 & ray_d, const float3 &inv_ray_dir)
{
	///////////////////////////////////////////////////////////////////////////
	// Node center is 0.5 * (aabb_min + aabb_max), which reduces to:
	// NOTE: This is probably of by 0.5 right now...
	///////////////////////////////////////////////////////////////////////////
	uint32_t shift = nof_levels - (stack_level);
	uint32_t node_radius = 1 << (shift - 1);
	float3 node_center = make_float3(stack_path << shift) + make_float3(node_radius);

	///////////////////////////////////////////////////////////////////////////
	// Find the t-values at which the ray intersects the axis-planes of the 
	// node: ray_o + tmid * ray_d = node_center   (component wise)
	///////////////////////////////////////////////////////////////////////////
	float3 tmid = (node_center - ray_o) * inv_ray_dir;

	///////////////////////////////////////////////////////////////////////////
	// Now find the t values at which the ray intersects the parent node,
	// by calculating the t-range for traveling through a "slab" from the 
	// center to each side of the node. 
	///////////////////////////////////////////////////////////////////////////
	float ray_tmin, ray_tmax;
	{
		float3 slab_radius = node_radius * abs(inv_ray_dir);
		float3 tmin = tmid - slab_radius;
		ray_tmin = fmax(max(tmin), 0.0f);
		float3 tmax = tmid + slab_radius;
		ray_tmax = min(tmax);
	}

	///////////////////////////////////////////////////////////////////////////
	// Find the first child that is intersected. 
	// NOTE: We assume that we WILL hit one child, since we assume that the 
	//       parents bounding box is hit.
	// NOTE: To safely get the correct node, we cannot use o+ray_tmin*d as the
	//       intersection point, since this point might lie too close to an 
	//       axis plane. Instead, we use the midpoint between max and min which
	//       will lie in the correct node IF the ray only intersects one node. 
	//       Otherwise, it will still lie in an intersected node, so there are 
	//       no false positives from this. 
	///////////////////////////////////////////////////////////////////////////
	uint8_t intersection_mask = 0;
	{
		uint8_t first_hit_child;
		float3 point_on_ray_segment = ray_o + 0.5f * (ray_tmin + ray_tmax) * ray_d;
		first_hit_child = point_on_ray_segment.x >= node_center.x ? 4 : 0;
		first_hit_child += point_on_ray_segment.y >= node_center.y ? 2 : 0;
		first_hit_child += point_on_ray_segment.z >= node_center.z ? 1 : 0;
		intersection_mask |= (1 << first_hit_child);
	}

	///////////////////////////////////////////////////////////////////////////
	// We now check the points where the ray intersects the X, Y and Z plane. 
	// If the intersection is within (ray_tmin, ray_tmax) then the intersection
	// point implies that two voxels will be touched by the ray. We find out 
	// which voxels to mask for an intersection point at +X, +Y by setting
	// ALL voxels at +X and ALL voxels at +Y and ANDing these two masks. 
	// 
	// NOTE: When the intersection point is close enough to another axis plane, 
	//       we must check both sides or we will get robustness issues. 
	///////////////////////////////////////////////////////////////////////////
	const float epsilon = 0.0001f;
	float3 pointOnRaySegment0 = ray_o + tmid.x*ray_d;
	uint32_t A = (abs(node_center.y - pointOnRaySegment0.y) < epsilon) ? (0xCC | 0x33) : ((pointOnRaySegment0.y >= node_center.y ? 0xCC : 0x33));
	uint32_t B = (abs(node_center.z - pointOnRaySegment0.z) < epsilon) ? (0xAA | 0x55) : ((pointOnRaySegment0.z >= node_center.z ? 0xAA : 0x55));
	intersection_mask |= (tmid.x < ray_tmin || tmid.x > ray_tmax) ? 0 : (A & B);
	float3 pointOnRaySegment1 = ray_o + tmid.y*ray_d;
	uint32_t C = (abs(node_center.x - pointOnRaySegment1.x) < epsilon) ? (0xF0 | 0x0F) : ((pointOnRaySegment1.x >= node_center.x ? 0xF0 : 0x0F));
	uint32_t D = (abs(node_center.z - pointOnRaySegment1.z) < epsilon) ? (0xAA | 0x55) : ((pointOnRaySegment1.z >= node_center.z ? 0xAA : 0x55));
	intersection_mask |= (tmid.y < ray_tmin || tmid.y > ray_tmax) ? 0 : (C & D);
	float3 pointOnRaySegment2 = ray_o + tmid.z*ray_d;
	uint32_t E = (abs(node_center.x - pointOnRaySegment2.x) < epsilon) ? (0xF0 | 0x0F) : ((pointOnRaySegment2.x >= node_center.x ? 0xF0 : 0x0F));
	uint32_t F = (abs(node_center.y - pointOnRaySegment2.y) < epsilon) ? (0xCC | 0x33) : ((pointOnRaySegment2.y >= node_center.y ? 0xCC : 0x33));
	intersection_mask |= (tmid.z < ray_tmin || tmid.z > ray_tmax) ? 0 : (E & F);

	///////////////////////////////////////////////////////////////////////////
	// Checking ray_tmin > ray_tmax is not generally safe (it happens sometimes
	// due to float precision). But we still use this test to stop rays that do 
	// not hit the root node. 
	///////////////////////////////////////////////////////////////////////////	
	return (stack_level == 0 && ray_tmin >= ray_tmax) ? 0 : intersection_mask;
};



__global__ void
primary_rays_kernel(
	uint32_t width,
	uint32_t height, 
	float3 camera_pos,
	float3 ray_d_min,
	double3 ray_d_dx,
	double3 ray_d_dy,
	uint32_t *dag,
	uint32_t nof_levels,
	cudaSurfaceObject_t path_buffer
#if NEXT_CHILD_LUT
	,uint8_t *d_next_child_lookup_table
#endif
	, int color_lookup_level
)
{
	///////////////////////////////////////////////////////////////////////////
	// Screen coordinates, discard outside
	///////////////////////////////////////////////////////////////////////////
	uint2 coord = make_uint2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
	if (coord.x >= width || coord.y >= height) return; 

	///////////////////////////////////////////////////////////////////////////
	// Calculate ray for pixel
	///////////////////////////////////////////////////////////////////////////
	const float3 ray_o = camera_pos; 
	const float3 ray_d = make_float3(normalize((make_double3(ray_d_min) + coord.x * ray_d_dx + coord.y * ray_d_dy) - make_double3(ray_o))); 
	float3 inv_ray_dir = 1.0f / ray_d;

	///////////////////////////////////////////////////////////////////////////
	// Stack
	///////////////////////////////////////////////////////////////////////////
	struct StackEntry{
		uint32_t node_idx;
		uint32_t masks; 
	} stack[MAX_LEVELS];
	uint3		stack_path = make_uint3(0,0,0);
	int			stack_level = 0; 


	///////////////////////////////////////////////////////////////////////////
	// If intersecting the root node, push it on stack
	///////////////////////////////////////////////////////////////////////////
	stack[0].node_idx = 0; 
	stack[0].masks = dag[0] & 0xFF;
	stack[0].masks |= (stack[0].masks & getIntersectionMask(stack_path, nof_levels, stack_level, ray_o, ray_d, inv_ray_dir)) << 8;

	///////////////////////////////////////////////////////////////////////////
	// Traverse until stack is empty (all root children are processed), or 
	// until a leaf node is intersected. 
	///////////////////////////////////////////////////////////////////////////
	uint64_t current_leafmask; 
	StackEntry curr_se = stack[0]; // Current stack entry

	while (stack_level >= 0)
	{
		///////////////////////////////////////////////////////////////////////
		// If no children left to test, roll back stack
		///////////////////////////////////////////////////////////////////////
		if ((curr_se.masks & 0xFF00) == 0x0) {
			stack_level = stack_level - 1;
			stack_path = make_uint3(stack_path.x >> 1, stack_path.y >> 1, stack_path.z >> 1);
			while ((stack[stack_level].masks & 0xFF00) == 0x0 && stack_level >= 0) {
				stack_level = stack_level - 1;
				stack_path = make_uint3(stack_path.x >> 1, stack_path.y >> 1, stack_path.z >> 1);
			}
			if (stack_level < 0) break;
			curr_se = stack[stack_level];
		}

		///////////////////////////////////////////////////////////////////////
		// Figure out which child to test next and create its path and 
		// bounding box
		///////////////////////////////////////////////////////////////////////
#if IN_ORDER_TRAVERSAL
		int child_order =	ray_d.x < 0 ? 4 : 0;
		child_order		+=	ray_d.y < 0 ? 2 : 0;
		child_order		+=	ray_d.z < 0 ? 1 : 0;
#if NEXT_CHILD_LUT
		uint8_t next_child = d_next_child_lookup_table[child_order * (256 * 256) + (curr_se.masks & 0xFF) * 256 + (curr_se.masks >> 8)];
#else
		uint8_t next_child = getNextChild(child_order, node_testmask, node_childmask);
#endif
#else
		uint8_t next_child = 31 - __clz(node_testmask);
#endif

		curr_se.masks &= ~(1 << (next_child + 8));

		///////////////////////////////////////////////////////////////////////
		// Check this child for intersection with ray
		///////////////////////////////////////////////////////////////////////
		{
			uint32_t node_offset = __popc((curr_se.masks & 0xFF) & ((1 << next_child) - 1)) + 1;

			stack_path = make_uint3(
				(stack_path.x << 1) | ((next_child & 0x4) >> 2),
				(stack_path.y << 1) | ((next_child & 0x2) >> 1),
				(stack_path.z << 1) | ((next_child & 0x1) >> 0));
			stack_level += 1;

			///////////////////////////////////////////////////////////////////
			// If we are at the final level where we have intersected a 1x1x1
			// voxel. We are done. 
			///////////////////////////////////////////////////////////////////
			if (stack_level ==  color_lookup_level) {
				break;
			}
			
			///////////////////////////////////////////////////////////////////
			// As I have intersected (and I am not finished) the current 
			// stack entry must be saved to the stack
			///////////////////////////////////////////////////////////////////
			stack[stack_level - 1] = curr_se;
			
			///////////////////////////////////////////////////////////////////
			// Now let's see if we are at the 2x2x2 level, in which case we
			// need to pick a child-mask from previously fetched leaf-masks
			// (node_idx does not matter until we are back)
			///////////////////////////////////////////////////////////////////
			if (stack_level == nof_levels - 1) {
				curr_se.masks = (current_leafmask >> next_child * 8) & 0xFF;
			}
			///////////////////////////////////////////////////////////////////
			// If the child level is the "leaf" level (containing 4x4x4 
			// leafmasks), create a fake node for the next pass
			///////////////////////////////////////////////////////////////////
			else if (stack_level == nof_levels - 2) {
				uint32_t leafmask_address = dag[curr_se.node_idx + node_offset]; 
				///////////////////////////////////////////////////////////////
				// Shouldn't there be a faster way to get the 8 bit mask from 
				// a 64 bit word... Without a bunch of compares? 
				///////////////////////////////////////////////////////////////
				uint32_t leafmask0 = dag[leafmask_address];
				uint32_t leafmask1 = dag[leafmask_address + 1];
				current_leafmask = uint64_t(leafmask1) << 32 | uint64_t(leafmask0); 
				curr_se.masks =
					((current_leafmask & 0x00000000000000FF) == 0 ? 0 : 1 << 0) |
					((current_leafmask & 0x000000000000FF00) == 0 ? 0 : 1 << 1) |
					((current_leafmask & 0x0000000000FF0000) == 0 ? 0 : 1 << 2) |
					((current_leafmask & 0x00000000FF000000) == 0 ? 0 : 1 << 3) |
					((current_leafmask & 0x000000FF00000000) == 0 ? 0 : 1 << 4) |
					((current_leafmask & 0x0000FF0000000000) == 0 ? 0 : 1 << 5) |
					((current_leafmask & 0x00FF000000000000) == 0 ? 0 : 1 << 6) |
					((current_leafmask & 0xFF00000000000000) == 0 ? 0 : 1 << 7);
			}
			///////////////////////////////////////////////////////////////
			// If we are at an internal node, push the child on the stack
			///////////////////////////////////////////////////////////////
			else {
				curr_se.node_idx = dag[curr_se.node_idx + node_offset]; 
				curr_se.masks = dag[curr_se.node_idx] & 0xFF;
			}

			curr_se.masks |= (curr_se.masks & getIntersectionMask(stack_path, nof_levels, stack_level, ray_o, ray_d, inv_ray_dir)) << 8;

		}
		///////////////////////////////////////////////////////////////////////
		// If it does not intersect, do nothing. Proceed at same level.
		///////////////////////////////////////////////////////////////////////
	}
	///////////////////////////////////////////////////////////////////////////
	// Done, stack_path is closest voxel (or 0 if none found)
	///////////////////////////////////////////////////////////////////////////
	stack_path.x = stack_path.x << (nof_levels - color_lookup_level);
	stack_path.y = stack_path.y << (nof_levels - color_lookup_level);
	stack_path.z = stack_path.z << (nof_levels - color_lookup_level);
	surf2Dwrite(make_uint4(stack_path, 0), path_buffer, coord.x  * sizeof(uint4), coord.y);
	return;
}

inline __device__ float3 rgb888_to_float3(uint32_t rgb) {
		return make_float3(	((rgb >> 0) & 0xFF) / 255.0f, 
							((rgb >> 8) & 0xFF) / 255.0f,
							((rgb >> 16) & 0xFF) / 255.0f);
	}

inline __device__ float3 rgb101210_to_float3(uint32_t rgb) {
		return make_float3(	((rgb >> 0) & 0x3FF) / 1023.0f, 
							((rgb >> 10) & 0xFFF) / 4095.0f,
											((rgb >> 22) & 0x3FF) / 1023.0f);
	}

inline __device__ float3 rgb565_to_float3(uint32_t rgb) {
		return make_float3(	((rgb >> 0) & 0x1F) / 31.0f, 
							((rgb >> 5) & 0x3F) / 63.0f,
							((rgb >> 11) & 0x1F) / 31.0f);
	}

inline __device__ uint32_t float3_to_rgb888(float3 c) {
			float R = fmin(1.0f, fmax(0.0f, c.x));
			float G = fmin(1.0f, fmax(0.0f, c.y));
			float B = fmin(1.0f, fmax(0.0f, c.z));
			return	(uint32_t(R * 255.0f) << 0) | 
				(uint32_t(G * 255.0f) << 8) | 
				(uint32_t(B * 255.0f) << 16);
		}

inline __device__ uint32_t float3_to_rgb101210(float3 c) {
			float R = fmin(1.0f, fmax(0.0f, c.x));
			float G = fmin(1.0f, fmax(0.0f, c.y));
			float B = fmin(1.0f, fmax(0.0f, c.z));
			return	(uint32_t(R * 1023.0f) << 0) | 
				(uint32_t(G * 4095.0f) << 10)| 
				(uint32_t(B * 1023.0f) << 22);
		}

inline __device__ uint32_t float3_to_rgb565(float3 c) {
			float R = fmin(1.0f, fmax(0.0f, c.x));
			float G = fmin(1.0f, fmax(0.0f, c.y));
			float B = fmin(1.0f, fmax(0.0f, c.z));
			return	(uint32_t(R * 31.0f) << 0) | 
				(uint32_t(G * 63.0f) << 5) | 
				(uint32_t(B * 31.0f) << 11);
		}


__global__ void 
color_lookup_kernel_morton(
	uint32_t width, 
	uint32_t height , 
	int nof_levels, 
	cudaSurfaceObject_t path_buffer,
	uint32_t *dag, 
	uint32_t *dag_color, 
	uint32_t *enclosed_leaves,
	uint32_t nof_top_levels, 
	cudaSurfaceObject_t output_image,
	bool all_colors,
	int stop_level
) 
{
	///////////////////////////////////////////////////////////////////////////
	// Screen coordinates, discard outside
	///////////////////////////////////////////////////////////////////////////
	uint2 coord = make_uint2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
	if (coord.x >= width || coord.y >= height) return;

	uint32_t color = 0x0000FF;
	uint3 path = make_uint3(surf2Dread<uint4>(path_buffer, coord.x * sizeof(uint4), coord.y));
	uint32_t nof_leaves = 0;
	uint32_t final_color_idx = 0;
	if (path != make_uint3(0, 0, 0)) {
		uint32_t level = 0;
		uint32_t node_index = 0;
		while (level < nof_levels - 2)
		{
			level += 1;
			//////////////////////////////////////////////////////////////////////////
			// Find the current childmask and which subnode we are in
			//////////////////////////////////////////////////////////////////////////
			uint32_t child_mask = dag[node_index];
			uint8_t child_idx = (((path.x >> (nof_levels - level) & 0x1) == 0) ? 0 : 4) |
												(((path.y >> (nof_levels - level) & 0x1) == 0) ? 0 : 2) |
												(((path.z >> (nof_levels - level) & 0x1) == 0) ? 0 : 1);
			if (level == stop_level && all_colors) {
				final_color_idx = nof_leaves; 
				break;
			}
			//////////////////////////////////////////////////////////////////////////
			// Make sure the node actually exists
			//////////////////////////////////////////////////////////////////////////
			if ((0xFF & (child_mask & (1 << child_idx))) == 0) {
				// We have traveled to an unexisting node!!
				color = 0xFF00FF;
				surf2Dwrite(color, output_image, (int)sizeof(uint32_t)* coord.x, coord.y, cudaBoundaryModeClamp);
				return;
			}
			//////////////////////////////////////////////////////////////////////////
			// Find out how many leafs are in the children preceeding this
			//////////////////////////////////////////////////////////////////////////
			if (level == nof_levels - 2) 
			{
				// If only voxel colors, just count the number of preceeding leaves
				// If all colors are stored, need to count the number of preceeding
				// _nodes_, i.e., add one for the 4x4x4 node and one for each 
				// preceeding 2x2x2 node (and all voxels)
				uint32_t n_offset = __popc(child_mask & ((1 << child_idx) - 1));
				uint32_t n_child_offset = all_colors ? 1 : 0; // One for this internal node
				for (uint32_t i = 0; i < n_offset; ++i) {
					if (all_colors) n_child_offset += 1; // 4x4x4 node
					uint32_t n_index = dag[node_index + 1 + i];
					uint32_t leafmask0 = dag[n_index];
					uint32_t leafmask1 = dag[n_index + 1];
					uint64_t current_leafmask = uint64_t(leafmask1) << 32 | uint64_t(leafmask0);
					if (all_colors)
					{ // 2x2x2 nodes
						// This is way better:
						// nof_leaves += __popc(__vsetgtu4(leafmask0, 0));
						// nof_leaves += __popc(__vsetgtu4(leafmask1, 0));
						n_child_offset += (current_leafmask & 0x00000000000000FF) == 0 ? 0 : 1; 
						n_child_offset += (current_leafmask & 0x000000000000FF00) == 0 ? 0 : 1;
						n_child_offset += (current_leafmask & 0x0000000000FF0000) == 0 ? 0 : 1;
						n_child_offset += (current_leafmask & 0x00000000FF000000) == 0 ? 0 : 1;
						n_child_offset += (current_leafmask & 0x000000FF00000000) == 0 ? 0 : 1;
						n_child_offset += (current_leafmask & 0x0000FF0000000000) == 0 ? 0 : 1;
						n_child_offset += (current_leafmask & 0x00FF000000000000) == 0 ? 0 : 1;
						n_child_offset += (current_leafmask & 0xFF00000000000000) == 0 ? 0 : 1;
					}
					// 1x1x1 nodes
					n_child_offset += __popcll(current_leafmask);
				}
				if (stop_level == nof_levels - 2 && all_colors) {
					final_color_idx = nof_leaves + n_child_offset;
					break;
				}
				if (all_colors) n_child_offset += 1; // 4x4x4 node
				uint32_t n_index = dag[node_index + 1 + n_offset];
				uint32_t leafmask0 = dag[n_index];
				uint32_t leafmask1 = dag[n_index + 1];
				uint64_t current_leafmask = uint64_t(leafmask1) << 32 | uint64_t(leafmask0);
				uint8_t final_idx = (((path.x & 0x1) == 0) ? 0 : 4) |
					(((path.y & 0x1) == 0) ? 0 : 2) |
					(((path.z & 0x1) == 0) ? 0 : 1) |
					(((path.x & 0x2) == 0) ? 0 : 32) |
					(((path.y & 0x2) == 0) ? 0 : 16) |
					(((path.z & 0x2) == 0) ? 0 : 8);
				// Masked leafmask must contain our bit since we need to account for the 2x2x2
				// node that we reside in. 
				if (all_colors){
					uint64_t masked_leafmask;
					if (stop_level == nof_levels - 1) {
						masked_leafmask = current_leafmask & ((1ull << ((final_idx / 8) * 8)) - 1ull);
					}
					else {
						masked_leafmask = current_leafmask & ((uint64_t(1) << final_idx) - uint64_t(1));
						masked_leafmask |= ((uint64_t(1) << final_idx));
					}
					if (all_colors) { // 2x2x2 nodes
						n_child_offset += (masked_leafmask & 0x00000000000000FF) == 0 ? 0 : 1;
						n_child_offset += (masked_leafmask & 0x000000000000FF00) == 0 ? 0 : 1;
						n_child_offset += (masked_leafmask & 0x0000000000FF0000) == 0 ? 0 : 1;
						n_child_offset += (masked_leafmask & 0x00000000FF000000) == 0 ? 0 : 1;
						n_child_offset += (masked_leafmask & 0x000000FF00000000) == 0 ? 0 : 1;
						n_child_offset += (masked_leafmask & 0x0000FF0000000000) == 0 ? 0 : 1;
						n_child_offset += (masked_leafmask & 0x00FF000000000000) == 0 ? 0 : 1;
						n_child_offset += (masked_leafmask & 0xFF00000000000000) == 0 ? 0 : 1;
					}
					// 1x1x1 nodes
					n_child_offset += __popcll(masked_leafmask);
					if (stop_level >= nof_levels) n_child_offset -= 1;
					final_color_idx = nof_leaves + n_child_offset;
				}
				else {
					final_color_idx = nof_leaves + n_child_offset+__popcll(current_leafmask & ((uint64_t(1) << final_idx) - uint64_t(1)));;// +n_offset + ;// 
				}
				break;
			}

			// Otherwise, fetch the next node (and accumulate leaves we pass by)
			uint8_t test_mask = child_mask & ((1 << child_idx) - 1);
			uint32_t node_offset = 0;
			if (all_colors) nof_leaves += 1; // This node
			while (test_mask != 0) {
				uint8_t next_child = 31 - __clz(test_mask);

				uint32_t upper_bits = dag[dag[node_index + 1 + node_offset++]] >> 8;

				// If we are in the top-levels, the obtained value is an index into an 
				// external array
				if (level < nof_top_levels) {
					nof_leaves += enclosed_leaves[upper_bits];
				}
				else {
					nof_leaves += upper_bits;
				}

				test_mask &= ~(1 << next_child);
			}
			node_offset = __popc(child_mask & ((1 << child_idx) - 1));
			node_index = dag[node_index + 1 + node_offset];
		}

		color = dag_color[final_color_idx];
	}
	surf2Dwrite(color, output_image, (int)sizeof(uint32_t)*coord.x, coord.y, cudaBoundaryModeClamp);
}


///////////////////////////////////////////////////////////////////////////////
// Trace primary rays
///////////////////////////////////////////////////////////////////////////////
struct render_param {
	glm::vec3 camera_pos;
	glm::vec3 p_bottom_left;
	double3 d_dx;
	double3 d_dy;
	render_param(const chag::view &camera, const dag::DAG &dag, uint32_t w, uint32_t h) {
		///////////////////////////////////////////////////////////////////////////
		// Calculate the camera position and three points spanning the near quad
		///////////////////////////////////////////////////////////////////////////
		camera_pos                   = camera.pos;
		glm::vec3 camera_dir         = -camera.R[2];
		glm::vec3 camera_up          = camera.R[1];
		glm::vec3 camera_right       = camera.R[0];
		float camera_fov             = camera.m_fov / 2.0f * (float(M_PI) / 180.0f);
		float camera_aspect_ratio    = float(w) / float(h);
		glm::vec3 Z                  = camera_dir * cos(camera_fov);
		glm::vec3 X                  = camera_right * sin(camera_fov) * camera_aspect_ratio;
		glm::vec3 Y                  = camera_up * sin(camera_fov);
		p_bottom_left                = camera_pos + Z - Y - X;
		glm::vec3 p_top_left         = camera_pos + Z + Y - X;
		glm::vec3 p_bottom_right     = camera_pos + Z - Y + X;

		///////////////////////////////////////////////////////////////////////////
		// Transform these points into "DAG" space and generate pixel dx/dy
		///////////////////////////////////////////////////////////////////////////
		glm::vec3 translation = -dag.m_aabb.min;
		float fres            = float(dag.geometryResolution());
		glm::vec3 scale       = glm::vec3(fres) / glm::vec3(dag.m_aabb.getHalfSize() * 2.0f);
		camera_pos            = (camera_pos     + translation) * scale;
		p_bottom_left         = (p_bottom_left  + translation) * scale;
		p_top_left            = (p_top_left     + translation) * scale;
		p_bottom_right        = (p_bottom_right + translation) * scale;

		auto to_double3 = [](auto v) {return make_double3(v.x, v.y, v.z); };
		d_dx                  = to_double3(p_bottom_right - p_bottom_left) * (1.0 / double(w));
		d_dy                  = to_double3(p_top_left     - p_bottom_left) * (1.0 / double(h));
	}
};

void DAGTracer::resolve_paths(const dag::DAG &dag, const chag::view & camera, int color_lookup_level)
{
	m_color_buffer.mapSurfaceObject();
	m_path_buffer.mapSurfaceObject();
	m_depth_buffer.mapSurfaceObject();

		auto to_float3 = [](const glm::vec3 &v){
				return make_float3(v.x, v.y, v.z);
		};

	render_param rp(camera, dag, m_width, m_height);

	dim3 block_dim = dim3(8, 32);
	dim3 grid_dim = dim3(m_width / block_dim.x + 1, m_height / block_dim.y + 1);
	primary_rays_kernel <<<grid_dim, block_dim >>>(
		m_width,
		m_height,
		to_float3(rp.camera_pos),
		to_float3(rp.p_bottom_left),
		rp.d_dx,
		rp.d_dy,
		dag.d_data,
		dag.nofGeometryLevels(),
		m_path_buffer.m_cuda_surface_object
#if NEXT_CHILD_LUT
		, d_next_child_lookup_table
#endif
		,
		dag.colors_in_all_nodes ? color_lookup_level : dag.nofGeometryLevels()
		);
	m_color_buffer.unmapSurfaceObject(); 
	m_path_buffer.unmapSurfaceObject(); 
	m_depth_buffer.unmapSurfaceObject();
}

void DAGTracer::resolve_colors(const dag::DAG &dag, int color_lookup_level)
{
	m_color_buffer.mapSurfaceObject();
	m_path_buffer.mapSurfaceObject();
	dim3 block_dim = dim3(8, 32);
	dim3 grid_dim = dim3(m_width / block_dim.x + 1, m_height / block_dim.y + 1);
	color_lookup_kernel_morton << <grid_dim, block_dim >> >(
			m_width,
			m_height,
			dag.nofGeometryLevels(),
			m_path_buffer.m_cuda_surface_object,
			dag.d_data,
			dag.d_color_data,
			dag.d_enclosed_leaves,
			dag.m_top_levels,
			m_color_buffer.m_cuda_surface_object,
			dag.colors_in_all_nodes,
			dag.colors_in_all_nodes ? color_lookup_level : dag.nofGeometryLevels()
		);
	m_color_buffer.unmapSurfaceObject(); 
	m_path_buffer.unmapSurfaceObject(); 
}
