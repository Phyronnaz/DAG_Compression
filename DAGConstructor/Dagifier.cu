#include "Dagifier.h"
// C-STD
#include <stdint.h>
#include <intrin.h>
#include <algorithm>
// CUDA
#include <cuda_runtime.h>
// THRUST
#include <thrust/functional.h>
#include <thrust/device_ptr.h>
#include <thrust/functional.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/gather.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/scan.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/unique.h>
#include <thrust/adjacent_difference.h>
// PROJECT
#include "../CudaHelpers.h" //FIXME: Proper search paths
#include "../hash.h"        //FIXME: Proper search paths

struct dagifier::impl {
	std::size_t m_num_colors;
	int m_cached_num_colors;
	int m_cached_frag_count;
	float4 *m_cu_cmpct_col_buf    = nullptr;
	uint64_t *m_cu_cmpct_mask_buf = nullptr;
	thrust::device_vector<float4> compact_colors;
	thrust::device_vector<uint64_t> compact_masks;

	thrust::device_vector<uint64_t> child_sort_key;
	thrust::device_vector<uint64_t> parent_sort_key;
	thrust::device_vector<uint32_t> unique_pos;
	thrust::device_vector<uint32_t> first_child_pos;
	thrust::device_vector<uint32_t> child_dag_idx;
	thrust::device_vector<uint32_t> compact_dag;
	thrust::device_vector<uint32_t> path;
	thrust::device_vector<float4>   color;
	thrust::device_vector<uint32_t> parent_paths;
	thrust::device_vector<uint32_t> parent_node_size;
	thrust::device_vector<uint32_t> parent_svo_idx;
	thrust::device_vector<uint32_t> sorted_orig_pos;
	thrust::device_vector<uint32_t> sorted_parent_node_size;
	thrust::device_vector<uint32_t> sorted_parent_svo_idx;
	thrust::device_vector<uint32_t> unique_size;
	thrust::device_vector<uint32_t> parent_dag_idx;

	thrust::device_vector<uint32_t> parent_svo_nodes;

	//////////////////
	uint32_t *m_child_paths;
	uint64_t *m_childmasks;
	uint32_t *m_parent_paths;
	uint32_t *m_first_child_pos;
	uint32_t *m_parent_node_size;
	uint32_t *m_parent_svo_idx;
	uint32_t *m_parent_svo_nodes;
	uint64_t *m_parent_sort_key;
	uint32_t *m_sorted_parent_node_size;
	uint32_t *m_sorted_parent_svo_idx;
	uint64_t *m_sorted_parent_sort_key;
	uint64_t *m_child_sort_key;
	uint32_t *m_unique_size;
	uint32_t *m_unique_pos;
	uint32_t *m_parent_dag_idx;
	uint32_t *m_child_dag_idx;
	uint32_t *m_sorted_orig_pos;
	int *m_unique_parent_nodes;

	uint32_t *m_compact_dag;

	int m_parent_svo_size;
	std::size_t m_child_svo_size;

	// New ------
	int m_levels;
	//-----------

	void map_resources(size_t child_svo_size);
	void initDag(int *child_level_start_offset, int *parent_level_start_offset);
	void buildDAG(int bottomLevel, int *parent_level_start_offset, int *child_level_start_offset);
	void printStatisticsComplete(int total_dag_size, int total_svo_size);
	void printStatisticsPartial(int lvl, int bottomLevel, int unique_parent_nodes, int final_level_size,
	                            int *total_dag_size, int *total_svo_size);

	void build_parent_level(int lvl, int bottomLevel);
	void create_dag_nodes(int parent_level_start_offset, int child_level_start_offset);
	uint32_t count_child_nodes(int lvl, int bottomlevel, uint32_t node_idx, std::vector<uint32_t> *dag);

	std::size_t sort_and_merge_fragments(std::size_t count);
	dag::DAG build_dag(int count, int depth, const chag::Aabb &aabb);
};

__device__ uint32_t childBits(const uint32_t &val) { return uint32_t(val & 0x7); }


__global__ void build_parent_level_kernel(uint32_t *d_parent_svo_nodes,
                                          uint32_t *d_parent_svo_idx,
										  uint32_t *d_child_paths, 
                                          uint32_t *d_first_child_pos,
										  uint32_t *d_child_dag_idx,
                                          uint64_t *d_child_sort_key,
										  uint64_t *d_parent_sort_key, 
                                          int parent_svo_size,
                                          int child_svo_size,
                                          int lvl,
										  int bottomLevel) 
{
	int thread = getGlobalIdx_1D_1D();
	if (thread < parent_svo_size) {
		uint32_t nof_children;
		if (thread == parent_svo_size - 1)
			nof_children = child_svo_size - d_first_child_pos[thread];
		else
			nof_children = d_first_child_pos[thread + 1] - d_first_child_pos[thread];

		uint64_t hash_key[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
		int ctr              = 0;

		d_parent_svo_nodes[d_parent_svo_idx[thread]] = 0;
		for (uint32_t i = 0; i < nof_children; i++) {
			uint32_t childpath = d_child_paths[d_first_child_pos[thread] + i];
			uint32_t child_idx = childBits(childpath);
			d_parent_svo_nodes[d_parent_svo_idx[thread]] |= (1 << child_idx);
			if (lvl != bottomLevel) {
				d_parent_svo_nodes[d_parent_svo_idx[thread] + 1 + i] = d_child_dag_idx[d_first_child_pos[thread] + i];
				hash_key[ctr++]                                      = d_child_sort_key[d_first_child_pos[thread] + i];
			}
		}
		if (lvl == bottomLevel) {
			d_parent_sort_key[thread] = d_parent_svo_nodes[d_parent_svo_idx[thread]];
		} else {
			hash_key[ctr++] = d_parent_svo_nodes[d_parent_svo_idx[thread]];
			uint64_t hash_value[2];
			MurmurHash3_x64_128(&hash_key[0], 9 * sizeof(uint64_t), 0x0, hash_value);
			d_parent_sort_key[thread] = hash_value[0];
		}
	}
}

__global__ void create_dag_nodes_kernel(uint64_t *d_parent_sort_key, 
                                        uint32_t *d_parent_node_size,
										uint32_t *d_unique_pos, 
                                        uint32_t *d_parent_svo_idx,
										uint32_t *d_parent_svo_nodes,
                                        uint32_t *d_parent_dag_idx,
										uint32_t *d_compact_dag, 
                                        uint32_t *d_sorted_orig_pos,
										uint32_t parent_level_start_offset, 
                                        uint32_t child_level_start_offset,
										int parent_svo_size)
{
	int thread = getGlobalIdx_1D_1D();
	if (thread < parent_svo_size) {
		bool first_unique = true;
		if (thread != 0) first_unique = d_parent_sort_key[thread] != d_parent_sort_key[thread - 1];
		uint32_t dag_index = d_parent_dag_idx[d_sorted_orig_pos[thread]] = d_unique_pos[thread];
		if (first_unique) {
			// Write childmask
			d_compact_dag[parent_level_start_offset + dag_index] = d_parent_svo_nodes[d_parent_svo_idx[thread]];
			// Write indices
			for (uint32_t i = 1; i < d_parent_node_size[thread]; i++) {
				d_compact_dag[parent_level_start_offset + dag_index + i] =
					child_level_start_offset + d_parent_svo_nodes[d_parent_svo_idx[thread] + i];
			}
		}
	}
}

void dagifier::impl::build_parent_level(int lvl, int bottomLevel) {
	dim3 blockDim = dim3(256);
	int a         = m_parent_svo_size / blockDim.x;
	int b         = m_parent_svo_size % blockDim.x != 0 ? 1 : 0;
	int c         = a + b;
	dim3 gridDim = dim3(c);
	build_parent_level_kernel<<<gridDim, blockDim>>>(m_parent_svo_nodes,
                                                     m_parent_svo_idx,
                                                     m_child_paths,
                                                     m_first_child_pos,
                                                     m_child_dag_idx,
                                                     m_child_sort_key,
                                                     m_parent_sort_key,
                                                     m_parent_svo_size,
                                                     m_child_svo_size,
                                                     lvl,
                                                     bottomLevel);
}


void dagifier::impl::create_dag_nodes(int parent_level_start_offset, int child_level_start_offset) {
	dim3 blockDim = dim3(256);
	int a         = m_parent_svo_size / blockDim.x;
	int b         = m_parent_svo_size % blockDim.x != 0 ? 1 : 0;
	int c         = a + b;
	dim3 gridDim  = dim3(c);

	create_dag_nodes_kernel<<<gridDim, blockDim>>>(m_parent_sort_key,
                                                   m_sorted_parent_node_size,
                                                   m_unique_pos,
												   m_sorted_parent_svo_idx,
                                                   m_parent_svo_nodes,
                                                   m_parent_dag_idx,
												   m_compact_dag,
                                                   m_sorted_orig_pos,
                                                   parent_level_start_offset,
												   child_level_start_offset,
                                                   m_parent_svo_size);
}



void dagifier::impl::map_resources(size_t child_svo_size) {
	m_child_svo_size = child_svo_size;
	size_t free, total;
	cudaMemGetInfo(&free, &total);

    compact_dag.resize(10*1024*1024);                 m_compact_dag             = compact_dag.data().get();
    parent_svo_nodes.resize(3*m_child_svo_size);      m_parent_svo_nodes        = parent_svo_nodes.data().get();
	sorted_parent_node_size.resize(m_child_svo_size); m_sorted_parent_node_size = sorted_parent_node_size.data().get();
	sorted_parent_svo_idx.resize(m_child_svo_size);   m_sorted_parent_svo_idx   = sorted_parent_svo_idx.data().get();
    parent_node_size.resize(m_child_svo_size);        m_parent_node_size        = parent_node_size.data().get();
    first_child_pos.resize(m_child_svo_size);         m_first_child_pos         = first_child_pos.data().get();
    parent_sort_key.resize(m_child_svo_size);         m_parent_sort_key         = parent_sort_key.data().get();
    sorted_orig_pos.resize(m_child_svo_size);         m_sorted_orig_pos         = sorted_orig_pos.data().get();
    parent_svo_idx.resize(m_child_svo_size);          m_parent_svo_idx          = parent_svo_idx.data().get();
	parent_dag_idx.resize(m_child_svo_size);          m_parent_dag_idx          = parent_dag_idx.data().get();
	child_sort_key.resize(m_child_svo_size);          m_child_sort_key          = child_sort_key.data().get();
    child_dag_idx.resize(m_child_svo_size);           m_child_dag_idx           = child_dag_idx.data().get();
    parent_paths.resize(m_child_svo_size);            m_parent_paths            = parent_paths.data().get();
	unique_size.resize(m_child_svo_size);             m_unique_size             = unique_size.data().get();
    unique_pos.resize(m_child_svo_size);              m_unique_pos              = unique_pos.data().get();
	cudaMemGetInfo(&free, &total);
}

void dagifier::impl::initDag(int *child_level_start_offset, int *parent_level_start_offset) {
	constexpr std::size_t child_start_offset = 9;
    thrust::copy_n(compact_masks.cbegin(), m_child_svo_size, child_sort_key.begin());  // Used for later
	thrust::sequence(unique_pos.begin(), unique_pos.begin() + m_child_svo_size);        // Index to masks
	thrust::sort_by_key(compact_masks.begin(), compact_masks.begin() + m_child_svo_size, unique_pos.begin());  // Sort masks AND index

	thrust::adjacent_difference(compact_masks.cbegin(), compact_masks.cbegin() + m_child_svo_size, first_child_pos.begin(), 2 * (thrust::placeholders::_1 != thrust::placeholders::_2)); // Mark unique masks as 2, since the 64-bit mask will be represented as 2x32-bit words in the DAG.
    first_child_pos[0] = 0;                                                                                                                                         // Set first to zero for inclusive scan (prefix sum)

    thrust::inclusive_scan(first_child_pos.cbegin(), first_child_pos.cbegin() + m_child_svo_size, thrust::make_permutation_iterator(child_dag_idx.begin(), unique_pos.cbegin())); // child_dag_idx[unique_pos[i]] = sum_at[i]
	const std::size_t num_unique = thrust::distance(compact_masks.begin(),thrust::unique(compact_masks.begin(), compact_masks.begin() + m_child_svo_size));                       // Reduce to unique masks
    cudaMemcpy((compact_dag.data().get() + child_start_offset), compact_masks.data().get(), num_unique*sizeof(uint64_t), cudaMemcpyDeviceToDevice);          // Quick and dirty memcpy as I do not know the thrust way of doing this..

	*child_level_start_offset  = child_start_offset;                   // Not used in first pass
	*parent_level_start_offset = child_start_offset + 2 * num_unique;  // Make room for a copied root node
}

namespace IdentifyParents{
using InParam  = thrust::tuple<uint32_t, uint32_t>;
struct equal_to{
	__host__ __device__ bool operator()(const InParam &lhs, const InParam &rhs) const { return lhs.get<1>() == rhs.get<1>(); }
};

using Iter = thrust::device_vector<uint32_t>::iterator;
auto InBegin(Iter path){
	return thrust::make_zip_iterator(
        thrust::make_tuple(thrust::make_counting_iterator(0ull),
                           thrust::make_transform_iterator(path, thrust::placeholders::_1 >> 3)));
}
auto InEnd(Iter path, std::size_t count) {
	return thrust::make_zip_iterator(
	    thrust::make_tuple(thrust::make_counting_iterator(count),
	                       thrust::make_transform_iterator(path + count, thrust::placeholders::_1 >> 3)));
}
auto OutBegin(Iter first_child_pos_, Iter parent_paths_){
	return thrust::make_zip_iterator(thrust::make_tuple(first_child_pos_, parent_paths_));
}
}  // namespace IdentifyParents


namespace FirstUnique {
using InParam1 = uint64_t;
using InParam2 = thrust::tuple<uint64_t, uint32_t>;
using OutParam = uint32_t;
struct functor : public thrust::binary_function<InParam1, InParam2, OutParam> {
	__host__ __device__ OutParam operator()(const InParam1 &lhs, const InParam2 &rhs) const {
		return lhs != rhs.get<0>() ? rhs.get<1>() : 0;
	}
};
}  // namespace FirstUnique


void dagifier::impl::buildDAG(int bottomLevel, int *parent_level_start_offset, int *child_level_start_offset) {
	initDag(child_level_start_offset, parent_level_start_offset);
	int prev_child_svo_size = m_child_svo_size;

	for (int lvl = bottomLevel - 1; lvl >= 0; lvl--) {
		int final_level_size;
		{
            // Copy unique parent paths and their children indexes
			m_parent_svo_size = thrust::distance(IdentifyParents::OutBegin(first_child_pos.begin(), parent_paths.begin()), 
                                    thrust::unique_copy(IdentifyParents::InBegin(path.begin()), 
                                                        IdentifyParents::InEnd(path.begin(), m_child_svo_size),
			                                            IdentifyParents::OutBegin(first_child_pos.begin(), parent_paths.begin()),
                                                        IdentifyParents::equal_to()));

			if(lvl == bottomLevel){
                thrust::fill_n(parent_node_size.begin(), m_parent_svo_size, 1);
            } else {
                using namespace thrust::placeholders;
				auto first_child_pos_back   = first_child_pos.begin()  + m_parent_svo_size-1;
				auto first_child_pos_end    = first_child_pos.begin()  + m_parent_svo_size;
				auto first_child_pos_second = first_child_pos.begin()  + 1;
				auto parent_node_size_back  = parent_node_size.begin() + m_parent_svo_size-1;
                thrust::transform(first_child_pos_second, first_child_pos_end, first_child_pos.begin(), parent_node_size.begin(), _1 - _2 + 1);
				*parent_node_size_back = m_child_svo_size - *first_child_pos_back + 1;
            }
            
            thrust::exclusive_scan(parent_node_size.begin(), parent_node_size.begin() + m_parent_svo_size, parent_svo_idx.begin());

			build_parent_level(lvl, bottomLevel);

			// Need to copy m_parent_sort_key to m_child_sort_key now, as m_parent_sort_key will become sorted.
			// We don't want in m_child_sort_key to be sorted..
			thrust::copy_n(parent_sort_key.begin(), m_parent_svo_size, child_sort_key.begin());

            thrust::sequence(sorted_orig_pos.begin(), sorted_orig_pos.begin() + m_parent_svo_size);
            thrust::sort_by_key(parent_sort_key.begin(), parent_sort_key.begin()+m_parent_svo_size, sorted_orig_pos.begin());


            thrust::gather(sorted_orig_pos.begin(), sorted_orig_pos.begin()+m_parent_svo_size, parent_node_size.begin(), sorted_parent_node_size.begin());
            thrust::gather(sorted_orig_pos.begin(), sorted_orig_pos.begin()+m_parent_svo_size, parent_svo_idx.begin(), sorted_parent_svo_idx.begin());

            thrust::transform(parent_sort_key.begin() + 1, parent_sort_key.begin() + m_parent_svo_size,
                              thrust::make_zip_iterator(thrust::make_tuple(parent_sort_key.begin(), sorted_parent_node_size.begin())),
                              unique_size.begin() + 1, FirstUnique::functor());
            unique_size[0] = 0;

            thrust::inclusive_scan(unique_size.begin(), unique_size.begin() + m_parent_svo_size, unique_pos.begin());

			final_level_size = unique_pos[m_parent_svo_size-1] + sorted_parent_node_size[m_parent_svo_size-1];

			create_dag_nodes(*parent_level_start_offset, *child_level_start_offset);

            thrust::copy_n(parent_dag_idx.begin(), m_parent_svo_size, child_dag_idx.begin());
			thrust::copy_n(parent_paths.begin(), prev_child_svo_size, path.begin());
		}

		///////////////////////////////////////////////////////////////////
		// New level
		///////////////////////////////////////////////////////////////////
		prev_child_svo_size       = m_child_svo_size;
		m_child_svo_size          = m_parent_svo_size;
		*child_level_start_offset = *parent_level_start_offset;
		*parent_level_start_offset += final_level_size;
	}
}

uint32_t dagifier::impl::count_child_nodes(int lvl, int bottomlevel, uint32_t node_idx, std::vector<uint32_t> *dag) {
	if (lvl == bottomlevel) {
		auto a = __popcnt((*dag)[node_idx]);
		auto b = __popcnt((*dag)[node_idx + 1]);
		return a+b; 
	}
	uint32_t n_children = 0;

	uint32_t set_bit            = 1;
	uint32_t idx                = 0;
	uint32_t *current_node_mask = &(*dag)[node_idx];
	do {
		if (set_bit & *current_node_mask) {
			const auto &child_idx = (*dag)[++idx + node_idx];
			n_children += count_child_nodes(lvl + 1, bottomlevel, child_idx, dag);
		}
		set_bit = set_bit << 1;
	} while (set_bit != (1 << 8));

	*current_node_mask |= n_children << 8;

	return n_children;
}


void dagifier::impl::printStatisticsPartial(int lvl, int bottomLevel, int unique_parent_nodes, int final_level_size,
									  int *total_dag_size, int *total_svo_size) {
	///////////////////////////////////////////////////////////////////
	// Statistics (assuming 4x4x4 leaf nodes)
	///////////////////////////////////////////////////////////////////
	if (lvl != bottomLevel) {
		if (lvl == bottomLevel - 1) {
			*total_dag_size = unique_parent_nodes * sizeof(uint64_t);
			*total_svo_size = m_parent_svo_size * sizeof(uint64_t);
		} else {
			*total_dag_size += final_level_size * sizeof(uint32_t);
			*total_svo_size += m_parent_svo_size * 2 * sizeof(uint32_t);
		}
	}
	// Assuming an SVO node is two uints
	int percent =
		int(100.0f * float(final_level_size * sizeof(uint32_t)) / float(m_parent_svo_size * 2 * sizeof(uint32_t)));
	std::cout << "Level " << lvl << " (bytes): SVO = " << m_parent_svo_size * 2 * sizeof(uint32_t)
			  << ", DAG = " << final_level_size * sizeof(uint32_t) << " [" << percent << "%]\n";
}


void dagifier::impl::printStatisticsComplete(int total_dag_size, int total_svo_size) {
	int percent = int(100.0f * float(total_dag_size) / float(total_svo_size));
	std::cout << "Total Size (bytes, assuming 4x4x4 leafs): SVO = " << total_svo_size << ", DAG = " << total_dag_size
			  << " [" << percent << "%]\n";
}

struct scopedStream {
	const cudaStream_t &get() const noexcept { return s; }
	scopedStream() { cudaStreamCreate(&s); }
	~scopedStream() { cudaStreamDestroy(s); }

 private:
	cudaStream_t s;
};

__device__ inline float4 &operator+=(float4 &a, const float4 &b) {
	a.x += b.x;
	a.y += b.y;
	a.z += b.z;
	a.w += b.w;
	return a;
}

__device__ inline float4 operator+(const float4 &a, const float4 &b) {
	return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}

struct average_color : public thrust::unary_function<float4, float4> { //FIXME: thrust unary deprecatedm use std..
	__device__ float4 operator()(const float4 &a) { return make_float4(a.x / a.w, a.y / a.w, a.z / a.w, 1.0f); }
};


std::size_t dagifier::impl::sort_and_merge_fragments(std::size_t count) {
	namespace tc = thrust::cuda;
    thrust::device_vector<uint32_t> th_sorted_index(count, 0);
	thrust::device_vector<uint32_t> th_unique(count,0);
	thrust::device_vector<uint32_t> th_pre_inc_sum(count,0);
	thrust::device_vector<uint64_t> th_sorted_masks(count,0);
	thrust::device_vector<float4>   th_sorted_color(count, make_float4(0.0f, 0.0f, 0.0f, 0.0f));

	thrust::sequence(tc::par, th_sorted_index.begin(), th_sorted_index.end());       // IOTA.
	thrust::sort_by_key(tc::par, path.begin(), path.begin() + count, th_sorted_index.begin());  // Sorts value AND key.
	{
		// scopedStream s1, s2, s3;
		// TODO: Read up on streams
		thrust::transform(tc::par, path.begin(), path.begin() + count, th_sorted_masks.begin(), 1ull << (thrust::placeholders::_1 & 0x3F)); // Figure out set bit in 64-bit leaf
		thrust::gather(tc::par, th_sorted_index.cbegin(), th_sorted_index.cend(), color.begin(), th_sorted_color.begin());           // Sort colors
	    thrust::adjacent_difference(tc::par, path.begin(), path.begin() + count, th_unique.begin(), thrust::not_equal_to<uint32_t>());      // Mark unique paths with 1
	}
	th_unique[0] = 0;                                                                                                           // First element need to be zero for inclusive scan (prefix sum)
	std::size_t unique_frag =
	    1 + *(thrust::inclusive_scan(tc::par, th_unique.cbegin(), th_unique.cend(), th_pre_inc_sum.begin()) - 1);               // Creates a mapping to compact elements
	m_num_colors = unique_frag;


    if(compact_colors.size() < m_num_colors){ 
        compact_colors.resize(m_num_colors); 
        m_cu_cmpct_col_buf = compact_colors.data().get();
    }
    

	// Reduce and unique works here because data is are sorted.
	// TODO: These can be done in paralell by later transforms..
	auto compact_colors_end = 
        thrust::reduce_by_key(tc::par, th_pre_inc_sum.cbegin(), th_pre_inc_sum.cend(), 
                              th_sorted_color.begin(), 
                              thrust::make_discard_iterator(), compact_colors.begin()).second;                       // Reduce identical colors by sum
	thrust::transform(tc::par, compact_colors.begin(), compact_colors_end, compact_colors.begin(), average_color()); // And average the sum (N is in color.w)

	// Truncate positions and collect unique
	thrust::transform(tc::par, path.begin(), path.begin() + count, path.begin(), thrust::placeholders::_1 >> 6);                          // Calculate parent paths
	thrust::adjacent_difference(tc::par, path.begin(), path.begin() + count, th_unique.begin(), thrust::not_equal_to<uint32_t>());  // Mark unique parents with 1
	th_unique[0] = 0;                                                                                                   // First element need to be zero for inclusive scan (prefix sum)

	{
		// scopedStream s1, s2;
		// TODO: Read up on streams
		thrust::inclusive_scan(tc::par, th_unique.cbegin(), th_unique.cend(), th_pre_inc_sum.begin());  // Creates a mapping to compact elements
		m_child_svo_size = thrust::distance(path.begin(), thrust::unique(tc::par, path.begin(), path.begin() + count));   // Reduce identical parents
	}

	// FIXME: Guess we need to move these allocations elsewhere..
	if (compact_masks.size() < m_child_svo_size) {
        compact_masks.resize(m_child_svo_size);
        m_cu_cmpct_mask_buf = compact_masks.data().get();
	}

	thrust::reduce_by_key(tc::par, th_pre_inc_sum.cbegin(), th_pre_inc_sum.cend(), th_sorted_masks.cbegin(),                                // Reduce them according to parents by bitwise or
	                      thrust::make_discard_iterator(), compact_masks.begin(), thrust::equal_to<uint32_t>(), thrust::bit_or<uint64_t>());
   
	m_child_paths = path.data().get();
	m_childmasks  = m_cu_cmpct_mask_buf;

	return m_child_svo_size;
}

dag::DAG dagifier::impl::build_dag(int count, int depth, const chag::Aabb &aabb) {
	dag::DAG result;
	result.m_levels = depth;
	result.m_aabb   = aabb;
	size_t n_paths  = sort_and_merge_fragments(count);
	// size_t n_paths = sort_and_merge_fragments(d_pos, d_mask, d_color, count);

	///////////////////////////////////////////////////////////////////////////
	// Setup
	///////////////////////////////////////////////////////////////////////////
	result.m_colors.resize(m_num_colors);
	std::vector<glm::vec4> f4_colors(m_num_colors);
	map_resources(n_paths);
	int child_level_start_offset;
	int parent_level_start_offset;
	buildDAG(result.m_levels, &parent_level_start_offset, &child_level_start_offset);
	{
		result.m_data.resize(parent_level_start_offset);  // Allocate huge output DAG

		cudaMemcpy(result.m_data.data(), compact_dag.data().get(), result.m_data.size() * sizeof(uint32_t),
		           cudaMemcpyDeviceToHost);
		// TODO: float4/vec4 conversion safety..
		cudaMemcpy(f4_colors.data(), compact_colors.data().get(), f4_colors.size() * sizeof(glm::vec4),
		           cudaMemcpyDeviceToHost);

		auto to_uint = [](float v) { return uint32_t(std::min(1.0f, std::max(v, 0.0f)) * 255.0f); };
		for (size_t i = 0; i < f4_colors.size(); ++i) {
			const auto &c      = f4_colors[i];
			result.m_colors[i] = to_uint(c.z / c.w) << 16 | to_uint(c.y / c.w) << 8 | to_uint(c.x / c.w);
		}

		///////////////////////////////////////////////////////////////////////
		// Copy root node to beginning
		///////////////////////////////////////////////////////////////////////
		int total_size = parent_level_start_offset;
		int root_start = child_level_start_offset;
		for (int i = 0; i < total_size - root_start; i++) { result.m_data[i] = result.m_data[root_start + i]; }

		count_child_nodes(0, result.m_levels, 0, &result.m_data);
		result.m_data.shrink_to_fit();
		result.m_colors.shrink_to_fit();
	}
	return result;
}


// Base wrapper.
dag::DAG dagifier::build_dag(const std::vector<uint32_t> &morton_paths,
                             const std::vector<float> &colors,
                             int count, int depth, const chag::Aabb &aabb){
	p_impl_->path.resize(count);
	cudaMemcpy(p_impl_->path.data().get(), morton_paths.data(), count * sizeof(uint32_t), cudaMemcpyHostToDevice);
	p_impl_->color.resize(count);
    cudaMemcpy(p_impl_->color.data().get(), colors.data(), count * sizeof(float4), cudaMemcpyHostToDevice);
	return p_impl_->build_dag(count, depth, aabb);
}

dag::DAG dagifier::build_dag(uint32_t *d_pos, float4 *d_color, int count, int depth, const chag::Aabb &aabb) {
	p_impl_->path.resize(count);
	thrust::copy_n(thrust::device_ptr<uint32_t>{d_pos}, count, p_impl_->path.begin());
	p_impl_->color.resize(count);
	thrust::copy_n(thrust::device_ptr<float4>{d_color}, count, p_impl_->color.begin());
    return p_impl_->build_dag(count, depth, aabb);
}
dagifier::dagifier() : p_impl_{std::make_unique<impl>()} {}
dagifier::~dagifier() {}
