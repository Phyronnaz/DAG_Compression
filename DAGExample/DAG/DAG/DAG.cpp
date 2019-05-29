#include "DAG.h"
#include <algorithm>
#include <bitset>
#include <limits>

using namespace std;

template <typename T>
unsigned popcnt_safe(T v) {
	return static_cast<unsigned>(std::bitset<std::numeric_limits<T>::digits>(v).count());
}


namespace dag {

bool calculate_sizes(const int node_idx, const int level, const std::vector<uint32_t> &dag, const uint32_t &nof_levels,
										 uint64_t &pointerless_svo_size, uint64_t &traversable_svo_size, uint64_t &dag_size,
										 uint64_t &leaf_voxels) {
	static std::vector<bool> visited(dag.size(), false);

	if (level == nof_levels - 2) {
		// This is a 4x4x4 leaf node (64 bits)
		pointerless_svo_size += 8;
		traversable_svo_size += 8;
		leaf_voxels += popcnt_safe(dag[node_idx]) + popcnt_safe(dag[node_idx + 1]);
		if (!visited[node_idx]) dag_size += 8;
		visited[node_idx] = true;
		return true;
	} else {
		uint32_t childmask = dag[node_idx] & 0xFF;
		int nof_subnodes   = popcnt_safe(childmask);
		pointerless_svo_size += 1;
		traversable_svo_size += 4;
		if (!visited[node_idx]) dag_size += 4;
		if (nof_subnodes > 0) traversable_svo_size += 4;
		for (int i = 0; i < nof_subnodes; i++) {
			if (!visited[node_idx]) dag_size += 4;
			calculate_sizes(dag[node_idx + 1 + i], level + 1, dag, nof_levels, pointerless_svo_size,
											traversable_svo_size, dag_size, leaf_voxels);
		}
		visited[node_idx] = true;
		return true;
	}
}

void DAG::calculateColorForAllNodes() {
	colors_in_all_nodes = true;
	// Traverse the whole DAG. Keep track of:
	// * number of leaf-nodes so far
	// * number of nodes so far
	// At each visited node:
	// * if in toplevels, update there
	// * else update upper 24 bits of node
	// * at the final level (64 bits) update the node, but no
	//   extra info is written for each 2x2x2 node. These are handled
	//   by the color lookup.
	//
	// When we hit an actual leaf node, push the color to the (current)
	// parents list. when have processed a whole node, calculate the
	// average color and push it to the parents list.

	////////////////////////////////////////////////////
	// PASS 1: Calculate new color index for each node
	////////////////////////////////////////////////////
	uint32_t n_child_offset = 0;
	for(int level = m_levels - 1 ; level >=0; --level)
	{
		for( uint32_t node_index = 0; node_index<m_data[level].size();)
		{
			uint32_t &node_mask = m_data[level][node_index];
			unsigned n_children = popcnt_safe(node_mask & 0xFF);
			// Current node
			n_child_offset = 1;
			if (level == m_levels - 1) 
			{
				for(int i = 0; i<n_children; ++i)
				{
					n_child_offset += 1; // 4x4x4 node
					const uint32_t n_index          = m_data[level][node_index + 1 + i];
					const uint32_t leafmask0        = m_data[level+1][n_index];
					const uint32_t leafmask1        = m_data[level+1][n_index + 1];
					const uint64_t current_leafmask = uint64_t(leafmask1) << 32 | uint64_t(leafmask0);
					{  // 2x2x2 nodes
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
					n_child_offset += popcnt_safe(current_leafmask);
				}
			}
			else
			{
				for(int i = 0; i<n_children; ++i)
				{
					const uint32_t n_index    = m_data[level][node_index + 1 + i];
					const uint32_t child_mask = m_data[level+1][n_index];
					if(level + 1 < m_top_levels)
					{
						n_child_offset += m_enclosed_leaves[(child_mask >> 8)];
					}
					else 
					{
						n_child_offset += (child_mask >> 8);
					}
				}
			}

			// We don't store nodes in subtree for root.
			if(level != 0)
			{
				if(level < m_top_levels)
				{
					m_enclosed_leaves[node_mask >> 8] = n_child_offset;
				}
				else 
				{
					node_mask = (node_mask & 0xFF) | (n_child_offset << 8);
				}
			}
			node_index += n_children + 1;
		}
	}
	int total_num_nodes = n_child_offset;

	using vec_t = std::vector<uint32_t>;
	vec_t all_base_colors(total_num_nodes);

	////////////////////////////////////////////////////
	// PASS 2: Create new color array
	////////////////////////////////////////////////////
	{
		uint32_t voxel_counter = 0;
		uint32_t node_counter  = 1;
		int nof_levels         = nofGeometryLevels();
		int nof_top_levels     = m_top_levels;
		struct StackEntry {
			uint32_t node_idx;
			uint8_t child_mask;
			uint8_t test_mask;
			uint32_t color_idx;
			vec_t child_base_colors;
		};
		vector<StackEntry> stack(100);
		int stack_level     = 0;
		stack[0].node_idx   = 0;
		stack[0].child_mask = m_data[0][0] & 0xFF;
		stack[0].test_mask  = stack[0].child_mask;
		stack[0].color_idx  = 0;
		StackEntry curr_se  = stack[0];
		uint64_t current_leafmask;

		while (stack_level >= 0) {
			// If no children left
			while (curr_se.test_mask == 0x0) {
				auto rgb888_to_float3 = [](uint32_t rgb) {
					return glm::vec3(
												((rgb >> 0) & 0xFF)  / 255.0f, 
												((rgb >> 8) & 0xFF)  / 255.0f, 
												((rgb >> 16) & 0xFF) / 255.0f
										);
				};

				auto float3_to_rgb888 = [](const glm::vec3 &c) {
					float R = min(1.0f, max(0.0f, c.x));
					float G = min(1.0f, max(0.0f, c.y));
					float B = min(1.0f, max(0.0f, c.z));
					return (uint32_t(R * 255.0f) << 0) | (uint32_t(G * 255.0f) << 8) | (uint32_t(B * 255.0f) << 16);
				};

				// Resolve colors and add to color list
				auto resolve_and_add = [&](vec_t &child_vec, vec_t &all_vec, bool ignore_black = false){
					glm::vec3 color = glm::vec3(0.0f, 0.0f, 0.0f);
					float N{0.0f};
					for (auto &c : child_vec) {
						if(!ignore_black || c != 0){
							color += rgb888_to_float3(c);
							++N;
						}
					}
					color /= N;
					uint32_t i_color = float3_to_rgb888(color);
					all_vec[curr_se.color_idx] = i_color;
					return i_color;
				};
				uint32_t i_base_color = resolve_and_add(curr_se.child_base_colors, all_base_colors);

				if (stack_level > 0) stack[stack_level - 1].child_base_colors.push_back(i_base_color);

				stack[stack_level].child_base_colors.clear();

				stack_level -= 1;
				if (stack_level < 0) break;
				curr_se = stack[stack_level];
			}
			if (stack_level < 0) break;
			// There are children left to test in current node.
			unsigned long next_child;
			//_BitScanReverse(&next_child, curr_se.child_mask);
			_BitScanForward(&next_child, curr_se.test_mask);
			curr_se.test_mask &= ~(1 << next_child);
			uint32_t node_offset = __popcnt((curr_se.child_mask & 0xFF) & ((1 << next_child) - 1)) + 1;

			stack[stack_level] = curr_se;
			stack_level += 1;

			if (stack_level == nof_levels) {
				// We found a voxel
				auto push_color = [&](vec_t &all, vec_t &child, uint32_t elem){
					all[node_counter] = elem;
					child.push_back(elem);
				};

				push_color(all_base_colors, curr_se.child_base_colors, m_base_colors[voxel_counter]);

				voxel_counter += 1;
				stack_level -= 1;
			} else if (stack_level == nof_levels - 1) {
				// We found a 2x2x2 node
				curr_se.color_idx  = node_counter;
				curr_se.child_mask = (current_leafmask >> next_child * 8) & 0xFF;
				curr_se.test_mask  = curr_se.child_mask;
				curr_se.child_base_colors.clear();
			} else if (stack_level == nof_levels - 2) {
				// We found a leaf node
				uint32_t leafmask_address = m_data[stack_level-1][curr_se.node_idx + node_offset];
				///////////////////////////////////////////////////////////////
				// FIXME: Shouldn't there be a faster way to get the 8 bit mask from
				// a 64 bit word... Without a bunch of compares?
				// Why, yes! Yes there is:
				// nof_leaves += __popc(__vsetgtu4((uint32_t)(masked_leafmask >> 0), 0));
				// nof_leaves += __popc(__vsetgtu4((uint32_t)(masked_leafmask >> 32), 0));
				///////////////////////////////////////////////////////////////
				uint32_t leafmask0 = m_data[stack_level][leafmask_address];
				uint32_t leafmask1 = m_data[stack_level][leafmask_address + 1];
				current_leafmask   = uint64_t(leafmask1) << 32 | uint64_t(leafmask0);
				curr_se.color_idx  = node_counter;
				curr_se.child_mask = ((current_leafmask & 0x00000000000000FFull) == 0 ? 0 : 1 << 0) |
														 ((current_leafmask & 0x000000000000FF00ull) == 0 ? 0 : 1 << 1) |
														 ((current_leafmask & 0x0000000000FF0000ull) == 0 ? 0 : 1 << 2) |
														 ((current_leafmask & 0x00000000FF000000ull) == 0 ? 0 : 1 << 3) |
														 ((current_leafmask & 0x000000FF00000000ull) == 0 ? 0 : 1 << 4) |
														 ((current_leafmask & 0x0000FF0000000000ull) == 0 ? 0 : 1 << 5) |
														 ((current_leafmask & 0x00FF000000000000ull) == 0 ? 0 : 1 << 6) |
														 ((current_leafmask & 0xFF00000000000000ull) == 0 ? 0 : 1 << 7);
				curr_se.test_mask = curr_se.child_mask;
				curr_se.child_base_colors.clear();

			}
			///////////////////////////////////////////////////////////////
			// If we are at an internal node, push the child on the stack
			///////////////////////////////////////////////////////////////
			else {
				curr_se.node_idx   = m_data[stack_level-1][curr_se.node_idx + node_offset];
				curr_se.color_idx  = node_counter;
				curr_se.child_mask = m_data[stack_level][curr_se.node_idx] & 0xFF;
				curr_se.test_mask  = curr_se.child_mask;
				curr_se.child_base_colors.clear();
			}
			node_counter += 1;
		}
	}

	m_base_colors = all_base_colors;
}
}  // namespace dag
