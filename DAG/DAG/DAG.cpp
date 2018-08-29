#include "DAG.h"
#include <algorithm>
#include <bitset>
#include <limits>

using namespace std;
template <typename T> 
unsigned int popcnt(T &&v) {
  return static_cast<unsigned int>(std::bitset<std::numeric_limits<T>::digits>(v).count());
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
		leaf_voxels += popcnt(dag[node_idx]) + popcnt(dag[node_idx + 1]);
		if (!visited[node_idx]) dag_size += 8;
		visited[node_idx] = true;
		return true;
	} else {
		uint32_t childmask = dag[node_idx] & 0xFF;
		int nof_subnodes   = popcnt(childmask);
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
}  // namespace dag
