#pragma once

#include <iostream>
#include <cstdint>
#include <vector>
#include <utils/Aabb.h>

struct float4;

namespace dag {

enum class DagType { STANDARD, ALL_COLORS };
class DAG {
 public:
	DAG()  = default;
	~DAG() = default;
	DAG(DAG &&d)                 = default;
	DAG(const DAG &d)            = default;
	DAG &operator=(DAG &&d)      = default;
	DAG &operator=(const DAG &d) = default;

	uint32_t m_levels = 0;
	inline uint32_t nofGeometryLevels()  const { return m_levels + 2; };
	inline uint32_t geometryResolution() const { return (1 << nofGeometryLevels()); }
	void calculateColorForAllNodes();

	uint32_t *d_data       = nullptr;
	uint32_t *d_color_data = nullptr;

	std::vector<std::vector<uint32_t>> m_data;
	std::vector<std::vector<uint64_t>> m_hashes;
	std::vector<uint32_t> m_base_colors;

	///////////////////////////////////////////////////////////////////////////
	// For the "top levels" (which currently coincide with the top levels we
	// added when merging the DAGs), we do not store the "number of enclosed
	// leaves" in the upper 24 bits of the mask. Istead, we store an index
	// there, into a separate array of 32 bit "enclosed leaves" entries.
	///////////////////////////////////////////////////////////////////////////
	uint32_t m_top_levels = 0;
	uint32_t *d_enclosed_leaves = nullptr;
	std::vector<uint32_t> m_enclosed_leaves;
	bool colors_in_all_nodes{false};

	chag::Aabb m_aabb;
};
}  // namespace dag
