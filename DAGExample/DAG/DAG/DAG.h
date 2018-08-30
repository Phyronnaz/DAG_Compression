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
	uint32_t m_levels = 0;
	uint32_t nofGeometryLevels()  const { return m_levels + 2; };
	uint32_t nofColorLevels()     const { return m_levels; }
	uint32_t geometryResolution() const { return (1 << nofGeometryLevels()); }
	uint32_t colorResolution()    const { return (1 << nofColorLevels()); }

	uint32_t *d_data       = nullptr;
	uint32_t *d_color_data = nullptr;

	std::vector<uint32_t> m_data;
	std::vector<uint32_t> m_colors;

	///////////////////////////////////////////////////////////////////////////
	// For the "top levels" (which currently coincide with the top levels we
	// added when merging the DAGs), we do not store the "number of enclosed
	// leaves" in the upper 24 bits of the mask. Istead, we store an index
	// there, into a separate array of 32 bit "enclosed leaves" entries.
	///////////////////////////////////////////////////////////////////////////
	uint32_t m_top_levels = 0;
	uint32_t *d_enclosed_leaves = nullptr;
	std::vector<uint32_t> m_enclosed_leaves;

	chag::Aabb m_aabb;
};
}  // namespace dag
