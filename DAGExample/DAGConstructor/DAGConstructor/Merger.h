#pragma once
#include <DAG/DAG.h>
#include <vector>
#include <array>
#include <optional>
#include <stdint.h>

namespace merger {

inline std::vector<chag::Aabb>
split_aabb(const std::vector<chag::Aabb> parent) {
	std::vector<chag::Aabb> retval;
	retval.reserve(parent.size() * 8);

	for (const auto &p : parent) {
		auto c = p.getCentre();
		auto d = p.getHalfSize();

		for (int i = 0; i < 8; ++i) {
			auto new_c = c;
			new_c.x += d.x * (i & 4 ? 0.5f : -0.5f);
			new_c.y += d.y * (i & 2 ? 0.5f : -0.5f);
			new_c.z += d.z * (i & 1 ? 0.5f : -0.5f);

			retval.push_back(chag::make_aabb(new_c - 0.5f * d, new_c + 0.5f * d));
		}
	}
	return retval;
}

std::optional<dag::DAG> merge(const std::array<std::optional<dag::DAG>, 8> &in);
}  // namespace merger
