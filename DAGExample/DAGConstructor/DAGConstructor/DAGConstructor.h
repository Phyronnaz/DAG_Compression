#pragma once
#include <DAG/DAG.h>
#include <memory>
#include <optional>
#include <tuple>
#include <utils/Aabb.h>
#include <vector>
#include <chrono>
#include "tracy/Tracy.hpp"

#ifndef __CUDACC__
#include "Merger.h"
#endif
uint32_t count_child_nodes(int lvl, int bottomlevel, uint32_t node_idx, std::vector<std::vector<uint32_t>>* dag);
std::tuple<uint32_t, uint32_t> count_child_nodes2(int lvl, int bottomlevel, uint32_t node_idx, std::vector<uint32_t>* dag);

namespace dag {
	class DAG;
}
class DAGConstructor {
	struct impl;
	std::unique_ptr<impl> p_impl_;

 public:
	DAGConstructor();
	~DAGConstructor();
	DAGConstructor(DAGConstructor &&)      = delete;
	DAGConstructor(const DAGConstructor &) = delete;
	DAGConstructor &operator=(DAGConstructor &&) = delete;
	DAGConstructor &operator=(const DAGConstructor &) = delete;

	dag::DAG build_dag(
		uint32_t *d_pos,
		uint32_t *d_base_color,
		int count,
		int depth,
		const chag::Aabb &aabb
	);

	dag::DAG build_dag(	
		const std::vector<uint32_t> &morton_paths,
		const std::vector<uint32_t> &base_color,
		int count,
		int depth,
		const chag::Aabb &aabb
	);

// including std::optional seems fine, but having it declared makes cuda sad.
#ifndef __CUDACC__
	template <typename Fn_>
	std::optional<dag::DAG> inline generate_DAG(
		Fn_ fn,
		int geometry_resolution,
		int max_subdag_resolution,
		int LevelsExcluding64BitLeafs,
		chag::Aabb aabb_in
	)
	{
		ZoneScoped
		
		auto aabb_list = std::vector<chag::Aabb>(1, aabb_in);
		// If the geometry resolution is too high, we need to split the geometry into smaller sub volumes
		// and process them independently. They are later merged to the final result.
		int nof_splits = std::max(0, static_cast<int>(log2(geometry_resolution) - log2(max_subdag_resolution)));
		for (unsigned i = 0; i < (unsigned)nof_splits; ++i) { aabb_list = merger::split_aabb(std::move(aabb_list)); }

		// Create sub DAGs from the sub volumes. Note that not all volumes may contain geometry, hence std::optional.
		std::vector<std::optional<dag::DAG>> dags(aabb_list.size());

        const auto startTime = std::chrono::high_resolution_clock::now();

        const auto printSeconds = [](uint64_t input_seconds)
        {
            size_t minutes = input_seconds / 60;
            size_t seconds = input_seconds % 60;

            std::cout << minutes << ":" << seconds << " ";
        };

        for (int i{ 0 }; i < aabb_list.size(); ++i)
        {
            if (i % 100 == 0)
            {
                std::cout << "Generating sub DAG " << i << " of " << aabb_list.size() << ". ";
                const double elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::high_resolution_clock::now() - startTime).count();
                std::cout << " Elapsed: ";
                printSeconds(uint64_t(elapsed));
                std::cout << "Remaining: ";
                printSeconds(uint64_t(elapsed * aabb_list.size() / i - elapsed));
                std::cout << '\n';
            }
			auto &aabb  = aabb_list[i];
			auto voxels = fn(aabb, std::min(max_subdag_resolution, geometry_resolution));
			if (voxels.count > 0) {
				dags[i] =
					build_dag(
						voxels.positions,
						voxels.base_color,
						voxels.count,
						LevelsExcluding64BitLeafs,
						aabb);
			}
		}
		std::cout << "done.\n";

		// The way the sub volumes are split, is in a morton order.
		// 8 consecutive volumes hence compose a larger super volume.
		// We thus create a batch of 8 subvolumes and merge them,
		// and place them in a new array to be recursively processed in
		// the next iteration.
		std::vector<std::optional<dag::DAG>> merged_dags(dags.size() / 8);
		std::size_t dags_left{dags.size()};
		std::cout << "Start merging DAGs...\n";
		while (dags_left != 1) {
			std::cout << "Passes left: " << (int)(std::log(dags_left)/std::log(8)) << ".\n";
            const auto passStartTime = std::chrono::high_resolution_clock::now();
			for (std::size_t i{0}; i < dags_left / 8; ++i) {
                std::cout << "Sub - pass " << i+1 << " of " << dags_left / 8;
                const double elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::high_resolution_clock::now() - passStartTime).count();
                std::cout << " Elapsed: ";
                printSeconds(uint64_t(elapsed));
                std::cout << "Remaining: ";
                printSeconds(uint64_t(elapsed * dags_left / 8 / (i + 1) - elapsed));
                std::cout << '\n';

				std::array<std::optional<dag::DAG>, 8> batch;
				for (int j{0}; j < 8; ++j) { batch[j] = std::move(dags[8 * i + j]); }
				merged_dags[i] = merger::merge(batch);
				std::cout << "Sub - pass " << i+1 << " of " << dags_left / 8 << " done.\n";
			}
			dags_left /= 8;
			std::swap(dags, merged_dags);
		}
		std::cout << "done.\n";

        {
            const double elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::high_resolution_clock::now() - startTime).count();
            std::cout << "Total Elapsed: ";
            printSeconds(uint64_t(elapsed));
            std::cout << std::endl;
        }

		// When all DAGs have been merged, the result resides in the
		// first slot of the array.
		if (dags[0]) {
			dags[0]->m_aabb = aabb_in;
			return std::move(dags[0]);
		}
		return {};
	}
#endif
};
