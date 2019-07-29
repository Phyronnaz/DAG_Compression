#include "Merger.h"
#include <limits>
#include <algorithm>
#include <numeric>
#include <bitset>
#include <vector>
#include <array>
#include <cassert>
#include <optional>
#include <tuple>
#include "../hash.h"

using namespace std;


namespace merger {

uint32_t
splitBy3_32(uint32_t a)
{
	uint32_t x = a               & 0x000003ff; // x = ---- ---- ---- ---- ---- --98 7654 3210
					 x = (x | (x << 16)) & 0xff0000ff; // x = ---- --98 ---- ---- ---- ---- 7654 3210
					 x = (x | (x << 8))  & 0x0300f00f; // x = ---- --98 ---- ---- 7654 ---- ---- 3210
					 x = (x | (x << 4))  & 0x030c30c3; // x = ---- --98 ---- 76-- --54 ---- 32-- --10
					 x = (x | (x << 2))  & 0x09249249; // x = ---- 9--8 --7- -6-- 5--4 --3- -2-- 1--0
	return x;
}

uint32_t 
mortonEncode32(uint32_t x, uint32_t y, uint32_t z)
{
	uint32_t answer = 0;
	answer |= splitBy3_32(x) << 2 | splitBy3_32(y) << 1 | splitBy3_32(z);
	return answer;
}

uint64_t
splitBy3_64(uint32_t a) 
{
	uint64_t x = a             & 0x00000000001ffffful;  // 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0001 1111 1111 1111 1111 1111
					 x = (x | x << 32) & 0x001f00000000fffful;  // 0000 0000 0001 1111 0000 0000 0000 0000 0000 0000 0000 0000 1111 1111 1111 1111
					 x = (x | x << 16) & 0x001f0000ff0000fful;  // 0000 0000 0001 1111 0000 0000 0000 0000 1111 1111 0000 0000 0000 0000 1111 1111
					 x = (x | x << 8)  & 0x100f00f00f00f00ful;  // 0001 0000 0000 1111 0000 0000 1111 0000 0000 1111 0000 0000 1111 0000 0000 1111
					 x = (x | x << 4)  & 0x10c30c30c30c30c3ul;  // 0001 0000 1100 0011 0000 1100 0011 0000 1100 0011 0000 1100 0011 0000 1100 0011
					 x = (x | x << 2)  & 0x1249249249249249ul;  // 0001 0010 0100 1001 0010 0100 1001 0010 0100 1001 0010 0100 1001 0010 0100 1001
	return x;
}

uint64_t
mortonEncode64(uint32_t x, uint32_t y, uint32_t z)
{
	uint64_t answer = 0;
	answer |= splitBy3_64(x) << 2 | splitBy3_64(y) << 1 | splitBy3_64(z);
	return answer;
}

template <typename T>
unsigned
popcnt_safe(T v)
{
	return static_cast<unsigned>(std::bitset<std::numeric_limits<T>::digits>(v).count());
}

template <class InputIt1, class InputIt2, class OutputIt>
OutputIt 
index_offset(InputIt1 first1, InputIt1 last1, InputIt2 first2, OutputIt d_first)
{
	typename std::iterator_traits<InputIt2>::value_type ctr{0};
	while (first1 != last1)
	{
		if (*first1 == *first2)
		{
			*(d_first++) = (uint32_t)ctr;
			++first1;
		}
		++first2;
		++ctr;
	}
	return d_first;
}


std::pair<std::vector<uint32_t>, std::vector<uint32_t>>
get_pointers(const std::vector<std::vector<uint32_t>> &dag, const uint32_t lvl)
{
	std::pair<std::vector<uint32_t>, std::vector<uint32_t>> result;
	struct PointerData
	{
		uint32_t ptr;
		uint32_t index;
		uint32_t offset;
	};

	std::vector<PointerData> pointers;

	std::vector<uint32_t> child_offsets;
	if (lvl + 1 == dag.size()-1)
	{
		child_offsets.resize(dag[lvl + 1].size() / 2);
		std::iota(child_offsets.begin(), child_offsets.end(), 0);
		std::transform(
			child_offsets.begin(),
			child_offsets.end(),
			child_offsets.begin(),
			[](uint32_t v) { return 2 * v; }
		);
	} 
	else
	{
		for (int i{0}; i < dag[lvl + 1].size();) 
		{
			uint32_t n_children = popcnt_safe(dag[lvl + 1][i] & 0xFF);
			child_offsets.emplace_back(i);
			i += n_children + 1;
		}
	}

	const auto &current_dag_lvl = dag[lvl];
	uint32_t idx{0};
	for(int i{0}; i < current_dag_lvl.size(); )
	{
		result.second.emplace_back(current_dag_lvl[i]);
		uint32_t n_children = popcnt_safe(current_dag_lvl[i] & 0xFF);
		for(int c{1}; c <= (int)n_children; ++c)
		{
			pointers.emplace_back(PointerData{current_dag_lvl[i+c], idx++, 0});
		}
		i += n_children + 1;
	}

	std::sort(
		begin(pointers),
		end(pointers),
		[](const auto &lhs, const auto &rhs) {
			return lhs.ptr < rhs.ptr;
		}
	);

	// TODO(dan): I can make an algorithm out of this.
	auto first1 = pointers.begin();
	auto last1 = pointers.end();
	auto first2 = child_offsets.begin();
	uint32_t ctr{0};
	while (first1 != last1)
	{
		while(first1->ptr > *first2)
		{
			++ctr;
			++first2;
		}
		first1->offset = ctr;
		if(++first1 == last1){break;}
		if(first1->ptr != *first2) { ++first2; }
	}

	std::adjacent_difference(
		begin(pointers),
		end(pointers),
		begin(pointers),
		[](const auto &lhs, const auto &rhs) -> PointerData {
			return PointerData{lhs.ptr == rhs.ptr ? 0u : 1u, lhs.index, lhs.offset};
		}
	);
	pointers[0].ptr = 0;

	std::partial_sum(
		begin(pointers),
		end(pointers),
		begin(pointers),
		[](const auto &sum, const auto &el) -> PointerData {
			return PointerData{sum.ptr + el.ptr, el.index, el.offset};
		}
	);

	std::sort(
		begin(pointers),
		end(pointers),
		[](const auto &lhs, const auto &rhs) {
			return lhs.index < rhs.index;
		}
	);

	for(uint32_t i{0}; i<pointers.size(); ++i)
	{
		pointers[i].ptr += pointers[i].offset;
	}

	result.first.reserve(pointers.size());
	for(const auto &v:pointers)
	{
		result.first.emplace_back(v.ptr);
	}
	return result;
}



dag::DAG
shallow_merge(const dag::DAG &lhs, const dag::DAG &rhs)
{
	std::vector<std::vector<uint32_t>> merged_dag;
	std::vector<std::vector<uint64_t>> merged_hashes;
	std::vector<uint32_t> merged_enclosed_leaves;
	std::vector<uint32_t> lhs_index_offsets;
	std::vector<uint32_t> rhs_index_offsets;
	std::vector<uint32_t> lhs_node_start;
	std::vector<uint32_t> rhs_node_start;
	std::vector<uint32_t> node_start;
	std::vector<uint32_t> node_start_new;
	uint32_t enclosed_ctr{0};

	auto get_top_level_leaf_offset = [&]() {
		const int l_offset = static_cast<int>(lhs.m_data.size()) - lhs.m_top_levels;
		const int r_offset = static_cast<int>(rhs.m_data.size()) - rhs.m_top_levels;
		// If we have top levels in both DAGs, they should start at the same offset
		// from the leaf per definition.
		if (lhs.m_top_levels != 0 && rhs.m_top_levels != 0)
		{
			if (l_offset != r_offset) { throw "Top level leaf offset does not match."; }
			return l_offset;
		} 
		// If only one DAG have top levels, return the offset.
		else if (lhs.m_top_levels != 0) {
			return l_offset;
		} else if (rhs.m_top_levels != 0) {
			return r_offset;
		}
		// If no DAG had top levels, the offset is at the root.
		// So just set the offset to be the max. It shouldn't really
		// matter which we choose, because as is now, we define the 
		// top levels to begin when we merge sub DAGs.
		return std::max(l_offset, r_offset);
	};

	const int n_levels = static_cast<int>(std::max(lhs.m_data.size(), rhs.m_data.size()));
	const int top_level_leaf_offset = get_top_level_leaf_offset();
	for (int lvl_from_leaf{0}; lvl_from_leaf < n_levels; ++lvl_from_leaf)
	{
		int l_lvl = static_cast<int>(lhs.m_data.size()) - 1 - lvl_from_leaf;
		int r_lvl = static_cast<int>(rhs.m_data.size()) - 1 - lvl_from_leaf;
		std::vector<uint64_t> empty_hash_set(0);
		const std::vector<uint64_t> &lhs_hashes = l_lvl < 0 ? empty_hash_set : lhs.m_hashes[l_lvl];
		const std::vector<uint64_t> &rhs_hashes = r_lvl < 0 ? empty_hash_set : rhs.m_hashes[r_lvl];
		std::vector<uint64_t> combined_hashes;

		// Combine hashes, discard identical.
		std::set_union(
			lhs_hashes.begin(),
			lhs_hashes.end(),
			rhs_hashes.begin(),
			rhs_hashes.end(),
			back_inserter(combined_hashes)
		);

		// Figure out new placement of child nodes in the DAGs,
		// as well the parent child mask.
		std::pair<std::vector<uint32_t>, std::vector<uint32_t>> lhs_pointers_mask, rhs_pointers_mask;
		if(lvl_from_leaf > 0)
		{
			if (l_lvl >= 0) { lhs_pointers_mask = get_pointers(lhs.m_data, l_lvl); }
			if (r_lvl >= 0) { rhs_pointers_mask = get_pointers(rhs.m_data, r_lvl); }
		}

		std::size_t lhs_hash_index{0}, rhs_hash_index{0};
		std::vector<uint32_t> new_level;
		std::size_t lhs_dag_index{0}, rhs_dag_index{0};
		for (int i{0}; i < combined_hashes.size(); ++ i)
		{
			auto current_hash = combined_hashes[i];
			bool lhs_hash_match = (l_lvl >= 0 && lhs_hashes[lhs_hash_index] == current_hash);
			bool rhs_hash_match = (r_lvl >= 0 && rhs_hashes[rhs_hash_index] == current_hash);
			if(!(lhs_hash_match || rhs_hash_match)){ throw "Hashes mismatch!";}

			if(lvl_from_leaf > 0)
			{
				auto &new_child_id  = lhs_hash_match ? lhs_index_offsets     : rhs_index_offsets;
				auto &enclosed      = lhs_hash_match ? lhs.m_enclosed_leaves : rhs.m_enclosed_leaves;
				auto &pointers_mask = lhs_hash_match ? lhs_pointers_mask     : rhs_pointers_mask;
				auto &hash_index    = lhs_hash_match ? lhs_hash_index        : rhs_hash_index;
				auto &dag_index     = lhs_hash_match ? lhs_dag_index         : rhs_dag_index;

				uint32_t child_mask{pointers_mask.second[hash_index]};
				// We are merging top levels.
				if(lvl_from_leaf >= top_level_leaf_offset)
				{
					// Look up enclosed leaves.
					uint32_t idx = child_mask >> 8;
					uint32_t node_enclosed = enclosed[idx];
					// Update index part of mask, and add to array.
					child_mask &= 0xFF;
					child_mask |= ((enclosed_ctr++)<<8);
					merged_enclosed_leaves.push_back(node_enclosed);
				}
				new_level.emplace_back(child_mask);
				const unsigned n_children = popcnt_safe(child_mask & 0xFF);
				// Increase offsset by the number of pointers plus the mask.
				node_start_new.emplace_back(n_children + 1);

				for(int n{0}; n < (int)n_children; ++n)
				{
					uint32_t original_child_id = pointers_mask.first[dag_index];

					new_level.emplace_back(node_start[new_child_id[original_child_id]]);

					if (lhs_hash_match && lhs_dag_index + 1 < lhs_pointers_mask.first.size()) { ++lhs_dag_index; }
					if (rhs_hash_match && rhs_dag_index + 1 < rhs_pointers_mask.first.size()) { ++rhs_dag_index; }
				}
			}
			else
			{
				uint32_t dag_index = 2*uint32_t(lhs_hash_match ? lhs_hash_index : rhs_hash_index);
				auto &dag = lhs_hash_match ? lhs.m_data[l_lvl] : rhs.m_data[r_lvl];
				new_level.emplace_back(dag[dag_index]);
				new_level.emplace_back(dag[dag_index + 1]);
				node_start_new.emplace_back(2);
			}
			if (lhs_hash_match && lhs_hash_index + 1 < lhs_hashes.size()) { ++lhs_hash_index; }
			if (rhs_hash_match && rhs_hash_index + 1 < rhs_hashes.size()) { ++rhs_hash_index; }
		}
		
		node_start.resize(node_start_new.size());
		node_start[0] = 0;
		std::partial_sum(
			node_start_new.begin(),
			prev(node_start_new.end()),
			next(node_start.begin())
		);
		node_start_new.clear();

		// Compute offsets for next level
		rhs_index_offsets.clear();
		index_offset(
			rhs_hashes.begin(),
			rhs_hashes.end(),
			combined_hashes.begin(),
			back_inserter(rhs_index_offsets)
		);
		lhs_index_offsets.clear();
		index_offset(
			lhs_hashes.begin(),
			lhs_hashes.end(),
			combined_hashes.begin(),
			back_inserter(lhs_index_offsets)
		);
		merged_dag.emplace_back(std::move(new_level));
		merged_hashes.emplace_back(std::move(combined_hashes));
	}

	std::reverse(merged_dag.begin(), merged_dag.end());
	std::reverse(merged_hashes.begin(), merged_hashes.end());

	dag::DAG result = lhs;
	result.m_data = merged_dag;
	result.m_hashes = merged_hashes;
	result.m_enclosed_leaves = merged_enclosed_leaves;
	return result;
}

inline 
std::optional<uint32_t>
find_offset(const dag::DAG &dag, uint32_t lvl, uint64_t hash)
{
	auto hash_begin = dag.m_hashes[lvl].begin();
	auto hash_end   = dag.m_hashes[lvl].end();
	auto found      = std::find(hash_begin, hash_end, hash);
	if (found != hash_end)
	{
		uint32_t hash_idx = (uint32_t)std::distance(hash_begin, found);
		uint32_t current_offset{0};
		for (uint32_t i{0}; i < hash_idx; ++i)
		{
			// mask + #childs
			current_offset += 1 + popcnt_safe(0xFF & dag.m_data[lvl][current_offset]);
		}
		return {current_offset};
	}
	return {};
}

std::optional<dag::DAG>
merge(const std::array<std::optional<dag::DAG>, 8> &batch)
{
	std::optional<dag::DAG> dag;
	std::vector<const dag::DAG*> dags;
	struct NodeInfo
	{
		uint64_t hash;
		uint32_t node_size;
		uint32_t node_index;
		uint32_t node_offset;
	};
	std::vector<NodeInfo> node_info;
	uint32_t mask{0}, enclosed{0}, ctr{0}, ctr2{0};
	for(auto & v:batch)
	{
		if(v)
		{
			mask |= 1u << ctr;
			uint32_t enclosed_mask = v->m_data[0][0] >> 8;
			enclosed += v->m_enclosed_leaves.size() == 0u ? enclosed_mask : v->m_enclosed_leaves[enclosed_mask];
			dags.push_back(&v.value());
			node_info.emplace_back(NodeInfo{
				v->m_hashes[0][0],
				popcnt_safe(v->m_data[0][0] & 0xFF) + 1,
				ctr2++,
				0}
			);
		}
		++ctr;
	}
	std::size_t n_dags{dags.size()};
	if(n_dags > 0) 
	{
		dag = *(dags[0]);

		// Merges DAGs up to the roots, but not further.
		for(int i{1}; i<n_dags; ++i)
		{
			dag = shallow_merge(*dag, *dags[i]);
		}

		// Construct upper leves, similar to
		// the DAG construction on the gpu.
		const std::size_t required_size =
			std::accumulate(
				dags.begin(),
				dags.end(),
				std::size_t(0),
				[](std::size_t acc, const dag::DAG *dag) { return acc + dag->m_base_colors.size(); }
			);

		dag->m_base_colors.reserve(required_size);

		for (int i{1}; i < n_dags; ++i)
		{
			auto insert_data = [](std::vector<uint32_t> &dst, const std::vector<uint32_t> &src) {
				dst.insert( dst.end(), src.begin(), src.end() );
			};
			insert_data(dag->m_base_colors, dags[i]->m_base_colors);
		}

		std::reverse(dag->m_data.begin(),   dag->m_data.end());
		std::reverse(dag->m_hashes.begin(), dag->m_hashes.end());
		std::vector<uint32_t> root;
		dag->m_levels++;
		dag->m_top_levels++;
		dag->m_enclosed_leaves.push_back(enclosed);
		root.push_back(mask | (uint32_t(dag->m_enclosed_leaves.size() -1) << 8));
		std::sort(
			node_info.begin(),
			node_info.end(),
			[](const NodeInfo &lhs, const NodeInfo &rhs) {
				return lhs.hash < rhs.hash;
			}
		);
		node_info[0].node_offset = 0;
		
		for(int i{1}; i<node_info.size(); ++i)
		{
			bool skip = node_info[i].hash == node_info[i-1].hash;
			node_info[i].node_offset = node_info[i-1].node_offset + (skip ? 0 : node_info[i-1].node_size);
		}

		std::sort(
			node_info.begin(),
			node_info.end(),
			[](const NodeInfo &lhs, const NodeInfo &rhs) {
				return lhs.node_index < rhs.node_index;
			}
		);

		uint64_t hash_key[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
		int ctr{0};
		uint64_t hash_value[2];
		for(const NodeInfo &v : node_info)
		{
			root.push_back(v.node_offset);
			hash_key[ctr++] = v.hash;
		}
		MurmurHash3_x64_128(&hash_key[0], 9 * sizeof(uint64_t), 0x0, hash_value);
		dag->m_hashes.push_back(std::vector<uint64_t>{hash_value[0]});
		dag->m_data.push_back(root);
		std::reverse(dag->m_data.begin(),   dag->m_data.end());
		std::reverse(dag->m_hashes.begin(), dag->m_hashes.end());
	}
	return dag;
}
}
