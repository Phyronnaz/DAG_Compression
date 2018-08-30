#pragma once
#include <memory>
#include <vector>
#include <tuple>
#include <utils/Aabb.h>
#include <DAG/DAG.h>
namespace dag{ class DAG;}
class DAGConstructor {
    struct impl;
    std::unique_ptr<impl> p_impl_;
 public:
	DAGConstructor();
	~DAGConstructor();
	DAGConstructor(DAGConstructor &&)  = delete;
	DAGConstructor(const DAGConstructor &) = delete;
	DAGConstructor &operator=(DAGConstructor &&) = delete; 
	DAGConstructor &operator=(const DAGConstructor &) = delete;
    dag::DAG build_dag(uint32_t *d_pos, float4 *d_color, int count, int depth, const chag::Aabb &aabb);
	dag::DAG build_dag(const std::vector<uint32_t> &morton_paths, 
                       const std::vector<float> &colors,
                       int count, int depth, const chag::Aabb &aabb);
};
