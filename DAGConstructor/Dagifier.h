#pragma once
#include <memory>
#include <vector>
#include <tuple>
#include "../utils/Aabb.h" //FIXME: Proper search paths
#include "../DAG/DAG.h"    //FIXME: Proper search paths
namespace dag{ class DAG;}
class dagifier {
    struct impl;
    std::unique_ptr<impl> p_impl_;
 public:
	dagifier();
	~dagifier();
	dagifier(dagifier &&)  = delete;
	dagifier(const dagifier &) = delete;
	dagifier &operator=(dagifier &&) = delete; 
	dagifier &operator=(const dagifier &) = delete;
    dag::DAG build_dag(uint32_t *d_pos, float4 *d_color, int count, int depth, const chag::Aabb &aabb);
	dag::DAG build_dag(const std::vector<uint32_t> &morton_paths, 
                       const std::vector<float> &colors,
                       int count, int depth, const chag::Aabb &aabb);
};
