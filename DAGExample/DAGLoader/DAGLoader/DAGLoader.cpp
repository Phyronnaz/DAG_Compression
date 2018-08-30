#include "DAGLoader.h"
#include <cereal/archives/binary.hpp>
#include <cereal/archives/json.hpp>
#include <cereal/types/vector.hpp>
#include <fstream>

#include <glm/vec3.hpp>

#include <utils/Aabb.h>

namespace glm {
template <class Archive>
void serialize(Archive& archive, vec3::type& v) {
	archive(cereal::make_nvp("x", v.x), cereal::make_nvp("y", v.y), cereal::make_nvp("z", v.z));
}
}  // namespace glm

namespace chag {
template <class Archive>
void serialize(Archive& archive, Aabb& aabb) {
	archive(cereal::make_nvp("min", aabb.min), cereal::make_nvp("max", aabb.max));
}
}  // namespace chag

namespace dag {
template <class Archive>
void serialize(Archive& archive, DAG& dag) {
	archive(cereal::make_nvp("AABB", dag.m_aabb), cereal::make_nvp("levels", dag.m_levels),
	        cereal::make_nvp("top levels", dag.m_top_levels),
	        cereal::make_nvp("enclosed leaves", dag.m_enclosed_leaves), cereal::make_nvp("nodes", dag.m_data));
}

namespace cerealization {
namespace JSON {
using OutputArchive = cereal::JSONOutputArchive;
using InputArchive  = cereal::JSONInputArchive;
void save(const DAG& dag, const std::string& file) {
	std::ofstream ofs(file);
	OutputArchive ar(ofs);
	ar(dag);
}

template <typename T>
void save_vec(const std::vector<T>& vec, const std::string& file) {
	std::ofstream ofs(file);
	OutputArchive ar(ofs);
	ar(cereal::make_nvp("array", vec));
}
template void save_vec(const std::vector<uint32_t>& vec, const std::string& file);
template void save_vec(const std::vector<vec3>& vec, const std::string& file);
template void save_vec(const std::vector<float>& vec, const std::string& file);

DAG load(const std::string& file) {
	DAG result;
	std::ifstream ifs(file);
	InputArchive ar(ifs);
	ar(result);
	return result;
}

template <typename T>
std::vector<T> load_vec(const std::string& file) {
	std::vector<T> result;
	std::ifstream ifs(file, std::ios::binary);
	InputArchive ar(ifs);
	ar(result);
	return result;
}
template std::vector<uint32_t> load_vec(const std::string& file);
template std::vector<vec3> load_vec(const std::string& file);
template std::vector<float> load_vec(const std::string& file);
}  // namespace JSON

namespace bin {
using OutputArchive = cereal::BinaryOutputArchive;
using InputArchive  = cereal::BinaryInputArchive;
void save(const DAG& dag, const std::string& file) {
	std::ofstream ofs(file, std::ios::binary);
	OutputArchive ar(ofs);
	ar(dag);
}
DAG load(const std::string& file) {
	DAG result;
	std::ifstream ifs(file, std::ios::binary);
	InputArchive ar(ifs);
	ar(result);
	return result;
}

template <typename T>
void save_vec(const std::vector<T>& vec, const std::string& file) {
	std::ofstream ofs(file, std::ios::binary);
	OutputArchive ar(ofs);
	ar(cereal::make_nvp("array", vec));
}
template void save_vec(const std::vector<uint32_t>& vec, const std::string& file);
template void save_vec(const std::vector<vec3>& vec, const std::string& file);
template void save_vec(const std::vector<float>& vec, const std::string& file);

template <typename T>
std::vector<T> load_vec(const std::string& file) {
	std::vector<T> result;
	std::ifstream ifs(file, std::ios::binary);
	InputArchive ar(ifs);
	ar(result);
	return result;
}
template std::vector<uint32_t> load_vec(const std::string& file);
template std::vector<vec3> load_vec(const std::string& file);
template std::vector<float> load_vec(const std::string& file);

}  // namespace bin
}  // namespace cerealization
}  // namespace dag
