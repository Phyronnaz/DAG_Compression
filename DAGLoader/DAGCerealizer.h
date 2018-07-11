#include <string>
#include <vector>
#include <glm/vec3.hpp>
#include "../DAG/DAG.h" //FIXME: Proper search paths
namespace dag {
namespace cerealization {
using glm::vec3;
namespace JSON {
void save(const DAG& dag, const std::string& file);
DAG load(const std::string& file);

template <typename T>
void save_vec(const std::vector<T>& vec, const std::string& file);
template <typename T>
std::vector<T> load_vec(const std::string& file);
}  // namespace JSON

namespace bin {
void save(const DAG& dag, const std::string& file);
DAG load(const std::string& file);

template <typename T>
void save_vec(const std::vector<T>& vec, const std::string& file);
template <typename T>
std::vector<T> load_vec(const std::string& file);
}  // namespace bin
}  // namespace cerealization
}  // namespace dag