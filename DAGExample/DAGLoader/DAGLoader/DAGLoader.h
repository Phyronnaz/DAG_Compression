#include <string>
#include <vector>

namespace cerealization
{
  namespace JSON
  {
    template<typename T>
    void save(const T& obj, const std::string& file);

    template<typename T>
    T load(const std::string& file);

    template <typename T>
    void save_vec(const std::vector<T>& vec, const std::string& file);

    template <typename T>
    std::vector<T> load_vec(const std::string& file);
  }  // namespace JSON

  namespace bin
  {
    template<typename T>
    void save(const T& obj, const std::string& file);

    template<typename T>
    T load(const std::string& file);

    template <typename T>
    void save_vec(const std::vector<T>& vec, const std::string& file);

    template <typename T>
    std::vector<T> load_vec(const std::string& file);
  }  // namespace bin
}  // namespace cerealization

