#pragma once
#include <optional>
#include <string>
#include "DAG/DAG.h"
std::optional<dag::DAG> DAG_from_scene(const int dag_resolution, const std::string scene_folder, const std::string scene_file);
