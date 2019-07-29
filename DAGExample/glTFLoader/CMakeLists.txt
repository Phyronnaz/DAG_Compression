cmake_minimum_required(VERSION 3.7.1)
project(glTFLoader LANGUAGES CXX)

set(CMAKE_INCLUDE_CURRENT_DIR ON)

find_package( OpenGL REQUIRED )
find_package( GLEW REQUIRED )

set(Project_VERSION 0.0.1)

add_library(glTFLoader STATIC
	glTFLoader/glTFLoader.cpp
	glTFLoader/glTFLoader.h
	glTFLoader/nlohmann/json.hpp
	)

target_link_libraries(glTFLoader
	PRIVATE
		OpenGL::GL
		GLEW::GLEW
		stb
		glm
)

target_link_libraries(glTFLoader
	PRIVATE 
		utils
)

target_compile_features(glTFLoader PUBLIC cxx_std_17)
set_property(TARGET glTFLoader PROPERTY VERSION ${Project_VERSION})

target_include_directories( glTFLoader 
	PUBLIC
		$<BUILD_INTERFACE:${glTFLoader_BINARY_DIR}>
		$<BUILD_INTERFACE:${glTFLoader_SOURCE_DIR}>
		$<INSTALL_INTERFACE:include>
)

############# Installing #############
# install(TARGETS glTFLoader
#   RUNTIME DESTINATION bin
# )
