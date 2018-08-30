cmake_minimum_required ( VERSION 3.1 )

project(utils)

find_package(GLM REQUIRED)
find_package(GLEW REQUIRED)
find_package(OpenGL REQUIRED)

add_library(utils STATIC
			utils/Aabb.h
			utils/Aabb.cpp
			utils/orientation.h
			utils/orientation.cpp
			utils/view.h
			utils/view.cpp
			utils/glm_extensions.h
			utils/glm_extensions.cpp
)

target_link_libraries(utils 
	PUBLIC
		glm 
	PRIVATE
		OpenGL::GL 
		GLEW::GLEW 
		glm
)

target_include_directories( utils 
	PUBLIC
		$<BUILD_INTERFACE:${utils_BINARY_DIR}>
		$<BUILD_INTERFACE:${utils_SOURCE_DIR}>
		$<INSTALL_INTERFACE:include>
)

set_property(TARGET utils PROPERTY CXX_STANDARD_REQUIRED ON)
set_property(TARGET utils PROPERTY CXX_EXTENSIONS ON)
set_property(TARGET utils PROPERTY CXX_STANDARD 17)
