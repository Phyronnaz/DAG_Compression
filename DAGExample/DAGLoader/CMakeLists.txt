cmake_minimum_required ( VERSION 3.1 )

project(DAGLoader)

add_library(DAGLoader STATIC
			DAGLoader/DAGLoader.h
			DAGLoader/DAGLoader.cpp
)

target_link_libraries(DAGLoader 
	PRIVATE 
		cereal 
		glm
)

target_link_libraries(DAGLoader 
	PRIVATE 
		utils 
		DAG
		ColorCompression
)

target_include_directories( DAGLoader 
	PUBLIC
		$<BUILD_INTERFACE:${DAGLoader_BINARY_DIR}>
		$<BUILD_INTERFACE:${DAGLoader_SOURCE_DIR}>
		$<INSTALL_INTERFACE:include>
)

set_property(TARGET DAGLoader PROPERTY CXX_STANDARD_REQUIRED ON)
set_property(TARGET DAGLoader PROPERTY CXX_EXTENSIONS ON)
set_property(TARGET DAGLoader PROPERTY CXX_STANDARD 11)
