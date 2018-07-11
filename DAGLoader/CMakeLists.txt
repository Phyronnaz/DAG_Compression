cmake_minimum_required ( VERSION 3.1 )

project(DAGLoader)

find_package(cereal REQUIRED)
find_package(GLM REQUIRED)

add_library(DAGLoader STATIC
			DAGCerealizer.h
			DAGCerealizer.cpp
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
)

set_property(TARGET DAGLoader PROPERTY CXX_STANDARD_REQUIRED ON)
set_property(TARGET DAGLoader PROPERTY CXX_EXTENSIONS ON)
set_property(TARGET DAGLoader PROPERTY CXX_STANDARD 11)
