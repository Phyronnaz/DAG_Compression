cmake_minimum_required ( VERSION 3.1 )

project(DAG)

add_library(DAG STATIC
			DAG.h
			DAG.cpp
)

target_link_libraries(DAG 
	PRIVATE 
		utils
)
set_property(TARGET DAG PROPERTY CXX_STANDARD_REQUIRED ON)
set_property(TARGET DAG PROPERTY CXX_EXTENSIONS ON)
set_property(TARGET DAG PROPERTY CXX_STANDARD 17)
