cmake_minimum_required ( VERSION 3.1 )

project(DAGTracer LANGUAGES CXX CUDA)

find_package( OpenGL REQUIRED )
find_package( GLEW REQUIRED )
find_package( CUDA REQUIRED )
list(APPEND CMAKE_CUDA_FLAGS " -gencode=arch=compute_50,code=sm_50")


add_library(DAGTracer STATIC
			DAGTracer.cu
			DAGTracer.h
)

target_link_libraries(DAGTracer
	PRIVATE 
		OpenGL::GL 
		GLEW::GLEW 
		glm
)
target_link_libraries(DAGTracer
	PRIVATE 
		DAG
		utils
)
set_property ( TARGET DAGTracer PROPERTY CXX_STANDARD_REQUIRED ON )
set_property ( TARGET DAGTracer PROPERTY CXX_EXTENSIONS ON )
set_property ( TARGET DAGTracer PROPERTY CXX_STANDARD 17 )
