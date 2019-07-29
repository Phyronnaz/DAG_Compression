cmake_minimum_required ( VERSION 3.1 )

project(Voxelizer LANGUAGES CXX )

find_package( OpenGL REQUIRED ) #FIXME: Remove. Need dependencies because of CudaHelpers....
find_package( GLEW REQUIRED )	#FIXME: Remove. Need dependencies because of CudaHelpers....
list( APPEND CMAKE_CUDA_FLAGS ${PROJECT_CUDA_FLAGS})
#list( APPEND CMAKE_CUDA_FLAGS " -gencode=arch=compute_75,code=sm_75 -use_fast_math")


add_library(Voxelizer STATIC
			Voxelizer/Voxelizer.cpp
			Voxelizer/Voxelizer.h
)

target_link_libraries(Voxelizer
	PRIVATE
		OpenGL::GL #FIXME: Remove. Need dependencies because of CudaHelpers....
		GLEW::GLEW #FIXME: Remove. Need dependencies because of CudaHelpers....
		glm
)
target_link_libraries(Voxelizer
	PRIVATE 
		DAG
		utils
)

target_include_directories( Voxelizer 
	PUBLIC
		$<BUILD_INTERFACE:${Voxelizer_BINARY_DIR}>
		$<BUILD_INTERFACE:${Voxelizer_SOURCE_DIR}>
		$<INSTALL_INTERFACE:include>
)

set_property ( TARGET Voxelizer PROPERTY CXX_STANDARD_REQUIRED ON )
set_property ( TARGET Voxelizer PROPERTY CXX_EXTENSIONS ON )
set_property ( TARGET Voxelizer PROPERTY CXX_STANDARD 17 )
