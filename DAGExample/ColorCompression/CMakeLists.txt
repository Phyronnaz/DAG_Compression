cmake_minimum_required ( VERSION 3.1 )

project(ColorCompression LANGUAGES CXX CUDA)

find_package(OpenGL REQUIRED) #FIXME: Remove. Need dependencies because of CudaHelpers....
find_package(GLEW REQUIRED)	#FIXME: Remove. Need dependencies because of CudaHelpers....
list(APPEND CMAKE_CUDA_FLAGS " -gencode=arch=compute_52,code=sm_52 -use_fast_math")
#list( APPEND CMAKE_CUDA_FLAGS " -gencode=arch=compute_75,code=sm_75 -use_fast_math")


add_library(ColorCompression STATIC
			ColorCompression/BlockBuild.h
			ColorCompression/ours.cu
			ColorCompression/ours_varbit.cpp
			ColorCompression/ours_varbit.h
)

target_link_libraries(ColorCompression
	PRIVATE
		OpenGL::GL #FIXME: Remove. Need dependencies because of CudaHelpers....
		GLEW::GLEW #FIXME: Remove. Need dependencies because of CudaHelpers....
		glm
)
target_link_libraries(ColorCompression
	PRIVATE 
		DAG
		utils
)

target_include_directories( ColorCompression 
	PUBLIC
		$<BUILD_INTERFACE:${ColorCompression_BINARY_DIR}>
		$<BUILD_INTERFACE:${ColorCompression_SOURCE_DIR}>
		$<INSTALL_INTERFACE:include>
)

set_property ( TARGET ColorCompression PROPERTY CXX_STANDARD_REQUIRED ON )
set_property ( TARGET ColorCompression PROPERTY CXX_EXTENSIONS ON )
set_property ( TARGET ColorCompression PROPERTY CXX_STANDARD 17 )
