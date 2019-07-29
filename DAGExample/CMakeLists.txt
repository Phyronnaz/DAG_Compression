cmake_minimum_required(VERSION 3.7.1)
project(DAGExample)

set(CMAKE_INCLUDE_CURRENT_DIR ON)

find_package(SDL2 REQUIRED)
find_package(GLEW REQUIRED)
find_package(OpenGL REQUIRED)

add_executable(my_exe
	src/my_exe.cpp
	src/voxelize_and_merge.h
	src/voxelize_and_merge.cpp
)

target_link_libraries(my_exe 
	PRIVATE 
		SDL2::SDL2 
		SDL2::SDL2main 
		OpenGL::GL 
		GLEW::GLEW 
		glm
)

target_link_libraries(my_exe
	PRIVATE 
		utils
		DAG 
		DAGTracer 
		DAGLoader
		DAGConstructor
		Voxelizer
		glTFLoader
		ColorCompression
)
add_custom_target(copy_dlls
    COMMAND ${CMAKE_COMMAND} -E copy_directory
	"$<TARGET_FILE_DIR:SDL2::SDL2>"
	"${CMAKE_CURRENT_BINARY_DIR}/$<CONFIGURATION>"
	COMMAND ${CMAKE_COMMAND} -E copy
	"$<TARGET_FILE_DIR:GLEW::GLEW>/../bin/glew32.dll"
	"${CMAKE_CURRENT_BINARY_DIR}/$<CONFIGURATION>"
)
