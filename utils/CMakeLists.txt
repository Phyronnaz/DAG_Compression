cmake_minimum_required ( VERSION 3.1 )

project(utils)
find_package(GLM REQUIRED)
add_library(utils STATIC
			Aabb.h
			Aabb.cpp
			orientation.h
			orientation.cpp
			view.h
			view.cpp
			glm_extensions.h
			glm_extensions.cpp
)

target_link_libraries(utils 
	PUBLIC
		glm 
	PRIVATE
		OpenGL::GL 
		GLEW::GLEW 
		glm
)

set_property(TARGET utils PROPERTY CXX_STANDARD_REQUIRED ON)
set_property(TARGET utils PROPERTY CXX_EXTENSIONS ON)
set_property(TARGET utils PROPERTY CXX_STANDARD 17)
