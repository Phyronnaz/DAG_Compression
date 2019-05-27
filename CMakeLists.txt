cmake_minimum_required(VERSION 3.7.1)
project(Project)

#########################################################
# Ignore find_package for sub projects.
#--------------------------------------------------------
# Not really needed here, but present for somewhere
# in the future when/if the subprojects should
# be separate libraries.
#--------------------------------------------------------
set(as_subproject
		utils
		DAG
		DAGConstructor
		DAGLoader
		DAGTracer
)
macro(find_package)
	if(NOT "${ARGV0}" IN_LIST as_subproject)
		_find_package(${ARGV})
	endif()
endmacro()

# Since GLM has decided to act like a little prick
# which can not be trusted, I will bundle the
# header only libraries here.
# I will at least then know that everyone gets a
# version which will work..
############ GLM ############
add_definitions(-DGLM_FORCE_PURE)
add_library(glm INTERFACE)
target_include_directories(glm INTERFACE thirdparty/glm)

############ CEREAL ############
add_library(cereal INTERFACE)
target_include_directories(cereal INTERFACE thirdparty/cereal/include/)



add_subdirectory("DAGExample/utils")
add_subdirectory("DAGExample/DAG")
add_subdirectory("DAGExample/DAGConstructor")
add_subdirectory("DAGExample/DAGTracer")
add_subdirectory("DAGExample/DAGLoader")
#########################################################

add_subdirectory(DAGExample)
