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
add_subdirectory("DAGExample/utils")
add_subdirectory("DAGExample/DAG")
add_subdirectory("DAGExample/DAGConstructor")
add_subdirectory("DAGExample/DAGTracer")
add_subdirectory("DAGExample/DAGLoader")
#########################################################

add_subdirectory(DAGExample)
