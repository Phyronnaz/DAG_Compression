cmake_minimum_required(VERSION 3.14)
project(DAG_Compression LANGUAGES CUDA CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")
set(CMAKE_CXX_FLAGS_RELEASE "-g -O3")

# Ignore some warnings that add a lot of spam and aren't that critical:
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-variable -Wno-unused-parameter")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-parentheses -Wall -Wextra -Wconversion")

find_package(glm REQUIRED)
find_package(SDL2 REQUIRED)
find_package(OpenGL REQUIRED)
find_package(CUDA REQUIRED)
find_package(GLEW REQUIRED)

include_directories(${SDL2_INCLUDE_DIRS})
include_directories(${CUDA_INCLUDE_DIRS})

set(CMAKE_CUDA_HOST_COMPILER gcc-8)
#XXX add -g to debug
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -gencode arch=compute_61,code=sm_61)

# Add --expt-relaxed-constexpr, allowing us to call constexpr functions in
# CUDA device code (without warnings)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")


include_directories(DAGExample)
include_directories(DAGExample/ColorCompression/)
include_directories(DAGExample/DAG/)
include_directories(DAGExample/DAGConstructor/)
include_directories(DAGExample/DAGLoader/)
include_directories(DAGExample/DAGTracer)
include_directories(DAGExample/glTFLoader/)
include_directories(DAGExample/src)
include_directories(DAGExample/utils/)
include_directories(DAGExample/Voxelizer/)
include_directories(thirdparty/cereal)
include_directories(thirdparty/cereal/include)
include_directories(thirdparty/glm/)
include_directories(thirdparty/stb/)

add_executable(DAG_Compression
        DAGExample/ColorCompression/ColorCompression/BlockBuild.h
        DAGExample/ColorCompression/ColorCompression/ours.cpp
        DAGExample/ColorCompression/ColorCompression/ours.h
        DAGExample/ColorCompression/ColorCompression/ours.cu
        DAGExample/ColorCompression/ColorCompression/ours_varbit.cpp
        DAGExample/ColorCompression/ColorCompression/ours_varbit.h
        DAGExample/ColorCompression/ColorCompression/svd.cpp
        DAGExample/ColorCompression/ColorCompression/svd.h
        DAGExample/DAG/DAG/DAG.cpp
        DAGExample/DAG/DAG/DAG.h
        DAGExample/DAGConstructor/DAGConstructor/DAGConstructor.cu
        DAGExample/DAGConstructor/DAGConstructor/DAGConstructor.h
        DAGExample/DAGConstructor/DAGConstructor/Merger.cpp
        DAGExample/DAGConstructor/DAGConstructor/Merger.h
        DAGExample/DAGLoader/DAGLoader/DAGLoader.cpp
        DAGExample/DAGLoader/DAGLoader/DAGLoader.h
        DAGExample/DAGTracer/DAGTracer/DAGTracer.cu
        DAGExample/DAGTracer/DAGTracer/DAGTracer.h
        DAGExample/glTFLoader/glTFLoader/nlohmann/json.hpp
        DAGExample/glTFLoader/glTFLoader/glTFLoader.cpp
        DAGExample/glTFLoader/glTFLoader/glTFLoader.h
        DAGExample/src/my_exe.cpp
        DAGExample/src/voxelize_and_merge.cpp
        DAGExample/src/voxelize_and_merge.h
        DAGExample/utils/utils/Aabb.cpp
        DAGExample/utils/utils/Aabb.h
        DAGExample/utils/utils/glm_extensions.cpp
        DAGExample/utils/utils/glm_extensions.h
        DAGExample/utils/utils/orientation.cpp
        DAGExample/utils/utils/orientation.h
        DAGExample/utils/utils/view.cpp
        DAGExample/utils/utils/view.h
        DAGExample/Voxelizer/Voxelizer/Voxelizer.cpp
        DAGExample/Voxelizer/Voxelizer/Voxelizer.h
        DAGExample/bits_in_uint_array.h
        DAGExample/CudaHelpers.h
        DAGExample/hash.h
        DAGExample/morton.h
        thirdparty/cereal/include/cereal/archives/adapters.hpp
        thirdparty/cereal/include/cereal/archives/binary.hpp
        thirdparty/cereal/include/cereal/archives/json.hpp
        thirdparty/cereal/include/cereal/archives/portable_binary.hpp
        thirdparty/cereal/include/cereal/archives/xml.hpp
        thirdparty/cereal/include/cereal/details/helpers.hpp
        thirdparty/cereal/include/cereal/details/polymorphic_impl.hpp
        thirdparty/cereal/include/cereal/details/polymorphic_impl_fwd.hpp
        thirdparty/cereal/include/cereal/details/static_object.hpp
        thirdparty/cereal/include/cereal/details/traits.hpp
        thirdparty/cereal/include/cereal/details/util.hpp
        thirdparty/cereal/include/cereal/external/rapidjson/error/en.h
        thirdparty/cereal/include/cereal/external/rapidjson/error/error.h
        thirdparty/cereal/include/cereal/external/rapidjson/internal/biginteger.h
        thirdparty/cereal/include/cereal/external/rapidjson/internal/diyfp.h
        thirdparty/cereal/include/cereal/external/rapidjson/internal/dtoa.h
        thirdparty/cereal/include/cereal/external/rapidjson/internal/ieee754.h
        thirdparty/cereal/include/cereal/external/rapidjson/internal/itoa.h
        thirdparty/cereal/include/cereal/external/rapidjson/internal/meta.h
        thirdparty/cereal/include/cereal/external/rapidjson/internal/pow10.h
        thirdparty/cereal/include/cereal/external/rapidjson/internal/regex.h
        thirdparty/cereal/include/cereal/external/rapidjson/internal/stack.h
        thirdparty/cereal/include/cereal/external/rapidjson/internal/strfunc.h
        thirdparty/cereal/include/cereal/external/rapidjson/internal/strtod.h
        thirdparty/cereal/include/cereal/external/rapidjson/internal/swap.h
        thirdparty/cereal/include/cereal/external/rapidjson/msinttypes/inttypes.h
        thirdparty/cereal/include/cereal/external/rapidjson/msinttypes/stdint.h
        thirdparty/cereal/include/cereal/external/rapidjson/allocators.h
        thirdparty/cereal/include/cereal/external/rapidjson/document.h
        thirdparty/cereal/include/cereal/external/rapidjson/encodedstream.h
        thirdparty/cereal/include/cereal/external/rapidjson/encodings.h
        thirdparty/cereal/include/cereal/external/rapidjson/filereadstream.h
        thirdparty/cereal/include/cereal/external/rapidjson/filewritestream.h
        thirdparty/cereal/include/cereal/external/rapidjson/fwd.h
        thirdparty/cereal/include/cereal/external/rapidjson/istreamwrapper.h
        thirdparty/cereal/include/cereal/external/rapidjson/memorybuffer.h
        thirdparty/cereal/include/cereal/external/rapidjson/memorystream.h
        thirdparty/cereal/include/cereal/external/rapidjson/ostreamwrapper.h
        thirdparty/cereal/include/cereal/external/rapidjson/pointer.h
        thirdparty/cereal/include/cereal/external/rapidjson/prettywriter.h
        thirdparty/cereal/include/cereal/external/rapidjson/rapidjson.h
        thirdparty/cereal/include/cereal/external/rapidjson/reader.h
        thirdparty/cereal/include/cereal/external/rapidjson/schema.h
        thirdparty/cereal/include/cereal/external/rapidjson/stream.h
        thirdparty/cereal/include/cereal/external/rapidjson/stringbuffer.h
        thirdparty/cereal/include/cereal/external/rapidjson/writer.h
        thirdparty/cereal/include/cereal/external/rapidxml/rapidxml.hpp
        thirdparty/cereal/include/cereal/external/rapidxml/rapidxml_iterators.hpp
        thirdparty/cereal/include/cereal/external/rapidxml/rapidxml_print.hpp
        thirdparty/cereal/include/cereal/external/rapidxml/rapidxml_utils.hpp
        thirdparty/cereal/include/cereal/external/base64.hpp
        thirdparty/cereal/include/cereal/types/concepts/pair_associative_container.hpp
        thirdparty/cereal/include/cereal/types/array.hpp
        thirdparty/cereal/include/cereal/types/base_class.hpp
        thirdparty/cereal/include/cereal/types/bitset.hpp
        thirdparty/cereal/include/cereal/types/boost_variant.hpp
        thirdparty/cereal/include/cereal/types/chrono.hpp
        thirdparty/cereal/include/cereal/types/common.hpp
        thirdparty/cereal/include/cereal/types/complex.hpp
        thirdparty/cereal/include/cereal/types/deque.hpp
        thirdparty/cereal/include/cereal/types/forward_list.hpp
        thirdparty/cereal/include/cereal/types/functional.hpp
        thirdparty/cereal/include/cereal/types/list.hpp
        thirdparty/cereal/include/cereal/types/map.hpp
        thirdparty/cereal/include/cereal/types/memory.hpp
        thirdparty/cereal/include/cereal/types/polymorphic.hpp
        thirdparty/cereal/include/cereal/types/queue.hpp
        thirdparty/cereal/include/cereal/types/set.hpp
        thirdparty/cereal/include/cereal/types/stack.hpp
        thirdparty/cereal/include/cereal/types/string.hpp
        thirdparty/cereal/include/cereal/types/tuple.hpp
        thirdparty/cereal/include/cereal/types/unordered_map.hpp
        thirdparty/cereal/include/cereal/types/unordered_set.hpp
        thirdparty/cereal/include/cereal/types/utility.hpp
        thirdparty/cereal/include/cereal/types/valarray.hpp
        thirdparty/cereal/include/cereal/types/vector.hpp
        thirdparty/cereal/include/cereal/access.hpp
        thirdparty/cereal/include/cereal/cereal.hpp
        thirdparty/cereal/include/cereal/macros.hpp
        thirdparty/glm/glm/detail/_features.hpp
        thirdparty/glm/glm/detail/_fixes.hpp
        thirdparty/glm/glm/detail/_noise.hpp
        thirdparty/glm/glm/detail/_swizzle.hpp
        thirdparty/glm/glm/detail/_swizzle_func.hpp
        thirdparty/glm/glm/detail/_vectorize.hpp
        thirdparty/glm/glm/detail/compute_common.hpp
        thirdparty/glm/glm/detail/compute_vector_relational.hpp
        thirdparty/glm/glm/detail/func_common.inl
        thirdparty/glm/glm/detail/func_common_simd.inl
        thirdparty/glm/glm/detail/func_exponential.inl
        thirdparty/glm/glm/detail/func_exponential_simd.inl
        thirdparty/glm/glm/detail/func_geometric.inl
        thirdparty/glm/glm/detail/func_geometric_simd.inl
        thirdparty/glm/glm/detail/func_integer.inl
        thirdparty/glm/glm/detail/func_integer_simd.inl
        thirdparty/glm/glm/detail/func_matrix.inl
        thirdparty/glm/glm/detail/func_matrix_simd.inl
        thirdparty/glm/glm/detail/func_packing.inl
        thirdparty/glm/glm/detail/func_packing_simd.inl
        thirdparty/glm/glm/detail/func_trigonometric.inl
        thirdparty/glm/glm/detail/func_trigonometric_simd.inl
        thirdparty/glm/glm/detail/func_vector_relational.inl
        thirdparty/glm/glm/detail/func_vector_relational_simd.inl
        thirdparty/glm/glm/detail/glm.cpp
        thirdparty/glm/glm/detail/qualifier.hpp
        thirdparty/glm/glm/detail/setup.hpp
        thirdparty/glm/glm/detail/type_float.hpp
        thirdparty/glm/glm/detail/type_half.hpp
        thirdparty/glm/glm/detail/type_half.inl
        thirdparty/glm/glm/detail/type_mat2x2.hpp
        thirdparty/glm/glm/detail/type_mat2x2.inl
        thirdparty/glm/glm/detail/type_mat2x3.hpp
        thirdparty/glm/glm/detail/type_mat2x3.inl
        thirdparty/glm/glm/detail/type_mat2x4.hpp
        thirdparty/glm/glm/detail/type_mat2x4.inl
        thirdparty/glm/glm/detail/type_mat3x2.hpp
        thirdparty/glm/glm/detail/type_mat3x2.inl
        thirdparty/glm/glm/detail/type_mat3x3.hpp
        thirdparty/glm/glm/detail/type_mat3x3.inl
        thirdparty/glm/glm/detail/type_mat3x4.hpp
        thirdparty/glm/glm/detail/type_mat3x4.inl
        thirdparty/glm/glm/detail/type_mat4x2.hpp
        thirdparty/glm/glm/detail/type_mat4x2.inl
        thirdparty/glm/glm/detail/type_mat4x3.hpp
        thirdparty/glm/glm/detail/type_mat4x3.inl
        thirdparty/glm/glm/detail/type_mat4x4.hpp
        thirdparty/glm/glm/detail/type_mat4x4.inl
        thirdparty/glm/glm/detail/type_mat4x4_simd.inl
        thirdparty/glm/glm/detail/type_quat.hpp
        thirdparty/glm/glm/detail/type_quat.inl
        thirdparty/glm/glm/detail/type_quat_simd.inl
        thirdparty/glm/glm/detail/type_vec1.hpp
        thirdparty/glm/glm/detail/type_vec1.inl
        thirdparty/glm/glm/detail/type_vec2.hpp
        thirdparty/glm/glm/detail/type_vec2.inl
        thirdparty/glm/glm/detail/type_vec3.hpp
        thirdparty/glm/glm/detail/type_vec3.inl
        thirdparty/glm/glm/detail/type_vec4.hpp
        thirdparty/glm/glm/detail/type_vec4.inl
        thirdparty/glm/glm/detail/type_vec4_simd.inl
        thirdparty/glm/glm/ext/matrix_clip_space.hpp
        thirdparty/glm/glm/ext/matrix_clip_space.inl
        thirdparty/glm/glm/ext/matrix_double2x2.hpp
        thirdparty/glm/glm/ext/matrix_double2x2_precision.hpp
        thirdparty/glm/glm/ext/matrix_double2x3.hpp
        thirdparty/glm/glm/ext/matrix_double2x3_precision.hpp
        thirdparty/glm/glm/ext/matrix_double2x4.hpp
        thirdparty/glm/glm/ext/matrix_double2x4_precision.hpp
        thirdparty/glm/glm/ext/matrix_double3x2.hpp
        thirdparty/glm/glm/ext/matrix_double3x2_precision.hpp
        thirdparty/glm/glm/ext/matrix_double3x3.hpp
        thirdparty/glm/glm/ext/matrix_double3x3_precision.hpp
        thirdparty/glm/glm/ext/matrix_double3x4.hpp
        thirdparty/glm/glm/ext/matrix_double3x4_precision.hpp
        thirdparty/glm/glm/ext/matrix_double4x2.hpp
        thirdparty/glm/glm/ext/matrix_double4x2_precision.hpp
        thirdparty/glm/glm/ext/matrix_double4x3.hpp
        thirdparty/glm/glm/ext/matrix_double4x3_precision.hpp
        thirdparty/glm/glm/ext/matrix_double4x4.hpp
        thirdparty/glm/glm/ext/matrix_double4x4_precision.hpp
        thirdparty/glm/glm/ext/matrix_float2x2.hpp
        thirdparty/glm/glm/ext/matrix_float2x2_precision.hpp
        thirdparty/glm/glm/ext/matrix_float2x3.hpp
        thirdparty/glm/glm/ext/matrix_float2x3_precision.hpp
        thirdparty/glm/glm/ext/matrix_float2x4.hpp
        thirdparty/glm/glm/ext/matrix_float2x4_precision.hpp
        thirdparty/glm/glm/ext/matrix_float3x2.hpp
        thirdparty/glm/glm/ext/matrix_float3x2_precision.hpp
        thirdparty/glm/glm/ext/matrix_float3x3.hpp
        thirdparty/glm/glm/ext/matrix_float3x3_precision.hpp
        thirdparty/glm/glm/ext/matrix_float3x4.hpp
        thirdparty/glm/glm/ext/matrix_float3x4_precision.hpp
        thirdparty/glm/glm/ext/matrix_float4x2.hpp
        thirdparty/glm/glm/ext/matrix_float4x2_precision.hpp
        thirdparty/glm/glm/ext/matrix_float4x3.hpp
        thirdparty/glm/glm/ext/matrix_float4x3_precision.hpp
        thirdparty/glm/glm/ext/matrix_float4x4.hpp
        thirdparty/glm/glm/ext/matrix_float4x4_precision.hpp
        thirdparty/glm/glm/ext/matrix_projection.hpp
        thirdparty/glm/glm/ext/matrix_projection.inl
        thirdparty/glm/glm/ext/matrix_relational.hpp
        thirdparty/glm/glm/ext/matrix_relational.inl
        thirdparty/glm/glm/ext/matrix_transform.hpp
        thirdparty/glm/glm/ext/matrix_transform.inl
        thirdparty/glm/glm/ext/quaternion_common.hpp
        thirdparty/glm/glm/ext/quaternion_common.inl
        thirdparty/glm/glm/ext/quaternion_common_simd.inl
        thirdparty/glm/glm/ext/quaternion_double.hpp
        thirdparty/glm/glm/ext/quaternion_double_precision.hpp
        thirdparty/glm/glm/ext/quaternion_exponential.hpp
        thirdparty/glm/glm/ext/quaternion_exponential.inl
        thirdparty/glm/glm/ext/quaternion_float.hpp
        thirdparty/glm/glm/ext/quaternion_float_precision.hpp
        thirdparty/glm/glm/ext/quaternion_geometric.hpp
        thirdparty/glm/glm/ext/quaternion_geometric.inl
        thirdparty/glm/glm/ext/quaternion_relational.hpp
        thirdparty/glm/glm/ext/quaternion_relational.inl
        thirdparty/glm/glm/ext/quaternion_transform.hpp
        thirdparty/glm/glm/ext/quaternion_transform.inl
        thirdparty/glm/glm/ext/quaternion_trigonometric.hpp
        thirdparty/glm/glm/ext/quaternion_trigonometric.inl
        thirdparty/glm/glm/ext/scalar_common.hpp
        thirdparty/glm/glm/ext/scalar_common.inl
        thirdparty/glm/glm/ext/scalar_constants.hpp
        thirdparty/glm/glm/ext/scalar_constants.inl
        thirdparty/glm/glm/ext/scalar_int_sized.hpp
        thirdparty/glm/glm/ext/scalar_relational.hpp
        thirdparty/glm/glm/ext/scalar_relational.inl
        thirdparty/glm/glm/ext/scalar_uint_sized.hpp
        thirdparty/glm/glm/ext/scalar_ulp.hpp
        thirdparty/glm/glm/ext/scalar_ulp.inl
        thirdparty/glm/glm/ext/vector_bool1.hpp
        thirdparty/glm/glm/ext/vector_bool1_precision.hpp
        thirdparty/glm/glm/ext/vector_bool2.hpp
        thirdparty/glm/glm/ext/vector_bool2_precision.hpp
        thirdparty/glm/glm/ext/vector_bool3.hpp
        thirdparty/glm/glm/ext/vector_bool3_precision.hpp
        thirdparty/glm/glm/ext/vector_bool4.hpp
        thirdparty/glm/glm/ext/vector_bool4_precision.hpp
        thirdparty/glm/glm/ext/vector_common.hpp
        thirdparty/glm/glm/ext/vector_common.inl
        thirdparty/glm/glm/ext/vector_double1.hpp
        thirdparty/glm/glm/ext/vector_double1_precision.hpp
        thirdparty/glm/glm/ext/vector_double2.hpp
        thirdparty/glm/glm/ext/vector_double2_precision.hpp
        thirdparty/glm/glm/ext/vector_double3.hpp
        thirdparty/glm/glm/ext/vector_double3_precision.hpp
        thirdparty/glm/glm/ext/vector_double4.hpp
        thirdparty/glm/glm/ext/vector_double4_precision.hpp
        thirdparty/glm/glm/ext/vector_float1.hpp
        thirdparty/glm/glm/ext/vector_float1_precision.hpp
        thirdparty/glm/glm/ext/vector_float2.hpp
        thirdparty/glm/glm/ext/vector_float2_precision.hpp
        thirdparty/glm/glm/ext/vector_float3.hpp
        thirdparty/glm/glm/ext/vector_float3_precision.hpp
        thirdparty/glm/glm/ext/vector_float4.hpp
        thirdparty/glm/glm/ext/vector_float4_precision.hpp
        thirdparty/glm/glm/ext/vector_int1.hpp
        thirdparty/glm/glm/ext/vector_int1_precision.hpp
        thirdparty/glm/glm/ext/vector_int2.hpp
        thirdparty/glm/glm/ext/vector_int2_precision.hpp
        thirdparty/glm/glm/ext/vector_int3.hpp
        thirdparty/glm/glm/ext/vector_int3_precision.hpp
        thirdparty/glm/glm/ext/vector_int4.hpp
        thirdparty/glm/glm/ext/vector_int4_precision.hpp
        thirdparty/glm/glm/ext/vector_relational.hpp
        thirdparty/glm/glm/ext/vector_relational.inl
        thirdparty/glm/glm/ext/vector_uint1.hpp
        thirdparty/glm/glm/ext/vector_uint1_precision.hpp
        thirdparty/glm/glm/ext/vector_uint2.hpp
        thirdparty/glm/glm/ext/vector_uint2_precision.hpp
        thirdparty/glm/glm/ext/vector_uint3.hpp
        thirdparty/glm/glm/ext/vector_uint3_precision.hpp
        thirdparty/glm/glm/ext/vector_uint4.hpp
        thirdparty/glm/glm/ext/vector_uint4_precision.hpp
        thirdparty/glm/glm/ext/vector_ulp.hpp
        thirdparty/glm/glm/ext/vector_ulp.inl
        thirdparty/glm/glm/gtc/bitfield.hpp
        thirdparty/glm/glm/gtc/bitfield.inl
        thirdparty/glm/glm/gtc/color_space.hpp
        thirdparty/glm/glm/gtc/color_space.inl
        thirdparty/glm/glm/gtc/constants.hpp
        thirdparty/glm/glm/gtc/constants.inl
        thirdparty/glm/glm/gtc/epsilon.hpp
        thirdparty/glm/glm/gtc/epsilon.inl
        thirdparty/glm/glm/gtc/integer.hpp
        thirdparty/glm/glm/gtc/integer.inl
        thirdparty/glm/glm/gtc/matrix_access.hpp
        thirdparty/glm/glm/gtc/matrix_access.inl
        thirdparty/glm/glm/gtc/matrix_integer.hpp
        thirdparty/glm/glm/gtc/matrix_inverse.hpp
        thirdparty/glm/glm/gtc/matrix_inverse.inl
        thirdparty/glm/glm/gtc/matrix_transform.hpp
        thirdparty/glm/glm/gtc/matrix_transform.inl
        thirdparty/glm/glm/gtc/noise.hpp
        thirdparty/glm/glm/gtc/noise.inl
        thirdparty/glm/glm/gtc/packing.hpp
        thirdparty/glm/glm/gtc/packing.inl
        thirdparty/glm/glm/gtc/quaternion.hpp
        thirdparty/glm/glm/gtc/quaternion.inl
        thirdparty/glm/glm/gtc/quaternion_simd.inl
        thirdparty/glm/glm/gtc/random.hpp
        thirdparty/glm/glm/gtc/random.inl
        thirdparty/glm/glm/gtc/reciprocal.hpp
        thirdparty/glm/glm/gtc/reciprocal.inl
        thirdparty/glm/glm/gtc/round.hpp
        thirdparty/glm/glm/gtc/round.inl
        thirdparty/glm/glm/gtc/type_aligned.hpp
        thirdparty/glm/glm/gtc/type_precision.hpp
        thirdparty/glm/glm/gtc/type_precision.inl
        thirdparty/glm/glm/gtc/type_ptr.hpp
        thirdparty/glm/glm/gtc/type_ptr.inl
        thirdparty/glm/glm/gtc/ulp.hpp
        thirdparty/glm/glm/gtc/ulp.inl
        thirdparty/glm/glm/gtc/vec1.hpp
        thirdparty/glm/glm/gtx/associated_min_max.hpp
        thirdparty/glm/glm/gtx/associated_min_max.inl
        thirdparty/glm/glm/gtx/bit.hpp
        thirdparty/glm/glm/gtx/bit.inl
        thirdparty/glm/glm/gtx/closest_point.hpp
        thirdparty/glm/glm/gtx/closest_point.inl
        thirdparty/glm/glm/gtx/color_encoding.hpp
        thirdparty/glm/glm/gtx/color_encoding.inl
        thirdparty/glm/glm/gtx/color_space.hpp
        thirdparty/glm/glm/gtx/color_space.inl
        thirdparty/glm/glm/gtx/color_space_YCoCg.hpp
        thirdparty/glm/glm/gtx/color_space_YCoCg.inl
        thirdparty/glm/glm/gtx/common.hpp
        thirdparty/glm/glm/gtx/common.inl
        thirdparty/glm/glm/gtx/compatibility.hpp
        thirdparty/glm/glm/gtx/compatibility.inl
        thirdparty/glm/glm/gtx/component_wise.hpp
        thirdparty/glm/glm/gtx/component_wise.inl
        thirdparty/glm/glm/gtx/dual_quaternion.hpp
        thirdparty/glm/glm/gtx/dual_quaternion.inl
        thirdparty/glm/glm/gtx/easing.hpp
        thirdparty/glm/glm/gtx/easing.inl
        thirdparty/glm/glm/gtx/euler_angles.hpp
        thirdparty/glm/glm/gtx/euler_angles.inl
        thirdparty/glm/glm/gtx/extend.hpp
        thirdparty/glm/glm/gtx/extend.inl
        thirdparty/glm/glm/gtx/extended_min_max.hpp
        thirdparty/glm/glm/gtx/extended_min_max.inl
        thirdparty/glm/glm/gtx/exterior_product.hpp
        thirdparty/glm/glm/gtx/exterior_product.inl
        thirdparty/glm/glm/gtx/fast_exponential.hpp
        thirdparty/glm/glm/gtx/fast_exponential.inl
        thirdparty/glm/glm/gtx/fast_square_root.hpp
        thirdparty/glm/glm/gtx/fast_square_root.inl
        thirdparty/glm/glm/gtx/fast_trigonometry.hpp
        thirdparty/glm/glm/gtx/fast_trigonometry.inl
        thirdparty/glm/glm/gtx/float_notmalize.inl
        thirdparty/glm/glm/gtx/functions.hpp
        thirdparty/glm/glm/gtx/functions.inl
        thirdparty/glm/glm/gtx/gradient_paint.hpp
        thirdparty/glm/glm/gtx/gradient_paint.inl
        thirdparty/glm/glm/gtx/handed_coordinate_space.hpp
        thirdparty/glm/glm/gtx/handed_coordinate_space.inl
        thirdparty/glm/glm/gtx/hash.hpp
        thirdparty/glm/glm/gtx/hash.inl
        thirdparty/glm/glm/gtx/integer.hpp
        thirdparty/glm/glm/gtx/integer.inl
        thirdparty/glm/glm/gtx/intersect.hpp
        thirdparty/glm/glm/gtx/intersect.inl
        thirdparty/glm/glm/gtx/io.hpp
        thirdparty/glm/glm/gtx/io.inl
        thirdparty/glm/glm/gtx/log_base.hpp
        thirdparty/glm/glm/gtx/log_base.inl
        thirdparty/glm/glm/gtx/matrix_cross_product.hpp
        thirdparty/glm/glm/gtx/matrix_cross_product.inl
        thirdparty/glm/glm/gtx/matrix_decompose.hpp
        thirdparty/glm/glm/gtx/matrix_decompose.inl
        thirdparty/glm/glm/gtx/matrix_factorisation.hpp
        thirdparty/glm/glm/gtx/matrix_factorisation.inl
        thirdparty/glm/glm/gtx/matrix_interpolation.hpp
        thirdparty/glm/glm/gtx/matrix_interpolation.inl
        thirdparty/glm/glm/gtx/matrix_major_storage.hpp
        thirdparty/glm/glm/gtx/matrix_major_storage.inl
        thirdparty/glm/glm/gtx/matrix_operation.hpp
        thirdparty/glm/glm/gtx/matrix_operation.inl
        thirdparty/glm/glm/gtx/matrix_query.hpp
        thirdparty/glm/glm/gtx/matrix_query.inl
        thirdparty/glm/glm/gtx/matrix_transform_2d.hpp
        thirdparty/glm/glm/gtx/matrix_transform_2d.inl
        thirdparty/glm/glm/gtx/mixed_product.hpp
        thirdparty/glm/glm/gtx/mixed_product.inl
        thirdparty/glm/glm/gtx/norm.hpp
        thirdparty/glm/glm/gtx/norm.inl
        thirdparty/glm/glm/gtx/normal.hpp
        thirdparty/glm/glm/gtx/normal.inl
        thirdparty/glm/glm/gtx/normalize_dot.hpp
        thirdparty/glm/glm/gtx/normalize_dot.inl
        thirdparty/glm/glm/gtx/number_precision.hpp
        thirdparty/glm/glm/gtx/number_precision.inl
        thirdparty/glm/glm/gtx/optimum_pow.hpp
        thirdparty/glm/glm/gtx/optimum_pow.inl
        thirdparty/glm/glm/gtx/orthonormalize.hpp
        thirdparty/glm/glm/gtx/orthonormalize.inl
        thirdparty/glm/glm/gtx/perpendicular.hpp
        thirdparty/glm/glm/gtx/perpendicular.inl
        thirdparty/glm/glm/gtx/polar_coordinates.hpp
        thirdparty/glm/glm/gtx/polar_coordinates.inl
        thirdparty/glm/glm/gtx/projection.hpp
        thirdparty/glm/glm/gtx/projection.inl
        thirdparty/glm/glm/gtx/quaternion.hpp
        thirdparty/glm/glm/gtx/quaternion.inl
        thirdparty/glm/glm/gtx/range.hpp
        thirdparty/glm/glm/gtx/raw_data.hpp
        thirdparty/glm/glm/gtx/raw_data.inl
        thirdparty/glm/glm/gtx/rotate_normalized_axis.hpp
        thirdparty/glm/glm/gtx/rotate_normalized_axis.inl
        thirdparty/glm/glm/gtx/rotate_vector.hpp
        thirdparty/glm/glm/gtx/rotate_vector.inl
        thirdparty/glm/glm/gtx/scalar_multiplication.hpp
        thirdparty/glm/glm/gtx/scalar_relational.hpp
        thirdparty/glm/glm/gtx/scalar_relational.inl
        thirdparty/glm/glm/gtx/spline.hpp
        thirdparty/glm/glm/gtx/spline.inl
        thirdparty/glm/glm/gtx/std_based_type.hpp
        thirdparty/glm/glm/gtx/std_based_type.inl
        thirdparty/glm/glm/gtx/string_cast.hpp
        thirdparty/glm/glm/gtx/string_cast.inl
        thirdparty/glm/glm/gtx/texture.hpp
        thirdparty/glm/glm/gtx/texture.inl
        thirdparty/glm/glm/gtx/transform.hpp
        thirdparty/glm/glm/gtx/transform.inl
        thirdparty/glm/glm/gtx/transform2.hpp
        thirdparty/glm/glm/gtx/transform2.inl
        thirdparty/glm/glm/gtx/type_aligned.hpp
        thirdparty/glm/glm/gtx/type_aligned.inl
        thirdparty/glm/glm/gtx/type_trait.hpp
        thirdparty/glm/glm/gtx/type_trait.inl
        thirdparty/glm/glm/gtx/vec_swizzle.hpp
        thirdparty/glm/glm/gtx/vector_angle.hpp
        thirdparty/glm/glm/gtx/vector_angle.inl
        thirdparty/glm/glm/gtx/vector_query.hpp
        thirdparty/glm/glm/gtx/vector_query.inl
        thirdparty/glm/glm/gtx/wrap.hpp
        thirdparty/glm/glm/gtx/wrap.inl
        thirdparty/glm/glm/simd/common.h
        thirdparty/glm/glm/simd/exponential.h
        thirdparty/glm/glm/simd/geometric.h
        thirdparty/glm/glm/simd/integer.h
        thirdparty/glm/glm/simd/matrix.h
        thirdparty/glm/glm/simd/packing.h
        thirdparty/glm/glm/simd/platform.h
        thirdparty/glm/glm/simd/trigonometric.h
        thirdparty/glm/glm/simd/vector_relational.h
        thirdparty/glm/glm/common.hpp
        thirdparty/glm/glm/exponential.hpp
        thirdparty/glm/glm/ext.hpp
        thirdparty/glm/glm/fwd.hpp
        thirdparty/glm/glm/geometric.hpp
        thirdparty/glm/glm/glm.hpp
        thirdparty/glm/glm/integer.hpp
        thirdparty/glm/glm/mat2x2.hpp
        thirdparty/glm/glm/mat2x3.hpp
        thirdparty/glm/glm/mat2x4.hpp
        thirdparty/glm/glm/mat3x2.hpp
        thirdparty/glm/glm/mat3x3.hpp
        thirdparty/glm/glm/mat3x4.hpp
        thirdparty/glm/glm/mat4x2.hpp
        thirdparty/glm/glm/mat4x3.hpp
        thirdparty/glm/glm/mat4x4.hpp
        thirdparty/glm/glm/matrix.hpp
        thirdparty/glm/glm/packing.hpp
        thirdparty/glm/glm/trigonometric.hpp
        thirdparty/glm/glm/vec2.hpp
        thirdparty/glm/glm/vec3.hpp
        thirdparty/glm/glm/vec4.hpp
        thirdparty/glm/glm/vector_relational.hpp
        thirdparty/stb/stb/stb.h
        thirdparty/stb/stb/stb_c_lexer.h
        thirdparty/stb/stb/stb_connected_components.h
        thirdparty/stb/stb/stb_divide.h
        thirdparty/stb/stb/stb_ds.h
        thirdparty/stb/stb/stb_dxt.h
        thirdparty/stb/stb/stb_easy_font.h
        thirdparty/stb/stb/stb_herringbone_wang_tile.h
        thirdparty/stb/stb/stb_image.h
        thirdparty/stb/stb/stb_image_resize.h
        thirdparty/stb/stb/stb_image_write.h
        thirdparty/stb/stb/stb_leakcheck.h
        thirdparty/stb/stb/stb_perlin.h
        thirdparty/stb/stb/stb_rect_pack.h
        thirdparty/stb/stb/stb_sprintf.h
        thirdparty/stb/stb/stb_textedit.h
        thirdparty/stb/stb/stb_tilemap_editor.h
        thirdparty/stb/stb/stb_truetype.h
        thirdparty/stb/stb/stb_vorbis.c
        thirdparty/stb/stb/stb_voxel_render.h
        thirdparty/stb/stb/stretchy_buffer.h)


target_link_libraries(DAG_Compression
        SDL2::SDL2
        SDL2::SDL2main
        OpenGL::GL
        GLEW::GLEW
        glm)
