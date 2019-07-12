#ifndef PROJECT_CUDAHELPERS_H_
#define PROJECT_CUDAHELPERS_H_

#include <GL/glew.h>
#include <cuda_gl_interop.h>
#include <cuda_runtime.h>


#ifdef __CUDACC__
__device__ inline int getGlobalIdx_1D_1D() { return blockIdx.x * blockDim.x + threadIdx.x; }
#endif
__host__ __device__ float3 inline operator * (const float a, const float3 &b) { return make_float3(a * b.x, a * b.y, a * b.z); };
__host__ __device__ double3 inline operator * (const double a, const double3 &b) { return make_double3(a * b.x, a * b.y, a * b.z); };
__host__ __device__ float3 inline operator * (const float3 &b, const float a) { return a * b; };
__host__ __device__ double3 inline operator * (const double3 &b, const double a) { return a * b; };
__host__ __device__ float3 inline operator - (const float3 &a, const float3 &b) { return make_float3(a.x - b.x, a.y - b.y, a.z - b.z); };
__host__ __device__ double3 inline operator - (const double3 &a, const double3 &b) { return make_double3(a.x - b.x, a.y - b.y, a.z - b.z); };
__host__ __device__ float3 inline operator + (const float3 &a, const float3 &b) { return make_float3(a.x + b.x, a.y + b.y, a.z + b.z); };
__host__ __device__ double3 inline operator + (const double3 &a, const double3 &b) { return make_double3(a.x + b.x, a.y + b.y, a.z + b.z); };
__host__ __device__ float3 inline operator * (const float3 &a, const float3 &b) { return make_float3(a.x * b.x, a.y * b.y, a.z * b.z); };
__host__ __device__ double3 inline operator * (const double3 &a, const double3 &b) { return make_double3(a.x * b.x, a.y * b.y, a.z * b.z); };
__host__ __device__ float3 inline operator / (const float3 &a, const float3 &b) { return make_float3(a.x / b.x, a.y / b.y, a.z / b.z); };
__host__ __device__ float3 inline operator - (float3 a) { return make_float3(-a.x, -a.y, -a.z); };
__host__ __device__ float3 inline make_float3(float a) { return make_float3(a,a,a); };
__host__ __device__ float  inline dot(const float3 &a, const float3 &b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
__host__ __device__ double  inline dot(const double3&a, const double3&b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
__host__ __device__ float  inline length(float3 a) { return sqrt(dot(a, a)); }
__host__ __device__ double  inline length(double3 a) { return sqrt(dot(a, a)); }
__host__ __device__ float3 inline normalize(float3 a) { return (1.0f / length(a)) * a; }
__host__ __device__ double3 inline normalize(double3 a) { return (1.0 / length(a)) * a; }
__host__ __device__ uint3  inline operator << (const uint3 &v, const int shift) { return make_uint3(v.x << shift, v.y << shift, v.z << shift); }
__host__ __device__ uint3  inline operator + (const uint3 &a, const uint3 &b) { return make_uint3(a.x + b.x, a.y + b.y, a.z + b.z); };
__host__ __device__ uint3  inline operator - (const uint3 &a, const uint3 &b) { return make_uint3(a.x - b.x, a.y - b.y, a.z - b.z); };
__host__ __device__ uint3  inline make_uint3(uint32_t a) { return make_uint3(a, a, a); };
__host__ __device__ bool   inline operator == (const uint3 & a, const uint3 & b) { return (a.x == b.x) && (a.y == b.y) && (a.z == b.z); }
__host__ __device__ bool   inline operator != (const uint3 & a, const uint3 & b) { return !(a == b); }
__host__ __device__ float3 inline make_float3(const uint3 &a) { return make_float3(float(a.x), float(a.y), float(a.z)); };
__host__ __device__ float3 inline make_float3(const double3 &a) { return make_float3(float(a.x), float(a.y), float(a.z)); };
__host__ __device__ double3 inline make_double3(const float3 &a) { return make_double3(double(a.x), double(a.y), double(a.z)); };
__host__ __device__ uint3  inline make_uint3(const uint4 &v) { return make_uint3(v.x, v.y, v.z); }
__host__ __device__ uint4  inline make_uint4(const uint3 &v, uint32_t a) { return make_uint4(v.x, v.y, v.z, a); }
__host__ __device__ float3 inline operator / (const float f, const float3 &v) { return make_float3(f / v.x, f / v.y, f / v.z); };
__host__ __device__ float3 inline abs(const float3 &v) { return make_float3(abs(v.x), abs(v.y), abs(v.z)); };
__host__ __device__ float  inline max(const float3 &v) { return fmax(v.x, fmax(v.y, v.z)); };
__host__ __device__ float  inline min(const float3 &v) { return fmin(v.x, fmin(v.y, v.z)); };


class CUDAGLInteropSurface
{
public:
	bool					m_registered;
	uint32_t				m_gl_idx;
	GLenum					m_gl_target;
	uint32_t				m_cuda_register_flags;
	cudaGraphicsResource_t	m_cuda_resource;
	cudaSurfaceObject_t		m_cuda_surface_object;
	CUDAGLInteropSurface(){
		m_gl_target = GL_TEXTURE_2D;
		m_cuda_register_flags = cudaGraphicsRegisterFlagsSurfaceLoadStore;
		m_gl_idx = 0;
		m_registered = false;
	};
	~CUDAGLInteropSurface(){
		unregisterResource();
	};
	void registerResource(){
		//assert(!m_registered);
		m_registered = true;
		cudaGraphicsGLRegisterImage(&m_cuda_resource, m_gl_idx, m_gl_target, m_cuda_register_flags);
	};
	void unregisterResource(){
		if (m_registered) {
			m_registered = false;
			cudaGraphicsUnregisterResource(m_cuda_resource);
		}
	};
	void mapSurfaceObject(){
		cudaGraphicsMapResources(1, &m_cuda_resource);
		cudaArray_t cuda_array;
		cudaGraphicsSubResourceGetMappedArray(&cuda_array, m_cuda_resource, 0, 0);
		cudaResourceDesc cuda_array_resource_desc;
		memset(&cuda_array_resource_desc, 0, sizeof(cuda_array_resource_desc));
		cuda_array_resource_desc.resType = cudaResourceTypeArray;
		cuda_array_resource_desc.res.array.array = cuda_array;
		cudaCreateSurfaceObject(&m_cuda_surface_object, &cuda_array_resource_desc);
	};
	void unmapSurfaceObject(){
		cudaDestroySurfaceObject(m_cuda_surface_object);
		cudaGraphicsUnmapResources(1, &m_cuda_resource);
	};
};

/////////////////////////////////////////////////////////////
// Plain buffer interop
/////////////////////////////////////////////////////////////
template <typename T>
class CUDAGLInteropBuffer {
 public:
	bool                   m_registered;
	uint32_t               m_gl_idx;
	GLenum                 m_gl_target;
	uint32_t               m_cuda_register_flags;
	cudaGraphicsResource_t m_cuda_resource;
	T                     *m_cuda_buffer_ptr;
	uint32_t              *m_sorted_index_ptr;
	int                    m_nelem;
	CUDAGLInteropBuffer() : m_gl_target(0), m_cuda_resource(nullptr), m_sorted_index_ptr(nullptr), m_nelem(0) {
		m_cuda_register_flags = cudaGraphicsMapFlagsNone;
		m_gl_idx              = 0;
		m_registered          = false;
	};
	~CUDAGLInteropBuffer() { unregisterResource(); };
	void registerResource() {
		assert(!m_registered);
		m_registered = true;
		cudaGraphicsGLRegisterBuffer(&m_cuda_resource, m_gl_idx, m_cuda_register_flags);
	};
	void unregisterResource() {
		if (m_registered) {
			m_registered = false;
			cudaGraphicsUnregisterResource(m_cuda_resource);
		}
	};
	void mapBuffer() {
		cudaGraphicsMapResources(1, &m_cuda_resource);
		size_t tmp_size;
		cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void **>(&m_cuda_buffer_ptr), &tmp_size,
		                                                       m_cuda_resource);
	};
	void unmapBuffer() { cudaGraphicsUnmapResources(1, &m_cuda_resource); };
};
#endif  // PROJECT_CUDAHELPERS_H_
