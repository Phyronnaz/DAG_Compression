#pragma once

//#include "ray.h"
#include <glm/glm.hpp>

namespace chag
{

	/**
	* An aabb defined by the min and max extrema.
	*/
	class Aabb
	{
	public:
		using vec3 = glm::vec3;
		vec3 min;
		vec3 max;

		/**
		*/
		const vec3 getCentre() const { return (min + max) * 0.5f; }

		/**
		*/
		const vec3 getHalfSize() const { return (max - min) * 0.5f; }

		/**
		*/
		float getVolume() const { vec3 d = max - min; return d.x * d.y * d.z; }

		/**
		*/
		float getArea() const { vec3 d = max - min; return d.x*d.y*2.0f + d.x*d.z*2.0f + d.z*d.y*2.0f; }

	};



	/**
	*/
	inline Aabb combine(const Aabb &a, const Aabb &b)
	{
		return { glm::min(a.min, b.min), glm::max(a.max, b.max) };
	}

	/**
	*
	*/
	inline Aabb combine(const Aabb &a, const glm::vec3 &pt)
	{
		return { glm::min(a.min, pt), glm::max(a.max, pt) };
	}
	/**
	* creates an aabb that has min = FLT_MAX and max = -FLT_MAX.
	*/
	Aabb make_inverse_extreme_aabb();


	/**
	*/
	Aabb make_aabb(const glm::vec3 &min, const glm::vec3 &max);

	/**
	*/
	inline Aabb make_aabb(const glm::vec3 &position, const float radius)
	{
		return { position - radius, position + radius };
	}

	/**
	*/
	Aabb make_aabb(const glm::vec3 *positions, const size_t numPositions);

	/**
	*/
	inline bool overlaps(const Aabb &a, const Aabb &b)
	{
		return a.max.x > b.min.x && a.min.x < b.max.x
			&& a.max.y > b.min.y && a.min.y < b.max.y
			&& a.max.z > b.min.z && a.min.z < b.max.z;

	}

	///**
	//* Intersect with a ray (from pbrt)
	//*/

	//inline bool intersect(const Aabb &a, const ray &r, float *hitt0 = nullptr, float *hitt1 = nullptr)
	//{
	//	float t0 = r.mint, t1 = r.maxt;
	//	for(int i=0; i<3; i++){
	//		float invRayDir = 1.0f / r.d[i];
	//		float tNear = (a.min[i] - r.o[i]) * invRayDir; 
	//		float tFar =  (a.max[i] - r.o[i]) * invRayDir; 
	//		if(tNear > tFar) { //swap(tNear, tFar); 
	//			float temp = tNear; 
	//			tNear = tFar; 
	//			tFar = temp; 
	//		}
	//		t0 = tNear > t0 ? tNear : t0; 
	//		t1 = tFar < t1 ? tFar : t1; 
	//		if(t0 > t1) return false; 
	//	}
	//	if(hitt0) *hitt0 = t0;
	//	if(hitt1) *hitt1 = t1;
	//	return true; 
	//}



} // namespace chag
