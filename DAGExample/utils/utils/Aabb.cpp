#include "Aabb.h"
#include <float.h>
#include <algorithm>

namespace chag
{


	Aabb make_aabb(const glm::vec3 &min, const glm::vec3 &max)
	{
		return { min, max };
	}


	Aabb make_inverse_extreme_aabb()
	{
		return { glm::vec3(FLT_MAX), glm::vec3(-FLT_MAX) };
	}



	Aabb make_aabb(const glm::vec3 *positions, const size_t numPositions)
	{
		Aabb result = make_inverse_extreme_aabb();

		for (size_t i = 0; i < numPositions; ++i)
		{
			result = combine(result, positions[i]);
		}

		return result;
	}

} // namespace chag
