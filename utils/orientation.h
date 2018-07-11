#pragma once

#include <glm/glm.hpp>
namespace chag {
class orientation {
 public:
	using mat3 = glm::mat3;
	using mat4 = glm::mat4;
	using vec3 = glm::vec3;

    orientation() {};
	~orientation() {};
	orientation(vec3 pos, vec3 dir, vec3 up);

	mat3 R     = mat3(1.0f);
	vec3 pos   = vec3(0.0f);
	vec3 scale = vec3(1.0f);
	void lookAt(vec3 pos, vec3 center, vec3 up);
	void lookAt(vec3 pos, vec3 center);

	mat4 get_MV() const;
	mat4 get_MV_inv() const;
	void set_as_modelview();

	void roll(float angle);
	void yaw(float angle);
	void pitch(float angle);
};
}  // namespace chag
