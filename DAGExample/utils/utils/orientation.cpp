#include "orientation.h"
#include "glm_extensions.h"
#include <GL/glew.h>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <iostream>
using namespace std;

namespace chag {
using mat4 = orientation::mat4;

orientation::orientation(vec3 pos, vec3 dir, vec3 up) {
	lookAt(pos, dir, up);
	scale = vec3(1.0f);
}

void orientation::lookAt(vec3 eye, vec3 center, vec3 up) {
	pos        = eye;
	vec3 dir   = normalize(eye - center);
	vec3 right = normalize(cross(up, normalize(dir)));
	vec3 newup = normalize(cross(dir, right));
	R          = mat3(right, newup, dir);
}

void orientation::lookAt(vec3 eye, vec3 center) {
	this->pos  = eye;
	vec3 dir   = normalize(eye - center);
	vec3 right = normalize(perp(dir));
	vec3 newup = normalize(cross(dir, right));
	this->R    = mat3(right, newup, dir);
}

void orientation::set_as_modelview() {
	mat4 MV = get_MV();
	glMatrixLoadfEXT(GL_MODELVIEW, glm::value_ptr(MV));
}

mat4 orientation::get_MV() const {
	mat4 invrot = mat4(transpose(R));
	return glm::scale(mat4(1.0f), vec3(1.0f / scale.x, 1.0f / scale.y, 1.0f / scale.z)) * invrot *
	       glm::translate(mat4(1.0f), -pos);
}

mat4 orientation::get_MV_inv() const {
	return glm::translate(mat4(1.0f), pos) * mat4(R) * glm::scale(mat4(1.0f), scale);
}

void orientation::roll(float angle)  { R = R * mat3(glm::rotate(mat4(1.0f), angle, vec3(0.0f, 0.0f, -1.0f))); }
void orientation::yaw(float angle)   { R = R * mat3(glm::rotate(mat4(1.0f), angle, vec3(0.0f, 1.0f, 0.0f))); }
void orientation::pitch(float angle) { R = R * mat3(glm::rotate(mat4(1.0f), angle, vec3(1.0f, 0.0f, 0.0f))); }

}  // namespace chag
