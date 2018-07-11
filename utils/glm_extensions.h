#pragma once
#include <iostream>
#include <glm/vec3.hpp>
#include <glm/mat4x4.hpp>

namespace glm
{
	////////////////////////////////////////////////////////////
	// Extended functions
	////////////////////////////////////////////////////////////
    const mat4 make_matrix_from_zAxis(const vec3& pos, const vec3& zAxis, const vec3& yAxis);

	// Equivalent to glFrustum
    const mat4 make_frustum(float left, float right, float bottom, float top, float znear, float zfar);
    const mat4 make_frustum_inv(float left, float right, float bottom, float top, float znear, float zfar);

    // Equivalent to gluPerspective
	const mat4 make_perspective(float fov, float aspect_ratio, float near, float far);
	const mat4 make_perspective_inv(float fov, float aspect_ratio, float near, float far);
	
	// Equivalent to glOrtho
	const mat4 make_ortho(float l, float r, float b, float t, float n, float f);
	const mat4 make_ortho_inv(float l, float r, float b, float t, float n, float f);
	
	// Equivalent to gluOrtho2d
	const mat4 make_ortho2d(float l, float r, float b, float t);

    // FIXME: Perp and perpendicular is essentialy the same and should
	//        perhaps be refactored.
	const vec3 perp(const vec3& a);
	inline vec3 perpendicular(const vec3 &v) {
		if (fabsf(v.x) < fabsf(v.y)) { return vec3(0.0f, -v.z, v.y); }
		return vec3(-v.z, 0.0f, v.x);
	}
}
