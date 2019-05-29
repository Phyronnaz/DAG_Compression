#include "glm_extensions.h"
#include <glm/glm.hpp>

using namespace std;
namespace glm
{
	////////////////////////////////////////////////////////////
	// Extended functions
	////////////////////////////////////////////////////////////
		const mat4 make_matrix_from_zAxis(const vec3& pos, const vec3& zAxis, const vec3& yAxis) {
			vec3 z = normalize(zAxis);
			vec3 x = normalize(cross(yAxis, z));
			vec3 y = cross(z, x);
			mat4 m = {
			{x.x,   x.y,   x.z,   0.0f}, 
			{y.x,   y.y,   y.z,   0.0f}, 
			{z.x,   z.y,   z.z,   0.0f}, 
			{pos.x, pos.y, pos.z, 1.0f}
		};
			return m;
		}

	const mat4 make_frustum(float left, float right, float bottom,
		float top, float znear, float zfar)
	{
		float temp, temp2, temp3, temp4;
		temp = 2.0f * znear;
		temp2 = right - left;
		temp3 = top - bottom;
		temp4 = zfar - znear;
		mat4 m =
		{
			/*c1*/{ temp / temp2, 0.0f, 0.0f, 0.0f },
			/*c2*/{ 0.0f, temp / temp3, 0.0f, 0.0f },
			/*c3*/{ (right + left) / temp2, (top + bottom) / temp3, (-zfar - znear) / temp4, -1.0f },
			/*c4*/{ 0.0f, 0.0f, (-temp * zfar) / temp4, 0.0f }
		};
		return m;
	}
	const mat4 make_frustum_inv(float left, float right, float bottom,
		float top, float znear, float zfar)
	{
		float temp, temp2, temp3, temp4;
		temp = 2.0f * znear;
		temp2 = right - left;
		temp3 = top - bottom;
		temp4 = zfar - znear;
		mat4 m =
		{
			/*c1*/{ temp2 / temp, 0.0f, 0.0f, 0.0f },
			/*c2*/{ 0.0f, temp3 / temp, 0.0f, 0.0f },
			/*c3*/{ 0.0f, 0.0f, 0.0f, -temp4 / (temp*zfar) },
			/*c4*/{ (right + left) / temp, (top + bottom) / temp, -1.0f, (zfar + znear) / (temp*zfar) }
		};
		return m;
	}

	// Equivalent to gluPerspective
	const mat4 make_perspective(float fov, float aspect_ratio, float near, float far)
	{
		float ymax = near * tanf(0.5f*radians(fov));
		float xmax = ymax * aspect_ratio;
		return make_frustum(-xmax, xmax, -ymax, ymax, near, far);
	}
	const mat4 make_perspective_inv(float fov, float aspect_ratio, float near, float far)
	{
		float ymax = near * tanf(0.5f*radians(fov));
		float xmax = ymax * aspect_ratio;
		return make_frustum_inv(-xmax, xmax, -ymax, ymax, near, far);
	}

	// Equivalent to glOrtho
	const mat4 make_ortho(float l, float r, float b, float t, float n, float f)
	{
		mat4 m =
		{
			/*c1*/{ 2.0f / (r - l), 0.0f, 0.0f, 0.0f },
			/*c2*/{ 0.0f, 2.0f / (t - b), 0.0f, 0.0f },
			/*c3*/{ 0.0f, 0.0f, 2.0f / (f - n), 0.0f },
			/*c4*/{ -((r + l) / (r - l)), -((t + b) / (t - b)), -((f + n) / (f - n)), 1.0f }
		};
		return m;
	}
	const mat4 make_ortho_inv(float l, float r, float b, float t, float n, float f)
	{
		mat4 m =
		{
			/*c1*/{ (r - l) / 2.0f, 0.0f, 0.0f, 0.0f },
			/*c2*/{ 0.0f, (t - b) / 2.0f, 0.0f, 0.0f },
			/*c3*/{ 0.0f, 0.0f, (f - n) / -2.0f, 0.0f },
			/*c4*/{ (r + l) / 2.0f, (t + b) / 2.0f, (n + f) / 2.0f, 1.0 }
		};
		return m;
	}
	// Equivalent to gluOrtho2d
	const mat4 make_ortho2d(float l, float r, float b, float t)
	{
		return make_ortho(l, r, b, t, -1.0f, 1.0f);
	}

	const vec3 perp(const vec3& a) {
		vec3 nv = vec3(std::abs(a.x), std::abs(a.y), std::abs(a.z));
		if (nv.x < nv.y)
			if (nv.x < nv.z)
				return vec3(0.0f, -a.z, a.y);
			else
				return vec3(-a.y, a.x, 0.0f);
		else if (nv.y < nv.z)
			return vec3(-a.z, 0.0f, a.x);
		else
			return vec3(-a.y, a.x, 0.0f);
	}
}


