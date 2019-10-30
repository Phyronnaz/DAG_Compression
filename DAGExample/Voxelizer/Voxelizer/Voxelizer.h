#pragma once
#include <array>
#include <GL/glew.h>
#include <utils/Aabb.h>
#include <utils/view.h>
#include <glm/gtc/type_ptr.hpp>

#ifndef FRAMEBUFFER_PROGRAMMABLE_SAMPLE_LOCATIONS_NV
#define FRAMEBUFFER_PROGRAMMABLE_SAMPLE_LOCATIONS_NV 0x9342
#endif

namespace voxelizer {
class Voxelizer {
 public:
	GLuint m_frag_ctr_buffer;
	GLuint m_dummy_fbo;
	GLuint tex;
	int m_tex_dim = 64 * 1024 * 1024;
	int m_grid_size;
	GLuint voxelize_shader{0};

 public:
	GLuint m_position_ssbo;
	GLuint m_mask_ssbo;
	GLuint m_base_color_ssbo;
	GLuint m_frag_count;
	GLuint m_num_colors;
	chag::Aabb m_aabb;

 public:
	template<class Fn> inline uint32_t
		generate_fragments(Fn DrawFunc, const chag::Aabb& aabb, int grid_resolution)
	{
		m_aabb = aabb;
		// Get ortho camera for given axis.
		enum class Axis { X, Y, Z };
		auto get_camera = [&aabb](Axis axis) {
			// clang-format off
		const glm::vec3 pos =   axis == Axis::X ?    aabb.getCentre() - aabb.getHalfSize().x * glm::vec3{1.f, 0.f, 0.f}
												:   axis == Axis::Y ?    aabb.getCentre() - aabb.getHalfSize().y * glm::vec3{0.f, 1.f, 0.f}
												:/* axis == Axis::Z ? */ aabb.getCentre() - aabb.getHalfSize().z * glm::vec3{0.f, 0.f, 1.f};
		const glm::vec3 up =   axis == Axis::X ?     glm::vec3{0.f, 1.f, 0.f}
											 :   axis == Axis::Y ?     glm::vec3{0.f, 0.f, 1.f}
											 :/* axis == Axis::Z ? */  glm::vec3{1.f, 0.f, 0.f};
			// clang-format on

			// Figure out clipping planes.
			const std::array<const glm::vec3, 8> points{
					glm::vec3{aabb.min.x, aabb.max.y, aabb.min.z}, glm::vec3{aabb.min.x, aabb.max.y, aabb.max.z},
					glm::vec3{aabb.min.x, aabb.min.y, aabb.min.z}, glm::vec3{aabb.min.x, aabb.min.y, aabb.max.z},
					glm::vec3{aabb.max.x, aabb.max.y, aabb.min.z}, glm::vec3{aabb.max.x, aabb.max.y, aabb.max.z},
					glm::vec3{aabb.max.x, aabb.min.y, aabb.min.z}, glm::vec3{aabb.max.x, aabb.min.y, aabb.max.z}};

			float min_x = std::numeric_limits<float>::max();
			float min_y = std::numeric_limits<float>::max();
			float min_z = std::numeric_limits<float>::max();
			float max_x = std::numeric_limits<float>::lowest();
			float max_y = std::numeric_limits<float>::lowest();
			float max_z = std::numeric_limits<float>::lowest();

			chag::orthoview result;
			result.lookAt(pos, aabb.getCentre(), up);
			{
				const glm::mat4 MV = result.get_MV();
				for (const auto& v : points) {
					const glm::vec4 vec = MV * glm::vec4{v, 1.0f};

					min_x = std::min(min_x, vec.x);
					min_y = std::min(min_y, vec.y);
					min_z = std::min(min_z, vec.z);
					max_x = std::max(max_x, vec.x);
					max_y = std::max(max_y, vec.y);
					max_z = std::max(max_z, vec.z);
				}
			}

			result.m_right  = max_x;
			result.m_left   = min_x;
			result.m_top    = max_y;
			result.m_bottom = min_y;
			// TODO: Remember my reason for this..
			result.m_far  = min_z;
			result.m_near = max_z;

			return result;
		};
		const chag::orthoview o_x = get_camera(Axis::X);
		const chag::orthoview o_y = get_camera(Axis::Y);
		const chag::orthoview o_z = get_camera(Axis::Z);

		glUseProgram(voxelize_shader);
		{
			GLuint location;
			location = glGetUniformLocation(voxelize_shader, "proj_x");
			glUniformMatrix4fv(location, 1, false, glm::value_ptr(o_x.get_MVP()));
			location = glGetUniformLocation(voxelize_shader, "proj_y");
			glUniformMatrix4fv(location, 1, false, glm::value_ptr(o_y.get_MVP()));
			location = glGetUniformLocation(voxelize_shader, "proj_z");
			glUniformMatrix4fv(location, 1, false, glm::value_ptr(o_z.get_MVP()));

			location = glGetUniformLocation(voxelize_shader, "grid_dim");
			glUniform1i(location, grid_resolution);
			location = glGetUniformLocation(voxelize_shader, "aabb_size");
			glUniform3fv(location, 1,glm::value_ptr(m_aabb.getHalfSize()) );

			glBindBufferBase(GL_ATOMIC_COUNTER_BUFFER, 0, m_frag_ctr_buffer);
			glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, m_position_ssbo);
			glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, m_base_color_ssbo);

			glBindFramebuffer(GL_FRAMEBUFFER, m_dummy_fbo);
			{
				glViewport(0, 0, grid_resolution, grid_resolution);
				glDisable(GL_CULL_FACE);
				glDisable(GL_DEPTH_TEST);
				glDisable(GL_MULTISAMPLE);

				if (glewIsExtensionSupported("GL_NV_conservative_raster")) glEnable(GL_CONSERVATIVE_RASTERIZATION_NV);
				{
					glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
					DrawFunc(voxelize_shader);
				}
				if (glewIsExtensionSupported("GL_NV_conservative_raster")) glDisable(GL_CONSERVATIVE_RASTERIZATION_NV);
			}
			glBindFramebuffer(GL_FRAMEBUFFER, 0);

			// Get number of written fragments.
			glBindBuffer(GL_ATOMIC_COUNTER_BUFFER, m_frag_ctr_buffer);
			glGetBufferSubData(GL_ATOMIC_COUNTER_BUFFER, 0, sizeof(GLuint), &m_frag_count);
			uint32_t zero_counter(0);
			glBufferSubData(GL_ATOMIC_COUNTER_BUFFER, 0, sizeof(GLuint), &zero_counter);
			glBindBuffer(GL_ATOMIC_COUNTER_BUFFER, 0);
		}
		glUseProgram(0);
		return m_frag_count;
	}
	explicit Voxelizer(int grid_size);
	~Voxelizer();

 private:
	Voxelizer();
};
}  // namespace voxelizer
