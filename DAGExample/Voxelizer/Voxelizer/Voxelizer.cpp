#include "Voxelizer.h"
#include <fstream>
#include <vector>

std::string
get_file_contents(const char* filename)
{
	std::ifstream in(filename, std::ios::in | std::ios::binary);
	if (in) {
		std::string contents;
		in.seekg(0, std::ios::end);
		contents.resize(in.tellg());
		in.seekg(0, std::ios::beg);
		in.read(&contents[0], contents.size());
		in.close();
		return (contents);
	}
	throw(errno);
}

// Hackety hack. Just to avoid tedious filepaths.
const std::string vert_src = 
#include "voxelize.vert"
;
const std::string frag_src = 
#include "voxelize.frag"
;
const std::string geom_src = 
#include "voxelize.geom"
;

GLuint
compile_shader(GLenum shader_type, const GLchar* source)
{
	GLuint shader_id = glCreateShader(shader_type);
	glShaderSource(shader_id, 1, &source, 0);
	glCompileShader(shader_id);

	GLint bufflen{ 0 };
	glGetShaderiv(shader_id, GL_INFO_LOG_LENGTH, &bufflen);
	if (bufflen > 1) {
		std::vector<GLchar> log_string(static_cast<std::size_t>(bufflen) + 1ll);
		glGetShaderInfoLog(shader_id, bufflen, 0, log_string.data());
		printf("Log found for '%s.':\n%s", source, log_string.data());
	}

	GLint compile_status = GL_FALSE;
	glGetShaderiv(shader_id, GL_COMPILE_STATUS, &compile_status);

	if (compile_status != GL_TRUE) {
		throw std::runtime_error("Failed to compile shaders");
	}

	return shader_id;
}

GLuint
link_shaders(GLuint vs, GLuint fs, GLuint gs)
{
	GLuint program_id = glCreateProgram();
	glAttachShader(program_id, vs);
	glAttachShader(program_id, fs);
	glAttachShader(program_id, gs);

	glProgramParameteriEXT(program_id, GL_GEOMETRY_INPUT_TYPE, GL_TRIANGLES);
	glProgramParameteriEXT(program_id, GL_GEOMETRY_OUTPUT_TYPE, GL_TRIANGLE_STRIP);
	glProgramParameteriEXT(program_id, GL_GEOMETRY_VERTICES_OUT, 3);
	
	glLinkProgram(program_id);

	GLint linkStatus = GL_FALSE;
	glGetProgramiv(program_id, GL_LINK_STATUS, &linkStatus);

	if (linkStatus != GL_TRUE) {
		throw std::runtime_error("Failed to link shaders");
	}

	return program_id;
}

namespace voxelizer {

Voxelizer::Voxelizer() {}

Voxelizer::Voxelizer(int grid_size) : m_grid_size(grid_size) {
	assert(m_grid_size > 0 /* && m_grid_size < 1025*/);
	GLuint vert = compile_shader(GL_VERTEX_SHADER, vert_src.c_str());
	GLuint frag = compile_shader(GL_FRAGMENT_SHADER, frag_src.c_str());
	GLuint geom = compile_shader(GL_GEOMETRY_SHADER, geom_src.c_str());
	voxelize_shader = link_shaders(vert, frag, geom);

	{
		// Atomic counter.
		glGenBuffers(1, &m_frag_ctr_buffer);
		glBindBuffer(GL_ATOMIC_COUNTER_BUFFER, m_frag_ctr_buffer);
		glBufferData(GL_ATOMIC_COUNTER_BUFFER, sizeof(GLuint), NULL, GL_DYNAMIC_DRAW);
		GLuint just_zero(0);
		glBufferSubData(GL_ATOMIC_COUNTER_BUFFER, 0, sizeof(GLuint), &just_zero);
		glBindBuffer(GL_ATOMIC_COUNTER_BUFFER, 0);

		// Data buffer (pos).
		glGenBuffers(1, &m_position_ssbo);
		glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_position_ssbo);
		glBufferData(GL_SHADER_STORAGE_BUFFER, m_tex_dim * sizeof(uint32_t), NULL, GL_STATIC_DRAW);
		glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

		// Data buffer (base color)
		glGenBuffers(1, &m_base_color_ssbo);
		glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_base_color_ssbo);
		glBufferData(GL_SHADER_STORAGE_BUFFER, m_tex_dim * sizeof(uint32_t), NULL, GL_STATIC_DRAW);
		glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

		// Dummy framebuffer.
		glGenFramebuffers(1, &m_dummy_fbo);
		glNamedFramebufferParameteri(m_dummy_fbo, GL_FRAMEBUFFER_DEFAULT_WIDTH, m_grid_size);
		glNamedFramebufferParameteri(m_dummy_fbo, GL_FRAMEBUFFER_DEFAULT_HEIGHT, m_grid_size);
		glBindFramebuffer(GL_FRAMEBUFFER, m_dummy_fbo);
		glBindFramebuffer(GL_FRAMEBUFFER, 0);
	}
}

Voxelizer::~Voxelizer() {
	glDeleteBuffers(1, &m_frag_ctr_buffer);
	glDeleteBuffers(1, &m_position_ssbo);
	glDeleteBuffers(1, &m_base_color_ssbo);
	glDeleteBuffers(1, &m_dummy_fbo);
}
}  // namespace voxelizer
