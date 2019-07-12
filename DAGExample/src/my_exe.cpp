#include <algorithm>
#include <chrono>
#include <iostream>
#include <fstream>
#include <numeric>
#include <string>
#include <stack>

#include <GL/glew.h>
#include <SDL.h>
#include <glm/gtc/type_ptr.hpp>

#include "DAG/DAG.h"
#include "DAGTracer/DAGTracer.h"
#include "utils/view.h"

#include "glTFLoader/glTFLoader.h"
#include "voxelize_and_merge.h"

using glm::ivec2;
using glm::vec2;
using glm::vec3;
using glm::vec4;

SDL_Window *mainWindow;
SDL_GLContext mainContext;
ivec2 screen_dim{1024, 1024};
GLuint copy_shader{0};
GLint renderbuffer_uniform{-1};

GLuint compile_shader(GLenum shader_type, const std::string &src) {
	GLuint shader_id     = glCreateShader(shader_type);
	const GLchar *source = static_cast<const GLchar *>(src.c_str());
	glShaderSource(shader_id, 1, &source, 0);
	glCompileShader(shader_id);

	GLint compile_status = GL_FALSE;
	glGetShaderiv(shader_id, GL_COMPILE_STATUS, &compile_status);

	if (compile_status != GL_TRUE) { throw std::runtime_error("Failed to compile shaders"); }

	return shader_id;
}

GLuint link_shaders(GLuint vs, GLuint fs) {
	GLuint program_id = glCreateProgram();
	glAttachShader(program_id, vs);
	glAttachShader(program_id, fs);
	glLinkProgram(program_id);

	GLint linkStatus = GL_FALSE;
	glGetProgramiv(program_id, GL_LINK_STATUS, &linkStatus);

	if (linkStatus != GL_TRUE) { throw std::runtime_error("Failed to link shaders"); }

	return program_id;
}

void init() {		
	mainWindow = SDL_CreateWindow(
		"DAG",
		SDL_WINDOWPOS_CENTERED,
		SDL_WINDOWPOS_CENTERED,
		screen_dim.x,
		screen_dim.y,
		SDL_WINDOW_OPENGL
	);

	mainContext = SDL_GL_CreateContext(mainWindow);

	SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);
	SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 4);
	SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 5);
	SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);
	SDL_GL_SetSwapInterval(1);

	int value = 0;
	SDL_GL_GetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, &value);
	std::cout << "SDL_GL_CONTEXT_MAJOR_VERSION : " << value << std::endl;

	SDL_GL_GetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, &value);
	std::cout << "SDL_GL_CONTEXT_MINOR_VERSION: " << value << std::endl;

	glewInit();

	std::string copy_vertex_shader =
			"#version 400 compatibility                                           \n"
			"out vec2 texcoord;                                                   \n"
			"void main() {                                                        \n"
			"   if(gl_VertexID == 0){ texcoord = vec2(0.0, 2.0); }                \n"
			"   if(gl_VertexID == 1){ texcoord = vec2(0.0, 0.0); }                \n"
			"   if(gl_VertexID == 2){ texcoord = vec2(2.0, 0.0); }                \n"
			"   if(gl_VertexID == 0){ gl_Position = vec4(-1.0,  3.0, 0.0, 1.0); } \n"
			"   if(gl_VertexID == 1){ gl_Position = vec4(-1.0, -1.0, 0.0, 1.0); } \n"
			"   if(gl_VertexID == 2){ gl_Position = vec4( 3.0, -1.0, 0.0, 1.0); } \n"
			"}                                                                    \n";

	std::string copy_fragment_shader =
			"#version 400 compatibility                                  \n"
			"in vec2 texcoord;                                           \n"
			"uniform sampler2D renderbuffer;                             \n"
			"void main() {                                               \n"
			"    gl_FragColor.xyz = texture(renderbuffer, texcoord).xyz; \n"
			"}                                                           \n";

	GLuint vs            = compile_shader(GL_VERTEX_SHADER, copy_vertex_shader);
	GLuint fs            = compile_shader(GL_FRAGMENT_SHADER, copy_fragment_shader);
	copy_shader          = link_shaders(vs, fs);
	renderbuffer_uniform = glGetUniformLocation(copy_shader, "renderbuffer");
}

class Timer {
	using clock_t        = std::chrono::high_resolution_clock;
	using timepoint_t    = std::chrono::time_point<clock_t>;
	using seconds_ft     = std::chrono::duration<float>;
	using milliseconds_t = std::chrono::milliseconds;
	using nanoseconds_t  = std::chrono::nanoseconds;
	timepoint_t start_tp;
	timepoint_t end_tp;
	nanoseconds_t diff_ns;

 public:
	Timer() : start_tp(clock_t::now()), end_tp(clock_t::now()), diff_ns(0) {}
	float dt_seconds() const { return std::chrono::duration_cast<seconds_ft>(diff_ns).count(); }
	void start() { start_tp = clock_t::now(); }
	void end() {
		end_tp  = clock_t::now();
		diff_ns = end_tp - start_tp;
	}
	void reset() {
		start_tp = clock_t::now();
		end_tp   = clock_t::now();
		diff_ns  = nanoseconds_t{0};
	}
};

struct AppState {
	chag::view camera;
	ivec2 old_mouse{0, 0};
	ivec2 new_mouse{0, 0};
	bool loop = true;
	Timer frame_timer;

	void handle_events() {
		const float dt = frame_timer.dt_seconds();
		SDL_Event event;
		while (SDL_PollEvent(&event)) {
			if (event.type == SDL_QUIT) {
				loop = false;
				break;
			}
		}

		const float move_scale_factor{1000.0f * dt};
		const Uint8 *key_state = SDL_GetKeyboardState(NULL);
		if (key_state[SDL_SCANCODE_ESCAPE]) { loop = false; }
		if (key_state[SDL_SCANCODE_W]) { camera.pos -= move_scale_factor * camera.R[2]; }
		if (key_state[SDL_SCANCODE_S]) { camera.pos += move_scale_factor * camera.R[2]; }
		if (key_state[SDL_SCANCODE_D]) { camera.pos += move_scale_factor * camera.R[0]; }
		if (key_state[SDL_SCANCODE_A]) { camera.pos -= move_scale_factor * camera.R[0]; }
		if (key_state[SDL_SCANCODE_E]) { camera.pos += move_scale_factor * camera.R[1]; }
		if (key_state[SDL_SCANCODE_Q]) { camera.pos -= move_scale_factor * camera.R[1]; }

		Uint32 mouse_state = SDL_GetMouseState(&(new_mouse.x), &(new_mouse.y));
		constexpr Uint32 lmb{1 << 0};
		constexpr Uint32 mmb{1 << 1};
		constexpr Uint32 rmb{1 << 2};
		const bool ld = (lmb & mouse_state) != 0;
		const bool md = (mmb & mouse_state) != 0;
		const bool rd = (rmb & mouse_state) != 0;

		vec2 delta = (vec2(new_mouse) - vec2(old_mouse));

		const float mouse_scale_factor{0.25f * dt};
		if (ld && rd) {
			camera.pos += delta.y * mouse_scale_factor * camera.R[2];
		} else if (ld) {
			camera.pitch(-delta.y * mouse_scale_factor);
			camera.yaw(-delta.x * mouse_scale_factor);
		} else if (rd) {
			camera.roll(-delta.x * mouse_scale_factor);
		} else if (md) {
			camera.pos -= delta.y * mouse_scale_factor * camera.R[1];
			camera.pos += delta.x * mouse_scale_factor * camera.R[0];
		}
		old_mouse = new_mouse;
	}
};

int main(int argc, char *argv[]) {
	init();

	constexpr int dag_resolution{1 << 13};
	auto dag = DAG_from_scene(dag_resolution, R"(assets\EpicCitadel\glTF\)", "EpicCitadel.gltf");
	if (dag)
	{
		struct MyHeader
		{
			float MinX;
			float MinY;
			float MinZ;
			float MaxX;
			float MaxY;
			float MaxZ;
			uint32_t levels;
			uint32_t topLevels;
			uint32_t numNodes;
			uint32_t numEnclosedLeaves;
			uint32_t numColors;
		};
		auto nodes = upload_to_gpu(*dag);
		MyHeader myHeader{
			dag->m_aabb.min.r, dag->m_aabb.min.g, dag->m_aabb.min.b,
			dag->m_aabb.max.r, dag->m_aabb.max.g, dag->m_aabb.max.b,
			dag->m_levels,
			dag->m_top_levels,
			nodes.size(),
			dag->m_enclosed_leaves.size(),
			dag->m_base_colors.size()
		};
		std::ofstream of("result.bin", std::ios::binary);
		of.write((char*)&myHeader, sizeof(MyHeader));
		of.write((char*)nodes.data(), myHeader.numNodes * sizeof(uint32_t));
		of.write((char*)dag->m_enclosed_leaves.data(), myHeader.numEnclosedLeaves * sizeof(uint32_t));
		of.write((char*)dag->m_base_colors.data(), myHeader.numColors * sizeof(uint32_t));
		of.close();
	}
	//dag->calculateColorForAllNodes();
	DAGTracer dag_tracer;
	dag_tracer.resize(screen_dim.x, screen_dim.y);

	if(dag)
	{
		upload_to_gpu(*dag);
		AppState app;
		app.camera.lookAt(vec3{0.0f, 1.0f, 0.0f}, vec3{0.0f, 0.0f, 0.0f});
		while (app.loop) {
			app.frame_timer.start();
			app.handle_events();

			const int color_lookup_lvl = dag->nofGeometryLevels();
			dag_tracer.resolve_paths(*dag, app.camera, color_lookup_lvl);
			dag_tracer.resolve_colors(*dag, color_lookup_lvl);

			glUseProgram(copy_shader);
			glActiveTexture(GL_TEXTURE0);
			glBindTexture(GL_TEXTURE_2D, dag_tracer.m_color_buffer.m_gl_idx);
			glUniform1i(renderbuffer_uniform, 0);
			glActiveTexture(GL_TEXTURE0);
			glDrawArrays(GL_TRIANGLES, 0, 3);

			SDL_GL_SwapWindow(mainWindow);
			app.frame_timer.end();
		}
	}

	SDL_GL_DeleteContext(mainContext);
	SDL_DestroyWindow(mainWindow);
	SDL_Quit();
	return 0;
}

//// Load from file
//#include "DAGLoader/DAGLoader.h"
//#include "DAGConstructor/DAGConstructor.h"
//#include "morton.h"
//...
//constexpr int GRID_RESOLUTION = 512;
//constexpr float GRID_RESOLUTION_FLOAT = static_cast<float>(GRID_RESOLUTION);
//...
//bool load_entire_dag{false};
//dag::DAG dag;
//if (load_entire_dag) 
//{
//	dag          = dag::cerealization::bin::load("../../cache/dag.bin");
//	dag.m_colors = dag::cerealization::bin::load_vec<uint32_t>("../../cache/colors.raw.bin");
//}
//else
//{
//	auto points = dag::cerealization::bin::load_vec<glm::vec3>("../../cache/positions");
//	auto colors = dag::cerealization::bin::load_vec<float>("../../cache/colors");
//
//	auto make_square_aabb = [](chag::Aabb aabb) {
//		const glm::vec3 hsz    = aabb.getHalfSize();
//		const glm::vec3 centre = aabb.getCentre();
//		const glm::vec3 c{glm::max(hsz.x, glm::max(hsz.y, hsz.z))};
//		aabb.min = centre - c;
//		aabb.max = centre + c;
//		return aabb;
//	};
//
//	chag::Aabb aabb = std::accumulate(
//			begin(points), end(points),
//			chag::make_aabb(vec3{std::numeric_limits<float>::max()}, vec3{std::numeric_limits<float>::lowest()}),
//			[](const chag::Aabb &lhs, const vec3 &rhs) {
//				chag::Aabb result;
//				result.min.x = std::min(lhs.min.x, rhs.x);
//				result.min.y = std::min(lhs.min.y, rhs.y);
//				result.min.z = std::min(lhs.min.z, rhs.z);
//
//				result.max.x = std::max(lhs.max.x, rhs.x);
//				result.max.y = std::max(lhs.max.y, rhs.y);
//				result.max.z = std::max(lhs.max.z, rhs.z);
//				return result;
//			});
//
//	chag::Aabb square_aabb = make_square_aabb(aabb);
//
//	std::vector<uint32_t> morton(points.size());
//	std::transform(begin(points), end(points), begin(morton), [square_aabb](const vec3 &pos) {
//					// First make sure the positions are in the range [0, GRID_RESOLUTION-1].
//		const vec3 corrected_pos = clamp(
//			GRID_RESOLUTION_FLOAT * ((pos - square_aabb.min) / (square_aabb.max - square_aabb.min)), 
//			vec3(0.0f), 
//			vec3(GRID_RESOLUTION_FLOAT - 1.0f)
//		);
//		return morton_encode_32(
//			static_cast<uint32_t>(corrected_pos.x),
//			static_cast<uint32_t>(corrected_pos.y),
//			static_cast<uint32_t>(corrected_pos.z)
//		);
//	});
//
//	// Need to make sure colors and morton key and colors are sorted according to morton.
//	{
//		struct sort_elem 
//		{
//			uint32_t morton;
//			vec4 color;
//		};
//		std::vector<sort_elem> sortme(morton.size());
//		for(size_t i{0}; i<sortme.size(); ++i)
//		{
//			sortme[i].morton = morton[i];
//			sortme[i].color  = vec4{ colors[4 * i + 0], colors[4 * i + 1], colors[4 * i + 2], colors[4 * i + 3] };
//		}
//		std::sort(begin(sortme), end(sortme), [](const sort_elem &lhs, const sort_elem &rhs){ return lhs.morton < rhs.morton; });
//		for (size_t i{0}; i < sortme.size(); ++i) 
//		{
//			morton[i] = sortme[i].morton;
//			colors[4 * i + 0] = sortme[i].color.x;
//			colors[4 * i + 1] = sortme[i].color.y;
//			colors[4 * i + 2] = sortme[i].color.z;
//			colors[4 * i + 3] = sortme[i].color.w;
//		}
//	}
//	DAGConstructor tmp;
//	// The log2(GRID_RESOLUTION / 4) + 0 is because we use 4x4x4 leafs instead of 2x2x2. 
//	// The final + 0 is a placeholder for when we need to merge dags.
//	dag = tmp.build_dag(morton, colors, static_cast<int>(morton.size()), static_cast<int>(log2(GRID_RESOLUTION / 4) + 0), square_aabb);
//}
