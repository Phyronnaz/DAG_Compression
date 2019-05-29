#pragma once
#include <GL/glew.h>
#include <glm/gtc/quaternion.hpp>
#include <glm/vec3.hpp>
#include <optional>
#include <string>
#include <vector>
#include <utils/Aabb.h>

//#define FOO(A)                         \
//	A(const A &) = default;            \
//	A(A &&)      = default;            \
//	A &operator=(const A &) = default; \
//	A &operator=(A &&) = default;

namespace glTFLoader {

	GLuint readRGBA8888(const std::string &filename);

	extern GLuint MASTER_VAO;

constexpr GLuint NO_ID_SENTINEL = 0xFFFFFFFF;

struct BufferView {
	GLuint buffer = 0;
	GLsizei byteLength;
	GLsizeiptr byteOffset;
	std::optional<GLbitfield> target;
};

struct Accessor {
	enum class Type {
		SCALAR,
		VEC2,
		VEC3,
		VEC4,
		MAT2,
		MAT3,
		MAT4,
		ERROR_TYPE,
	};
	std::size_t bufferView;
	GLbitfield componentType;
	GLsizei count;
	GLsizeiptr byteOffset;
	Type type;
	GLint typeSize;
};

struct Attribute {
	enum Type : GLint {
		POSITION   = 0,
		NORMAL     = 1,
		TANGENT    = 2,
		TEXCOORD_0 = 3,
		TEXCOORD_1 = 4,
		ERROR_TYPE = -1,
	};
	Type index;
	std::size_t accsessor;
};

struct Primitive {
	std::vector<Attribute> attributes;
	std::size_t material;
	std::size_t indices;
	chag::Aabb aabb;
};

struct Mesh {
	std::vector<Primitive> primitives;
	std::string name;
};

struct Material {
	enum class AlphaMode { ALPHA_OPAQUE, ALPHA_MASK, ALPHA_BLEND };
	enum class Type { METALLIC_ROUGHNESS, SPECULAR_GLOSSINESS };
	union {
		struct pbrMetallicRoughness {
			GLuint baseColorTexture;
			GLuint metallicRoughnessTexture;
			float baseColorFactor[3];
		} metallic_roughness;
		struct pbrSpecularGlossiness {
			GLuint diffuseTexture;
			GLuint specularGlossinessTexture;
			float diffuseFactor[3];
		} specular_glossiness;
	};
	GLuint normalTexture;
	GLuint occlusionTexture;
	GLuint bakedTexture;
	AlphaMode alphaMode;
	Type type;
	std::string name;
};

struct SceneNode {
	using FlagType = std::uint16_t;
	enum Flag : FlagType {
		MESH        = 1 << 0,
		ROTATION    = 1 << 1,
		TRANSLATION = 1 << 2,
		SCALE       = 1 << 3,
		MATRIX      = 1 << 4,
		CHILDREN    = 1 << 5,
	};

	// c++ standard way
	// enum Flagmode{Flagmask = 0xff};
	// static constexpr Flagmode mesh  = (Flagmode)(1 << 0);
	// static constexpr Flagmode rot   = (Flagmode)(1 << 1);
	// static constexpr Flagmode trans = (Flagmode)(1 << 2);
	// static constexpr Flagmode scale = (Flagmode)(1 << 3);
	// static constexpr Flagmode mat   = (Flagmode)(1 << 4);
	// static constexpr Flagmode child = (Flagmode)(1 << 5);

	static bool hasProperties(const SceneNode &node, Flag flag) { return (node.properties & flag) != 0; }
	glm::quat rotation{1.0f, 0.0f, 0.0f, 0.0f};
	glm::vec3 translation{0.0f};
	glm::vec3 scale{1.0f};
	glm::mat4 matrix{1.0f};
	glm::mat4 cached_transform;
	std::size_t mesh_id;
	FlagType properties = 0;
	std::vector<std::size_t> children;
	std::string name;
	chag::Aabb aabb;
};

struct Scene {
	std::vector<Mesh> meshes;
	std::vector<BufferView> bufferViews;
	std::vector<Accessor> accsessors;
	std::vector<Material> materials;
	std::vector<GLuint> textures;
	std::vector<SceneNode> scene_nodes;
	std::vector<std::size_t> scene_nodes_roots;
	std::vector<GLuint> buffer_objects;
};

void inline free_scene(Scene &scene)
{
	glDeleteBuffers(scene.buffer_objects.size(), scene.buffer_objects.data());
	glDeleteVertexArrays(1, &MASTER_VAO);
	glDeleteTextures(scene.textures.size(), scene.textures.data());
}


	Scene load(const std::string &path, const std::string &file);
}  // namespace glTFLoader
