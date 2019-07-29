#include "glTFLoader.h"
#include "nlohmann/json.hpp"
#define STB_IMAGE_IMPLEMENTATION
#include <glm/gtc/type_ptr.hpp>
#include <stb/stb_image.h>

#include <fstream>
#include <iostream>
#include <stack>

using json = nlohmann::json;

namespace glTFLoader {

GLuint MASTER_VAO{0u};

// FIXME: NO GLOBALS!
json root;

struct SafeFree {
	void operator()(stbi_uc *data) const {
		if (data) free(data);
	}
};

GLuint readRGBA8888(const std::string &filename) {
	int width, height, components;
	stbi_set_flip_vertically_on_load(false);
	std::unique_ptr<stbi_uc, SafeFree> data(stbi_load(filename.c_str(), &width, &height, &components, 4));

	if (data) {
		GLuint tex;
		glGenTextures(1, &tex);
		glBindTexture(GL_TEXTURE_2D, tex);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, data.get());
		glGenerateMipmap(GL_TEXTURE_2D);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		return tex;
	}
	return 0;
	// throw std::exception(std::string("Failed to load texture: ").append(filename).c_str());
	// FIXME: Just log error?...
}

GLuint uploadBin(const std::string &file, GLsizeiptr count) {
	// If count is zero, figure out file size.
	std::ifstream binfile(file, std::ios::binary | ((count == 0) ? std::ios::ate : 0));
	if (count == 0) {
		GLsizeiptr count = binfile.tellg();
		binfile.seekg(0);
	}

	GLuint vboID;
	if(MASTER_VAO == 0u){
		glGenVertexArrays(1, &MASTER_VAO);
	}
	glBindVertexArray(MASTER_VAO);

	glGenBuffers(1, &vboID);
	glBindBuffer(GL_ARRAY_BUFFER, vboID);
	glBufferData(GL_ARRAY_BUFFER, count, nullptr, GL_STATIC_DRAW);
	void *tmp = glMapBuffer(GL_ARRAY_BUFFER, GL_WRITE_ONLY);
	binfile.read(reinterpret_cast<char *>(tmp), count);
	glUnmapBuffer(GL_ARRAY_BUFFER);
	return vboID;

	// NOTE: Don't really need persistent mapped vbo.
	// glBindBuffer(GL_ARRAY_BUFFER, vboID);
	// GLbitfield flags = GL_MAP_WRITE_BIT | GL_MAP_PERSISTENT_BIT | GL_MAP_COHERENT_BIT;
	// glBufferStorage(GL_ARRAY_BUFFER, count, 0, flags);
	// void *myPointer = glMapBufferRange(GL_ARRAY_BUFFER, 0, count, flags);
	// binfile.read(reinterpret_cast<char *>(myPointer), count);
	// return vboID;
}

template <typename T>
std::optional<T> getOptionalField(const json &j, const std::string &field) {
	auto it = j.find(field);
	if (it != j.end()) return {*it};
	return {};
}

// clang-format off
Attribute::Type getAttributeType(const std::string &typestring) {
				 if(typestring == "POSITION"  ) return Attribute::Type::POSITION;
		else if(typestring == "NORMAL"    ) return Attribute::Type::NORMAL;
		else if(typestring == "TANGENT"   ) return Attribute::Type::TANGENT;
		else if(typestring == "TEXCOORD_0") return Attribute::Type::TEXCOORD_0;
		else if(typestring == "TEXCOORD_1") return Attribute::Type::TEXCOORD_1;
		else /*Log error*/

		return Attribute::Type::ERROR_TYPE;
}
Accessor::Type getAccessorType(const std::string &typestring) {
				 if(typestring == "SCALAR") return Accessor::Type::SCALAR;
		else if(typestring == "VEC2"  ) return Accessor::Type::VEC2;
		else if(typestring == "VEC3"  ) return Accessor::Type::VEC3;
		else if(typestring == "VEC4"  ) return Accessor::Type::VEC4;
		else if(typestring == "MAT2"  ) return Accessor::Type::MAT2;
		else if(typestring == "MAT3"  ) return Accessor::Type::MAT3;
		else if(typestring == "MAT4"  ) return Accessor::Type::MAT4;
		else /*Log error*/

		return Accessor::Type::ERROR_TYPE;
}
GLint getTypeSize(const Accessor::Type type) {
	switch (type)
	{
		case Accessor::Type::SCALAR : return 1;
		case Accessor::Type::VEC2   : return 2;
		case Accessor::Type::VEC3   : return 3;
		case Accessor::Type::VEC4   : return 4;
		case Accessor::Type::MAT2   : return 4;
		case Accessor::Type::MAT3   : return 9;
		case Accessor::Type::MAT4   : return 16;
		case Accessor::Type::ERROR_TYPE:
		default: /*Log error*/
			break;
	}
	return -1;
}
// clang-format on

void from_json(const json &j, BufferView &bv) {
	j.at("buffer").get_to(bv.buffer);
	j.at("byteLength").get_to(bv.byteLength);
	bv.byteOffset = getOptionalField<GLuint>(j, "byteOffset").value_or(0);
	bv.target     = getOptionalField<GLuint>(j, "target");
}

void from_json(const json &j, Accessor &ac) {
	j.at("bufferView").get_to(ac.bufferView);
	j.at("componentType").get_to(ac.componentType);
	j.at("count").get_to(ac.count);
	ac.type       = getAccessorType(j.at("type"));
	ac.typeSize   = getTypeSize(ac.type);
	ac.byteOffset = getOptionalField<GLuint>(j, "byteOffset").value_or(0);
}

void from_json(const json &j, Primitive &p) {
	for (auto it = j["attributes"].begin(); it != j["attributes"].end(); ++it) {
		p.attributes.emplace_back(Attribute{getAttributeType(it.key()), it.value()});
	}
	j.at("indices").get_to(p.indices);
	j.at("material").get_to(p.material);
}

void from_json(const json &j, Mesh &m) {
	m.name = getOptionalField<std::string>(j, "name").value_or("Unnamed");
	j.at("primitives").get_to(m.primitives);
}

void from_json(const json &j, SceneNode &sn) {
	sn.name = getOptionalField<std::string>(j, "name").value_or("Unnamed");

	if (auto it = j.find("mesh"); it != j.end()) {
		it->get_to(sn.mesh_id);
		sn.properties |= SceneNode::Flag::MESH;
	}

	if (auto it = j.find("children"); it != j.end()) {
		it->get_to(sn.children);
		sn.properties |= SceneNode::Flag::CHILDREN;
	}

	if (auto it = j.find("translation"); it != j.end()) {
		sn.translation = glm::vec3{(*it)[0], (*it)[1], (*it)[2]};
		sn.properties |= SceneNode::Flag::TRANSLATION;
	}

	if (auto it = j.find("rotation"); it != j.end()) {
		// NOTE: Constructor for glm::quat is quat(w,x,y,z), whereas the glTF quaternion is stored as xyzw.
		sn.rotation = glm::quat{(*it)[3], (*it)[0], (*it)[1], (*it)[2]};
		sn.properties |= SceneNode::Flag::ROTATION;
	}

	if (auto it = j.find("scale"); it != j.end()) {
		sn.scale = glm::vec3{(*it)[0], (*it)[1], (*it)[2]};
		sn.properties |= SceneNode::Flag::SCALE;
	}

	if (auto it = j.find("matrix"); it != j.end()) {
		std::vector<float> v = *it;
		sn.matrix = glm::make_mat4(v.data());
		sn.properties |= SceneNode::Flag::MATRIX;
	}
}

GLuint getTextureId(const json &material, const std::string &name) {
	if (auto it = material.find(name); it != material.end()) {
		const unsigned source_id = (*it)["index"];
		const json &texture      = root["textures"][source_id];
		/* Sampler etc... */
		return texture["source"];
	}
	return NO_ID_SENTINEL;
}

Material::AlphaMode getAlphaMode(const std::optional<std::string> &mode) {
	if (mode) {
		if (mode == "OPAQUE") return Material::AlphaMode::ALPHA_OPAQUE;
		if (mode == "MASK") return Material::AlphaMode::ALPHA_OPAQUE;
		if (mode == "BLEND") return Material::AlphaMode::ALPHA_BLEND;
	}
	return Material::AlphaMode::ALPHA_OPAQUE;
}

void from_json(const json &j, Material &m) {
	//j.at("name").get_to(m.name);
	// FIXME: Move optional part into getAlphaMode
	m.alphaMode = getAlphaMode(getOptionalField<std::string>(j, "alphaMode"));

	// TODO: A glTF material can have both MetallicRoughness and SpecularGlossiness defined.
	//       Add support for both. Now we default to SpecularGlossiness.
	if (auto ext = j.find("extensions"); ext != j.end()) {
		if (auto pbr = ext->find("KHR_materials_pbrSpecularGlossiness"); pbr != ext->end()) {
			m.type                                          = Material::Type::SPECULAR_GLOSSINESS;
			m.specular_glossiness.diffuseTexture            = getTextureId(*pbr, "diffuseTexture");
			m.specular_glossiness.specularGlossinessTexture = getTextureId(*pbr, "specularGlossinessTexture");
		} else if (auto pbr = j.find("pbrMetallicRoughness"); pbr != j.end()) {
			m.type                                        = Material::Type::METALLIC_ROUGHNESS;
			m.metallic_roughness.baseColorTexture         = getTextureId(*pbr, "baseColorTexture");
			m.metallic_roughness.metallicRoughnessTexture = getTextureId(*pbr, "metallicRoughnessTexture");
			auto it = pbr->find("baseColorFactor");
			if(it != pbr->end()){
				m.metallic_roughness.baseColorFactor[0] = (*it)[0];
				m.metallic_roughness.baseColorFactor[1] = (*it)[1];
				m.metallic_roughness.baseColorFactor[2] = (*it)[2];
			}
		}
	} else if (auto pbr = j.find("pbrMetallicRoughness"); pbr != j.end()) {
		m.type                                        = Material::Type::METALLIC_ROUGHNESS;
		m.metallic_roughness.baseColorTexture         = getTextureId(*pbr, "baseColorTexture");
		m.metallic_roughness.metallicRoughnessTexture = getTextureId(*pbr, "metallicRoughnessTexture");
		auto it = pbr->find("baseColorFactor");
		if(it != pbr->end()){
			m.metallic_roughness.baseColorFactor[0] = (*it)[0];
			m.metallic_roughness.baseColorFactor[1] = (*it)[1];
			m.metallic_roughness.baseColorFactor[2] = (*it)[2];
		}
	}
	m.normalTexture    = getTextureId(j, "normalTexture");
	m.occlusionTexture = getTextureId(j, "occlusionTexture");
	m.bakedTexture     = getTextureId(j, "bakedTexture");
}

Scene load(const std::string &path, const std::string &file) {
	Scene result;
	std::ifstream myfile(path + file);

	// json root;
	myfile >> root;

	root["bufferViews"].get_to(result.bufferViews);
	root["accessors"].get_to(result.accsessors);
	root["meshes"].get_to(result.meshes);
	root["nodes"].get_to(result.scene_nodes);
	root["materials"].get_to(result.materials);
	unsigned scene_id = root["scene"];
	const json &nodes = root["scenes"][scene_id]["nodes"];
	nodes.get_to(result.scene_nodes_roots);

	for (const auto &image : root["images"]) {
		const std::string filename = image["uri"];
		result.textures.emplace_back(readRGBA8888(path + filename));
	}

	for (auto &buffer : root["buffers"]) {
		std::string filename = path + std::string(buffer["uri"]);
		GLsizeiptr count     = getOptionalField<GLsizeiptr>(buffer, "byteLength").value_or(0);
		result.buffer_objects.push_back(uploadBin(filename, count));
	}

	// Create a topologial order so we can create an aabb hierarchy bottom up.
	std::vector<std::size_t> topological_order;
	for (std::size_t root : result.scene_nodes_roots) { topological_order.emplace_back(root); }
	// We can do it as below because:
	// "For Version 2.0 conformance, the glTF node hierarchy is not a directed acyclic graph (DAG) or scene graph, but a
	// disjoint union of strict trees. That is, no node may be a direct descendant of more than one node. This
	// restriction is meant to simplify implementation and facilitate conformance." - glTF specification
	for (std::size_t head{0}; head < topological_order.size(); ++head) {
		const auto &children = result.scene_nodes[topological_order[head]].children;
		if (children.size() == 0) continue;
		for (const std::size_t child : children) { topological_order.emplace_back(child); }
	}
	///////////////// NOTE ///////////////////////////////////////////
	// Can be done on parsing of primitives.                        //
	//////////////////////////////////////////////////////////////////
	auto getAABB_ac = [&result](Accessor &ac, Accessor &indices) {
		if(ac.type != Accessor::Type::VEC3) throw "Invalid type";
		const auto  &bv = result.bufferViews[ac.bufferView];
		std::size_t offset = ac.byteOffset + bv.byteOffset;

		const auto  &ibv = result.bufferViews[indices.bufferView];
		std::size_t ioffset = indices.byteOffset + ibv.byteOffset;
		//if(indices.componentType != GL_UNSIGNED_SHORT){
		//	throw "Ble";
		//}

		GLuint vboID = result.buffer_objects[bv.buffer];

		// NOTE: Yeah... I know it might have been better to not punish the
		//       poor PCI-bus. 
		glBindBuffer(GL_ARRAY_BUFFER, vboID);
		std::byte *b = reinterpret_cast<std::byte*>(glMapBuffer(GL_ARRAY_BUFFER, GL_READ_ONLY)) + offset;
		float *tmp = reinterpret_cast<float*>(b);
		//GLushort *idx = reinterpret_cast<GLushort *>(tmp);
		chag::Aabb aabb = chag::make_inverse_extreme_aabb();
		for(std::size_t i{0}; i<ac.count; ++i){
			//GLushort vtx_idx = *(idx+i);
			aabb = chag::combine(aabb, glm::vec3(tmp[0 + 3*i],tmp[1 + 3*i],tmp[2 + 3*i]));
		}
		glUnmapBuffer(GL_ARRAY_BUFFER);
		return aabb;
	};

	auto createAABB_prim = [&](Primitive &prim) {
		chag::Aabb aabb = chag::make_inverse_extreme_aabb();
		for (const auto &attr : prim.attributes) {
			if (attr.index == Attribute::POSITION) {
				 aabb = chag::combine(aabb, getAABB_ac(result.accsessors[attr.accsessor], result.accsessors[prim.indices]));
			}
		}
		prim.aabb = aabb;
	};
	
	auto createAABB_node = [&](SceneNode &node) {
		chag::Aabb aabb = chag::make_inverse_extreme_aabb();
		// Construct AABB containing all primitive AABB.
		if (SceneNode::hasProperties(node, SceneNode::MESH)) {
			Mesh &mesh = result.meshes[node.mesh_id];
			for (auto &prim : mesh.primitives) {
				createAABB_prim(prim);
				aabb = chag::combine(aabb, prim.aabb);
			}
		}
		// Combine with childrens AABB.
		for (auto &child : node.children) { aabb = chag::combine(aabb, result.scene_nodes[child].aabb); }
		node.aabb = aabb;
	};
	std::reverse(topological_order.begin(), topological_order.end());
	for(auto &node : topological_order){
		createAABB_node(result.scene_nodes[node]);
	}

	return result;
}
}  // namespace glTFLoader
