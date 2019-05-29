R""(
#version 450 compatibility
layout(location = 0) in vec3 a_Position;
layout(location = 1) in vec3 a_Normal;
layout(location = 2) in vec4 a_Tangent;
layout(location = 3) in vec2 a_UV;
layout(location = 4) in vec2 a_UV2;
out vec2 uv_VS;
out vec3 gs_position;
out vec3 gs_normal;
uniform mat4 modelMatrix;
void main()
{
	vec3 world_space_position = (modelMatrix * vec4(a_Position, 1.0)).xyz;
	gl_Position = vec4(world_space_position, 1.0);
	gs_position = world_space_position; 
	gs_normal = (modelMatrix * vec4(a_Normal, 0.0)).xyz;
	uv_VS = a_UV;
}
)""
