R""(
#version 450 compatibility
layout ( triangles ) in;
layout ( triangle_strip, max_vertices = 3 ) out;

in vec2 uv_VS[];
in vec3 gs_position[];
in vec3 gs_normal[];
out vec2 uv;
out vec3 fs_position0;
out vec3 fs_position1;
out vec3 fs_position2;
out vec3 fs_normal;
out vec3 barycoords; 
flat out vec3 fs_gnormal; 
flat out int axis_id;

uniform mat4 proj_x;
uniform mat4 proj_y;
uniform mat4 proj_z;
uniform vec3 aabb_size; 

void main()
{
	vec3 faceNormal = normalize( cross( gl_in[1].gl_Position.xyz-gl_in[0].gl_Position.xyz,
										gl_in[2].gl_Position.xyz-gl_in[0].gl_Position.xyz));
	faceNormal *= aabb_size; 
	float NdX = abs( faceNormal.x );
	float NdY = abs( faceNormal.y );
	float NdZ = abs( faceNormal.z );
	mat4 proj;

	if(NdX >= NdZ){
		if(NdX >= NdY) {
			proj = proj_x;
			axis_id = 1;
			} else {
				proj = proj_y;
				axis_id = 2;
			}
	} else if(NdY >= NdZ) {
		proj = proj_y;
		axis_id = 2;
	} else {
		proj = proj_z;
		axis_id = 3;
	}

	fs_gnormal = faceNormal; 

	fs_position0 = gs_position[0];
	fs_position1 = gs_position[1];
	fs_position2 = gs_position[2];

	gl_Position = proj * gl_in[0].gl_Position;
	uv = uv_VS[0];
	fs_normal = gs_normal[0];
	EmitVertex();
	gl_Position = proj * gl_in[1].gl_Position;
	uv = uv_VS[1];
	fs_normal = gs_normal[1];
	EmitVertex();
	gl_Position = proj * gl_in[2].gl_Position;
	uv = uv_VS[2];
	fs_normal = gs_normal[2];
	EmitVertex();
}
)""
