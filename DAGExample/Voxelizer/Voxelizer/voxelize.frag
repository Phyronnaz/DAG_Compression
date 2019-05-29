R""( 
#version 450 compatibility
#extension GL_ARB_shader_atomic_counters : enable
#extension GL_NV_gpu_shader5 : enable
#extension GL_ARB_gpu_shader_int64 : enable
uint32_t splitBy3_32(uint32_t x) {
	x = x               & 0x000003ff; // x = ---- ---- ---- ---- ---- --98 7654 3210
	x = (x | (x << 16)) & 0xff0000ff; // x = ---- --98 ---- ---- ---- ---- 7654 3210
	x = (x | (x <<  8)) & 0x0300f00f; // x = ---- --98 ---- ---- 7654 ---- ---- 3210
	x = (x | (x <<  4)) & 0x030c30c3; // x = ---- --98 ---- 76-- --54 ---- 32-- --10
	x = (x | (x <<  2)) & 0x09249249; // x = ---- 9--8 --7- -6-- 5--4 --3- -2-- 1--0
	return x;
};

uint32_t mortonEncode32(uint32_t x, uint32_t y, uint32_t z) {
	return splitBy3_32(x) << 2 | splitBy3_32(y) << 1 | splitBy3_32(z);
};

void swap(inout int a[5], int i, int j) {
	int tmp = a[i];
	a[i] = a[j];
	a[j] = tmp;
}

void sort(inout int a[5]) {
	// Sort first two pairs
	if(a[1] < a[0]) swap(a, 0, 1);
	if(a[3] < a[2]) swap(a, 2, 3);
	// Sort pairs by larger element
	if(a[3] < a[1]) {
	    swap(a, 0, 2);
	    swap(a, 1, 3);
	}
	// A = [a,b,c,d,e] with a < b < d and c < d
	// insert e into [a,b,d]
	int b[4];
	if(a[4] < a[1]){
	    if(a[4] < a[0]) {
	        b[0] = a[4];
	        b[1] = a[0];
	        b[2] = a[1];
	        b[3] = a[3];
	    } else {
	        b[0] = a[0];
	        b[1] = a[4];
	        b[2] = a[1];
	        b[3] = a[3];
	    }
	} else {
	    if(a[4] < a[3]){
	        b[0] = a[0];
	        b[1] = a[1];
	        b[2] = a[4];
	        b[3] = a[3];
	    } else {
	        b[0] = a[0];
	        b[1] = a[1];
	        b[2] = a[3];
	        b[3] = a[4];
	    }
	}
	// insert c into the first three elements of B
	if(a[2] < b[1]){
	    if(a[2] < b[0]){
	        a[0] = a[2];
	        a[1] = b[0];
	        a[2] = b[1];
	        a[3] = b[2];
	        a[4] = b[3];
	    } else {
	        a[1] = a[2];
	        a[0] = b[0];
	        a[2] = b[1];
	        a[3] = b[2];
	        a[4] = b[3];
	    }
	} else {
	    if(a[2] < b[2]){
	        a[2] = a[2];
	        a[0] = b[0];
	        a[1] = b[1];
	        a[3] = b[2];
	        a[4] = b[3];
	    } else {
	        a[3] = a[2];
	        a[0] = b[0];
	        a[1] = b[1];
	        a[2] = b[2];
	        a[4] = b[3];
	    }
	}
}

flat in int axis_id; 
in vec2 uv;

layout ( binding = 0 ) uniform atomic_uint frag_count;
layout(binding = 0, std430) restrict coherent buffer item_buffer_block0{ uint32_t position_ssbo[]; };
layout(binding = 1, std430) restrict coherent buffer item_buffer_block2{ uint32_t base_color_ssbo[]; };

uniform int grid_dim;
layout(binding = 0) uniform sampler2D u_BaseColorSampler;

void main() {
	///////////////////////////////////////////////////////////////////////
	// Fetch color (once per shader invocation)
	///////////////////////////////////////////////////////////////////////
	uvec4 base_color;
	base_color.rgb  = clamp(uvec3(round(255.0 * texture2D(u_BaseColorSampler, uv).rgb)), uvec3(0), uvec3(255));

	vec3 subvoxel_pos = vec3((gl_FragCoord.x), 
							(gl_FragCoord.y), 
							(gl_FragCoord.z * grid_dim)); 

	// Conservative in z
	float dzdx = 0.5*dFdxFine(gl_FragCoord.z) * grid_dim;
	float dzdy = 0.5*dFdyFine(gl_FragCoord.z) * grid_dim;
	int apa[5] = {
		clamp(int(subvoxel_pos.z              ), 0, grid_dim-1),
		clamp(int(subvoxel_pos.z + dzdx + dzdy), 0, grid_dim-1),
		clamp(int(subvoxel_pos.z + dzdx - dzdy), 0, grid_dim-1),
		clamp(int(subvoxel_pos.z - dzdx + dzdy), 0, grid_dim-1),
		clamp(int(subvoxel_pos.z - dzdx - dzdy), 0, grid_dim-1)
	};
	sort(apa);
	for(int i = 0; i<5; ++i){ 
		if(i == 0 || apa[i] != apa[i-1]){
			uvec3 subvoxel_coord2 = uvec3(clamp(uvec2(subvoxel_pos.xy), uvec2(0), uvec2(grid_dim-1)), uint(apa[i]));
			if      (axis_id == 1) { subvoxel_coord2.xyz = subvoxel_coord2.zyx; }
			else if (axis_id == 2) { subvoxel_coord2.xyz = subvoxel_coord2.xzy; }
			else if (axis_id == 3) { subvoxel_coord2.xyz = subvoxel_coord2.yxz; }
			uint32_t idx   = atomicCounterIncrement(frag_count);
			position_ssbo[idx] = mortonEncode32(subvoxel_coord2.x, subvoxel_coord2.y, subvoxel_coord2.z);
			base_color_ssbo[idx]   = (base_color.r  << 24) | (base_color.g  << 16) | (base_color.b  << 8) | (base_color.a  << 0);
		}
	}
}
)""
