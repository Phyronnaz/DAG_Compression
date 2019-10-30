#pragma once
#include <stdint.h>
#include <tuple>

inline
uint32_t split3_32(uint32_t x) {
	x =  x              & 0x000003ff;  // x = ---- ---- ---- ---- ---- --98 7654 3210
	x = (x | (x << 16)) & 0xff0000ff;  // x = ---- --98 ---- ---- ---- ---- 7654 3210
	x = (x | (x <<  8)) & 0x0300f00f;  // x = ---- --98 ---- ---- 7654 ---- ---- 3210
	x = (x | (x <<  4)) & 0x030c30c3;  // x = ---- --98 ---- 76-- --54 ---- 32-- --10
	x = (x | (x <<  2)) & 0x09249249;  // x = ---- 9--8 --7- -6-- 5--4 --3- -2-- 1--0
	return x;
}

inline 
uint32_t  morton_encode_32(uint32_t x, uint32_t y, uint32_t z) {
	return split3_32(x) << 2 | split3_32(y) << 1 | split3_32(z);
}

inline
uint32_t compact3_32(uint32_t x) {
	x      =  x              & 0x09249249;  // x = ---- 9--8 --7- -6-- 5--4 --3- -2-- 1--0
	x      = (x | (x >> 2))  & 0x030c30c3;  // x = ---- --98 ---- 76-- --54 ---- 32-- --10
	x      = (x | (x >> 4))  & 0x0300f00f;  // x = ---- --98 ---- ---- 7654 ---- ---- 3210
	x      = (x | (x >> 8))  & 0xff0000ff;  // x = ---- --98 ---- ---- ---- ---- 7654 3210
	x      = (x | (x >> 16)) & 0x000003ff;  // x = ---- ---- ---- ---- ---- --98 7654 3210
	return x;
};

template<typename T>
inline
T morton_decode_32(uint32_t x) {
	return T{compact3_32(x >> 2), compact3_32(x >> 1), compact3_32(x)};
};

inline uint64_t splitBy3_64(uint32_t a) {
	uint64_t x = a    & 0x00000000001ffffful; // we only look at the first 21 bits
	x = (x | x << 32) & 0x001f00000000fffful; // shift left 32 bits, OR with self, and 00011111000000000000000000000000000000001111111111111111
	x = (x | x << 16) & 0x001f0000ff0000fful; // shift left 32 bits, OR with self, and 00011111000000000000000011111111000000000000000011111111
	x = (x | x << 8)  & 0x100f00f00f00f00ful; // shift left 32 bits, OR with self, and 0001000000001111000000001111000000001111000000001111000000000000
	x = (x | x << 4)  & 0x10c30c30c30c30c3ul; // shift left 32 bits, OR with self, and 0001000011000011000011000011000011000011000011000011000100000000
	x = (x | x << 2)  & 0x1249249249249249ul;
	return x;
};

inline uint64_t mortonEncode64(uint32_t x, uint32_t y, uint32_t z) {
	return splitBy3_64(x) << 2 | splitBy3_64(y) << 1 | splitBy3_64(z);
};

inline
uint32_t compact3_64(uint64_t x) {
	x      =  x              & 0x1249249249249249ul;
	x      = (x | (x >> 2))  & 0x10c30c30c30c30c3ul; 
	x      = (x | (x >> 4))  & 0x100f00f00f00f00ful; 
	x      = (x | (x >> 8))  & 0x001f0000ff0000fful; 
	x      = (x | (x >> 16)) & 0x001f00000000fffful; 
	x      = (x | (x >> 32)) & 0x00000000001ffffful; 
	return x;
};

template<typename T>
inline
T morton_decode_64(uint64_t x) {
	return T{compact3_64(x >> 2), compact3_64(x >> 1), compact3_64(x)};
};