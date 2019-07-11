#pragma once
#include <inttypes.h>
#include <assert.h>

#if __CUDACC__
#define PREAMBLE __host__ __device__
#else
#define PREAMBLE
#endif

PREAMBLE inline void clear_bits(uint32_t &val, const int from_bit, const int bits) {
	assert(from_bit + bits <= 32);
	uint32_t t0 = (0xFFFFFFFFu >> from_bit);
	// Need to be careful not to try to shift >= 32 steps (undefined)
	uint32_t t1 = (from_bit + bits == 32) ? 0xFFFFFFFFu : ~(0xFFFFFFFFu >> (from_bit + bits));
	val &= ~(t0 & t1);
};

PREAMBLE inline uint64_t insert_bits(uint32_t val, int bits, uint32_t * array, uint64_t bitptr) {
	if (bits == 0) return bitptr;
	val &= (0xFFFFFFFFu >> (32 - bits));
	uint32_t ptr_word = static_cast<uint32_t>(bitptr / 32ull);
	uint32_t ptr_bit  = static_cast<uint32_t>(bitptr % 32ull);
	int bits_left = 32 - ptr_bit;
	if (bits_left >= bits) {
		clear_bits(array[ptr_word], ptr_bit, bits);
		array[ptr_word] |= val << (bits_left - bits);
	}
	else {
		clear_bits(array[ptr_word], ptr_bit, bits_left);
		uint32_t upper_val = val >> (bits - bits_left);
		array[ptr_word] |= upper_val;
		clear_bits(array[ptr_word + 1], 0, bits - bits_left);
		uint32_t lower_val = val << (32 - (bits - bits_left));
		array[ptr_word + 1] |= lower_val;
	}
	return bitptr + bits;
};

PREAMBLE inline uint32_t extract_bits(int bits, uint32_t * array, uint64_t bitptr) {
	if (bits == 0) return 0;
	uint32_t ptr_word = uint32_t(bitptr / 32ull);
	uint32_t ptr_bit = uint32_t(bitptr % 32ull);
	int bits_left = 32 - ptr_bit;
	// Need to be careful not to try to shift >= 32 steps (undefined)
	uint32_t upper_mask = (bits_left == 32) ? 0xFFFFFFFF : (~(0xFFFFFFFFu << bits_left));
	if (bits_left >= bits) {
		uint32_t val = upper_mask & array[ptr_word];
		val >>= (bits_left - bits);
		return val;
	}
	else {
		uint32_t val = (upper_mask & array[ptr_word]) << (bits - bits_left);
		val |= array[ptr_word + 1] >> (32 - (bits - bits_left));
		return val;
	}
};