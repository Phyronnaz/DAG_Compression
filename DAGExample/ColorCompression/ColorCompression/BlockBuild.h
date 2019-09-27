#ifndef BLOCK_BUILD_H
#define BLOCK_BUILD_H
#include <cstdint>
#include <cuda_runtime.h>
#include <glm/vec3.hpp>
#include <vector>
struct BlockBuild {
	BlockBuild(size_t blockStart, size_t blockLength) :
		blockStart(blockStart),
		blockLength(blockLength),
		dirty(true)
	{};

	BlockBuild(size_t blockStart) :
		BlockBuild(blockStart, 1)
	{};

	size_t blockStart;
	size_t blockLength;
	bool dirty;
};
enum ColorLayout { R_4, R_8, R_16, RG_8_8, RG_16_16, RGB_8_8_8, RGB_10_12_10, RGB_5_6_5, NONE };

void uploadColors(const std::vector<float3> &colors);
void scores_gpu(
	const std::vector<BlockBuild> &blocks,
	std::vector<float> &scores,
	std::vector<uint8_t> &weights,
	std::vector<float3> &colorRanges,
	float error_treshold,
	bool minmaxcorrection,
	bool laberr,
	ColorLayout layout,
	int K,
	bool finalEval = false
);
#endif  // BLOCK_BUILD_H
