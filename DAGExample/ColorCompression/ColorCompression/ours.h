//#pragma once
//#include "BlockBuild.h"
//#include <vector>
//#include <cstdint>
//#include <utility>  // std::pair, std::make_pair
//#include <utils/ProgressListener.h>
//
//#define USE_INDIRECT_WEIGHTS_ALWAYS 0
//
//namespace ours {
//using ColorLayout = ColorLayout;
//void compressColors(std::vector<uint32_t> &original_colors, bool USE_SINGLE_COLOR_BLOCKS,
//                    ProgressListener *progress_listener = NULL);
//void compressColors_alternative(std::vector<uint32_t> &original_colors);
//void compressColors_alternative_par(std::vector<uint32_t> &original_colors);
//struct OursData {
//	OursData() : d_block_headers(NULL){};
//	uint32_t nof_blocks;
//	uint32_t *d_block_headers;
//	uint32_t *h_block_headers;
//	uint32_t nof_colors;
//	uint32_t weights_size;
//	uint32_t headers_size;
//	uint32_t macro_offset_size;
//	uint32_t *d_weights;
//	uint32_t *h_weights;
//	uint32_t *d_macro_w_offset;
//	uint32_t *h_macro_w_offset;
//	uint32_t bits_per_weight;
//	ColorLayout color_layout;
//	bool use_single_color_blocks;
//};
//
//struct CacheHeader {
//	uint32_t headers_size;
//	uint32_t weights_size;
//	uint32_t nof_blocks;
//	uint32_t nof_colors;
//	uint32_t bits_per_weight;
//	bool use_single_color_blocks;
//};
//
//extern OursData ours_data;
//size_t sizeInBytes();
//bool getErrInfo(const std::vector<uint32_t> &colors, const std::string filename, float *mse, float *maxR, float *maxG,
//                float *maxB, float *maxLength);
//float getPSNR(float mse);
//std::string generateMetaData(std::string filename);
//
/////////////////////////////////////////////////////////////////////////////
//// Caching
/////////////////////////////////////////////////////////////////////////////
//std::string getCacheIdentifier();
//void upload_to_gpu();
//bool loadCached(const std::string &filename);
//void saveCached(const std::string &filename);
//};  // namespace ours
