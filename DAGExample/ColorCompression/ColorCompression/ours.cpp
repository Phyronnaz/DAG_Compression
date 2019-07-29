//#include "ours.h"
//#include "../bits_in_uint_array.h"
//#include "svd.h"
//#include <CHAGApp/Settings.h>
//#include <inttypes.h>
//#include <utils/ScopeTimer.h>
////#include "colorspace.h"
//#include <algorithm>
//#include <array>
//#include <sstream>
//
//#include <glm/glm.hpp>
//
//#define _USE_MATH_DEFINES
//#include <math.h>
//
//#define GPU_SCORE_COMP
//
//using namespace std;
//using glm::mat3;
//using glm::vec3;
//static float max_error_eval = 0.f;
//
//namespace ours {
/////////////////////////////////////////////////////////////////////////////
//// Temp storage required while building and global variables
/////////////////////////////////////////////////////////////////////////////
//vector<vec3> colors;
//vector<uint32_t> w;
//vector<vec3> subtracted;
//std::vector<vec3> V(3);
//int K;
//
/////////////////////////////////////////////////////////////////////////////
//// Allocate temp storage
/////////////////////////////////////////////////////////////////////////////
//void init(int nof_colors) {
//	colors.resize(nof_colors);
//	subtracted.resize(nof_colors);
//	w.resize(nof_colors);
//}
//
/////////////////////////////////////////////////////////////////////////////
//// Free temp storage
/////////////////////////////////////////////////////////////////////////////
//void deinit() {
//	subtracted.clear();
//	w.clear();
//	colors.clear();
//}
//
//static bool g_laberr;
//static ColorLayout g_layout;
//static bool g_minmax_correction;
//
//ColorLayout getColorLayout() {
//	auto layout = Settings::instance().get<std::string>("our_compression", "minmax_color_layout");
//	if      (layout == "8_8_8")    return RGB_8_8_8;
//	else if (layout == "10_12_10") return RGB_10_12_10;
//	else if (layout == "5_6_5")    return RGB_5_6_5;
//	return NONE;
//}
//
//vec3 rgb888_to_float3(uint32_t rgb) {
//	return vec3(
//		((rgb >> 0)  & 0xFF) / 255.0f,
//		((rgb >> 8)  & 0xFF) / 255.0f,
//		((rgb >> 16) & 0xFF) / 255.0f
//	);
//}
//
//vec3 rgb101210_to_float3(uint32_t rgb) {
//	return vec3(
//		((rgb >> 0)  & 0x3FF) / 1023.0f,
//		((rgb >> 10) & 0xFFF) / 4095.0f,
//		((rgb >> 22) & 0x3FF) / 1023.0f
//	);
//}
//
//vec3 rgb565_to_float3(uint32_t rgb) {
//	return vec3(
//		((rgb >> 0)  & 0x1F) / 31.0f,
//		((rgb >> 5)  & 0x3F) / 63.0f,
//		((rgb >> 11) & 0x1F) / 31.0f
//	);
//}
//
//uint32_t float3_to_rgb888(vec3 c) {
//	float R = min(1.0f, max(0.0f, c.x));
//	float G = min(1.0f, max(0.0f, c.y));
//	float B = min(1.0f, max(0.0f, c.z));
//	return (uint32_t(round(R * 255.0f)) << 0) |
//		     (uint32_t(round(G * 255.0f)) << 8) |
//	       (uint32_t(round(B * 255.0f)) << 16);
//}
//
//uint32_t float3_to_rgb101210(vec3 c) {
//	float R = min(1.0f, max(0.0f, c.x));
//	float G = min(1.0f, max(0.0f, c.y));
//	float B = min(1.0f, max(0.0f, c.z));
//	return (uint32_t(round(R * 1023.0f)) << 0)  |
//				 (uint32_t(round(G * 4095.0f)) << 10) |
//	       (uint32_t(round(B * 1023.0f)) << 22);
//}
//
//uint32_t float3_to_rgb565(vec3 c) {
//	float R = min(1.0f, max(0.0f, c.x));
//	float G = min(1.0f, max(0.0f, c.y));
//	float B = min(1.0f, max(0.0f, c.z));
//	return (uint32_t(round(R * 31.0f)) << 0) |
//		     (uint32_t(round(G * 63.0f)) << 5) |
//		     (uint32_t(round(B * 31.0f)) << 11);
//}
//
//vec3 rgbxxx_to_float3(uint32_t rgb, ColorLayout layout) {
//	switch (layout) {
//		case RGB_8_8_8: {
//			return vec3(
//				((rgb >> 0)  & 0xFF) / 255.0f,
//				((rgb >> 8)  & 0xFF) / 255.0f,
//				((rgb >> 16) & 0xFF) / 255.0f
//			);
//		} break;
//		case RGB_10_12_10: {
//			return vec3(
//				((rgb >> 0)  & 0x3FF) / 1023.0f,
//				((rgb >> 10) & 0xFFF) / 4095.0f,
//				((rgb >> 22) & 0x3FF) / 1023.0f
//			);
//		} break;
//		case RGB_5_6_5: {
//			return vec3(
//				((rgb >> 0)  & 0x1F) / 31.0f,
//				((rgb >> 5)  & 0x3F) / 63.0f,
//				((rgb >> 11) & 0x1F) / 31.0f
//			);
//		} break;
//		default: break;
//	}
//	LOG_ERROR("Invalid color format!");
//	return vec3(0.0);
//}
//
//uint32_t float3_to_rgbxxx(vec3 c, ColorLayout layout) {
//	float R = min(1.0f, max(0.0f, c.x));
//	float G = min(1.0f, max(0.0f, c.y));
//	float B = min(1.0f, max(0.0f, c.z));
//
//	switch (layout) {
//		case RGB_8_8_8: {
//			return (uint32_t(round(R * 255.0f)) << 0) |
//				     (uint32_t(round(G * 255.0f)) << 8) |
//			       (uint32_t(round(B * 255.0f)) << 16);
//		} break;
//		case RGB_10_12_10: {
//			return (uint32_t(round(R * 1023.0f)) << 0)  |
//				     (uint32_t(round(G * 4095.0f)) << 10) |
//			       (uint32_t(round(B * 1023.0f)) << 22);
//		} break;
//		case RGB_5_6_5: {
//			return (uint32_t(round(R * 31.0f)) << 0) |
//				     (uint32_t(round(G * 63.0f)) << 5) |
//			       (uint32_t(round(B * 31.0f)) << 11);
//
//		} break;
//		default: break;
//	}
//	LOG_ERROR("Invalid color format!");
//	return 0;
//}
//
/////////////////////////////////////////////////////////////////////////////
//// Output data
/////////////////////////////////////////////////////////////////////////////
//OursData ours_data;
//
//size_t final_total_size_in_bytes;
//size_t sizeInBytes() { return final_total_size_in_bytes; }
//
//bool getErrInfo(const vector<uint32_t> &colors, const string filename, float *mse, float *maxR, float *maxG,
//                float *maxB, float *maxLength) {
//	ColorLayout layout = getColorLayout();
//	///////////////////////////////////////////////////////////////////////////
//	// Read the data from disk...
//	///////////////////////////////////////////////////////////////////////////
//	ifstream is(filename, ios::binary);
//	if (!is.good()) return false;
//
//	CacheHeader nfo;
//	is.read(reinterpret_cast<char *>(&nfo), sizeof(CacheHeader));
//	vector<uint32_t> block_headers(nfo.headers_size / sizeof(uint32_t));
//	is.read(reinterpret_cast<char *>(block_headers.data()), nfo.headers_size);
//
//	vector<uint32_t> weights(nfo.weights_size / sizeof(uint32_t));
//	is.read(reinterpret_cast<char *>(weights.data()), nfo.weights_size);
//	is.close();
//
//	///////////////////////////////////////////////////////////////////////////
//	// Get next color...
//	///////////////////////////////////////////////////////////////////////////
//	size_t position              = 0;
//	uint32_t color_idx           = 0;
//	bool USE_SINGLE_COLOR_BLOCKS = Settings::instance().get<bool>("our_compression", "use_single_color_blocks");
//
//	auto getNextColor = [&]() {
//		///////////////////////////////////////////////////////////////////////////
//		// Binary search through headers to find the block containing my node
//		///////////////////////////////////////////////////////////////////////////
//		int header_size = (USE_INDIRECT_WEIGHTS_ALWAYS || USE_SINGLE_COLOR_BLOCKS) ? 4 : 3;
//		header_size -= layout == RGB_5_6_5 ? 1 : 0;
//
//		///////////////////////////////////////////////////////////////////////////
//		// Fetch min/max and weight and interpolate out the color
//		///////////////////////////////////////////////////////////////////////////
//		vec3 mincolor, maxcolor;
//		switch (layout) {
//			case RGB_8_8_8: {
//				mincolor = rgb888_to_float3(block_headers[position * header_size + 1]);
//				maxcolor = rgb888_to_float3(block_headers[position * header_size + 2]);
//			} break;
//			case RGB_10_12_10: {
//				mincolor = rgb101210_to_float3(block_headers[position * header_size + 1]);
//				maxcolor = rgb101210_to_float3(block_headers[position * header_size + 2]);
//			} break;
//			case RGB_5_6_5: {
//				auto tmp = block_headers[position * header_size + 1];
//				mincolor = rgb565_to_float3(tmp & 0xFFFF);
//				maxcolor = rgb565_to_float3((tmp >> 16) & 0xFFFF);
//			} break;
//			default: break;
//		}
//
//		int weight;
//		bool is_single_color_block = false;
//		if (USE_INDIRECT_WEIGHTS_ALWAYS || USE_SINGLE_COLOR_BLOCKS) {
//			uint32_t range;
//			uint32_t start_node = block_headers[position * header_size] & 0x7FFFFFFF;
//			if (USE_SINGLE_COLOR_BLOCKS)
//				is_single_color_block = (block_headers[position * header_size] & 0x80000000) != 0;
//			if (!is_single_color_block) {
//				uint32_t local_index = color_idx - start_node;
//				if (position < nfo.nof_blocks - 1)
//					range = (block_headers[(position + 1) * header_size] & 0x7FFFFFFF) - start_node;
//				else
//					range = nfo.nof_colors - start_node;
//				int weights_offset = 2 + ((layout == RGB_5_6_5) ? 0 : 1);
//				if (range <= 32 / nfo.bits_per_weight) {
//					uint32_t weights = block_headers[position * header_size + weights_offset];
//					weight           = extract_bits(nfo.bits_per_weight, &weights, local_index * nfo.bits_per_weight);
//				} else {
//					uint32_t w_index = block_headers[position * header_size + weights_offset];
//					weight           = extract_bits(nfo.bits_per_weight, weights.data(),
//                                          (w_index + local_index) * nfo.bits_per_weight);
//				}
//			}
//		} else {
//			weight = extract_bits(nfo.bits_per_weight, weights.data(), color_idx * nfo.bits_per_weight);
//		}
//		vec3 decompressed_color =
//		    is_single_color_block ? mincolor
//		                          : mincolor + (weight / float((1 << nfo.bits_per_weight) - 1)) * (maxcolor - mincolor);
//
//		++color_idx;
//		if (((position + 1) * header_size) < block_headers.size()) {
//			if (color_idx >= (block_headers[(position + 1) * header_size] & 0x7FFFFFFF)) { ++position; }
//		}
//
//		return float3_to_rgb888(decompressed_color);
//	};
//
//	///////////////////////////////////////////////////////////////////////////
//	// Actually compute MSE...
//	///////////////////////////////////////////////////////////////////////////
//	auto to_float3 = [](uint32_t rgb) { return vec3(((rgb >> 0) & 0xFF), ((rgb >> 8) & 0xFF), ((rgb >> 16) & 0xFF)); };
//	size_t N       = colors.size();
//	double errsq   = 0.f;
//
//	double max_errR   = -std::numeric_limits<double>::max();
//	double max_errG   = -std::numeric_limits<double>::max();
//	double max_errB   = -std::numeric_limits<double>::max();
//	double max_length = -std::numeric_limits<double>::max();
//
//	for (size_t i = 0; i < N; ++i) {
//		auto a3     = to_float3(colors[i]);
//		auto b3     = to_float3(getNextColor());
//		double errR = double(abs(a3.x - b3.x));
//		double errG = double(abs(a3.y - b3.y));
//		double errB = double(abs(a3.z - b3.z));
//
//		errsq += errR * errR + errG * errG + errB * errB;
//
//		max_errR   = max(errR, max_errR);
//		max_errG   = max(errG, max_errG);
//		max_errB   = max(errB, max_errB);
//		max_length = max(double(glm::length(a3 - b3)), max_length);
//	}
//
//	*mse       = errsq / double(N * 3);
//	*maxR      = max_errR;
//	*maxG      = max_errG;
//	*maxB      = max_errB;
//	*maxLength = max_length;
//	return true;
//}
//
//float getPSNR(float mse) { return mse > 0.f ? 20.f * log10(255.f) - 10.f * log10(mse) : -1.f; }
//
//std::string generateMetaData(std::string filename) {
//	ifstream is(filename, ios::binary);
//	if (!is.good()) {
//		LOG_ERROR("BAD DATA");
//		return "{ }";
//	}
//
//	CacheHeader nfo;
//	is.read(reinterpret_cast<char *>(&nfo), sizeof(CacheHeader));
//	is.close();
//
//	std::stringstream ss;
//	ss << "{\n"
//	   << "\"headers_size\":" << nfo.headers_size << ",\n"
//	   << "\"weights_size\":" << nfo.weights_size << ",\n"
//	   << "\"ratio\":" << float(nfo.headers_size) / float(nfo.weights_size) << "\n}";
//	return ss.str();
//}
//
/////////////////////////////////////////////////////////////////////////////
//// Fit a line to a set of 3D points
/////////////////////////////////////////////////////////////////////////////
//void leastSquaresFit(const vec3 *points, int nof_points, vec3 &o, vec3 &d) {
//	// Find mean of all points
//	o = vec3(0.0f);
//	for (int i = 0; i < nof_points; i++) o += (1.0f / float(nof_points)) * points[i];
//	for (int i = 0; i < nof_points; i++) { subtracted[i] = points[i] - o; }
//	array<float, 3> w;
//	dsvd(&subtracted[0].x, nof_points, 3, w.data(), &V[0].x);
//
//	auto max_index = std::distance(w.begin(), std::max_element(w.begin(), w.end()));
//	d = glm::normalize(
//		vec3(
//			reinterpret_cast<float *>(&V[0])[max_index],
//			reinterpret_cast<float *>(&V[1])[max_index],
//			reinterpret_cast<float *>(&V[2])[max_index])
//	);
//}
//
/////////////////////////////////////////////////////////////////////////
//// Get the "error" between two colors. Should be perceptually sane.
/////////////////////////////////////////////////////////////////////////
//float getError(const vec3 &a_, const vec3 &b_) {
//	auto a = a_;
//	auto b = b_;
//	if (g_minmax_correction) {
//		a = rgb888_to_float3(float3_to_rgb888(a));
//		b = rgb888_to_float3(float3_to_rgb888(b));
//	}
//	return length(a - b);
//};
//
//float getErrorPerChannel(const vec3 &a_, const vec3 &b_) {
//	auto a = a_;
//	auto b = b_;
//	if (g_minmax_correction) {
//		a = rgb888_to_float3(float3_to_rgb888(a));
//		b = rgb888_to_float3(float3_to_rgb888(b));
//	}
//	float x = a.x - b.x;
//	float y = a.y - b.y;
//	float z = a.z - b.z;
//	return x * x + y * y + z * z;
//};
//
/////////////////////////////////////////////////////////////////////////
//// Project a point onto a line defined by points A and B
/////////////////////////////////////////////////////////////////////////
//vec3 project(const vec3 &p, const vec3 &A, const vec3 &B) {
//	return A + (dot(p - A, B - A) / dot(B - A, B - A)) * (B - A);
//};
//
/////////////////////////////////////////////////////////////////////////
//// home-made least-square fit
/////////////////////////////////////////////////////////////////////////
//
//template <class T>
//T compensatedSum(T val, T &sum, T &error) {
//	T y   = val - error;
//	T t   = sum + y;
//	error = (t - sum) - y;
//	sum   = t;
//	return sum;
//}
//
//inline float trace(const mat3 m) { return m[0][0]+ m[1][1] + m[2][2]; }
//
//void leastSquaresFit_alternative(const vector<vec3> &points, uint32_t start, uint32_t range, vec3 &o, vec3 &d) {
//	// Find mean of all points
//	o = vec3(0.0f);
//#if 1
//	float s = 1.0f / range;
//	for (uint32_t i = start; i < start + range; i++) o += points[i];
//#else
//	float3 error = make_vector(0.0f, 0.0f, 0.0f);
//	for (int i = start; i < start + range; i++) o = compensatedSum(points[i], o, error);
//#endif
//
//	o /= range;
//
//	float3 zeros           = make_vector(0.0f, 0.0f, 0.0f);
//	float3x3 scatterMatrix = make_matrix(zeros, zeros, zeros);
//	float3x3 merror        = scatterMatrix;
//
//	for (uint32_t i = start; i < start + range; i++) {
//		float3 relpos      = points[i] - o;
//		float3x3 outerProd = make_matrix(relpos.x * relpos, relpos.y * relpos, relpos.z * relpos);
//		scatterMatrix      = scatterMatrix + outerProd;
//		// scatterMatrix = compensatedSum(outerProd, scatterMatrix, merror);
//	}
//
//	scatterMatrix.c1.y = scatterMatrix.c2.x;
//	scatterMatrix.c1.z = scatterMatrix.c3.x;
//	scatterMatrix.c2.z = scatterMatrix.c3.y;
//	if (trace(scatterMatrix) == 0.f) {
//		d.x = float(rand()) / float(RAND_MAX);
//		d.y = float(rand()) / float(RAND_MAX);
//		d.z = float(rand()) / float(RAND_MAX);
//		d   = normalize(d);
//		return;
//	}
//	// power method to find eigenvector of largest eigenvalue
//	vec3 v =
//	    make_vector(float(rand()) / float(RAND_MAX), float(rand()) / float(RAND_MAX), float(rand()) / float(RAND_MAX));
//	for (int i = 0; i < 20; i++) {
//		if (length(v) == 0.f) {
//			v = make_vector(float(rand()) / float(RAND_MAX), float(rand()) / float(RAND_MAX),
//			                float(rand()) / float(RAND_MAX));
//			i = 0;
//		}
//		v = normalize(v);
//		v = scatterMatrix * v;
//	}
//
//	float3 eigenvector = normalize(v);
//	float eigenvalue   = dot(scatterMatrix * v, v);
//
//	d = eigenvector;
//};
//
/////////////////////////////////////////////////////////////////////////////
//// Find a suitable min and max point for a range of colors
/////////////////////////////////////////////////////////////////////////////
//void fit(int start, int range, vec3 &minpoint, vec3 &maxpoint) {
//	if (range == 1) {
//		minpoint = colors[start];
//		maxpoint = colors[start];
//	} else if (range == 2) {
//		minpoint = colors[start];
//		maxpoint = colors[start + 1];
//	} else {
//		vec3 o, d;
//		leastSquaresFit((vec3 *)&colors[start].x, range, o, d);
//		// leastSquaresFit_alternative(colors, start, range, o, d);
//		float mindist = FLT_MAX;
//		float maxdist = -FLT_MAX;
//		const vec3 A  = o;
//		const vec3 B  = o + d;
//		for (int i = start; i < start + range; i++) {
//			float distance =
//			    dot(colors[i] - o, d);  // / dot(B - A, B - A); // B-A = d, which is *should be* normalized.
//			mindist = min(mindist, distance);
//			maxdist = max(maxdist, distance);
//		}
//		minpoint = o + mindist * d;
//		maxpoint = o + maxdist * d;
//	}
//	if (g_minmax_correction) {
//		minpoint = rgbxxx_to_float3(float3_to_rgbxxx(minpoint, g_layout), g_layout);
//		maxpoint = rgbxxx_to_float3(float3_to_rgbxxx(maxpoint, g_layout), g_layout);
//	}
//};
//
//bool assert_fit(int start, int range, const vec3 &A, const vec3 &B, const float error_treshold,
//                int vals_per_weight = K) {
//	int K = vals_per_weight;
//	if (range == 1) {
//		const vec3 p = colors[start];
//		if (getError(A, p) > error_treshold || getError(B, p) > error_treshold) return false;
//		return true;
//	} else if (range == 2 && K > 1) {
//		const vec3 p1 = colors[start];
//		const vec3 p2 = colors[start + 1];
//		if (getError(A, p1) > error_treshold || getError(B, p2) > error_treshold) return false;
//		return true;
//	}
//
//	bool bEval = true;
//	if (K > 1) {
//		for (int i = start; i < start + range; i++) {
//			// const vec3 & p = colors[i];
//			const vec3 p = colors[i];
//
//			float distance;
//			// Since A and B can be extremely close, we need to bail out
//			// This is safe since we are talking about colors that will be equal when truncated.
//			if (length(B - A) < (1e-4))
//				distance = 0.0f;
//			else
//				distance = dot(p - A, B - A) / dot(B - A, B - A);
//			vec3 interpolated_color = A + (float(w[i]) / float(K - 1)) * (B - A);
//			float error             = getError(p, interpolated_color);
//			if (error > error_treshold) { bEval = false; }
//		}
//	} else {
//		for (int i = start; i < start + range; i++) {
//			// const vec3 & p = colors[i];
//			const vec3 p = colors[i];
//
//			vec3 interpolated_color = A;
//			float error             = getError(p, interpolated_color);
//			// if (error > error_treshold) return false;
//			if (error > error_treshold) { bEval = false; }
//		}
//	}
//
//	return bEval;
//}
/////////////////////////////////////////////////////////////////////////////
//// Evaluate if a range of colors can be represented as interpolations of
//// two given min and max points, and update the corresponding weights.
/////////////////////////////////////////////////////////////////////////////
//bool evaluate(int start, int range, const vec3 &A, const vec3 &B, const float error_treshold, float *max_error = NULL,
//              float *mse = NULL) {
//	if (range == 1) {
//		w[start] = 0;
//		if (max_error != NULL) *max_error = 0.0f;
//		if (mse != NULL) *mse = 0.0f;
//		if (getError(A, colors[start]) > error_treshold || getError(B, colors[start]) > error_treshold) return false;
//		return true;
//	} else if (range == 2) {
//		w[start]     = 0;
//		w[start + 1] = K - 1;
//		if (max_error != NULL) *max_error = 0.0f;
//		if (mse != NULL) *mse = 0.0f;
//		if (getError(A, colors[start]) > error_treshold || getError(B, colors[start + 1]) > error_treshold)
//			return false;
//		return true;
//	}
//	if (max_error != NULL) *max_error = -FLT_MAX;
//	float msesum = 0.0f;
//
//	bool bEval = true;
//	for (int i = start; i < start + range; i++) {
//		const vec3 &p = colors[i];
//		float distance;
//		// Since A and B can be extremely close, we need to bail out
//		// This is safe since we are talking about colors that will be equal when truncated.
//		if (length(B - A) < (1e-4))
//			distance = 0.0f;
//		else
//			distance = dot(p - A, B - A) / dot(B - A, B - A);
//		int _w                  = int(round(distance * float(K - 1)));
//		_w                      = min(max(_w, 0), K - 1);
//		vec3 interpolated_color = A + (float(_w) / float(K - 1)) * (B - A);
//		float error             = getError(p, interpolated_color);
//		max_error_eval          = max(error, max_error_eval);
//		if (max_error != NULL) *max_error = std::max(*max_error, error);
//		msesum += getErrorPerChannel(p, interpolated_color);
//		// if (error > error_treshold) return false;
//		if (error > error_treshold) { bEval = false; }
//		w[i] = _w;
//	}
//	if (mse != NULL) *mse = msesum / double(range * 3);
//	return bEval;
//};
//
/////////////////////////////////////////////////////////////////////////////
//// Compress colors
/////////////////////////////////////////////////////////////////////////////
//void compressColors(std::vector<uint32_t> &original_colors, bool USE_SINGLE_COLOR_BLOCKS,
//                    ProgressListener *attached_progress_listener) {
//	g_progress->push_task("Compress colors with our algo", true);
//	PROFILE_CPU("compressColors");
//
//	///////////////////////////////////////////////////////////////////////
//	// Initialization
//	///////////////////////////////////////////////////////////////////////
//	init(original_colors.size());
//	for (int i = 0; i < colors.size(); i++) {
//		colors[i] =
//		    chag::make_vector(float(original_colors[i] & 0xFF) / 255.f, float((original_colors[i] >> 8) & 0xFF) / 255.f,
//		                      float((original_colors[i] >> 16) & 0xFF) / 255.f);
//	}
//	// Fetch accepted error and number of bits from settings
//	K                        = Settings::instance().get<int>("our_compression", "weight_bits");
//	uint32_t bits_per_weight = uint32_t(log2(K));
//	float error_treshold     = Settings::instance().get<float>("our_compression", "error_treshold");
//
//	g_layout            = getColorLayout();
//	g_laberr            = Settings::instance().get<bool>("our_compression", "use_lab_error");
//	g_minmax_correction = Settings::instance().get<bool>("our_compression", "minmax_correction");
//
//	///////////////////////////////////////////////////////////////////////
//	// The block information used when building
//	///////////////////////////////////////////////////////////////////////
//	struct block {
//		block(bool single_color, uint32_t start, uint32_t _range, vec3 minp, vec3 maxp)
//		    : is_single_color_block(single_color), start_node(start), range(_range), minpoint(minp), maxpoint(maxp){};
//		bool is_single_color_block;
//		uint32_t start_node;
//		uint32_t range;
//		vec3 minpoint, maxpoint;
//	};
//
//	///////////////////////////////////////////////////////////////////////
//	// If enabled, first search for blocks that can be represented by
//	// a single color and store them in a separate list
//	///////////////////////////////////////////////////////////////////////
//	struct sblock {
//		sblock(uint32_t s, uint32_t r, vec3 c) : start(s), range(r), color(c){};
//		uint32_t start, range;
//		vec3 color;
//	};
//	vector<sblock> sblocks;
//
//	if (USE_SINGLE_COLOR_BLOCKS) {
//		PROFILE_CPU("Find single color blocks");
//		g_progress->push_task("Find single-color blocks");
//		g_progress->setRange(colors.size());
//
//		///////////////////////////////////////////////////////////////////
//		// Calculate the number of elements required for a single-color
//		// block to be worth it.
//		///////////////////////////////////////////////////////////////////
//		uint32_t elements_required = uint32_t(ceil(80 / bits_per_weight));
//		auto lowerPowerOfTwo       = [](uint32_t val) { return pow(2, ceil(log(val) / log(2)) - 1); };
//		uint32_t lowest_range      = lowerPowerOfTwo(elements_required);
//		///////////////////////////////////////////////////////////////////
//		// No point in searching for smaller blocks.
//		///////////////////////////////////////////////////////////////////
//		uint32_t current_start   = 0;
//		uint32_t current_range   = lowest_range;
//		uint32_t last_good_range = 1;
//		vec3 last_good_color;
//		bool mode_find_largest_failing = true;
//		int step_size;
//
//		while (true) {
//			if (sblocks.size() > 0) g_progress->update(sblocks.back().start + sblocks.back().range);
//			///////////////////////////////////////////////////////////////
//			// Find average of current range
//			///////////////////////////////////////////////////////////////
//			uint32_t capped_range = min(int(colors.size()) - current_start, current_range);
//			vec3 avg_color        = make_vector(0.0f, 0.0f, 0.0f);
//			for (uint32_t i = current_start; i < current_start + capped_range; i++) {
//				avg_color += colors[i] * (1.0f / float(capped_range));
//			}
//			///////////////////////////////////////////////////////////////
//			// Se if all colors in range can be represented by that color
//			///////////////////////////////////////////////////////////////
//			bool range_good = true;
//			for (uint32_t i = current_start; i < current_start + capped_range; i++) {
//				if (getError(avg_color, colors[i]) > error_treshold) {
//					range_good = false;
//					break;
//				}
//			}
//			if (range_good) {
//				last_good_range = capped_range;
//				last_good_color = avg_color;
//			}
//
//			if (mode_find_largest_failing) {
//				if (range_good) {
//					///////////////////////////////////////////////////////
//					// If we reach the end of the list we are done. If the
//					// current range is large enough to be a single color
//					// block, push it.
//					///////////////////////////////////////////////////////
//					if (current_start + capped_range == colors.size()) {
//						if (capped_range <= elements_required) {
//							sblocks.push_back(sblock(current_start, capped_range, avg_color));
//						}
//						break;
//					} else {
//						current_range *= 2;
//					}
//				} else {
//					mode_find_largest_failing = false;
//					current_range             = (current_range / 2) + (current_range / 4);
//					step_size                 = (current_range / 8);
//				}
//			} else {
//				if (step_size == 0 || current_range < lowest_range) {
//					///////////////////////////////////////////////////////////
//					// If we have finished, check if the found block (if any)
//					// is large enough, then push it
//					///////////////////////////////////////////////////////////
//					if (last_good_range >= elements_required) {
//						sblocks.push_back(sblock(current_start, last_good_range, last_good_color));
//						current_start = current_start + last_good_range;
//					} else {
//						current_start = current_start + 1;
//					}
//					current_range             = lowest_range;
//					last_good_range           = 1;
//					mode_find_largest_failing = true;
//				} else {
//					if (range_good) {
//						current_range += step_size;
//					} else {
//						current_range -= step_size;
//					}
//					step_size /= 2;
//				}
//			}
//		}
//		g_progress->pop_task();
//	}
//
//	///////////////////////////////////////////////////////////////////////
//	// Start finding blocks. First we will double the range until we find a
//	// blockthat fails the error treshold. Then we will binary search to
//	// find the maximum block that fits.
//	///////////////////////////////////////////////////////////////////////
//	vector<block> blocks;
//	uint32_t current_start   = 0;
//	uint32_t current_range   = 4;  // No need to check "2", it always fits.
//	uint32_t last_good_range = 2;
//	int step_size;
//	bool mode_find_largest_failing = true;
//
//	///////////////////////////////////////////////////////////////////////
//	// Below are needed only if using single color blocks
//	///////////////////////////////////////////////////////////////////////
//	int next_sblock            = sblocks.size() > 0 ? 0 : -1;
//	uint32_t next_sblock_start = sblocks.size() > 0 ? sblocks[0].start : UINT_MAX;
//	uint32_t next_sblock_range = sblocks.size() > 0 ? sblocks[0].range : UINT_MAX;
//
//	{
//		PROFILE_CPU("Compress Blocks");
//		g_progress->push_task("Compress Blocks");
//		g_progress->setRange(colors.size());
//		while (true) {
//			g_progress->update(current_start);
//			///////////////////////////////////////////////////////////////////
//			// If we use single color blocks, don't attempt to build normal
//			// blocks in these ranges.
//			///////////////////////////////////////////////////////////////////
//			if (USE_SINGLE_COLOR_BLOCKS) {
//				// If we have reached an sblock, push it to the block list
//				if (current_start == next_sblock_start) {
//					blocks.push_back(block(true, sblocks[next_sblock].start, sblocks[next_sblock].range,
//					                       sblocks[next_sblock].color, make_vector(0.0f, 0.0f, 0.0f)));
//					current_start = next_sblock_start + next_sblock_range;
//					if (current_start >= colors.size()) break;
//				}
//
//				// Find the next sblock start from current_start
//				while (next_sblock >= 0) {
//					if (sblocks[next_sblock].start > current_start) {
//						next_sblock_start = sblocks[next_sblock].start;
//						next_sblock_range = sblocks[next_sblock].range;
//						break;
//					} else {
//						next_sblock += 1;
//						if (next_sblock >= sblocks.size()) {
//							next_sblock       = -1;
//							next_sblock_start = INT_MAX;
//							next_sblock_range = INT_MAX;
//						}
//					}
//				}
//			}
//
//			///////////////////////////////////////////////////////////////
//			// Fit and evaluate current range
//			///////////////////////////////////////////////////////////////
//			vec3 minpoint, maxpoint;
//			int capped_range =
//			    min(next_sblock_start - current_start, min(int(colors.size()) - current_start, current_range));
//			fit(current_start, capped_range, minpoint, maxpoint);
//			bool range_good = evaluate(current_start, capped_range, minpoint, maxpoint, error_treshold);
//			if (range_good) last_good_range = capped_range;
//
//			if (mode_find_largest_failing) {
//				if (range_good) {
//					///////////////////////////////////////////////////////////
//					// If we find that the block is good all the way up to the
//					// end of the list, push the block and exit.
//					///////////////////////////////////////////////////////////
//					if (current_start + capped_range == colors.size()) {
//						blocks.push_back(block(false, current_start, last_good_range, minpoint, maxpoint));
//						break;
//					}
//					///////////////////////////////////////////////////////////
//					// If the block is good up until the next single-color
//					// block, finish this block, and continue.
//					///////////////////////////////////////////////////////////
//					else if (current_start + capped_range == next_sblock_start) {
//						blocks.push_back(block(false, current_start, last_good_range, minpoint, maxpoint));
//						current_start += last_good_range;
//						current_range             = 4;
//						last_good_range           = 2;
//						mode_find_largest_failing = true;
//						continue;
//					}
//					///////////////////////////////////////////////////////////
//					// Otherwise, double the range and continue searching for
//					// first failing.
//					///////////////////////////////////////////////////////////
//					else {
//						current_range *= 2;
//					}
//				} else {  // !range_good
//					///////////////////////////////////////////////////////////
//					// We found a range that does not work, start binary search
//					// to find the largest block size that does.
//					///////////////////////////////////////////////////////////
//					mode_find_largest_failing = false;
//					current_range             = (current_range / 2) + (current_range / 4);
//					step_size                 = (current_range / 8);
//				}
//			} else {
//				if (step_size == 0) {
//					///////////////////////////////////////////////////////////
//					// We have finished our search, and now we know the largest
//					// range that works. Refit and reevaluate that block and
//					// push it.
//					// NOTE: Special case for 2 colors where fitting does not
//					//       seem to work.
//					///////////////////////////////////////////////////////////
//					if (last_good_range == 2) {
//						w[current_start + 0] = 0;
//						w[current_start + 1] = K - 1;
//						blocks.push_back(block(false, current_start, last_good_range, colors[current_start + 0],
//						                       colors[current_start + 1]));
//					} else {
//						fit(current_start, last_good_range, minpoint, maxpoint);
//						evaluate(current_start, last_good_range, minpoint, maxpoint, error_treshold);
//						blocks.push_back(block(false, current_start, last_good_range, minpoint, maxpoint));
//					}
//					if (current_start + last_good_range >= colors.size()) break;
//					current_start += last_good_range;
//					current_range             = 4;
//					last_good_range           = 2;
//					mode_find_largest_failing = true;
//				} else {
//					///////////////////////////////////////////////////////////
//					// Step up or down to find the best block
//					///////////////////////////////////////////////////////////
//					if (range_good) {
//						current_range += step_size;
//					} else {
//						current_range -= step_size;
//					}
//					step_size /= 2;
//				}
//			}
//		}
//		g_progress->pop_task();
//	}
//
//	g_progress->push_task("Finalize");
//	///////////////////////////////////////////////////////////////////////
//	// Create the compressed data
//	///////////////////////////////////////////////////////////////////////
//	if (ours_data.d_block_headers != NULL) {
//		cudaFree(ours_data.d_block_headers);
//		cudaFree(ours_data.d_weights);
//	}
//	ours_data.nof_blocks              = blocks.size();
//	ours_data.nof_colors              = colors.size();
//	ours_data.bits_per_weight         = bits_per_weight;
//	ours_data.use_single_color_blocks = USE_SINGLE_COLOR_BLOCKS;
//
//	vector<uint32_t> h_block_headers;
//	vector<uint32_t> h_weights;
//	uint32_t bits_required = w.size() * log2(K);
//	h_weights.resize(((bits_required - 1) / 32) + 1);
//	uint32_t weights_added = 0;
//	uint32_t global_bptr   = 0;
//	for (auto b : blocks) {
//		if (USE_SINGLE_COLOR_BLOCKS)
//			h_block_headers.push_back(b.start_node | (b.is_single_color_block ? 0x80000000 : 0x0));
//		else
//			h_block_headers.push_back(b.start_node);
//		switch (g_layout) {
//			case RGB_8_8_8:
//			case RGB_10_12_10: {
//				h_block_headers.push_back(float3_to_rgbxxx(b.minpoint, g_layout));
//				h_block_headers.push_back(float3_to_rgbxxx(b.maxpoint, g_layout));
//			} break;
//			case RGB_5_6_5: {
//				uint32_t minC = float3_to_rgbxxx(b.minpoint, g_layout);
//				uint32_t maxC = float3_to_rgbxxx(b.maxpoint, g_layout);
//				h_block_headers.push_back((minC & 0xFFFF) | ((maxC & 0xFFFF) << 16));
//			} break;
//		}
//
//		if (USE_INDIRECT_WEIGHTS_ALWAYS || USE_SINGLE_COLOR_BLOCKS) {
//			if (USE_SINGLE_COLOR_BLOCKS && b.is_single_color_block) {
//				// Single color block. Don't push any weights at all
//				h_block_headers.push_back(0x0);
//			} else {
//				// With indirect weights, there is another word which will
//				// either be a pointer to where the weights are, OR, id there
//				// are few enough weights, the weights themselves
//				if (b.range <= 32 / bits_per_weight) {
//					uint32_t weights;
//					int bptr = 0;
//					for (uint32_t i = b.start_node; i < b.start_node + b.range; i++) {
//						bptr = insert_bits(w[i], bits_per_weight, &weights, bptr);
//					}
//					h_block_headers.push_back(weights);
//				} else {
//					h_block_headers.push_back(weights_added);
//					for (uint32_t i = b.start_node; i < b.start_node + b.range; i++) {
//						global_bptr = insert_bits(w[i], bits_per_weight, &h_weights[0], global_bptr);
//					}
//					weights_added += b.range;
//				}
//			}
//		} else {
//			for (uint32_t i = b.start_node; i < b.start_node + b.range; i++) {
//				global_bptr = insert_bits(w[i], bits_per_weight, &h_weights[0], global_bptr);
//			}
//		}
//	}
//
//	///////////////////////////////////////////////////////////////////////
//	// Put in final data structure
//	///////////////////////////////////////////////////////////////////////
//	uint32_t headers_size     = h_block_headers.size() * sizeof(uint32_t);
//	ours_data.h_block_headers = (uint32_t *)malloc(headers_size);
//	memcpy(ours_data.h_block_headers, &h_block_headers[0], headers_size);
//	uint32_t weights_size = (((global_bptr - 1) / 32) + 1) * sizeof(uint32_t);
//	ours_data.h_weights   = (uint32_t *)malloc(weights_size);
//	memcpy(ours_data.h_weights, &h_weights[0], weights_size);
//	ours_data.weights_size = weights_size;
//	ours_data.headers_size = headers_size;
//	ours_data.color_layout = g_layout;
//
//	{
//		LOG_INFO("Headers size: " << headers_size << " bytes.");
//		LOG_INFO("Weights size: " << weights_size << " bytes.");
//		float compression = double(headers_size + weights_size) / double(colors.size() * 3);
//		LOG_INFO("Total: " << headers_size + weights_size << " bytes (" << compression * 100.0f << "%).");
//	}
//
//	final_total_size_in_bytes = headers_size + weights_size;
//
//	///////////////////////////////////////////////////////////////////////
//	// Free working data
//	///////////////////////////////////////////////////////////////////////
//	deinit();
//	g_progress->pop_task();
//	g_progress->pop_task();
//}
//
/////////////////////////////////////////////////////////////////////////////
//// Compress colors (Alternative Take)
/////////////////////////////////////////////////////////////////////////////
//void compressColors_alternative(std::vector<uint32_t> &original_colors) {
//	g_progress->push_task("Compress colors (ours, alternative)");
//	PROFILE_CPU("compressColors");
//	///////////////////////////////////////////////////////////////////////
//	// Initialization
//	///////////////////////////////////////////////////////////////////////
//	init(original_colors.size());
//	for (int i = 0; i < colors.size(); i++) {
//		colors[i] =
//		    chag::make_vector(float(original_colors[i] & 0xFF) / 255.f, float((original_colors[i] >> 8) & 0xFF) / 255.f,
//		                      float((original_colors[i] >> 16) & 0xFF) / 255.f);
//	}
//	// Fetch accepted error and number of bits from settings
//	K                        = Settings::instance().get<int>("our_compression", "weight_bits");
//	uint32_t bits_per_weight = uint32_t(log2(K));
//	float error_treshold     = Settings::instance().get<float>("our_compression", "error_treshold");
//	g_layout                 = getColorLayout();
//	g_laberr                 = Settings::instance().get<bool>("our_compression", "use_lab_error");
//	g_minmax_correction      = Settings::instance().get<bool>("our_compression", "minmax_correction");
//
//	struct block {
//		block() : start_node(0xBADC0DE){};
//		block(uint32_t start, uint32_t rng, block *l = NULL, block *r = NULL)
//		    : start_node(start), range(rng), left(l), right(r), dirty(true){};
//		block *left, *right;
//		uint32_t start_node;
//		uint32_t range;
//		vec3 minpoint, maxpoint;
//		bool dirty;
//	};
//	vector<block> blocks;
//
//	///////////////////////////////////////////////////////////////////////
//	// To avoid running out of memory, split the compression into a number
//	// of very large parts.
//	// NOTE: These parts may not be merge together, but that won't affect
//	//       memory much.
//	///////////////////////////////////////////////////////////////////////
//	const uint64_t max_part_size = (100 * 1024 * 1024);
//	int nof_parts                = (colors.size() - 1) / max_part_size + 1;
//	uint32_t part_offset         = 0;
//	for (int part = 0; part < nof_parts; part++) {
//		const uint64_t part_size  = (part == nof_parts - 1) ? (colors.size() % max_part_size) : max_part_size;
//		const uint64_t part_start = part * max_part_size;
//		///////////////////////////////////////////////////////////////////////
//		// Build an original block-list where every block contains a single
//		// color
//		///////////////////////////////////////////////////////////////////////
//		vector<block> all_blocks(max_part_size);
//		LOG_VERBOSE("Compressing part " << part << ".");
//		block *head       = NULL;
//		block *left_block = NULL;
//		for (int i = part_start; i < part_start + part_size; i++) {
//			block *b = new (&all_blocks[i - part_start]) block(i, 1, left_block, NULL);
//			if (left_block != NULL)
//				left_block->right = b;
//			else {
//				head = b;
//			}
//			left_block = b;
//		}
//		assert(head != NULL);
//
//		///////////////////////////////////////////////////////////////////////
//		// Until no block can be merged, merge every other block with either
//		// its left or its right neighbour.
//		///////////////////////////////////////////////////////////////////////
//		int nof_blocks_merged = 1;
//
//		// A _highly_ approximate meassurement of how done we are
//		uint32_t number_of_merges_done = 0;
//		g_progress->push_task("Compress part " + part);
//		g_progress->setRange(max_part_size);
//
//		auto score = [error_treshold](uint32_t start, uint32_t range) {
//			assert(range > 1);
//			if (range == 2) {
//				// We cannot say anything about how fitting a line is to two colors, so in that case
//				// we only use the distance between colors.
//				return 1.0f / (length(colors[start] - colors[start + 1]) + 1.0f);
//			} else {
//				float max_error, mse;
//				vec3 minpoint, maxpoint;
//				fit(start, range, minpoint, maxpoint);
//				if (!evaluate(start, range, minpoint, maxpoint, error_treshold, &max_error, &mse)) return -1.0f;
//				// else return 1.0f / (max_error + 1.0f);
//				else
//					return 1.0f / (mse + 1.0f);
//			}
//		};
//
//		static int passes;
//		passes = 0;
//		while (nof_blocks_merged > 0) {
//			LOG_VERBOSE("Compressing, pass: " << passes++);
//			nof_blocks_merged    = 0;
//			block *current_block = head->right;
//			while (current_block != NULL) {
//				bool left_dirty  = !(current_block->left == NULL || !current_block->left->dirty);
//				bool right_dirty = !(current_block->right == NULL || !current_block->right->dirty);
//				if (left_dirty || right_dirty || current_block->dirty) {
//					float left_score =
//					    (current_block->left == NULL)
//					        ? -1.0f
//					        : score(current_block->left->start_node, current_block->left->range + current_block->range);
//					float right_score =
//					    (current_block->right == NULL)
//					        ? -1.0f
//					        : score(current_block->start_node, current_block->range + current_block->right->range);
//
//					if (left_score >= 0.0f || right_score >= 0.0f) {
//						auto merge = [](block *left, block *right) {
//							left->range = left->range + right->range;
//							left->right = right->right;
//							if (right->right != NULL) right->right->left = left;
//							return left;
//						};
//						if (left_score > right_score) {
//							nof_blocks_merged += 1;
//							current_block = merge(current_block->left, current_block);
//						} else {
//							nof_blocks_merged += 1;
//							current_block = merge(current_block, current_block->right);
//						}
//						current_block->dirty = true;
//					} else {
//						current_block->dirty = false;
//					}
//				}
//
//				// Advance two steps from the current block (unless that would
//				// mean we fall of the end of the list)
//				if (current_block->right != NULL) {
//					if (current_block->right->right != NULL) {
//						current_block = current_block->right->right;
//					} else {
//						current_block = current_block->right;
//					}
//				} else {
//					current_block = NULL;
//				}
//			}
//			number_of_merges_done += nof_blocks_merged;
//			g_progress->update(number_of_merges_done);
//		}
//		g_progress->pop_task();
//
//		///////////////////////////////////////////////////////////////////////
//		// Create the compressed data
//		///////////////////////////////////////////////////////////////////////
//		block *current_block = head;
//
//		while (current_block != NULL) {
//			fit(current_block->start_node, current_block->range, current_block->minpoint, current_block->maxpoint);
//			evaluate(current_block->start_node, current_block->range, current_block->minpoint, current_block->maxpoint,
//			         error_treshold);
//			blocks.push_back(*current_block);
//			current_block = current_block->right;
//		}
//	}
//
//	g_progress->push_task("Finalize");
//	LOG_INFO("Nof blocks: " << blocks.size());
//	LOG_INFO("Average nof colors/block: " << colors.size() / float(blocks.size()));
//
//	if (ours_data.d_block_headers != NULL) {
//		cudaFree(ours_data.d_block_headers);
//		cudaFree(ours_data.d_weights);
//	}
//	ours_data.nof_blocks              = blocks.size();
//	ours_data.nof_colors              = colors.size();
//	ours_data.bits_per_weight         = bits_per_weight;
//	ours_data.use_single_color_blocks = false;
//
//	vector<uint32_t> h_block_headers;
//	vector<uint32_t> h_weights;
//	uint32_t bits_required = w.size() * log2(K);
//	h_weights.resize(((bits_required - 1) / 32) + 1);
//	uint32_t weights_added = 0;
//	uint32_t global_bptr   = 0;
//	for (auto b : blocks) {
//		h_block_headers.push_back(b.start_node);
//		switch (g_layout) {
//			case RGB_8_8_8:
//			case RGB_10_12_10: {
//				h_block_headers.push_back(float3_to_rgbxxx(b.minpoint, g_layout));
//				h_block_headers.push_back(float3_to_rgbxxx(b.maxpoint, g_layout));
//			} break;
//			case RGB_5_6_5: {
//				uint32_t minC = float3_to_rgbxxx(b.minpoint, g_layout);
//				uint32_t maxC = float3_to_rgbxxx(b.maxpoint, g_layout);
//				h_block_headers.push_back((minC & 0xFFFF) | ((maxC & 0xFFFF) << 16));
//			} break;
//		}
//		for (uint32_t i = b.start_node; i < b.start_node + b.range; i++) {
//			global_bptr = insert_bits(w[i], bits_per_weight, &h_weights[0], global_bptr);
//		}
//	}
//
//	///////////////////////////////////////////////////////////////////////
//	// Put in final data structure
//	///////////////////////////////////////////////////////////////////////
//	uint32_t headers_size     = h_block_headers.size() * sizeof(uint32_t);
//	ours_data.h_block_headers = (uint32_t *)malloc(headers_size);
//	memcpy(ours_data.h_block_headers, &h_block_headers[0], headers_size);
//	uint32_t weights_size = (((global_bptr - 1) / 32) + 1) * sizeof(uint32_t);
//	ours_data.h_weights   = (uint32_t *)malloc(weights_size);
//	memcpy(ours_data.h_weights, &h_weights[0], weights_size);
//	ours_data.weights_size = weights_size;
//	ours_data.headers_size = headers_size;
//	ours_data.color_layout = g_layout;
//	{
//		LOG_INFO("Headers size: " << headers_size << " bytes.");
//		LOG_INFO("Weights size: " << weights_size << " bytes.");
//		float compression = float(headers_size + weights_size) / float(colors.size() * 3);
//		LOG_INFO("Total: " << headers_size + weights_size << " bytes (" << compression * 100.0f << "%).");
//	}
//
//	final_total_size_in_bytes = headers_size + weights_size;
//
//	///////////////////////////////////////////////////////////////////////
//	// Free working data
//	///////////////////////////////////////////////////////////////////////
//	deinit();
//
//	g_progress->pop_task();
//	g_progress->pop_task();
//}
//
/////////////////////////////////////////////////////////////////////////////
//// Compress colors (Alternative Take) Parallel attempt
/////////////////////////////////////////////////////////////////////////////
//struct block {
//	block() : start_node(0xBADC0DE){};
//	block(uint32_t start, uint32_t rng) : start_node(start), range(rng), dirty(true){};
//	uint32_t start_node;
//	uint32_t range;
//	vec3 minpoint, maxpoint;
//	bool dirty;
//};
//std::vector<block> compress_range(size_t part_start, size_t part_size, const std::vector<uint32_t> &original_colors) {
//	const float error_treshold = Settings::instance().get<float>("our_compression", "error_treshold");
//
//	using vec3;
//	std::vector<float3> workingColorSet(part_size);
//	for (size_t i = part_start, j = 0; i < (part_start + part_size); i++, j++) {
//		workingColorSet[j] =
//		    chag::make_vector(float(original_colors[i] & 0xFF) / 255.f, float((original_colors[i] >> 8) & 0xFF) / 255.f,
//		                      float((original_colors[i] >> 16) & 0xFF) / 255.f);
//	}
//
//	int vals_per_weight = K;
//
//	uint64_t block_index = 0;
//	std::vector<BlockBuild> buildBlocks(workingColorSet.size(), BlockBuild(UINT32_MAX));
//
//	// start with one block per color
//	for (uint32_t colorIdx = 0; colorIdx < workingColorSet.size(); colorIdx++) {
//		buildBlocks[colorIdx] = BlockBuild(colorIdx);
//	}
//
//	uploadColors(workingColorSet);
//	int iteration              = 0;
//	uint32_t nof_blocks_merged = UINT32_MAX;
//	while (nof_blocks_merged > 0) {
//		LOG_INFO("Iteration " << iteration++);
//		nof_blocks_merged = 0;
//
//		// compute scores
//		vector<float> scores(buildBlocks.size(), 0.0f);
//		vector<vec3> colorRanges(buildBlocks.size());
//		vector<uint8_t> weights;
//		scores_gpu(buildBlocks, scores, weights, colorRanges, error_treshold, g_minmax_correction, g_laberr, g_layout,
//		           vals_per_weight);
//
//		{
//			PROFILE_CPU("merge blocks");
//			for (uint32_t blk_curr = 1; blk_curr < buildBlocks.size(); blk_curr += 2) {
//				// for (uint32_t blk_curr = 1; blk_curr < buildBlocks.size(); blk_curr += 3) {
//				bool seqDirty = buildBlocks[blk_curr - 1].dirty || buildBlocks[blk_curr].dirty ||
//				                (blk_curr + 1 < buildBlocks.size() && buildBlocks[blk_curr + 1].dirty);
//
//				if (seqDirty) {
//					float left_score  = scores[blk_curr - 1];
//					float right_score = blk_curr + 1 < buildBlocks.size() ? scores[blk_curr] : -1.0f;
//
//					if (left_score >= 0.0f || right_score >= 0.0f) {
//						if (left_score > right_score) {
//							buildBlocks[blk_curr - 1].blockLength += buildBlocks[blk_curr].blockLength;
//							buildBlocks[blk_curr - 1].dirty   = true;
//							buildBlocks[blk_curr].blockLength = 0;
//						} else {
//							buildBlocks[blk_curr].blockLength += buildBlocks[blk_curr + 1].blockLength;
//							buildBlocks[blk_curr].dirty           = true;
//							buildBlocks[blk_curr + 1].blockLength = 0;
//							blk_curr++;  // take 3 steps instead of 2 next time
//						}
//
//						nof_blocks_merged += 1;
//					} else {
//						buildBlocks[blk_curr].dirty = false;
//					}
//				}
//			}  // ~blk_curr
//		}      // end profile scope
//
//		std::vector<BlockBuild> newblocks;
//		newblocks.reserve(buildBlocks.size() - nof_blocks_merged);
//
//		for (uint32_t blk_curr = 0; blk_curr < buildBlocks.size(); blk_curr += 1) {
//			if (buildBlocks[blk_curr].blockLength > 0) newblocks.push_back(buildBlocks[blk_curr]);
//		}
//
//		buildBlocks = newblocks;
//
//		///////////////////////
//		//{
//		//	vector<float> scores;
//		//	vector<uint8_t> weights;
//		//	vector<vec3> colorRanges;
//		//	vector<vec3> asslord;
//		//	scores_gpu(buildBlocks, scores, weights, colorRanges, error_treshold,
//		//		g_minmax_correction, g_laberr, g_layout, vals_per_weight, true);
//		//	int bad = 0;
//		//	int good = 0;
//		//	for (uint64_t i = 0; i < buildBlocks.size(); i++) {
//		//		block tmp(buildBlocks[i].blockStart + part_start, buildBlocks[i].blockLength);
//		//		tmp.minpoint = colorRanges[2 * i + 0];
//		//		tmp.maxpoint = colorRanges[2 * i + 1];
//
//		//		uint64_t b_start = buildBlocks[i].blockStart;
//		//		uint64_t b_end = b_start + buildBlocks[i].blockLength;
//
//		//		for (uint64_t ci = b_start; ci < b_end; ci++) { w[part_start + ci] = weights[ci]; }
//		//		if (!evaluate(tmp.start_node, tmp.range, tmp.minpoint, tmp.maxpoint, error_treshold)){
//		//			LOG_ERROR("Block" << i << " was bad");
//		//			exit(0);
//		//		}
//
//		//	}
//		//	buildBlocks = newblocks;
//		//}
//		///////////////////////
//	}
//
//	vector<block> blocks;
//	blocks.reserve(buildBlocks.size());
//	{
//		PROFILE_CPU("Final fit and eval")
//		vector<float> scores;
//		vector<uint8_t> weights;
//		vector<vec3> colorRanges;
//		scores_gpu(buildBlocks, scores, weights, colorRanges, error_treshold, g_minmax_correction, g_laberr, g_layout,
//		           vals_per_weight, true);
//
//		for (uint64_t i = 0; i < buildBlocks.size(); i++) {
//			block tmp(buildBlocks[i].blockStart + part_start, buildBlocks[i].blockLength);
//			tmp.minpoint = colorRanges[2 * i + 0];
//			tmp.maxpoint = colorRanges[2 * i + 1];
//			blocks.push_back(tmp);
//
//			uint64_t b_start = buildBlocks[i].blockStart;
//			uint64_t b_end   = b_start + buildBlocks[i].blockLength;
//
//			for (uint64_t ci = b_start; ci < b_end; ci++) { w[part_start + ci] = weights[ci]; }
//		}
//	}
//
//	/////////////////////////////////////////////////////////////////////////
//	// SOLUTION FOR RANGE
//	/////////////////////////////////////////////////////////////////////////
//	return blocks;
//}
//
//void add_to_final(const vector<block> &solution, vector<uint32_t> &h_block_headers, vector<uint32_t> &h_weights,
//                  uint64_t &global_bptr, int &wrong_colors, int &ok_colors) {
//	const float error_treshold = Settings::instance().get<float>("our_compression", "error_treshold");
//	const int bitrate          = log2(K);
//
//	max_error_eval = 0.f;
//	for (auto b : solution) {
//#if 1
//		bool should_be_true = assert_fit(b.start_node, b.range, b.minpoint, b.maxpoint, error_treshold);
//		if (!should_be_true) {
//			wrong_colors += 1;
//		} else {
//			ok_colors += 1;
//		}
//#endif
//		h_block_headers.push_back(b.start_node);
//		switch (g_layout) {
//			case RGB_8_8_8:
//			case RGB_10_12_10: {
//				h_block_headers.push_back(float3_to_rgbxxx(b.minpoint, g_layout));
//				h_block_headers.push_back(float3_to_rgbxxx(b.maxpoint, g_layout));
//			} break;
//			case RGB_5_6_5: {
//				uint32_t minC = float3_to_rgbxxx(b.minpoint, g_layout);
//				uint32_t maxC = float3_to_rgbxxx(b.maxpoint, g_layout);
//				h_block_headers.push_back((minC & 0xFFFF) | ((maxC & 0xFFFF) << 16));
//			} break;
//		}
//		for (uint64_t i = b.start_node; i < b.start_node + b.range; i++) {
//			global_bptr = insert_bits(w[i], bitrate, &h_weights[0], global_bptr);
//		}
//	}
//}
//
//void compressColors_alternative_par(std::vector<uint32_t> &original_colors) {
//	K                   = Settings::instance().get<int>("our_compression", "weight_bits");
//	g_layout            = getColorLayout();
//	g_laberr            = Settings::instance().get<bool>("our_compression", "use_lab_error");
//	g_minmax_correction = Settings::instance().get<bool>("our_compression", "minmax_correction");
//
//	colors.resize(original_colors.size());
//	w.resize(original_colors.size());
//
//	const uint64_t max_part_size   = uint64_t(10 * 1024 * 1024);  // has to be this way in this method
//	const int nof_parts            = (original_colors.size() + max_part_size - 1) / max_part_size;
//	const uint64_t bits_required   = w.size() * log2(K);
//	const uint32_t bits_per_weight = uint32_t(log2(K));
//
//	for (int i = 0; i < colors.size(); i++) {
//		colors[i] =
//		    chag::make_vector(float(original_colors[i] & 0xFF) / 255.f, float((original_colors[i] >> 8) & 0xFF) / 255.f,
//		                      float((original_colors[i] >> 16) & 0xFF) / 255.f);
//	}
//
//	vector<uint32_t> h_weights(((bits_required - 1ull) / 32ull) + 1);
//	vector<uint32_t> h_block_headers;
//	uint64_t global_bptr = 0;
//
//	// For info
//	int wrong_colors    = 0;
//	int ok_colors       = 0;
//	uint32_t total_bits = 0;
//	uint32_t nof_blocks = 0;
//	float max_error     = 0.0f;
//	for (int part = 0; part < nof_parts; part++) {
//		LOG_INFO("Part: " << part << " of " << nof_parts << " header_size: " << h_block_headers.size());
//		const size_t part_size  = (part == nof_parts - 1) ? (original_colors.size() % max_part_size) : max_part_size;
//		const size_t part_start = part * max_part_size;
//
//		const vector<block> solution = compress_range(part_start, part_size, original_colors);
//		add_to_final(solution, h_block_headers, h_weights, global_bptr, wrong_colors, ok_colors);
//
//		// Info
//		nof_blocks += solution.size();
//		max_error = max(max_error_eval, max_error);
//		LOG_ERROR("error: " << max_error_eval);
//		if (max_error > 1.0f) exit(0);
//	}
//	h_weights.resize((global_bptr + 31) / 32);  // uint32_t elements
//
//	g_progress->push_task("Finalize");
//	LOG_INFO("Nof blocks: " << nof_blocks);
//	LOG_INFO("Average nof colors/block: " << colors.size() / float(nof_blocks));
//	LOG_INFO("h_weights_size: " << h_weights.size() << " = " << ((bits_required - 1) / 32) + 1);
//
//	LOG_INFO(wrong_colors << " wrong evals " << ok_colors << " was correct.. ("
//	                      << 100.f * float(wrong_colors) / float(wrong_colors + ok_colors) << " %)");
//	LOG_INFO("Max error is: " << max_error);
//
//	if (ours_data.d_block_headers != NULL) {
//		cudaFree(ours_data.d_block_headers);
//		cudaFree(ours_data.d_weights);
//	}
//	ours_data.nof_blocks              = nof_blocks;
//	ours_data.nof_colors              = colors.size();
//	ours_data.bits_per_weight         = bits_per_weight;
//	ours_data.use_single_color_blocks = false;
//
//	///////////////////////////////////////////////////////////////////////
//	// Put in final data structure
//	///////////////////////////////////////////////////////////////////////
//	size_t headers_size       = h_block_headers.size() * sizeof(uint32_t);
//	ours_data.h_block_headers = (uint32_t *)malloc(headers_size);
//	memcpy(ours_data.h_block_headers, &h_block_headers[0], headers_size);
//	size_t weights_size = (((global_bptr - 1) / 32) + 1) * sizeof(uint32_t);
//	ours_data.h_weights = (uint32_t *)malloc(weights_size);
//	memcpy(ours_data.h_weights, &h_weights[0], weights_size);
//	ours_data.weights_size = weights_size;
//	ours_data.headers_size = headers_size;
//	ours_data.color_layout = g_layout;
//	{
//		LOG_INFO("Headers size: " << headers_size << " bytes.");
//		LOG_INFO("Weights size: " << weights_size << " bytes.");
//		float compression = float(headers_size + weights_size) / float(colors.size() * 3);
//		LOG_INFO("Total: " << headers_size + weights_size << " bytes (" << compression * 100.0f << "%).");
//	}
//
//	final_total_size_in_bytes = headers_size + weights_size;
//
//	///////////////////////////////////////////////////////////////////////
//	// Free working data
//	///////////////////////////////////////////////////////////////////////
//	deinit();
//
//	g_progress->pop_task();
//	g_progress->pop_task();
//}
//
//std::string getCacheIdentifier() {
//	bool all_colors = Settings::instance().get<bool>("dag_compression", "colors_in_all_nodes");
//	int k           = Settings::instance().get<int>("our_compression", "weight_bits");
//	float e         = Settings::instance().get<float>("our_compression", "error_treshold");
//	string ret      = string(".b") + std::to_string(k) + string(".e") + std::to_string(e);
//	ret += (USE_INDIRECT_WEIGHTS_ALWAYS ? ".iwa" : "");
//	ret += Settings::instance().get<bool>("our_compression", "use_single_color_blocks") ? ".scb" : "";
//	ret += Settings::instance().get<bool>("our_compression", "use_lab_error") ? ".Lab" : ".RGB";
//	ret += Settings::instance().get<bool>("our_compression", "alternative_version") ? ".alternative" : "";
//	ret += std::string(".") + Settings::instance().get<std::string>("our_compression", "minmax_color_layout");
//	ret += Settings::instance().get<bool>("our_compression", "minmax_correction") ? ".corr" : ".nocorr";
//	ret += all_colors ? ".all" : "";
//	return ret + ".ours";
//}
//
//void upload_to_gpu() {
//	if (ours_data.d_block_headers != NULL) {
//		cudaFree(ours_data.d_block_headers);
//		cudaFree(ours_data.d_weights);
//	}
//	cudaMalloc((void **)&ours_data.d_block_headers, ours_data.headers_size);
//	cudaMemcpy(ours_data.d_block_headers, ours_data.h_block_headers, ours_data.headers_size, cudaMemcpyHostToDevice);
//	cudaMalloc((void **)&ours_data.d_weights, ours_data.weights_size);
//	cudaMemcpy(ours_data.d_weights, ours_data.h_weights, ours_data.weights_size, cudaMemcpyHostToDevice);
//}
//
//bool loadCached(const std::string &filename) {
//	// return false; // work around while working on color compression (when we do not want to use cached results)
//	CacheHeader nfo;
//	ifstream is(filename, ios::binary);
//	if (!is.good()) return false;
//
//	is.read(reinterpret_cast<char *>(&nfo), sizeof(CacheHeader));
//
//	if (ours_data.h_block_headers) free(ours_data.h_block_headers);
//	ours_data.h_block_headers = static_cast<uint32_t *>(malloc(nfo.headers_size));
//	is.read(reinterpret_cast<char *>(ours_data.h_block_headers), nfo.headers_size);
//
//	if (ours_data.h_weights) free(ours_data.h_weights);
//	ours_data.h_weights = static_cast<uint32_t *>(malloc(nfo.weights_size));
//	is.read(reinterpret_cast<char *>(ours_data.h_weights), nfo.weights_size);
//	is.close();
//
//	ours_data.headers_size            = nfo.headers_size;
//	ours_data.weights_size            = nfo.weights_size;
//	ours_data.nof_blocks              = nfo.nof_blocks;
//	ours_data.nof_colors              = nfo.nof_colors;
//	ours_data.bits_per_weight         = nfo.bits_per_weight;
//	ours_data.use_single_color_blocks = nfo.use_single_color_blocks;
//	ours_data.color_layout            = getColorLayout();
//
//	final_total_size_in_bytes = ours_data.headers_size + ours_data.weights_size;
//
//	LOG_INFO("Loaded cache: " << filename);
//	return true;
//}
//
//void saveCached(const std::string &filename) {
//	CacheHeader nfo;
//	nfo.headers_size            = ours_data.headers_size;
//	nfo.weights_size            = ours_data.weights_size;
//	nfo.nof_blocks              = ours_data.nof_blocks;
//	nfo.nof_colors              = ours_data.nof_colors;
//	nfo.bits_per_weight         = ours_data.bits_per_weight;
//	nfo.use_single_color_blocks = ours_data.use_single_color_blocks;
//	ofstream os(filename, ios::binary);
//	os.write(reinterpret_cast<const char *>(&nfo), sizeof(CacheHeader));
//	os.write(reinterpret_cast<const char *>(ours_data.h_block_headers), nfo.headers_size);
//	os.write(reinterpret_cast<const char *>(ours_data.h_weights), nfo.weights_size);
//	os.close();
//	LOG_INFO("Saved cache: " << filename);
//}
//}  // namespace ours
