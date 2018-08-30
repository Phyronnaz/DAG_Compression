#pragma once
#include "orientation.h"
#include <iostream>

/**
 * A view is a composite of its base class, the orientation
 * and the variables defining its frustrum.
 */
namespace chag {
class view : public orientation {
 public:
	view(){};
	~view(){};
	// Perspective view variables
	float m_fov          = 60.0;
	float m_aspect_ratio = 1.0;
	float m_near         = 1.0;
	float m_far          = 500.0;

	virtual void set_as_projection();
	void push();
	void pop();
	glm::mat4 get_MVP() const;
	glm::mat4 get_MVP_inv() const;
	virtual glm::mat4 get_P() const;
	virtual glm::mat4 get_P_inv() const;

	static void drawFullScreenQuad();
	void draw();
};

/**
 * Less than optimally, an orthoview inherits the "view" which is actually
 * a perspective view. Some day I should let perspectiveview inherit view as well
 * leaving view as an interface.
 */
class orthoview : public view {
 public:
	// Ortho view variables
	float m_right, m_left, m_top, m_bottom;
	virtual glm::mat4 get_P() const;
	virtual glm::mat4 get_P_inv() const;
};
}  // namespace chag
