#include "view.h"
#include <cstring>
#include <float.h>
#include <GL/glew.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include "glm_extensions.h"

using namespace std; 
using mat4 = glm::mat4;
using vec2 = glm::vec2;
using vec4 = glm::vec4;

#define PI 3.14159265f
namespace chag{
void view::set_as_projection()
{
	mat4 P = get_P();
	glMatrixLoadfEXT(GL_PROJECTION, glm::value_ptr(P));
}


void view::push()
{
	glMatrixPushEXT(GL_PROJECTION);
	set_as_projection(); 
	glMatrixPushEXT(GL_MODELVIEW); 
	set_as_modelview(); 
}

void view::pop()
{
	glMatrixPopEXT(GL_PROJECTION); 
	glMatrixPopEXT(GL_MODELVIEW);
}

mat4 view::get_P() const
{
	return glm::make_perspective(m_fov, m_aspect_ratio, m_near, m_far);
}

mat4 view::get_P_inv() const
{
	return glm::make_perspective_inv(m_fov, m_aspect_ratio, m_near, m_far);
}

mat4 view::get_MVP() const
{
	return get_P() * get_MV(); 
}


mat4 view::get_MVP_inv() const
{
	return get_MV_inv() * get_P_inv(); 
}

/**
 * Given an axis aligned bounding box in world coordinates, 
 * find an orientation and a post projection matrix the fits
 * the bounding box optimaly
 */
// Helpers for pack
bool isLeft(vec2 P0, vec2 P1, vec2 P2 ){
	return ((P1.x - P0.x)*(P2.y - P0.y) - (P2.x - P0.x)*(P1.y - P0.y)) > 0 ;
}
float isLeftf(vec2 P0, vec2 P1, vec2 P2 ){
	return ((P1.x - P0.x)*(P2.y - P0.y) - (P2.x - P0.x)*(P1.y - P0.y));
}
// Used to sort in pack()
vec2 p0; // Used only temporarily, should be in class
int angle_compare(const void *p1, const void *p2){
	// That element comes first that, with p0, makes the smallest angle
	// with the x-axis
	if((*((vec2 *)p1) == p0) || isLeft(p0, *((vec2 *)p1), *((vec2 *)p2))) return -1; // p1 before p2
	else return 1; // p2 before p1
}

void view::draw()
{
	mat4 MVP_inv = get_MVP_inv();
	vec4 p1 = MVP_inv * vec4(-1.0f, -1.0f, -1.0f, 1.0f); 
	vec4 p2 = MVP_inv * vec4(-1.0f, 1.0f, -1.0f, 1.0f); 
	vec4 p3 = MVP_inv * vec4(1.0f, 1.0f, -1.0f, 1.0f); 
	vec4 p4 = MVP_inv * vec4(1.0f, -1.0f, -1.0f, 1.0f); 

	vec4 p5 = MVP_inv * vec4(-1.0f, -1.0f, 1.0f, 1.0f); 
	vec4 p6 = MVP_inv * vec4(-1.0f, 1.0f, 1.0f, 1.0f); 
	vec4 p7 = MVP_inv * vec4(1.0f, 1.0f, 1.0f, 1.0f); 
	vec4 p8 = MVP_inv * vec4(1.0f, -1.0f, 1.0f, 1.0f); 

	// To disambiguate direction, draw a triangle
	// on the bottom edge of the camera-facing side.
	vec4 p0 = MVP_inv * vec4(0.0f, 0.0f, -1.0f, 1.0f);
	p1 = (1.0f/p1.w) * p1; 	p2 = (1.0f/p2.w) * p2; 
	p3 = (1.0f/p3.w) * p3; 	p4 = (1.0f/p4.w) * p4; 
	p5 = (1.0f/p5.w) * p5; 	p6 = (1.0f/p6.w) * p6; 
	p7 = (1.0f/p7.w) * p7; 	p8 = (1.0f/p8.w) * p8; 
	p0 = (1.0f/p0.w) * p0;

	// Draw the light area
	float size = 1.0; 

	glBegin(GL_LINE_LOOP);
	glVertex3fv(glm::value_ptr(pos + (size / 2.0f) * R[0] + (size / 2.0f) * R[1]));
	glVertex3fv(glm::value_ptr(pos + (size / 2.0f) * R[0] - (size / 2.0f) * R[1]));
	glVertex3fv(glm::value_ptr(pos - (size / 2.0f) * R[0] - (size / 2.0f) * R[1]));
	glVertex3fv(glm::value_ptr(pos - (size / 2.0f) * R[0] + (size / 2.0f) * R[1]));
	glEnd(); 

	// Draw a line from light that ends in a cross in the far plane
	vec4 middlebackplane = (p5+p6+p7+p8) * (1.0f/4.0f);
	glBegin(GL_LINES);
	glVertex3fv(glm::value_ptr(pos));
	glVertex3fv(glm::value_ptr(middlebackplane));
	glVertex3fv(glm::value_ptr(middlebackplane));
	glVertex3fv(glm::value_ptr(middlebackplane + 0.1f * (p5 - middlebackplane)));
	glVertex3fv(glm::value_ptr(middlebackplane));
	glVertex3fv(glm::value_ptr(middlebackplane + 0.1f * (p6 - middlebackplane)));
	glVertex3fv(glm::value_ptr(middlebackplane));
	glVertex3fv(glm::value_ptr(middlebackplane + 0.1f * (p7 - middlebackplane)));
	glVertex3fv(glm::value_ptr(middlebackplane));
	glVertex3fv(glm::value_ptr(middlebackplane + 0.1f * (p8 - middlebackplane)));
	glEnd();

	glBegin(GL_LINE_STRIP);
	// near loop
	glVertex3fv((float *)&p1); 	glVertex3fv((float *)&p2); 	
	glVertex3fv((float *)&p3); 	glVertex3fv((float *)&p4); 
	glVertex3fv((float *)&p1);
	// triangle
	glVertex3fv((float *)&p0); 	glVertex3fv((float *)&p4); 
	// far loop
	glVertex3fv((float *)&p8); 	glVertex3fv((float *)&p5); 
	glVertex3fv((float *)&p6); 	glVertex3fv((float *)&p7); 
	glVertex3fv((float *)&p8);
	// leftovers
	glVertex3fv((float *)&p5); 	glVertex3fv((float *)&p1); 
	glVertex3fv((float *)&p2); 	glVertex3fv((float *)&p6); 
	glVertex3fv((float *)&p7); 	glVertex3fv((float *)&p3); 
	glEnd(); 

	glColor3f(1.0, 0.0, 0.0); 
	glPointSize(2.0); 
	glBegin(GL_LINES); 
	glVertex3fv(&pos.x); 
	vec3 pv = pos + R[2]; 
	glVertex3fv(&pv.x);
	glEnd(); 
	glColor3f(1.0, 1.0, 1.0); 
}

void view::drawFullScreenQuad()
{	
	glm::mat4 ortho = glm::make_ortho2d(0.0f, 1.0f, 0.0f, 1.0f);

	glMatrixPushEXT(GL_PROJECTION); 
	glMatrixLoadIdentityEXT(GL_PROJECTION); 
	glMatrixLoadfEXT(GL_PROJECTION, glm::value_ptr(ortho));
	glMatrixPushEXT(GL_MODELVIEW); 
	glMatrixLoadIdentityEXT(GL_MODELVIEW); 
	glBegin(GL_QUADS); 
	glTexCoord2f(0,0); glVertex2f(0,0); 
	glTexCoord2f(1,0); glVertex2f(1,0); 
	glTexCoord2f(1,1); glVertex2f(1,1); 
	glTexCoord2f(0,1); glVertex2f(0,1); 
	glEnd(); 
	glMatrixPopEXT(GL_MODELVIEW); 
	glMatrixPopEXT(GL_PROJECTION); 
}

mat4 orthoview::get_P() const
{
	return glm::make_ortho(m_left, m_right, m_bottom, m_top, m_near, m_far);
}

mat4 orthoview::get_P_inv() const
{
	return glm::make_ortho_inv(m_left, m_right, m_bottom, m_top, m_near, m_far);
}
}  // namespace chag
