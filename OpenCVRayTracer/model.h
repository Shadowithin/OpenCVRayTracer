#pragma once
#ifndef __MODEL_H__
#define __MODEL_H__

#include <vector>
#include <opencv.hpp>

class Model {
private:
	std::vector<cv::Vec3f> verts_;
	std::vector<cv::Vec3i> faces_;
	cv::Vec3f mincorner, maxcorner;
public:
	Model(const char *filename);
	~Model();
	int nverts();
	int nfaces();
	cv::Vec3f vert(int i);
	cv::Vec3f vert(int iface, int nthvert);
	cv::Vec3i face(int idx);

	void get_bbox(cv::Vec3f &mincorner, cv::Vec3f &maxcorner);
	bool ray_bbox_intersect(const cv::Vec3f &orig, const cv::Vec3f &dir);
	bool ray_triangle_intersect(const int &fi, const cv::Vec3f &orig, const cv::Vec3f &dir, float &tnear);
};

#endif //__MODEL_H__
