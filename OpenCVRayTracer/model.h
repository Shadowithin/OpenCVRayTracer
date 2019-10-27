#pragma once
#ifndef __MODEL_H__
#define __MODEL_H__

#include <vector>
#include <opencv.hpp>

class Model {
private:
	std::vector<cv::Vec3f> verts_;
	std::vector<cv::Vec3f> norms_;
	std::vector<cv::Vec2f> uv_;
	std::vector<std::vector<cv::Vec3i>> faces_;

	cv::Mat diffusemap_;
	cv::Mat normalmap_;
	cv::Mat specularmap_;

	void load_texture(std::string filename, const char *suffix, cv::Mat &img);

public:
	Model(const char *filename);
	~Model();
	int nverts();
	int nfaces();
	cv::Vec3f vert(int i);
	cv::Vec3f vert(int iface, int nthvert);
	cv::Vec3f normal(cv::Vec2f uv);
	cv::Vec3f normal(int iface, int nthvert);
	cv::Vec2f uv(int iface, int nthvert);
	cv::Scalar diffuse(cv::Vec2f uv);
	float specular(cv::Vec2f uv);
	std::vector<cv::Vec3i> face(int idx);
};

#endif //__MODEL_H__
