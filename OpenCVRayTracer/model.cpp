#include <iostream>
#include <string>
#include <fstream>
#include <sstream>
#include <vector>
#include <algorithm>
#include "model.h"

using namespace cv;

Model::Model(const char *filename) : verts_(), faces_() {
	std::ifstream in;
	in.open(filename, std::ifstream::in);
	if (in.fail()) return;
	std::string line;
	while (!in.eof()) {
		std::getline(in, line);
		std::istringstream iss(line.c_str());
		char trash;
		if (!line.compare(0, 2, "v ")) {
			iss >> trash;
			Vec3f v;
			for (int i = 0; i < 3; i++) iss >> v[i];
			verts_.push_back(v);
		}
		else if (!line.compare(0, 2, "f ")) {
			Vec3i f;
			iss >> trash;
			for (int i = 0; i < 3; i++) iss >> f[i];
			f -= Vec3f(1, 1, 1);
			faces_.push_back(f);
		}
	}
	std::cerr << "# v# " << verts_.size() << " f# " << faces_.size() << std::endl;

	get_bbox(mincorner, maxcorner);
}

Model::~Model() {
}

int Model::nverts() {
    return (int)verts_.size();
}

int Model::nfaces() {
    return (int)faces_.size();
}

Vec3i Model::face(int idx) {
    return faces_[idx];
}

Vec3f Model::vert(int i) {
    return verts_[i];
}

Vec3f Model::vert(int iface, int nthvert)
{
	return verts_[faces_[iface][nthvert]];
}

void Model::get_bbox(Vec3f &mincorner, Vec3f &maxcorner) {
	mincorner = maxcorner = verts_[0];
	for (int i = 1; i < (int)verts_.size(); ++i) {
		for (int j = 0; j < 3; j++) {
			mincorner[j] = std::min(mincorner[j], verts_[i][j]);
			maxcorner[j] = std::max(maxcorner[j], verts_[i][j]);
		}
	}
	std::cerr << "bbox: [" << mincorner << " : " << maxcorner << "]" << std::endl;
}

bool Model::ray_triangle_intersect(const int &fi, const Vec3f &orig, const Vec3f &dir, float &tnear) {
	Vec3f edge1 = vert(fi, 1) - vert(fi, 0);
	Vec3f edge2 = vert(fi, 2) - vert(fi, 0);
	Vec3f pvec = dir.cross(edge2);
	float det = edge1 .dot(pvec);
	if (det < 1e-5) return false;

	Vec3f tvec = orig - vert(fi, 0);
	float u = tvec .dot(pvec);
	if (u < 0 || u > det) return false;

	Vec3f qvec = tvec.cross(edge1);
	float v = dir.dot(qvec);
	if (v < 0 || u + v > det) return false;

	tnear = edge2.dot(qvec) * (1. / det);
	return tnear > 1e-5;
}

bool Model::ray_bbox_intersect(const Vec3f &orig, const Vec3f &dir) {
	float tmin = -std::numeric_limits<float>::max();
	float tmax = std::numeric_limits<float>::max();
	for (int i = 0; i < 3; i++){
		Vec3f normal = Vec3f(0, 0, 0);
		normal[i] = 1;
		float t1 = -(-mincorner[i] + normal.dot(orig)) / (normal.dot(dir));
		float t2 = -(-maxcorner[i] + normal.dot(orig)) / (normal.dot(dir));
		t1 > t2 ? tmax = min(t1, tmax), tmin = min(t2, tmin) : tmax = min(t2, tmax), tmin = min(t1, tmin);
	}
	return tmin < tmax;
}