#include <iostream>
#include <string>
#include <fstream>
#include <sstream>
#include <vector>
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
		else if (!line.compare(0, 3, "vn ")) {
			iss >> trash >> trash;
			Vec3f n;
			for (int i = 0; i < 3; i++) iss >> n[i];
			norms_.push_back(n);
		}
		else if (!line.compare(0, 3, "vt ")) {
			iss >> trash >> trash;
			Vec2f uv;
			for (int i = 0; i < 2; i++) iss >> uv[i];
			uv_.push_back(uv);
		}
		else if (!line.compare(0, 2, "f ")) {
			std::vector<Vec3i> f;
			Vec3i tmp;
			iss >> trash;
			while (iss >> tmp[0] >> trash >> tmp[1] >> trash >> tmp[2]) {
				for (int i = 0; i < 3; i++) tmp[i]--; // in wavefront obj all indices start at 1, not zero
				f.push_back(tmp);
			}
			faces_.push_back(f);
		}
	}
	std::cerr << "# v# " << verts_.size() << " f# " << faces_.size() << " vt# " << uv_.size() << " vn# " << norms_.size() << std::endl;
	load_texture(filename, "_diffuse.bmp", diffusemap_);
	load_texture(filename, "_nm_tangent.bmp", normalmap_);
	load_texture(filename, "_spec.bmp", specularmap_);
}

Model::~Model() {
}

int Model::nverts() {
    return (int)verts_.size();
}

int Model::nfaces() {
    return (int)faces_.size();
}

std::vector<Vec3i> Model::face(int idx) {
    return faces_[idx];
}

Vec3f Model::vert(int i) {
    return verts_[i];
}

Vec3f Model::vert(int iface, int nthvert)
{
	return verts_[faces_[iface][nthvert][0]];
}

Vec3f Model::normal(int iface, int nthvert) {
	int idx = faces_[iface][nthvert][2];
	return normalize(norms_[idx]);
}

Vec2f Model::uv(int iface, int nthvert) {
	return uv_[faces_[iface][nthvert][1]];
}

Scalar Model::diffuse(Vec2f uvf) {
	Vec2i uv(uvf[0] * diffusemap_.cols, uvf[1] * diffusemap_.rows);
	Scalar ans = diffusemap_.at<Vec3b>(uv[1], uv[0]);
	return ans;
}

Vec3f Model::normal(Vec2f uvf) {
	Vec2i uv(uvf[0] * normalmap_.cols, uvf[1] * normalmap_.rows);
	Scalar c = normalmap_.at<Vec3b>(uv[1], uv[0]);
	Vec3f res;
	for (int i = 0; i < 3; i++)
		res[2 - i] = (float)c[i] / 255.f*2.f - 1.f;
	return res;
}

float Model::specular(Vec2f uvf) {
	Vec2i uv(uvf[0] * specularmap_.cols, uvf[1] * specularmap_.rows);
	return specularmap_.at<Vec3b>(uv[0], uv[1])[0] / 1.f;
}

void Model::load_texture(std::string filename, const char *suffix, Mat &img) {
	std::string texfile(filename);
	size_t dot = texfile.find_last_of(".");
	if (dot != std::string::npos) {
		texfile = texfile.substr(0, dot) + std::string(suffix);
		img = imread(texfile.c_str());
		std::cerr << "texture file " << texfile << " loading " <<  (img.empty() ? "failed" : "ok")<< std::endl;
		flip(img, img, 0);
	}
}

