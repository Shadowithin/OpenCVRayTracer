#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#include <iostream>
#include <vector>
#include <cmath>
#include <limits>
#include <opencv.hpp>

using namespace std;
using namespace cv;

const int width = 1024;
const int height = 768;
const int fov = M_PI/ 2.;

Mat background;
int bkwidth;
int bkheight;

struct Light {
	Light(const Vec3f &p, const float &i, const Vec3f &c) : position(p), intensity(i), color(c){}
	Vec3f color;
	Vec3f position;
	float intensity;
};

struct Material {
	Material(const float & r, const Vec4f &a, const Vec3f &color, const float &spec) : refractive_index(r), albedo(a), diffuse_color(color), specular_exponent(spec) {}
	Material() : albedo(1, 0), diffuse_color(), specular_exponent() {}

	float refractive_index;
	Vec4f albedo;
	Vec3f diffuse_color;
	float specular_exponent;
};

struct Sphere {
	Vec3f center;
	float radius;
	Material material;

	Sphere(const Vec3f &c, const float &r, const Material &m) : center(c), radius(r), material(m) {}

	bool ray_intersect(const Vec3f &orig, const Vec3f &dir, float &t0) const {
		Vec3f L = center - orig;
		float tca = L .dot(dir);
		float d2 = L.dot(L) - tca * tca;
		if (d2 > radius*radius) return false;
		float thc = sqrtf(radius*radius - d2);
		t0 = tca - thc;
		float t1 = tca + thc;
		if (t0 < 0) t0 = t1;
		if (t0 < 0) return false;
		return true;
	}
};

Vec3f reflect(const Vec3f &l, const Vec3f &n) {
	Vec3f L = normalize(l);
	Vec3f N = normalize(n);
	return L - 2 * N*(N.dot(L)) ;
}

Vec3f refract(const Vec3f &I, const Vec3f &N, const float &refractive_index) { // Snell's law
	float cosi = -max(-1.f, min(1.f, I.dot(N)));
	float etai = 1, etat = refractive_index;
	Vec3f n = N;
	if (cosi < 0) { 
		cosi = -cosi;
		std::swap(etai, etat); n = -N;
	}
	float eta = etai / etat;
	float k = 1 - eta * eta*(1 - cosi * cosi);
	return k < 0 ? Vec3f(0, 0, 0) : I * eta + n * (eta * cosi - sqrtf(k));
}

bool scene_intersect(const Vec3f &orig, const Vec3f &dir, const std::vector<Sphere> &spheres, Vec3f &hit, Vec3f &N, Material &material) {
	float spheres_dist = std::numeric_limits<float>::max();
	for (auto &sphere : spheres) {
		float dist_i;
		if (sphere.ray_intersect(orig, dir, dist_i) && dist_i < spheres_dist) {
			spheres_dist = dist_i;
			hit = orig + dir * dist_i;
			N = normalize(hit - sphere.center);
			material = sphere.material;
		}
	}

	float floor_dist = std::numeric_limits<float>::max();
	if (fabs(dir[1]) > 0) {
		float d = -(orig[1] + 4) / dir[1];
		Vec3f pt = orig + dir*d;
		if (d>0 && d < spheres_dist && fabs(pt[0]) < 10 && pt[2]<-10 && pt[2]>-30) {
			floor_dist = d;
			hit = pt;
			N = Vec3f(0, 1, 0);
			material.diffuse_color = (int(.5*hit[0]+10 ) + int(.5*hit[2])) & 1 ? Vec3f(.3, .3, .3) : Vec3f(.3, .2, .1);
			return floor_dist < 1000;
		}
	}

	return spheres_dist < 1000;
}

Vec3f cast_ray(const Vec3f &orig, const Vec3f &dir, const vector<Sphere> &spheres, const vector<Light> &lights, size_t depth=0) {
	Vec3f point, N;
	Material material;
	
	if (depth > 4 || !scene_intersect(orig, dir, spheres, point, N, material)) {
		int x = max(min(bkwidth - 1, int((atan2(dir[2], dir[0]) / (2 * M_PI) + 0.5) * bkwidth)), 0);
		int y = max(min(bkheight - 1, int(acos(dir[1]) / M_PI * bkheight)), 0);
		//return Vec3f(0.8, 0.7, 0.2); 
		return background.at<Vec3f>(y, x);
	}

	Vec3f reflect_dir = normalize(reflect(dir, N));
	Vec3f reflect_orig = point + N * 1e-3;
	Vec3f reflect_color = cast_ray(reflect_orig, reflect_dir, spheres, lights, depth + 1);

	Vec3f refract_dir = normalize(refract(dir, N, material.refractive_index));
	Vec3f refract_orig = refract_dir.dot(N) > 0 ? point + N * 1e-3 : point - N * 1e-3;
	Vec3f refract_color = cast_ray(refract_orig, refract_dir, spheres, lights, depth + 1);

	float diffuse_light_intensity = 0;
	Vec3f specular_light_intensity = Vec3f(0, 0, 0);
	for (auto &light : lights){
		Vec3f light_dir = normalize(light.position - point);
		Vec3f shadow_orig = point + N * 1e-3;
		Vec3f point_shadow, N_shadow;
		Material material_shadow;
		if (scene_intersect(shadow_orig, light_dir, spheres, point_shadow, N_shadow, material_shadow)) continue;
		diffuse_light_intensity += light.intensity * max(light_dir.dot(N), 0.f);
		specular_light_intensity += light.color * powf(max(0.f, -dir.dot(reflect(-light_dir, N))), material.specular_exponent);
	}

	return material.diffuse_color * diffuse_light_intensity * material.albedo[0] + specular_light_intensity * material.albedo[1]  + reflect_color * material.albedo[2] + refract_color * material.albedo[3];
}

void render(const vector<Sphere> &spheres, const vector<Light> &lights, Mat &frame) {
#pragma omp parallel for
	for (int j = 0; j < height; j++) {
		for (int i = 0; i < width; i++) {
			float x = (2 * (i + 0.5) / (float)width - 1)*tan(fov / 2.) * width / (float)height;
			float y = -(2 * (j + 0.5) / (float)height - 1)*tan(fov / 2.);
			Vec3f dir = normalize(Vec3f(x, y, -1));
			frame.at<Vec4b>(j, i) = Scalar(cast_ray(Vec3f(0, 0, 0), dir, spheres, lights)) * 255;
		}
	}
}

int main()
{
	Material            ivory(1.0, Vec4f(0.6, 0.3, 0.1, 0.0), Vec3f(0.3, 0.4, 0.4), 50.);
	Material       red_rubber(1.0, Vec4f(0.9, 0.1, 0.0, 0.0), Vec3f(0.1, 0.1, 0.3), 10.);
	Material        mirror(1.0, Vec4f(0.0, 10.0, 0.8, 0.0), Vec3f(1.0, 1.0, 1.0), 1425.);
	Material           glass(1.5, Vec4f(0.0, 0.5, 0.1, 0.8), Vec3f(1.0, 1.0, 1.0), 125.);

	Mat frame = Mat::zeros(height, width, CV_8UC4);
	background = imread("envmap.jpg", IMREAD_COLOR);
	background.convertTo(background, CV_32FC3);
	background = background / 255;
	bkwidth = background.cols;
	bkheight = background.rows;

	vector<Sphere> spheres;
    spheres.push_back(Sphere(Vec3f(-3,    0,   -16), 2, ivory));
    spheres.push_back(Sphere(Vec3f(-1.0, -1.5, -12), 2, glass));
    spheres.push_back(Sphere(Vec3f( 1.5, -0.5, -18), 3, red_rubber));
    spheres.push_back(Sphere(Vec3f( 7,    5,   -18), 4, mirror));

	vector<Light> lights;
	lights.push_back(Light(Vec3f(-20, 20, 20), 1.5, Vec3f(1.0, 1.0, 1.0)));
	lights.push_back(Light(Vec3f(30, 50, -25), 1.8, Vec3f(1.0, 1.0, 1.0)));
	lights.push_back(Light(Vec3f(30, 20, 30), 1.7, Vec3f(1.0, 1.0, 1.0)));

	render(spheres, lights, frame);

	//flip(frame, frame, 0);
	imshow("frame", frame);
	imwrite("frame.jpg", frame);
	waitKey(0);
	system("pause");
	return 0;
}

#endif