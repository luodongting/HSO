// This file is part of HSO: Hybrid Sparse Monocular Visual Odometry 
// With Online Photometric Calibration
//
// Copyright(c) 2021, Dongting Luo, Dalian University of Technology, Dalian
// Copyright(c) 2021, Robotics Group, Dalian University of Technology
//
// This program is highly based on the previous implementation 
// of SVO: https://github.com/uzh-rpg/rpg_svo
// and PL-SVO: https://github.com/rubengooj/pl-svo
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.


#include "hso/camera.h"
#include <Eigen/Eigen>
#include <opencv2/opencv.hpp>

namespace hso {

PinholeCamera::PinholeCamera(double width, double height,
              				 double fx, double fy,
              				 double cx, double cy,
              				 double d0, double d1, double d2, double d3, double d4) :
							 AbstractCamera(width, height),
              				 fx_(fx), fy_(fy), cx_(cx), cy_(cy),
              				 distortion_(fabs(d0) > 0.0000001),
              				 undist_map1_(height_, width_, CV_16SC2),
              				 undist_map2_(height_, width_, CV_16SC2)
{
	d_[0] = d0; d_[1] = d1; d_[2] = d2; d_[3] = d3; d_[4] = d4;

	cvK_ = (cv::Mat_<float>(3,3) << fx_, 0.0, cx_, 0.0, fy_, cy_, 0.0, 0.0, 1.0);

	cvD_ = (cv::Mat_<float>(1,5) << d_[0], d_[1], d_[2], d_[3], d_[4]);

	cv::initUndistortRectifyMap(cvK_, 
								cvD_, 
								cv::Mat_<double>::eye(3,3), 
								cvK_, 
								cv::Size(width_, height_), 
								CV_16SC2, 
								undist_map1_, 
								undist_map2_);

	K_ << fx_, 0.0, cx_, 0.0, fy_, cy_, 0.0, 0.0, 1.0;
	K_inv_ = K_.inverse();

	fxy_mean_ = (fx*fy < 0) ? fabs(fx) : fabs((fx+fy)*0.5);

	undistort_ = false;
}

PinholeCamera::~PinholeCamera()
{}

Vector3d PinholeCamera::cam2world(const double& u, const double& v) const
{
	Vector3d xyz;
	if(!distortion_)
	{
		xyz[0] = (u - cx_)/fx_;
		xyz[1] = (v - cy_)/fy_;
		xyz[2] = 1.0;
	}
	else
	{
		cv::Point2f uv(u,v), px;
		const cv::Mat src_pt(1, 1, CV_32FC2, &uv.x);
		cv::Mat dst_pt(1, 1, CV_32FC2, &px.x);
		cv::undistortPoints(src_pt, dst_pt, cvK_, cvD_);
		xyz[0] = px.x;
		xyz[1] = px.y;
		xyz[2] = 1.0;
	}
	return xyz.normalized();
}

Vector3d PinholeCamera::cam2world (const Vector2d& uv) const
{
	return cam2world(uv[0], uv[1]);
}

Vector2d PinholeCamera::world2cam(const Vector3d& xyz) const
{
	return world2cam(Vector2d(xyz[0]/xyz[2], xyz[1]/xyz[2]));
}

Vector2d PinholeCamera::world2cam(const Vector2d& uv) const
{
	Vector2d px;
	if(!distortion_)
	{
		px[0] = fx_*uv[0] + cx_;
		px[1] = fy_*uv[1] + cy_;
	}
	else
	{
		double x, y, r2, r4, r6, a1, a2, a3, cdist, xd, yd;
		x = uv[0];
		y = uv[1];
		r2 = x*x + y*y;
		r4 = r2*r2;
		r6 = r4*r2;
		a1 = 2*x*y;
		a2 = r2 + 2*x*x;
		a3 = r2 + 2*y*y;
		cdist = 1 + d_[0]*r2 + d_[1]*r4 + d_[4]*r6;
		xd = x*cdist + d_[2]*a1 + d_[3]*a2;
		yd = y*cdist + d_[2]*a3 + d_[3]*a1;
		px[0] = xd*fx_ + cx_;
		px[1] = yd*fy_ + cy_;
	}
	return px;
}

void PinholeCamera::undistortImage(const cv::Mat& raw, cv::Mat& rectified) const
{
	assert(undistort_);
	cv::remap(raw, rectified, undist_map1_, undist_map2_, cv::INTER_LINEAR);
}


FOVCamera::FOVCamera(double width, double height, 
					 double fx, double fy, 
					 double cx, double cy, 
					 double omega, bool undistort) : 
					 AbstractCamera(width, height),
					 undist_map1_(height_, width_, CV_16SC2),
              		 undist_map2_(height_, width_, CV_16SC2)
{
	if(cx < 1.0 && cy < 1.0)
	{
		fx_ = fx*width;
		fy_ = fy*height;
		cx_ = cx*width;
		cy_ = cy*height;
	}
	else
	{
		fx_ = fx; 
		fy_ = fy; 
		cx_ = cx; 
		cy_ = cy;
	}

	omega_ = omega;
	K_ << fx_, 0.0, cx_, 0.0, fy_, cy_, 0.0, 0.0, 1.0;
	K_inv_ = K_.inverse();

	fxy_mean_ = (fx_*fy_ < 0) ? fabs(fx_) : fabs((fx_+fy_)*0.5);

	undistort_ = undistort;

	this->getRemap();
}

FOVCamera::~FOVCamera()
{}

Vector3d FOVCamera::cam2world(const double& u, const double& v) const
{
	Vector3d xyz;
	if(undistort_)
	{
		xyz[0] = (u - cx_)/fx_;
		xyz[1] = (v - cy_)/fy_;
		xyz[2] = 1.0;
	}
	else
	{
		double ud = (u - cx_)/fx_;
		double vd = (v - cy_)/fy_;
		double dist = sqrt(ud*ud+vd*vd);

		double radial_distortion = tan(dist*omega_)/(2*dist*tan(omega_/2));
		xyz[0] = radial_distortion*ud;
		xyz[1] = radial_distortion*vd;
		xyz[2] = 1.0;
	}
	return xyz.normalized();
}

Vector3d FOVCamera::cam2world (const Vector2d& uv) const
{
 	return cam2world(uv[0], uv[1]);
}

Vector2d FOVCamera::world2cam(const Vector3d& xyz) const
{
 	return world2cam(Vector2d(xyz[0]/xyz[2], xyz[1]/xyz[2]));
}

Vector2d FOVCamera::world2cam(const Vector2d& uv) const
{
	Vector2d px;
	if(undistort_)
	{
		px[0] = fx_*uv[0] + cx_;
		px[1] = fy_*uv[1] + cy_;
	}
	else
	{
		double dist = sqrt(uv[0]*uv[0]+uv[1]*uv[1]);
		double ratio = (omega_==0 || dist==0) ? 1 : atan(2*dist*tan(omega_/2))/(dist*omega_);

		px[0] = ratio*fx_*uv[0] + cx_;
		px[1] = ratio*fy_*uv[1] + cy_;
	}
    return px;
}

void FOVCamera::getRemap()
{
	cv::Size resolution(cvRound(this->width()), cvRound(this->height()));

	cv::Mat map_x_float(resolution, CV_32FC1);
    cv::Mat map_y_float(resolution, CV_32FC1);

    // Compute the remap maps
    for(int v = 0; v < resolution.height; ++v)
        for(int u = 0; u < resolution.width; ++u)
        {
            Eigen::Vector2d pixel_location(u, v);
            Eigen::Vector2d distorted_pixel_location;
            this->distortPixelFOV(pixel_location, &distorted_pixel_location);

            // Insert in map
            map_x_float.at<float>(v, u) = static_cast<float>(distorted_pixel_location.x());
            map_y_float.at<float>(v, u) = static_cast<float>(distorted_pixel_location.y());
        }

    // convert to fixed point maps for increased speed
    cv::convertMaps(map_x_float, map_y_float, undist_map1_, undist_map2_, CV_16SC2);
}

void FOVCamera::distortPixelFOV(const Eigen::Vector2d& pixel_location, Eigen::Vector2d* distorted_pixel_location)
{
	float dist = omega_;
    float d2t = 2 * tan(dist/2);

    float x = pixel_location[0];
    float y = pixel_location[1];
    float ix = (x - cx_) / fx_;
    float iy = (y - cy_) / fy_;

    float r = sqrtf(ix*ix + iy*iy);
    float fac = (r==0 || dist==0) ? 1 : atanf(r * d2t)/(dist*r);

    ix = fx_*fac*ix+cx_;
    iy = fy_*fac*iy+cy_;

    distorted_pixel_location->x() = ix;
    distorted_pixel_location->y() = iy;
}

void FOVCamera::undistortImage(const cv::Mat& raw, cv::Mat& rectified) const
{
	assert(undistort_);
	cv::remap(raw, rectified, undist_map1_, undist_map2_, cv::INTER_LINEAR);
}


EquidistantCamera::EquidistantCamera(double width, double height, 
  					  				 double fx, double fy, double cx, double cy, 
  					  				 double k0, double k1, double k2, double k3) : 
					  				 AbstractCamera(width, height),
					  				 fx_(fx), fy_(fy), cx_(cx), cy_(cy),
					  				 undist_map1_(height_, width_, CV_16SC2),
              		  				 undist_map2_(height_, width_, CV_16SC2)
{
	d_[0] = k0; d_[1] = k1; d_[2] = k2; d_[3] = k3;

	K_ << fx_, 0.0, cx_, 0.0, fy_, cy_, 0.0, 0.0, 1.0;
	K_inv_ = K_.inverse();

	fxy_mean_ = (fx_*fy_ < 0) ? fabs(fx_) : fabs((fx_+fy_)*0.5);

	undistort_ = true;

	this->getRemap();
}

EquidistantCamera::~EquidistantCamera()
{}

Vector3d EquidistantCamera::cam2world(const double& u, const double& v) const
{
	return Vector3d((u - cx_)/fx_, (v - cy_)/fy_, 1.0).normalized();
}

Vector3d EquidistantCamera::cam2world (const Vector2d& uv) const
{
 	return cam2world(uv[0], uv[1]);
}

Vector2d EquidistantCamera::world2cam(const Vector3d& xyz) const
{
 	return world2cam(Vector2d(xyz[0]/xyz[2], xyz[1]/xyz[2]));
}

Vector2d EquidistantCamera::world2cam(const Vector2d& uv) const
{
	return Vector2d(fx_*uv[0] + cx_, fy_*uv[1] + cy_);
}

void EquidistantCamera::getRemap()
{
	const cv::Size resolution(cvRound(this->width()), cvRound(this->height()));

    // Initialize maps
    cv::Mat map_x_float(resolution, CV_32FC1);
    cv::Mat map_y_float(resolution, CV_32FC1);

    // Compute the remap maps
    for(int v = 0; v < resolution.height; ++v)
        for(int u = 0; u < resolution.width; ++u)
        {
            Eigen::Vector2d pixel_location(u, v);
            Eigen::Vector2d distorted_pixel_location;
            this->distortPixelEquidistant(pixel_location, &distorted_pixel_location);

            // Insert in map
            map_x_float.at<float>(v, u) = static_cast<float>(distorted_pixel_location.x());
            map_y_float.at<float>(v, u) = static_cast<float>(distorted_pixel_location.y());
        }

    // convert to fixed point maps for increased speed
    cv::convertMaps(map_x_float, map_y_float, undist_map1_, undist_map2_, CV_16SC2);
}

void EquidistantCamera::distortPixelEquidistant(const Eigen::Vector2d& pixel_location, 
												Eigen::Vector2d* distorted_pixel_location)
{
	float x = pixel_location[0];
    float y = pixel_location[1];

    // EQUI
    float ix = (x - cx_) / fx_;
    float iy = (y - cy_) / fy_;
    float r = sqrt(ix * ix + iy * iy);
    float theta = atan(r);
    float theta2 = theta * theta;
    float theta4 = theta2 * theta2;
    float theta6 = theta4 * theta2;
    float theta8 = theta4 * theta4;
    float thetad = theta * (1 + d_[0] * theta2 + d_[1] * theta4 + d_[2] * theta6 + d_[3] * theta8);
    float scaling = (r > 1e-8) ? thetad / r : 1.0;
    float ox = fx_*ix*scaling + cx_;
    float oy = fy_*iy*scaling + cy_;

    distorted_pixel_location->x() = ox;
    distorted_pixel_location->y() = oy;
}

void EquidistantCamera::undistortImage(const cv::Mat& raw, cv::Mat& rectified) const
{
	assert(undistort_);
	cv::remap(raw, rectified, undist_map1_, undist_map2_, cv::INTER_LINEAR);
}

}