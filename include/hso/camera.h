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


#pragma once

#include <Eigen/Eigen>
#include <opencv2/opencv.hpp>

namespace hso {

using namespace std;
using namespace Eigen;

class AbstractCamera
{
protected:

	int width_;   // TODO cannot be const because of omni-camera model
	int height_;
	Eigen::Matrix3d K_;
  	Eigen::Matrix3d K_inv_;


public:

	AbstractCamera(int width, int height) : width_(width), height_(height) {};

	virtual ~AbstractCamera() {};

	/// Project from pixels to world coordiantes. Returns a bearing vector of unit length.
	virtual Vector3d cam2world(const double& x, const double& y) const = 0;

	/// Project from pixels to world coordiantes. Returns a bearing vector of unit length.
  	virtual Vector3d cam2world(const Vector2d& px) const = 0;

  	virtual Vector2d world2cam(const Vector3d& xyz_c) const = 0;

  	/// projects unit plane coordinates to camera coordinates
  	virtual Vector2d world2cam(const Vector2d& uv) const = 0;

  	virtual const Matrix3d& K() const = 0;
  	virtual const Matrix3d& K_inv() const = 0;

  	virtual double errorMultiplier2() const = 0;
  	virtual double errorMultiplier() const = 0;

  	virtual bool getUndistort() const = 0;

  	virtual void undistortImage(const cv::Mat& raw, cv::Mat& rectified) const = 0;


  	inline int width() const { return width_; }

  	inline int height() const { return height_; }

  	virtual Vector2d focal_length() const = 0;

	inline bool isInFrame(const Vector2i & obs, int boundary=0) const
	{
		if(obs[0]>=boundary && obs[0]<width()-boundary && obs[1]>=boundary && obs[1]<height()-boundary) return true;
		return false;
	}

	inline bool isInFrame(const Vector2i &obs, int boundary, int level) const
	{
		if(obs[0] >= boundary && obs[0] < width()/(1<<level)-boundary && obs[1] >= boundary && obs[1] <height()/(1<<level)-boundary) return true;
		return false;
	}

}; // AbstractCamera


class PinholeCamera : public AbstractCamera {

public:
	double fx_, fy_;
	double cx_, cy_;
	bool distortion_;             //!< is it pure pinhole model or has it radial distortion?
	double d_[5];                 //!< distortion parameters, see http://docs.opencv.org/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html
	cv::Mat cvK_, cvD_;
	cv::Mat undist_map1_, undist_map2_;
	double fxy_mean_;
	bool undistort_;

public:
  	EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  	PinholeCamera(double width, double height,
                double fx, double fy, double cx, double cy,
                double d0=0, double d1=0, double d2=0, double d3=0, double d4=0);

  	~PinholeCamera();

  	void initUnistortionMap();

  	virtual Vector3d cam2world(const double& x, const double& y) const;

  	virtual Vector3d cam2world(const Vector2d& px) const;

  	virtual Vector2d world2cam(const Vector3d& xyz_c) const;

  	virtual Vector2d world2cam(const Vector2d& uv) const;

  	virtual Vector2d focal_length() const
	{
		return Vector2d(fx_, fy_);
	}

	virtual double errorMultiplier2() const 
	{
		return fxy_mean_;
	}

	virtual double errorMultiplier() const
	{
		return fabs(4.0*fx_*fy_);
	}

	virtual const Matrix3d& K() const { return K_; };
	virtual const Matrix3d& K_inv() const { return K_inv_; };


	inline double fx() const { return fx_; };
	inline double fy() const { return fy_; };
	inline double cx() const { return cx_; };
	inline double cy() const { return cy_; };
	inline double d0() const { return d_[0]; };
	inline double d1() const { return d_[1]; };
	inline double d2() const { return d_[2]; };
	inline double d3() const { return d_[3]; };
	inline double d4() const { return d_[4]; };

	virtual bool getUndistort() const
	{ 
		return undistort_; 
	}

	virtual void undistortImage(const cv::Mat& raw, cv::Mat& rectified) const;

}; // PinholeCamera


class FOVCamera : public AbstractCamera {

public:
	double fx_, fy_;
	double cx_, cy_;
	double fxy_mean_;
	double omega_;
	cv::Mat undist_map1_, undist_map2_;
  	bool undistort_;

public:
  	EIGEN_MAKE_ALIGNED_OPERATOR_NEW

	FOVCamera(double width, double height, double fx, double fy, double cx, double cy, double omega=0, bool undistort=false);

	~FOVCamera();

	virtual Vector3d cam2world(const double& x, const double& y) const;

	virtual Vector3d cam2world(const Vector2d& px) const;

	virtual Vector2d world2cam(const Vector3d& xyz_c) const;

	virtual Vector2d world2cam(const Vector2d& uv) const;

	virtual Vector2d focal_length() const
	{
		return Vector2d(fx_, fy_);
	}

	virtual double errorMultiplier2() const 
	{
		return fxy_mean_;
	}

	virtual double errorMultiplier() const
	{
		return fabs(4.0*fx_*fy_);
	}

	virtual const Matrix3d& K() const { return K_; };
	virtual const Matrix3d& K_inv() const { return K_inv_; };



	inline double fx() const { return fx_; };
	inline double fy() const { return fy_; };
	inline double cx() const { return cx_; };
	inline double cy() const { return cy_; };
	inline double omega() const { return omega_; };

	virtual bool getUndistort() const
	{ 
		return undistort_; 
	}

	void getRemap();
	void distortPixelFOV(const Eigen::Vector2d& pixel_location, Eigen::Vector2d* distorted_pixel_location);
	virtual void undistortImage(const cv::Mat& raw, cv::Mat& rectified) const;

}; // FOVCamera


class EquidistantCamera : public AbstractCamera {

public:
	double fx_, fy_;
	double cx_, cy_;
	double fxy_mean_;
	double d_[4];
	cv::Mat undist_map1_, undist_map2_;
  	bool undistort_;

 public:
  	EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  	
  	EquidistantCamera(double width, double height, 
  					  double fx, double fy, double cx, double cy, 
  					  double k0=0, double k1=0, double k2=0, double k3=0);
  	~EquidistantCamera(); 	


  	virtual Vector3d cam2world(const double& x, const double& y) const;

	virtual Vector3d cam2world(const Vector2d& px) const;

	virtual Vector2d world2cam(const Vector3d& xyz_c) const;

	virtual Vector2d world2cam(const Vector2d& uv) const;

	virtual Vector2d focal_length() const
	{
		return Vector2d(fx_, fy_);
	}

	virtual double errorMultiplier2() const 
	{
		return fxy_mean_;
	}

	virtual double errorMultiplier() const
	{
		return fabs(4.0*fx_*fy_);
	}

	virtual const Matrix3d& K() const { return K_; };
	virtual const Matrix3d& K_inv() const { return K_inv_; };


	inline double fx() const { return fx_; };
	inline double fy() const { return fy_; };
	inline double cx() const { return cx_; };
	inline double cy() const { return cy_; };
	inline double k0() const { return d_[0]; };
	inline double k1() const { return d_[1]; };
	inline double k2() const { return d_[2]; };
	inline double k3() const { return d_[3]; };

	virtual bool getUndistort() const
	{ 
		return undistort_; 
	}

	void getRemap();
	void distortPixelEquidistant(const Eigen::Vector2d& pixel_location, Eigen::Vector2d* distorted_pixel_location);
	virtual void undistortImage(const cv::Mat& raw, cv::Mat& rectified) const;

}; // EquidistantCamera

}