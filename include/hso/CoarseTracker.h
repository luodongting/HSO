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

#include "hso/global.h"
#include "hso/MatrixAccumulator.h"

namespace hso {

struct Feature;

class CoarseTracker
{
	typedef Eigen::Matrix<double,7,7> Matrix7d;
	typedef Eigen::Matrix<double,7,1> Vector7d;
	typedef Eigen::Matrix<double,6,6> Matrix6d;
	typedef Eigen::Matrix<double,6,1> Vector6d;

	int pattern_0[13][2] = {
		{0,-2}, {-1,-1}, {0,-1}, {1,-1}, {-2,0}, {-1,0}, {0,0}, {1,0}, {2,0}, {-1,1}, {0,1}, {1,1}, {0,2}
	};

	int pattern_L[13][2] = {
		{-2,-2}, {0,-2}, {2,-2}, {-1,-1}, {1,-1}, {-2,0}, {0,0}, {2,0}, {-1,1}, {1,1}, {-2,2}, {0,2}, {2,2}
	};


	// int pattern_0[9][2] = { {-1,-1}, {0,-1}, {1,-1}, {-1,0}, {0,0}, {1,0}, {-1,1}, {0,1}, {1,1} }; // top ~ top-1
	// int pattern_L[9][2] = { {0,-2}, {-1,-1}, {1,-1}, {-2,0}, {0,0},	{2,0}, {-1,1}, {0,2}, {1,1} };

	// fixed pattern
	int PATCH_AREA = 13; 
	int HALF_PATCH_SIZE = 2;



	int staticPattern[8][40][2] = {
		{{0,0}, 	  {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100},	// .
		 {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100},
		 {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100},
		 {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}},

		{{0,-1},	  {-1,0},	   {0,0},	    {1,0},	     {0,1}, 	  {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100},	// +
		 {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100},
		 {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100},
		 {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}},

		{{-1,-1},	  {-1,0},	   {-1,1},		{-1,0},		 {0,0},		  {0,1},	   {1,-1},		{1,0},		 {1,1},       {-100,-100},	// full-tight-9
		 {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100},
		 {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100},
		 {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}},

		{{0,-2},	  {-1,-1},	   {1,-1},		{-2,0},		 {0,0},		  {2,0},	   {-1,1},		{1,1},		 {0,2},       {0,-1},	    // full-tight-13
		 {-1,0},      {1,0},       {0,1},       {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100},
		 {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100},
		 {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}},

		{{0,-2},	  {-1,-1},	   {1,-1},		{-2,0},		 {0,0},		  {2,0},	   {-1,1},		{1,1},		 {0,2},       {-2,-2},   // full-spread-13
		 {-2,2},      {2,-2},      {2,2},       {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100},
		 {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100},
		 {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}},

		 {{0,-2},	  {-1,-1},	   {1,-1},		{-2,0},		 {0,0},		  {2,0},	   {-1,1},		{1,1},		 {0,2},       {-2,-2},   // full-spread-21
		 {-2,2},      {2,-2},      {2,2},       {-3,-1},     {-3,1},      {3,-1}, 	   {3,1},       {1,-3},      {-1,-3},     {1,3},
		 {-1,3},      {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100},
		 {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}},

		{{-2,-2},     {-2,-1}, {-2,-0}, {-2,1}, {-2,2}, {-1,-2}, {-1,-1}, {-1,-0}, {-1,1}, {-1,2}, 										// full-25
		 {-0,-2},     {-0,-1}, {-0,-0}, {-0,1}, {-0,2}, {+1,-2}, {+1,-1}, {+1,-0}, {+1,1}, {+1,2},
		 {+2,-2}, 	  {+2,-1}, {+2,-0}, {+2,1}, {+2,2}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100},
		 {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}},

		{{-4,-4},     {-4,-2}, {-4,-0}, {-4,2}, {-4,4}, {-2,-4}, {-2,-2}, {-2,-0}, {-2,2}, {-2,4}, 										// full-45-SPREAD
		 {-0,-4},     {-0,-2}, {-0,-0}, {-0,2}, {-0,4}, {+2,-4}, {+2,-2}, {+2,-0}, {+2,2}, {+2,4},
		 {+4,-4}, 	  {+4,-2}, {+4,-0}, {+4,2}, {+4,4}, {-200,-200}, {-200,-200}, {-200,-200}, {-200,-200}, {-200,-200},
		 {-200,-200}, {-200,-200}, {-200,-200}, {-200,-200}, {-200,-200}, {-200,-200}, {-200,-200}, {-200,-200}, {-200,-200}, {-200,-200}},
};

	int staticPatternNum[8] = {
		1,
		5,
		9,
		13,
		13,
		21,
		25,
		25
}; 

	int staticPatternPadding[8] = {
		1,
		1,
		1,
		2,
		2,
		3,
		2,
		4
};

	int m_pattern_offset = 2;

	int m_offset_all = 0;





	
public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

	CoarseTracker(bool inverse_composition, int max_level, int min_level, int n_iter, bool verbose);
	~CoarseTracker();
	bool m_inverse_composition;
	int m_max_level, m_min_level;
	int m_n_iter;
	bool m_verbose;

	size_t run(FramePtr ref_frame, FramePtr cur_frame);
	FramePtr m_ref_frame, m_cur_frame;
	SE3 m_T_cur_ref;

private:
	double computeResiduals(const SE3& T_cur_ref, float exposure_rat, double cutoff_error, float b=0);


	void computeGS(Matrix7d& H_out, Vector7d& b_out);
	void computeGS(Matrix6d& H_out, Vector6d& b_out);

	void precomputeReferencePatches();

	void selectRobustFunctionLevel(const SE3& T_cur_ref, float exposure_rat, float b=0);

	void makeDepthRef();

	Vector2f lineFit(vector<float>& X, vector<float>& Y, float a, float b=0);

	Accumulator7 m_acc7;

	float m_exposure_rat, m_b;

	int m_level;

	int m_iter;

	int m_total_terms, m_saturated_terms;

	std::vector<bool> m_visible_fts;
	cv::Mat m_ref_patch_cache;

	Matrix<double, 6, Dynamic, ColMajor> m_jacobian_cache_true;
    Matrix<double, 6, Dynamic, ColMajor> m_jacobian_cache_raw;

    std::vector<Vector7d> m_buf_jacobian;
    // std::vector<Vector6d> m_buf_jacobian;
    std::vector<double> m_buf_weight;
    std::vector<double> m_buf_error;

    std::vector<double> m_pt_ref;
    std::vector<Vector3d> m_pt_host;


    float m_huber_thresh;
    float m_outlier_thresh;


    std::vector<float> m_color_cur;
    std::vector<float> m_color_ref;

};
} // namespace hso