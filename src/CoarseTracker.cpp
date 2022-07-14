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



#include "hso/CoarseTracker.h"
#include "hso/frame.h"
#include "hso/feature.h"
#include "hso/point.h"

#include "hso/vikit/math_utils.h"

namespace hso {

CoarseTracker::CoarseTracker(bool inverse_composition, int max_level, int min_level, int n_iter, bool verbose): 
	m_inverse_composition(inverse_composition), 
    m_max_level(max_level), 
    m_min_level(min_level), 
    m_n_iter(n_iter), 
    m_verbose(verbose),
    m_iter(0), 
    m_total_terms(0), 
    m_saturated_terms(0)
{}

CoarseTracker::~CoarseTracker()
{

}

size_t CoarseTracker::run(FramePtr ref_frame, FramePtr cur_frame)
{
	if(ref_frame->fts_.empty())
		return 0;

	m_ref_frame = ref_frame;
	m_cur_frame = cur_frame;

	//[TODO]
	m_exposure_rat = m_cur_frame->integralImage_ / m_ref_frame->integralImage_;
    m_b = 0;

	m_T_cur_ref = m_cur_frame->T_f_w_ * m_ref_frame->T_f_w_.inverse();

	// m_ref_patch_cache = cv::Mat(m_ref_frame->fts_.size(), PATCH_AREA, CV_32F);
	// m_visible_fts.resize(m_ref_frame->fts_.size(), false);
	// m_jacobian_cache_true.resize(Eigen::NoChange, m_ref_patch_cache.rows*PATCH_AREA);
	// m_jacobian_cache_raw.resize(Eigen::NoChange, m_ref_patch_cache.rows*PATCH_AREA);

    makeDepthRef();



	for(m_level = m_max_level; m_level >= m_min_level; --m_level)
	{
		//reset
		std::fill(m_visible_fts.begin(), m_visible_fts.end(), false);


        m_offset_all    = m_max_level-m_level+m_pattern_offset;
        HALF_PATCH_SIZE = staticPatternPadding[m_offset_all];
        PATCH_AREA      = staticPatternNum[m_offset_all];

        m_ref_patch_cache = cv::Mat(m_ref_frame->fts_.size(), PATCH_AREA, CV_32F);
        m_visible_fts.resize(m_ref_frame->fts_.size(), false);
        m_jacobian_cache_true.resize(Eigen::NoChange, m_ref_patch_cache.rows*PATCH_AREA);
        m_jacobian_cache_raw.resize(Eigen::NoChange, m_ref_patch_cache.rows*PATCH_AREA);

        

        precomputeReferencePatches();
        selectRobustFunctionLevel(m_T_cur_ref, m_exposure_rat);

        // if(m_verbose)
        // {
        //     printf("\nPYRAMID LEVEL %i\n---------------\n", m_level);
        //     cout << "Huber = " << m_huber_thresh << "\tOutliers = " << m_outlier_thresh << endl;
        // }

        const double cutoff_error = m_outlier_thresh;

		double energy_old = computeResiduals(m_T_cur_ref, m_exposure_rat, cutoff_error);

		Matrix7d H; Vector7d b;
        // Matrix6d H; Vector6d b;
        computeGS(H,b);

		float lambda = 0.1;

		for(m_iter=0; m_iter < m_n_iter; m_iter++)
		{
			Matrix7d Hl = H;
			for(int i=0;i<7;i++) Hl(i,i) *= (1+lambda);
			Vector7d step = Hl.ldlt().solve(b);

            // Matrix6d Hl = H;
            // for(int i=0;i<6;i++) Hl(i,i) *= (1+lambda);
            // Vector6d step = Hl.ldlt().solve(b);

			float extrap_fac = 1;
			if(lambda < 0.001) extrap_fac = sqrt(sqrt(0.001 / lambda));
			step *= extrap_fac;

			if(!std::isfinite(step.sum()) || std::isnan(step[0])) step.setZero();

            float new_exposure_rat = m_exposure_rat + step[0];
            // Vector2f aff = lineFit(m_color_cur, m_color_ref, m_exposure_rat, m_b);

			SE3 new_T_cur_ref;
			if(!m_inverse_composition)
				new_T_cur_ref = Sophus::SE3::exp(-step.segment<6>(1))*m_T_cur_ref;
			else
				new_T_cur_ref = m_T_cur_ref*Sophus::SE3::exp(-step.segment<6>(1));

            // SE3 new_T_cur_ref;
            // if(!m_inverse_composition)
            //     new_T_cur_ref = Sophus::SE3::exp(-step)*m_T_cur_ref;
            // else
            //     new_T_cur_ref = m_T_cur_ref*Sophus::SE3::exp(-step);

			double energy_new = computeResiduals(new_T_cur_ref, new_exposure_rat, cutoff_error);

			if(energy_new < energy_old)
			{
				if(m_verbose)
		        {
		          cout << "It. " << m_iter
		               << "\t Success"
		               << "\t n_meas = " << m_total_terms
		               << "\t rejected = " << m_saturated_terms
		               << "\t new_chi2 = " << energy_new
		               << "\t exposure = " << new_exposure_rat
		               << "\t mu = " << lambda
		               << endl;
		        }

				computeGS(H,b);
				energy_old = energy_new;

				m_exposure_rat = new_exposure_rat;
                // m_b = aff[1];

				m_T_cur_ref = new_T_cur_ref;

                lambda *= 0.5;
                
			}
			else
			{
				if(m_verbose)
		        {
                    cout << "It. " << m_iter
                         << "\t Failure"
                         << "\t n_meas = " << m_total_terms
                         << "\t rejected = " << m_saturated_terms
                         << "\t new_chi2 = " << energy_new
                         << "\t exposure = " << new_exposure_rat
                         << "\t mu = " << lambda
                         << endl;
		        }

                lambda *= 4;
				if(lambda < 0.001) lambda = 0.001;
			}

            // double eps = m_level == m_max_level ? 1e-5 : 1e-4;

            if(!(step.norm() > 1e-4))
            {
                if(m_verbose)
                    printf("inc too small, break!\n");
                break;
            }
		}
	}


	m_cur_frame->T_f_w_ = m_T_cur_ref * m_ref_frame->T_f_w_;

    m_cur_frame->m_exposure_time = m_exposure_rat*m_ref_frame->m_exposure_time;

    if(m_exposure_rat > 0.99 && m_exposure_rat < 1.01) m_cur_frame->m_exposure_time = m_ref_frame->m_exposure_time;




    return float(m_total_terms) / PATCH_AREA;
}

void CoarseTracker::makeDepthRef()
{
    m_pt_ref.resize(m_ref_frame->fts_.size(), -1);

    size_t feature_counter = 0;
    for(auto it_ft=m_ref_frame->fts_.begin(); it_ft!=m_ref_frame->fts_.end(); ++it_ft, ++feature_counter)
    {
        if((*it_ft)->point == NULL) continue;

        Vector3d p_host = (*it_ft)->point->hostFeature_->f * (1.0/(*it_ft)->point->idist_);
        SE3 T_r_h = m_ref_frame->T_f_w_ * (*it_ft)->point->hostFeature_->frame->T_f_w_.inverse();
        Vector3d p_ref = T_r_h*p_host;
        if(p_ref[2] < 0.00001) continue;

        // double dist = p_ref.norm();
        // double dist2 = ((*it_ft)->point->pos_ - m_ref_frame->pos()).norm();
        // assert(fabsf(dist-dist2) < 0.01);


        // Vector3d f1 = p_ref.normalized();
        // double cosf1f2 = f1.dot((*it_ft)->f);
        // double dist = cosf1f2*p_ref.norm();
        // assert(dist > 0);


        m_pt_ref[feature_counter] = p_ref.norm();

        // if((*it_ft)->type == Feature::CORNER)
        //     m_nr_corner_inliers++;
    }
}

double CoarseTracker::computeResiduals(const SE3& T_cur_ref, float exposure_rat, double cutoff_error, float b)
{
	if(m_inverse_composition)
		m_jacobian_cache_true = exposure_rat*m_jacobian_cache_raw;

	const cv::Mat& cur_img = m_cur_frame->img_pyr_.at(m_level);
	const int stride = cur_img.cols;
    const int border = HALF_PATCH_SIZE+1;
    const float scale = 1.0f/(1<<m_level);
    const Vector3d ref_pos(m_ref_frame->pos());
    const double fxl = m_ref_frame->cam_->focal_length().x()*scale;
    const double fyl = m_ref_frame->cam_->focal_length().y()*scale;


    float setting_huberTH = m_huber_thresh;
    // if(m_level == m_max_level && m_iter == 0) setting_huberTH *= 2;


    const float max_energy = 2*setting_huberTH*cutoff_error-setting_huberTH*setting_huberTH;


    const int pattern_offset = m_offset_all;


    // reset
    m_buf_jacobian.clear();
    m_buf_weight.clear();
    m_buf_error.clear();
    m_total_terms = m_saturated_terms = 0;

    float E = 0;
    //[TODO] debug affine

    // HSO_2020
    m_color_cur.clear(); m_color_ref.clear();

    size_t feature_counter = 0;
    std::vector<bool>::iterator visiblity_it = m_visible_fts.begin();
    for(auto it_ft=m_ref_frame->fts_.begin(); it_ft!=m_ref_frame->fts_.end(); ++it_ft, ++feature_counter, ++visiblity_it)
    {
    	if(!*visiblity_it) continue;
        
        // if(m_edge_reject == EdgeRejectLevel::PARTLY && (*it_ft)->type == Feature::EDGELET && m_level == m_max_level)
        //     continue;

    	// double depth = ((*it_ft)->point->pos_ - ref_pos).norm();
     //    Vector3d xyz_ref((*it_ft)->f*depth);

        double dist = m_pt_ref[feature_counter]; if(dist < 0) continue;

        Vector3d xyz_ref((*it_ft)->f*dist);
        Vector3d xyz_cur(T_cur_ref * xyz_ref);

        // Vector3d p_host = m_pt_host[feature_counter];
        // assert(p_host[2] > 0);
        // SE3 T_t_h = T_cur_w * (*it_ft)->point->hostFeature_->frame->T_f_w_.inverse();
        // Vector3d xyz_cur(T_t_h*p_host);

        if(xyz_cur[2] < 0) continue;

        Vector2f uv_cur_0(m_cur_frame->cam_->world2cam(xyz_cur).cast<float>());
        Vector2f uv_cur_pyr(uv_cur_0 * scale);
        float u_cur = uv_cur_pyr[0];
        float v_cur = uv_cur_pyr[1];
        int u_cur_i = floorf(u_cur);
        int v_cur_i = floorf(v_cur);

        // check if projection is within the image
        if(u_cur_i-border < 0 || v_cur_i-border < 0 || u_cur_i+border >= cur_img.cols || v_cur_i+border >= cur_img.rows)
            continue;

        Matrix<double,2,6> frame_jac;
        if(!m_inverse_composition)
    		Frame::jacobian_xyz2uv(xyz_cur, frame_jac);

        // compute bilateral interpolation weights for the current image
        float subpix_u_cur = u_cur-u_cur_i;
        float subpix_v_cur = v_cur-v_cur_i;
        float w_cur_tl = (1.0-subpix_u_cur) * (1.0-subpix_v_cur);
        float w_cur_tr = subpix_u_cur * (1.0-subpix_v_cur);
        float w_cur_bl = (1.0-subpix_u_cur) * subpix_v_cur;
        float w_cur_br = subpix_u_cur * subpix_v_cur;

        float* ref_patch_cache_ptr = reinterpret_cast<float*>(m_ref_patch_cache.data) + PATCH_AREA*feature_counter;
        size_t pixel_counter = 0;

        for(int n=0; n<PATCH_AREA; ++n, ++ref_patch_cache_ptr, ++pixel_counter)
    	{
            // uint8_t* cur_img_ptr;
            // if(m_level == m_max_level)
            //     cur_img_ptr = (uint8_t*)cur_img.data + (v_cur_i+pattern_0[n][1])*stride + u_cur_i+pattern_0[n][0]; // top level
            // else
            //     cur_img_ptr = (uint8_t*)cur_img.data + (v_cur_i+pattern_L[n][1])*stride + u_cur_i+pattern_L[n][0];


            uint8_t* cur_img_ptr = (uint8_t*)cur_img.data + (v_cur_i + staticPattern[pattern_offset][n][1])*stride + u_cur_i + staticPattern[pattern_offset][n][0];

    		float cur_color = w_cur_tl*cur_img_ptr[0] 
    						+ w_cur_tr*cur_img_ptr[1] 
    						+ w_cur_bl*cur_img_ptr[stride] 
    						+ w_cur_br*cur_img_ptr[stride+1];
    		if(!std::isfinite(cur_color)) continue;

    		float residual = cur_color - (exposure_rat*(*ref_patch_cache_ptr) + b);


    		float hw = fabs(residual) < setting_huberTH ? 1 : setting_huberTH / fabs(residual);

    		if(fabs(residual) > cutoff_error && m_level < m_max_level) 
    		{
    			E += max_energy;
    			m_total_terms++;
    			m_saturated_terms++;
    		}
    		else
    		{ 
                if(m_level == m_max_level)
                    E += hw *residual*residual;
                else
                    E += hw *residual*residual*(2-hw);

				m_total_terms++;

				//save
				if(!m_inverse_composition)
				{
					float dx = 0.5f * ((w_cur_tl*cur_img_ptr[1]       + w_cur_tr*cur_img_ptr[2]        + w_cur_bl*cur_img_ptr[stride+1] + w_cur_br*cur_img_ptr[stride+2])
                                  	  -(w_cur_tl*cur_img_ptr[-1]      + w_cur_tr*cur_img_ptr[0]        + w_cur_bl*cur_img_ptr[stride-1] + w_cur_br*cur_img_ptr[stride]));
	        		float dy = 0.5f * ((w_cur_tl*cur_img_ptr[stride]  + w_cur_tr*cur_img_ptr[1+stride] + w_cur_bl*cur_img_ptr[stride*2] + w_cur_br*cur_img_ptr[stride*2+1])
	                                  -(w_cur_tl*cur_img_ptr[-stride] + w_cur_tr*cur_img_ptr[1-stride] + w_cur_bl*cur_img_ptr[0]        + w_cur_br*cur_img_ptr[1]));
        			Vector6d J_T = dx*frame_jac.row(0)*fxl + dy*frame_jac.row(1)*fyl;

                    double J_e = -(*ref_patch_cache_ptr);
                    // if(cur_color > 253 || (*ref_patch_cache_ptr) > 253 || cur_color < 2 || (*ref_patch_cache_ptr) < 2)
                    //     J_e = 0;


                    Vector7d J; J[0] = J_e;
                    J.segment<6>(1) = J_T;
                    // for(int i=1; i<7; ++i) J[i] = J_t[i-1];

        			m_buf_jacobian.push_back(J); // 7d
                    // m_buf_jacobian.push_back(J_T);  // 6d
        			m_buf_weight.push_back(hw);
        			m_buf_error.push_back(residual);
				}
				else
				{
					Vector6d J_T(m_jacobian_cache_true.col(feature_counter*PATCH_AREA + pixel_counter));

                    double J_e = -(*ref_patch_cache_ptr);
                    // if(cur_color > 253 || (*ref_patch_cache_ptr) > 253 || cur_color < 2 || (*ref_patch_cache_ptr) < 2)
                    //     J_e = 0;

                    Vector7d J; J[0] = J_e;
                    J.segment<6>(1) = J_T;
                    // for(int i=1; i<7; ++i) J[i] = J_t[i-1];

					m_buf_jacobian.push_back(J);  // 7d
                    // m_buf_jacobian.push_back(J_T);   // 6d
        			m_buf_weight.push_back(hw);
        			m_buf_error.push_back(residual);
				}
    		}

            // debug or linefit
            m_color_cur.push_back(cur_color);
            m_color_ref.push_back(*ref_patch_cache_ptr);
    	}
    }

    return E/m_total_terms;
}

void CoarseTracker::precomputeReferencePatches()
{
	const int border = HALF_PATCH_SIZE+1;
    const cv::Mat& ref_img = m_ref_frame->img_pyr_[m_level];
    const int stride = ref_img.cols;
    const float scale = 1.0f/(1<<m_level);
    const Vector3d ref_pos = m_ref_frame->pos();
    const double fxl = m_ref_frame->cam_->focal_length().x()*scale;
    const double fyl = m_ref_frame->cam_->focal_length().y()*scale;
    const int pattern_offset = m_offset_all;

    std::vector<bool>::iterator visiblity_it = m_visible_fts.begin();
    size_t feature_counter = 0;
    for(auto ft_it = m_ref_frame->fts_.begin(); ft_it != m_ref_frame->fts_.end(); ++ft_it, ++visiblity_it, ++feature_counter)
    {
    	// if((*ft_it)->point == NULL || (*ft_it)->level > m_level)
     //        continue;
        if((*ft_it)->point == NULL)
            continue;

        // check if reference with patch size is within image
        float u_ref = (*ft_it)->px[0]*scale;
        float v_ref = (*ft_it)->px[1]*scale;
        int u_ref_i = floorf(u_ref);
        int v_ref_i = floorf(v_ref);
        if(u_ref_i-border < 0 || v_ref_i-border < 0 || u_ref_i+border >= ref_img.cols || v_ref_i+border >= ref_img.rows)
            continue;

        *visiblity_it = true;

        Matrix<double,2,6> frame_jac;
        if(m_inverse_composition)
        {
        	// double depth(((*ft_it)->point->pos_ - ref_pos).norm());
        	// Vector3d xyz_ref((*ft_it)->f*depth);
            // Vector3d xyz_ref = m_pt_ref[feature_counter];
        	// assert(xyz_ref[2] > 0);

            double dist = m_pt_ref[feature_counter];
            if(dist < 0) continue;
            Vector3d xyz_ref((*ft_it)->f*dist);

        	Frame::jacobian_xyz2uv(xyz_ref, frame_jac);
        }

        // compute bilateral interpolation weights for reference image
        float subpix_u_ref = u_ref-u_ref_i;
        float subpix_v_ref = v_ref-v_ref_i;
        float w_ref_tl = (1.0-subpix_u_ref) * (1.0-subpix_v_ref);
        float w_ref_tr = subpix_u_ref * (1.0-subpix_v_ref);
        float w_ref_bl = (1.0-subpix_u_ref) * subpix_v_ref;
        float w_ref_br = 1.0-(w_ref_tl+w_ref_tr+w_ref_bl);

        size_t pixel_counter = 0;
        float* cache_ptr = reinterpret_cast<float*>(m_ref_patch_cache.data) + PATCH_AREA*feature_counter;
        for(int n=0; n<PATCH_AREA; ++n, ++cache_ptr, ++pixel_counter)
    	{
            // uint8_t* ref_img_ptr;
            // if(m_level == m_max_level)
            //     ref_img_ptr = (uint8_t*)ref_img.data + (v_ref_i+pattern_0[n][1])*stride + u_ref_i+pattern_0[n][0];
            // else
            //     ref_img_ptr = (uint8_t*)ref_img.data + (v_ref_i+pattern_L[n][1])*stride + u_ref_i+pattern_L[n][0];

            uint8_t* ref_img_ptr = (uint8_t*)ref_img.data + (v_ref_i + staticPattern[pattern_offset][n][1])*stride + u_ref_i + staticPattern[pattern_offset][n][0];


    		*cache_ptr = w_ref_tl*ref_img_ptr[0] 
    				   + w_ref_tr*ref_img_ptr[1] 
    				   + w_ref_bl*ref_img_ptr[stride] 
    				   + w_ref_br*ref_img_ptr[stride+1];

    		if(m_inverse_composition)
    		{
    			float dx = 0.5f * ((w_ref_tl*ref_img_ptr[1]       + w_ref_tr*ref_img_ptr[2]        + w_ref_bl*ref_img_ptr[stride+1] + w_ref_br*ref_img_ptr[stride+2])
                                  -(w_ref_tl*ref_img_ptr[-1]      + w_ref_tr*ref_img_ptr[0]        + w_ref_bl*ref_img_ptr[stride-1] + w_ref_br*ref_img_ptr[stride]));
            	float dy = 0.5f * ((w_ref_tl*ref_img_ptr[stride]  + w_ref_tr*ref_img_ptr[1+stride] + w_ref_bl*ref_img_ptr[stride*2] + w_ref_br*ref_img_ptr[stride*2+1])
                              	  -(w_ref_tl*ref_img_ptr[-stride] + w_ref_tr*ref_img_ptr[1-stride] + w_ref_bl*ref_img_ptr[0]        + w_ref_br*ref_img_ptr[1]));
            	m_jacobian_cache_raw.col(feature_counter*PATCH_AREA + pixel_counter) = dx*frame_jac.row(0)*fxl + dy*frame_jac.row(1)*fyl;
    		}
    	}
    }
}

void CoarseTracker::computeGS(Matrix7d& H_out, Vector7d& b_out)
{
	assert(m_buf_jacobian.size() == m_buf_weight.size());

    m_acc7.initialize();

	// H_out.setZero(); 
    b_out.setZero();
	for(size_t i=0; i<m_buf_jacobian.size(); ++i)
	{
		// H_out.noalias() += m_buf_jacobian[i]*m_buf_jacobian[i].transpose()*m_buf_weight[i];

        m_acc7.updateSingleWeighted(m_buf_jacobian[i][0],
                                    m_buf_jacobian[i][1],
                                    m_buf_jacobian[i][2],
                                    m_buf_jacobian[i][3],
                                    m_buf_jacobian[i][4],
                                    m_buf_jacobian[i][5],
                                    m_buf_jacobian[i][6],
                                    m_buf_weight[i], 0);

        b_out.noalias() -= m_buf_jacobian[i]*m_buf_error[i]*m_buf_weight[i];
	}

    m_acc7.finish();
    H_out = m_acc7.H.cast<double>();
}




void CoarseTracker::selectRobustFunctionLevel(const SE3& T_cur_ref, float exposure_rat, float b)
{
    const cv::Mat& cur_img = m_cur_frame->img_pyr_.at(m_level);
    const int stride = cur_img.cols;
    const int border = HALF_PATCH_SIZE+1;
    const float scale = 1.0f/(1<<m_level);
    const Vector3d ref_pos(m_ref_frame->pos());

    const int pattern_offset = m_offset_all;


    std::vector<float> errors;
    // int n_terms = 0;

    size_t feature_counter = 0;
    std::vector<bool>::iterator visiblity_it = m_visible_fts.begin();

    for(auto it_ft=m_ref_frame->fts_.begin(); it_ft!=m_ref_frame->fts_.end(); ++it_ft, ++feature_counter, ++visiblity_it)
    {
        if(!*visiblity_it) continue;

        // if(m_edge_reject == EdgeRejectLevel::PARTLY && (*it_ft)->type == Feature::EDGELET && m_level == m_max_level)
        //     continue;

        // double depth = ((*it_ft)->point->pos_ - ref_pos).norm();
        // Vector3d xyz_ref((*it_ft)->f*depth);
        // Vector3d xyz_ref = m_pt_ref[feature_counter];
        double dist = m_pt_ref[feature_counter]; if(dist < 0) continue;
        
        Vector3d xyz_ref((*it_ft)->f*dist);
        Vector3d xyz_cur(T_cur_ref * xyz_ref);

        // Vector3d p_host = m_pt_host[feature_counter];
        // SE3 T_t_h = T_cur_w * (*it_ft)->point->hostFeature_->frame->T_f_w_.inverse();
        // Vector3d xyz_cur(T_t_h*p_host);
        if(xyz_cur[2] < 0) continue;

        Vector2f uv_cur_pyr(m_cur_frame->cam_->world2cam(xyz_cur).cast<float>() * scale);
        float u_cur = uv_cur_pyr[0];
        float v_cur = uv_cur_pyr[1];
        int u_cur_i = floorf(u_cur);
        int v_cur_i = floorf(v_cur);

        // check if projection is within the image
        if(u_cur_i-border < 0 || v_cur_i-border < 0 || u_cur_i+border >= cur_img.cols || v_cur_i+border >= cur_img.rows)
            continue;

        // compute bilateral interpolation weights for the current image
        float subpix_u_cur = u_cur-u_cur_i;
        float subpix_v_cur = v_cur-v_cur_i;
        float w_cur_tl = (1.0-subpix_u_cur) * (1.0-subpix_v_cur);
        float w_cur_tr = subpix_u_cur * (1.0-subpix_v_cur);
        float w_cur_bl = (1.0-subpix_u_cur) * subpix_v_cur;
        float w_cur_br = subpix_u_cur * subpix_v_cur;

        float* ref_patch_cache_ptr = reinterpret_cast<float*>(m_ref_patch_cache.data) + PATCH_AREA*feature_counter;
        for(int n=0; n<PATCH_AREA; ++n, ++ref_patch_cache_ptr)
        {
            // uint8_t* cur_img_ptr;
            // if(m_level == m_max_level)
            //     cur_img_ptr = (uint8_t*)cur_img.data + (v_cur_i+pattern_0[n][1])*stride + u_cur_i+pattern_0[n][0]; // top level
            // else
            //     cur_img_ptr = (uint8_t*)cur_img.data + (v_cur_i+pattern_L[n][1])*stride + u_cur_i+pattern_L[n][0];

            uint8_t* cur_img_ptr = (uint8_t*)cur_img.data + (v_cur_i + staticPattern[pattern_offset][n][1])*stride + u_cur_i + staticPattern[pattern_offset][n][0];


            float cur_color = w_cur_tl*cur_img_ptr[0] 
                            + w_cur_tr*cur_img_ptr[1] 
                            + w_cur_bl*cur_img_ptr[stride] 
                            + w_cur_br*cur_img_ptr[stride+1];
            float residual = cur_color - (exposure_rat*(*ref_patch_cache_ptr) + b);

            errors.push_back(fabsf(residual));
            // n_terms++;
        }
    }

    if(errors.size() < 30)
    {
        m_huber_thresh = 5.2;
        m_outlier_thresh = 100;
        return;
    }

    float residual_median = hso::getMedian(errors);
    vector<float> absolute_deviation;
    for(size_t i=0; i<errors.size(); ++i)
        absolute_deviation.push_back(fabs(errors[i]-residual_median));

    float standard_deviation = 1.4826*hso::getMedian(absolute_deviation);


    m_huber_thresh = residual_median + standard_deviation;
    m_outlier_thresh = 3*m_huber_thresh;
    // m_outlier_thresh = 20;



    // if(m_huber_thresh < 1) m_huber_thresh = 1; 
    if(m_outlier_thresh < 10) m_outlier_thresh = 10;



    if(m_verbose)
    {
        printf("\nPYRAMID LEVEL %i\n---------------\n", m_level);
        cout << "Mid = "        << residual_median 
             << "\tStd = "      << standard_deviation 
             << "\tHuber = "    << m_huber_thresh 
             << "\tOutliers = " << m_outlier_thresh << endl;
    }


}


} //namespace hso