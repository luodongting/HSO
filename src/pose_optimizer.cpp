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

#include <stdexcept>
#include <hso/pose_optimizer.h>
#include <hso/frame.h>
#include <hso/feature.h>
#include <hso/point.h>
#include <hso/config.h>

#include "hso/vikit/math_utils.h"
#include "hso/vikit/robust_cost.h"

namespace hso {
namespace pose_optimizer {

void optimizeGaussNewton(
    const double reproj_thresh,
    const size_t n_iter,
    const bool verbose,
    FramePtr& frame,
    double& estimated_scale,
    double& error_init,
    double& error_final,
    size_t& num_obs)
{
  // init
  double chi2(0.0);
  vector<double> chi2_vec_init, chi2_vec_final;
  hso::robust_cost::TukeyWeightFunction weight_function;
  SE3 T_old(frame->T_f_w_);
  Matrix6d A;
  Vector6d b;

  // compute the scale of the error for robust estimation
  std::vector<float> errors; errors.reserve(frame->fts_.size());
  for(auto it=frame->fts_.begin(); it!=frame->fts_.end(); ++it)
  {
    if((*it)->point == NULL)
      continue;
    Vector2d e = hso::project2d((*it)->f)
               - hso::project2d(frame->T_f_w_ * (*it)->point->pos_);
    e *= 1.0 / (1<<(*it)->level);
    if((*it)->type == Feature::EDGELET)
    {
      errors.push_back( std::fabs((*it)->grad.transpose() * e) );
    }
    else
    {
      errors.push_back(e.norm());
    }
  }
  if(errors.empty())
    return;
  hso::robust_cost::MADScaleEstimator scale_estimator;
  estimated_scale = scale_estimator.compute(errors);

  num_obs = errors.size();
  chi2_vec_init.reserve(num_obs);
  chi2_vec_final.reserve(num_obs);
  double scale = estimated_scale;
  for(size_t iter=0; iter<n_iter; iter++)
  {
    // overwrite scale
    if(iter == 5)
      scale = 0.85/frame->cam_->errorMultiplier2();

    b.setZero();
    A.setZero();
    double new_chi2(0.0);

    // compute residual
    for(auto it=frame->fts_.begin(); it!=frame->fts_.end(); ++it)
    {
      if((*it)->point == NULL)
        continue;
      Matrix26d J;
      Vector3d xyz_f(frame->T_f_w_ * (*it)->point->pos_);
      Frame::jacobian_xyz2uv(xyz_f, J);
      Vector2d e = hso::project2d((*it)->f) - hso::project2d(xyz_f);
      double sqrt_inv_cov = 1.0 / (1<<(*it)->level);
      e *= sqrt_inv_cov;
      if(iter == 0)
      {
        if((*it)->type == Feature::EDGELET)
        {
          double err_edge = (*it)->grad.transpose() * e;
          chi2_vec_init.push_back(err_edge*err_edge);
        }
        else
          chi2_vec_init.push_back(e.squaredNorm()); 
      }
      J *= sqrt_inv_cov;
      if((*it)->type == Feature::EDGELET)
      {
        Matrix<double,1,6> J_edge = (*it)->grad.transpose() * J;
        double err_edge = (*it)->grad.transpose() * e;
        double weight = weight_function.value(std::fabs(err_edge)/scale);
        A.noalias() += J_edge.transpose()*J_edge*weight;
        b.noalias() -= J_edge.transpose()*err_edge*weight;
        new_chi2 += err_edge*err_edge*weight;
      }
      else
      {
        double weight = weight_function.value(e.norm()/scale);
        A.noalias() += J.transpose()*J*weight;
        b.noalias() -= J.transpose()*e*weight;
        new_chi2 += e.squaredNorm()*weight;
      }
    }

    // solve linear system
    const Vector6d dT(A.ldlt().solve(b));

    // check if error increased
    if((iter > 0 && new_chi2 > chi2) || (bool) std::isnan((double)dT[0]))
    {
      if(verbose)
        std::cout << "it " << iter
                  << "\t FAILURE \t new_chi2 = " << new_chi2 << std::endl;
      frame->T_f_w_ = T_old; // roll-back
      break;
    }

    // update the model
    SE3 T_new = SE3::exp(dT)*frame->T_f_w_;
    T_old = frame->T_f_w_;
    frame->T_f_w_ = T_new;
    chi2 = new_chi2;
    if(verbose)
      std::cout << "it " << iter
                << "\t Success \t new_chi2 = " << new_chi2
                << "\t norm(dT) = " << hso::norm_max(dT) << std::endl;

    // stop when converged
    if(hso::norm_max(dT) <= EPS)
      break;
  }

  // Set covariance as inverse information matrix. Optimistic estimator!
  const double pixel_variance=1.0;
  frame->Cov_ = pixel_variance*(A*std::pow(frame->cam_->errorMultiplier2(),2)).inverse();

  // Remove Measurements with too large reprojection error
  double reproj_thresh_scaled = reproj_thresh / frame->cam_->errorMultiplier2();
  size_t n_deleted_refs = 0;
  for(Features::iterator it=frame->fts_.begin(); it!=frame->fts_.end(); ++it)
  {
    if((*it)->point == NULL)
      continue;
    Vector2d e = hso::project2d((*it)->f) - hso::project2d(frame->T_f_w_ * (*it)->point->pos_);
    double sqrt_inv_cov = 1.0 / (1<<(*it)->level);
    e *= sqrt_inv_cov;
    if((*it)->type == Feature::EDGELET)
    {
      double err_edge = (*it)->grad.transpose() * e;
      chi2_vec_init.push_back(err_edge*err_edge);
      if(std::fabs(err_edge) > reproj_thresh_scaled)
      {
        (*it)->point = NULL;
        ++n_deleted_refs;
      }
    }
    else
    {
      chi2_vec_final.push_back(e.squaredNorm());
      if(e.norm() > reproj_thresh_scaled)
      {
        // we don't need to delete a reference in the point since it was not created yet
        (*it)->point = NULL;
        ++n_deleted_refs;
      }
    }
  }

  error_init=0.0;
  error_final=0.0;
  if(!chi2_vec_init.empty())
    error_init = sqrt(hso::getMedian(chi2_vec_init))*frame->cam_->errorMultiplier2();
  if(!chi2_vec_final.empty())
    error_final = sqrt(hso::getMedian(chi2_vec_final))*frame->cam_->errorMultiplier2();

  estimated_scale *= frame->cam_->errorMultiplier2();
  if(verbose)
    std::cout << "n deleted obs = " << n_deleted_refs
              << "\t scale = " << estimated_scale
              << "\t error init = " << error_init
              << "\t error end = " << error_final << std::endl;
  num_obs -= n_deleted_refs;
}

void optimizeLevenbergMarquardt2nd(
    const double reproj_thresh, const size_t n_iter, const bool verbose,
    FramePtr& frame, double& estimated_scale, double& error_init, double& error_final,
    size_t& num_obs)
{
    double chi2(0.0);
    double rho = 0;
    double mu = 0.01;
    double nu = 2.0;
    bool stop = false;
    int n_trials = 0;
    const int n_trials_max = 5;

    vector<double> chi2_vec_init, chi2_vec_final;
    hso::robust_cost::TukeyWeightFunction weight_function;
    Matrix6d A;
    Vector6d b;

    // compute scale
    std::vector<float> errors; errors.reserve(frame->fts_.size());
    for(auto it=frame->fts_.begin(); it!=frame->fts_.end(); ++it)
    {
        // if((*it)->point == NULL) continue;
        assert((*it)->point != NULL);

        Vector2d e = hso::project2d((*it)->f) - hso::project2d(frame->T_f_w_ * (*it)->point->pos_);
        e *= 1.0 / (1<<(*it)->level);
        errors.push_back(e.norm());
    }

    if(errors.empty()) return;
    hso::robust_cost::MADScaleEstimator scale_estimator;
    estimated_scale = scale_estimator.compute(errors);

    for(auto it=frame->fts_.begin(); it!=frame->fts_.end(); ++it)
    {
        // if((*it)->point == NULL) continue;
        assert((*it)->point != NULL);

        Vector2d e = hso::project2d((*it)->f) - hso::project2d(frame->T_f_w_ * (*it)->point->pos_);
        e *= 1.0 / (1<<(*it)->level);

        double weight = weight_function.value(e.norm()/estimated_scale);
        chi2 += e.squaredNorm() * weight;
    }

    num_obs = errors.size();
    chi2_vec_init.reserve(num_obs);
    chi2_vec_final.reserve(num_obs);
    double scale = estimated_scale;

    for(size_t iter=0; iter<n_iter; iter++)
    {
        // if(iter == 5) 
        //     scale = 0.85/frame->cam_->errorMultiplier2();

        rho = 0;
        n_trials = 0;
        do
        {
            SE3 T_new;
            double new_chi2 = 0.0;
            A.setZero();
            b.setZero();

            for(auto it=frame->fts_.begin(); it!=frame->fts_.end(); ++it)
            {
                // if((*it)->point == NULL) continue;
                assert((*it)->point != NULL);

                Matrix26d J;
                Vector3d xyz_f(frame->T_f_w_ * (*it)->point->pos_);
                Frame::jacobian_xyz2uv(xyz_f, J);
                Vector2d e = hso::project2d((*it)->f) - hso::project2d(xyz_f);
                double sqrt_inv_cov = 1.0 / (1<<(*it)->level);
                e *= sqrt_inv_cov;

                if(iter == 0 && n_trials == 0)
                    chi2_vec_init.push_back(e.squaredNorm()); 

                J *= sqrt_inv_cov;

                double weight = weight_function.value(e.norm()/scale);
                A.noalias() += J.transpose()*J*weight;
                b.noalias() -= J.transpose()*e*weight;
            }

            A += (A.diagonal()*mu).asDiagonal();
            const Vector6d dT(A.ldlt().solve(b));
            // const Vector6d dT(A.inverse()*b);

            if(!(bool) std::isnan((double)dT[0]))
            {
                T_new = SE3::exp(dT)*frame->T_f_w_;

                for(auto it=frame->fts_.begin(); it!=frame->fts_.end(); ++it)
                {
                    // if((*it)->point == NULL) continue;
                    assert((*it)->point != NULL);

                    Vector3d xyz_f(T_new * (*it)->point->pos_);
                    Vector2d e = hso::project2d((*it)->f) - hso::project2d(xyz_f);
                    double sqrt_inv_cov = 1.0 / (1<<(*it)->level);
                    e *= sqrt_inv_cov;

                    double weight = weight_function.value(e.norm()/scale);
                    new_chi2 += e.squaredNorm() * weight;

                }
                rho = chi2 - new_chi2;
            }
            else
                rho = -1;

            if(rho>0)
            {
                frame->T_f_w_ = T_new;
                chi2 = new_chi2;
                stop = hso::norm_max(dT) <= EPS;
                mu *= std::max(1./3., std::min(1.-std::pow(2*rho-1,3), 2./3.));
                nu = 2.;

                if(verbose)
                {
                    cout << "It. " << iter
                    << "\t Trial " << n_trials
                    << "\t Success"
                    << "\t new_chi2 = " << new_chi2
                    << "\t mu = " << mu
                    << "\t nu = " << nu
                    << endl;
                }
            }
            else
            {
                mu *= nu;
                nu *= 2.;
                ++n_trials;
                if(n_trials >= n_trials_max) stop = true;

                if(verbose)
                {
                    cout << "It. " << iter
                    << "\t Trial " << n_trials
                    << "\t Failure"
                    << "\t new_chi2 = " << new_chi2
                    << "\t mu = " << mu
                    << "\t nu = " << nu
                    << endl;
                }
            }
        }while(!(rho>0 || stop));

        if (stop) break;
    }

    const double pixel_variance=1.0;
    frame->Cov_ = pixel_variance*(A*std::pow(frame->cam_->errorMultiplier2(),2)).inverse();

    // Remove Measurements with too large reprojection error
    double reproj_thresh_scaled = reproj_thresh / frame->cam_->errorMultiplier2();
    size_t n_deleted_refs = 0;
    for(Features::iterator it=frame->fts_.begin(); it!=frame->fts_.end(); ++it)
    {
        Vector2d e = hso::project2d((*it)->f) - hso::project2d(frame->T_f_w_ * (*it)->point->pos_);
        double sqrt_inv_cov = 1.0 / (1<<(*it)->level);
        e *= sqrt_inv_cov;

        chi2_vec_final.push_back(e.squaredNorm());
        if(e.norm() > reproj_thresh_scaled)
        {
            (*it)->point = NULL;
            ++n_deleted_refs;
        }
    }

    error_init=0.0;
    error_final=0.0;
    if(!chi2_vec_init.empty())
        error_init = sqrt(hso::getMedian(chi2_vec_init))*frame->cam_->errorMultiplier2();
    if(!chi2_vec_final.empty())
        error_final = sqrt(hso::getMedian(chi2_vec_final))*frame->cam_->errorMultiplier2();

    estimated_scale *= frame->cam_->errorMultiplier2();
    num_obs -= n_deleted_refs;
}

void optimizeLevenbergMarquardt3rd(
    const double reproj_thresh, const size_t n_iter, const bool verbose,
    FramePtr& frame, double& estimated_scale, double& error_init, double& error_final,
    size_t& num_obs)
{
    double chi2=0.0, rho=0, mu=0.1, nu=2.0;
    bool stop = false;
    int n_trials = 0;
    const int n_trials_max = 5;

    vector<double> chi2_vec_init, chi2_vec_final;
    chi2_vec_init.reserve(frame->fts_.size());
    chi2_vec_final.reserve(frame->fts_.size());

    hso::robust_cost::HuberWeightFunction weight_function;
    Matrix6d A; A.setZero();
    Vector6d b; b.setZero();

    // compute scale
    vector<float> errors_pt; errors_pt.reserve(frame->fts_.size());
    vector<float> errors_ls; errors_ls.reserve(frame->fts_.size());

    // jointly scale
    // vector<float> errors_tt; errors_tt.reserve(frame->fts_.size());

    vector<Vector3d> v_host; v_host.reserve(frame->fts_.size());

    for(Features::iterator it=frame->fts_.begin(); it!=frame->fts_.end(); ++it)
    {
        // assert((*it)->point != NULL);
        if((*it)->point == NULL) continue;
        Feature* ft = *it;

        Frame* host = ft->point->hostFeature_->frame;
        Vector3d pHost = ft->point->hostFeature_->f * (1.0/ft->point->idist_);
        SE3 Tth = frame->T_f_w_ * host->T_f_w_.inverse();
        Vector3d pTarget = Tth * pHost;

        Vector2d e = hso::project2d(ft->f) - hso::project2d(pTarget);
        e *= 1.0 / (1<<ft->level);

        if(ft->type == Feature::EDGELET)
        {
            float error_ls = ft->grad.transpose()*e;
            errors_ls.push_back(fabs(error_ls));
            chi2_vec_init.push_back(error_ls*error_ls);
        }
        else
        {
            float error_pt = e.norm();
            errors_pt.push_back(error_pt);
            chi2_vec_init.push_back(error_pt*error_pt);
        }

        v_host.push_back(pHost);
    }

    if(errors_pt.empty() && errors_ls.empty()) return;
    // if(errors_tt.empty()) return;
    
    hso::robust_cost::MADScaleEstimator scale_estimator;

    float estimated_scale_pt, estimated_scale_ls;
    if(!errors_pt.empty() && !errors_ls.empty())
    {
        estimated_scale_pt = scale_estimator.compute(errors_pt);
        estimated_scale_ls = scale_estimator.compute(errors_ls);
    }
    else if(!errors_pt.empty() && errors_ls.empty())
    {
        estimated_scale_pt = scale_estimator.compute(errors_pt);
        estimated_scale_ls = 0.5*estimated_scale_pt;
    }
    else if(errors_pt.empty() && !errors_ls.empty())
    {
        estimated_scale_ls = scale_estimator.compute(errors_ls);
        estimated_scale_pt = 2*estimated_scale_ls;
    }
    else
    {
        // assert(1 > 2);
    }


    estimated_scale = estimated_scale_pt;




    int idx_host = 0;
    for(Features::iterator it=frame->fts_.begin(); it!=frame->fts_.end(); ++it)
    {
        // assert((*it)->point != NULL);
        if((*it)->point == NULL) continue;

        Frame* host = (*it)->point->hostFeature_->frame;
        // Vector3d pHost = (*it)->point->hostFeature_->f * (1.0/(*it)->point->idist_);
        Vector3d pHost = v_host[idx_host];
        SE3 Tth = frame->T_f_w_ * host->T_f_w_.inverse();
        Vector3d pTarget = Tth * pHost;

        Vector2d e = hso::project2d((*it)->f) - hso::project2d(pTarget);
        e *= 1.0 / (1<<(*it)->level);

        if((*it)->type == Feature::EDGELET)
        {
            double error_ls = (*it)->grad.transpose()*e;

            double weight = weight_function.value(fabs(error_ls)/estimated_scale_ls);
            // double weight = weight_function.value(e.norm()/estimated_scale); // jointly


            if((*it)->point->type_ == Point::TYPE_TEMPORARY) weight *= 0.5;

            chi2 += error_ls*error_ls * weight;
        }
        else
        {
            double error_pt = e.norm();
            double weight = weight_function.value(error_pt/estimated_scale_pt);

            if((*it)->point->type_ == Point::TYPE_TEMPORARY) weight *= 0.5;

            chi2 += error_pt*error_pt * weight;
        }

        ++idx_host;
    }

    // num_obs = errors_tt.size();
    num_obs = errors_pt.size()+errors_ls.size();

    for(size_t iter=0; iter<n_iter; iter++)
    {
        // if(iter == 5) scale = 0.85/frame->cam_->errorMultiplier2();

        rho = 0;
        n_trials = 0;
        do
        {
            SE3 T_new;
            double new_chi2 = 0.0;
            A.setZero();
            b.setZero();

            idx_host = 0;
            for(Features::iterator it=frame->fts_.begin(); it!=frame->fts_.end(); ++it)
            {
                // assert((*it)->point != NULL);
                if((*it)->point == NULL) continue;
                
                Frame* host = (*it)->point->hostFeature_->frame;
                // Vector3d pHost = (*it)->point->hostFeature_->f * (1.0/(*it)->point->idist_);
                Vector3d pHost = v_host[idx_host];
                SE3 Tth = frame->T_f_w_ * host->T_f_w_.inverse();
                Vector3d pTarget = Tth * pHost;



                Matrix26d J;
                Frame::jacobian_xyz2uv(pTarget, J);
                Vector2d e = hso::project2d((*it)->f) - hso::project2d(pTarget);
                double sqrt_inv_cov = 1.0 / (1<<(*it)->level);
                e *= sqrt_inv_cov;
                J *= sqrt_inv_cov;

                if((*it)->type == Feature::EDGELET)
                {
                    Matrix<double,1,6> J_edge = (*it)->grad.transpose()*J;
                    double e_edge = (*it)->grad.transpose()*e;

                    double weight = weight_function.value(fabs(e_edge)/estimated_scale_ls);
                    // double weight = weight_function.value(e.norm()/scale_tt);

       
                    if((*it)->point->type_ == Point::TYPE_TEMPORARY) weight *= 0.5;

                    A.noalias() += J_edge.transpose()*J_edge*weight;
                    b.noalias() -= J_edge.transpose()*e_edge*weight;
                }
                else
                {
                    // double weight = weight_function.value(e.norm()/scale_pt);
                    double weight = weight_function.value(e.norm()/estimated_scale_pt);

         
                    if((*it)->point->type_ == Point::TYPE_TEMPORARY) weight *= 0.5;

                    A.noalias() += J.transpose()*J*weight;
                    b.noalias() -= J.transpose()*e*weight;
                }

                ++idx_host;
            }

            A += (A.diagonal()*mu).asDiagonal();
            const Vector6d dT(A.ldlt().solve(b));

            if(!(bool) std::isnan((double)dT[0]))
            {
                T_new = SE3::exp(dT)*frame->T_f_w_;

                idx_host = 0;
                for(Features::iterator it=frame->fts_.begin(); it!=frame->fts_.end(); ++it)
                {
                    // assert((*it)->point != NULL);
                    if((*it)->point == NULL) continue;

                    Frame* host = (*it)->point->hostFeature_->frame;
                    // Vector3d pHost = (*it)->point->hostFeature_->f * (1.0/(*it)->point->idist_);
                    Vector3d pHost = v_host[idx_host];
                    SE3 Tth = T_new * host->T_f_w_.inverse();
                    Vector3d pTarget = Tth * pHost;

                    Vector2d e = hso::project2d((*it)->f) - hso::project2d(pTarget);
                    double sqrt_inv_cov = 1.0 / (1<<(*it)->level);
                    e *= sqrt_inv_cov;

                    if((*it)->type == Feature::EDGELET)
                    {
                        double error_ls = (*it)->grad.transpose()*e;

                        // double weight = weight_function.value(fabs(errEdge)/scale); // OMG!!!
                        double weight = weight_function.value(fabs(error_ls)/estimated_scale_ls);
                        // double weight = weight_function.value(e.norm()/scale_tt);

                        if((*it)->point->type_ == Point::TYPE_TEMPORARY) weight *= 0.5;

                        new_chi2 += error_ls*error_ls * weight;
                    }
                    else
                    {
                        double error_pt = e.norm();
                        // double weight = weight_function.value(e.norm()/scale_pt);
                        double weight = weight_function.value(error_pt/estimated_scale_pt);

                        if((*it)->point->type_ == Point::TYPE_TEMPORARY) weight *= 0.5;

                        new_chi2 += error_pt*error_pt * weight;
                    }

                    ++idx_host;
                }
                rho = chi2 - new_chi2;
            }
            else
                rho = -1;

            if(rho>0)
            {
                frame->T_f_w_ = T_new;
                chi2 = new_chi2;
                stop = hso::norm_max(dT) <= EPS;
                mu *= std::max(1./3., std::min(1.-std::pow(2*rho-1,3), 2./3.));
                nu = 2.;

                if(verbose)
                {
                    cout << "It. " << iter
                    << "\t Trial " << n_trials
                    << "\t Success"
                    << "\t new_chi2 = " << new_chi2
                    << "\t mu = " << mu
                    << "\t nu = " << nu
                    << endl;
                }
            }
            else
            {
                mu *= nu;
                nu *= 2.;
                if(mu < 0.0001) mu = 0.0001;
                
                ++n_trials;
                if(n_trials >= n_trials_max) stop = true;

                if(verbose)
                {
                    cout << "It. " << iter
                    << "\t Trial " << n_trials
                    << "\t Failure"
                    << "\t new_chi2 = " << new_chi2
                    << "\t mu = " << mu
                    << "\t nu = " << nu
                    << endl;
                }
            }
        }while(!(rho>0 || stop));

        if (stop) break;
    }

    const float pixel_variance=1.0;
    frame->Cov_ = pixel_variance*(A*std::pow(frame->cam_->errorMultiplier2(),2)).inverse();



    const float reproj_thresh_scaled_pt = (frame->fts_.size() < 80)? sqrt(5.991)/frame->cam_->errorMultiplier2() : reproj_thresh/frame->cam_->errorMultiplier2();


    const float reproj_thresh_scaled_ls = 1.3 / frame->cam_->errorMultiplier2();



    size_t n_deleted_refs = 0;

    idx_host = 0;
    for(Features::iterator it=frame->fts_.begin(); it!=frame->fts_.end(); ++it)
    {
        if((*it)->point == NULL) continue;
        // assert((*it)->point != NULL);
        
        Frame* host = (*it)->point->hostFeature_->frame;
        // Vector3d pHost = (*it)->point->hostFeature_->f * (1.0/(*it)->point->idist_);
        Vector3d pHost = v_host[idx_host];
        SE3 Tth = frame->T_f_w_ * host->T_f_w_.inverse();
        Vector3d pTarget = Tth * pHost;

        Vector2d e = hso::project2d((*it)->f) - hso::project2d(pTarget);
        double sqrt_inv_cov = 1.0 / (1<<(*it)->level);
        e *= sqrt_inv_cov;

        if((*it)->type == Feature::EDGELET)
        {
            double error_ls = (*it)->grad.transpose()*e;
            if(fabs(error_ls) > reproj_thresh_scaled_ls)
            {
                ++n_deleted_refs;
                (*it)->point = NULL;
            }
            chi2_vec_final.push_back(error_ls*error_ls);
        }
        else
        {
            float error_pt = e.norm();
            if(error_pt > reproj_thresh_scaled_pt)
            {
                ++n_deleted_refs;
                // if(error_pt < reproj_thresh_scaled_pt_2)
                //     feat_buffer.push_back(*it);
                // else
                (*it)->point = NULL;
            }
            chi2_vec_final.push_back(error_pt*error_pt);
        }

        ++idx_host;


        
    }






    error_init=0.0;
    error_final=0.0;
    if(!chi2_vec_init.empty())
        error_init = sqrt(hso::getMedian(chi2_vec_init))*frame->cam_->errorMultiplier2();
    if(!chi2_vec_final.empty())
        error_final = sqrt(hso::getMedian(chi2_vec_final))*frame->cam_->errorMultiplier2();

    estimated_scale *= frame->cam_->errorMultiplier2();
    num_obs -= n_deleted_refs;


    frame->m_error_in_px = error_final<1.5? 1.0 : 1.5/error_final;



}

void optimizeLevenbergMarquardtMagnitude(
    const double reproj_thresh, const size_t n_iter, const bool verbose,
    FramePtr& frame, double& estimated_scale, double& error_init, double& error_final,
    size_t& num_obs)
{
    vector<float> errors_pt, errors_ls;
    vector<double> chi2_vec_init, chi2_vec_final; // debug

    vector<Vector3d> v_pHost;

    for(auto it=frame->fts_.begin(); it!=frame->fts_.end(); ++it)
    {
        if((*it)->point == NULL) continue;
        Feature* feat = *it;

        Frame* host = feat->point->hostFeature_->frame;
        Vector3d pHost = feat->point->hostFeature_->f * (1.0/feat->point->idist_);
        SE3 Tth = frame->T_f_w_ * host->T_f_w_.inverse();
        Vector3d pTarget = Tth * pHost;

        Vector2d e = hso::project2d(feat->f) - hso::project2d(pTarget);
        e *= 1.0 / (1<<feat->level);

        if(feat->type == Feature::EDGELET)
        {
            double e_edge = feat->grad.transpose()*e;
            double e_magn = e_edge*e_edge;
            errors_ls.push_back(e_magn);
            chi2_vec_init.push_back(e_magn);
        }
        else
        {
            double p_magn = e.squaredNorm();
            errors_pt.push_back(p_magn);
            chi2_vec_init.push_back(p_magn);
        }

        v_pHost.push_back(pHost);
    }


    hso::robust_cost::MADScaleEstimator scale_estimator;
    float estimated_pt = scale_estimator.compute(errors_pt);

    float estimated_ls = estimated_pt*0.5;
    if(errors_ls.size() > 5)
        estimated_ls = scale_estimator.compute(errors_ls);

    estimated_scale = estimated_pt;

    hso::robust_cost::TukeyWeightFunction weight_function;
    // hso::robust_cost::HuberWeightFunction weight_function;

    double chi2 = 0;
    size_t index = 0;

    for(auto it=frame->fts_.begin(); it!=frame->fts_.end(); ++it)
    {
        if((*it)->point == NULL) continue;
        Feature* feat = *it;

        Frame* host = feat->point->hostFeature_->frame;
        // Vector3d pHost = (*it)->point->hostFeature_->f * (1.0/(*it)->point->idist_);
        Vector3d pHost = v_pHost[index];
        SE3 Tth = frame->T_f_w_ * host->T_f_w_.inverse();
        Vector3d pTarget = Tth * pHost;

        Vector2d e = hso::project2d(feat->f) - hso::project2d(pTarget);
        e *= 1.0 / (1<<feat->level);

        if(feat->type == Feature::EDGELET)
        {
            double e_edge = feat->grad.transpose()*e;
            double e_magn = e_edge*e_edge;
            double weight = weight_function.value(e_magn/estimated_ls);

            if(feat->point->type_ == Point::TYPE_TEMPORARY) weight *= 0.5;

            chi2 += e_magn*e_magn*weight;
        }
        else
        {
            double p_magn = e.squaredNorm();
            double weight = weight_function.value(p_magn/estimated_pt);

            if(feat->point->type_ == Point::TYPE_TEMPORARY) weight *= 0.5;

            chi2 += p_magn*p_magn*weight;
        }


        index++;
    }


    num_obs = errors_pt.size()+errors_ls.size();

    chi2_vec_final.reserve(num_obs);
    double scale_pt = estimated_pt;
    double scale_ls = estimated_ls;

    double rho=0, mu=0.1, nu=2.0;
    bool stop = false;
    int n_trials = 0;
    const int n_trials_max = 5;

    Matrix6d A; 
    Vector6d b;

    for(size_t iter=0; iter<n_iter; iter++)
    {
        rho = 0;
        n_trials = 0;

        do
        {
            SE3 T_new;
            double new_chi2 = 0.0;
            A.setZero();
            b.setZero();

            index = 0;
            for(auto it=frame->fts_.begin(); it!=frame->fts_.end(); ++it)
            {
                if((*it)->point == NULL) continue;
                Feature* feat = *it;
                
                Frame* host = feat->point->hostFeature_->frame;
                // Vector3d pHost = (*it)->point->hostFeature_->f * (1.0/(*it)->point->idist_);
                Vector3d pHost = v_pHost[index];
                SE3 Tth = frame->T_f_w_ * host->T_f_w_.inverse();
                Vector3d pTarget = Tth * pHost;

                // Vector3d xyz_f(frame->T_f_w_ * (*it)->point->pos_);

                Matrix26d J;
                Frame::jacobian_xyz2uv(pTarget, J);
                Vector2d e = hso::project2d(feat->f) - hso::project2d(pTarget);
                double sqrt_inv_cov = 1.0 / (1<<(*it)->level);
                e *= sqrt_inv_cov;
                J *= sqrt_inv_cov;



                if((*it)->type == Feature::EDGELET)
                {
                    Matrix<double,1,6> J_edge = (*it)->grad.transpose()*J;
                    double e_edge = (*it)->grad.transpose()*e;
                    double e_magn = e_edge*e_edge;
                    J_edge *= 2*e_edge;

                    double weight = weight_function.value(e_magn/scale_ls);

                    if((*it)->point->type_ == Point::TYPE_TEMPORARY)
                        weight *= 0.5;

                    A.noalias() += J_edge.transpose()*J_edge*weight;
                    b.noalias() -= J_edge.transpose()*e_magn*weight;
                }
                else
                {
                    double p_magn = e.squaredNorm();
                    double weight = weight_function.value(p_magn/scale_pt);

                    if((*it)->point->type_ == Point::TYPE_TEMPORARY)
                        weight *= 0.5;

                    Vector2d J_magn2err(2*e[0], 2*e[1]);
                    Vector6d J_new = J_magn2err.transpose()*J;

                    A.noalias() += J_new*J_new.transpose()*weight;
                    b.noalias() -= J_new*p_magn*weight;
                }

                ++index;
            }

            A += (A.diagonal()*mu).asDiagonal();
            const Vector6d dT(A.ldlt().solve(b));

            if(!(bool) std::isnan((double)dT[0]))
            {
                T_new = SE3::exp(dT)*frame->T_f_w_;

                index = 0;
                for(auto it=frame->fts_.begin(); it!=frame->fts_.end(); ++it)
                {
                    if((*it)->point == NULL) continue;

                    Frame* host = (*it)->point->hostFeature_->frame;
                    // Vector3d pHost = (*it)->point->hostFeature_->f * (1.0/(*it)->point->idist_);
                    Vector3d pHost = v_pHost[index];
                    SE3 Tth = T_new * host->T_f_w_.inverse();
                    Vector3d pTarget = Tth * pHost;

                    Vector2d e = hso::project2d((*it)->f) - hso::project2d(pTarget);
                    double sqrt_inv_cov = 1.0 / (1<<(*it)->level);
                    e *= sqrt_inv_cov;

                    if((*it)->type == Feature::EDGELET)
                    {
                        double e_edge = (*it)->grad.transpose()*e;
                        double e_magn = e_edge*e_edge;

                        double weight = weight_function.value(e_magn/scale_ls);

                        if((*it)->point->type_ == Point::TYPE_TEMPORARY)
                            weight *= 0.5;

                        new_chi2 += e_magn*e_magn*weight;
                    }
                    else
                    {
                        double p_magn = e.squaredNorm();
                        double weight = weight_function.value(p_magn/scale_pt);

                        if((*it)->point->type_ == Point::TYPE_TEMPORARY)
                            weight *= 0.5;

                        new_chi2 += p_magn*p_magn*weight;
                    }

                    ++index;
                }

                rho = chi2 - new_chi2;
            }
            else
                rho = -1;

            if(rho>0)
            {
                frame->T_f_w_ = T_new;
                chi2 = new_chi2;
                stop = hso::norm_max(dT) <= EPS;
                mu *= std::max(1./3., std::min(1.-std::pow(2*rho-1,3), 2./3.));
                nu = 2.;

                if(verbose)
                {
                    cout << "It. " << iter
                    << "\t Trial " << n_trials
                    << "\t Success"
                    << "\t new_chi2 = " << new_chi2
                    << "\t mu = " << mu
                    << "\t nu = " << nu
                    << endl;
                }
            }
            else
            {
                mu *= nu;
                nu *= 2.;
                if(mu < 0.0001) mu = 0.0001;
                
                ++n_trials;
                if(n_trials >= n_trials_max) stop = true;

                if(verbose)
                {
                    cout << "It. " << iter
                    << "\t Trial " << n_trials
                    << "\t Failure"
                    << "\t new_chi2 = " << new_chi2
                    << "\t mu = " << mu
                    << "\t nu = " << nu
                    << endl;
                }
            }
        } while(!(rho>0 || stop));

        if (stop) break;
    }


    const double pixel_variance=1.0;
    frame->Cov_ = pixel_variance*(A*std::pow(frame->cam_->errorMultiplier2(),2)).inverse();

    // Remove Measurements with too large reprojection error
    double reproj_thresh_scaled_pt = reproj_thresh / frame->cam_->errorMultiplier2();
    double reproj_thresh_scaled_ls = 1.5/frame->cam_->errorMultiplier2();
    size_t n_deleted_refs = 0;
    index = 0;
    for(Features::iterator it=frame->fts_.begin(); it!=frame->fts_.end(); ++it)
    {
        if((*it)->point == NULL) continue;
        // assert((*it)->point != NULL);
        
        Frame* host = (*it)->point->hostFeature_->frame;
        // Vector3d pHost = (*it)->point->hostFeature_->f * (1.0/(*it)->point->idist_);
        Vector3d pHost = v_pHost[index];
        SE3 Tth = frame->T_f_w_ * host->T_f_w_.inverse();
        Vector3d pTarget = Tth * pHost;

        Vector2d e = hso::project2d((*it)->f) - hso::project2d(pTarget);
        double sqrt_inv_cov = 1.0 / (1<<(*it)->level);
        e *= sqrt_inv_cov;

        if((*it)->type == Feature::EDGELET)
        {
            double e_edge = (*it)->grad.transpose()*e;
            chi2_vec_final.push_back(e_edge*e_edge);

            if(fabs(e_edge) > reproj_thresh_scaled_ls)
            {
                ++n_deleted_refs;
                (*it)->point = NULL;
            }
        }
        else
        {
            chi2_vec_final.push_back(e.squaredNorm());

            if(e.norm() > reproj_thresh_scaled_pt)
            {
                ++n_deleted_refs;
                (*it)->point = NULL;
            }
        }

        ++index;
    }


    error_init=0.0;
    error_final=0.0;
    if(!chi2_vec_init.empty())
        error_init = sqrt(hso::getMedian(chi2_vec_init))*frame->cam_->errorMultiplier2();
    if(!chi2_vec_final.empty())
        error_final = sqrt(hso::getMedian(chi2_vec_final))*frame->cam_->errorMultiplier2();

    estimated_scale *= frame->cam_->errorMultiplier2();
    num_obs -= n_deleted_refs;

}

} // namespace pose_optimizer
} // namespace hso
