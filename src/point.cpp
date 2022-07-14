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
#include <hso/point.h>
#include <hso/frame.h>
#include <hso/feature.h>
#include <hso/config.h>

#include "hso/vikit/math_utils.h"


namespace hso {

int Point::point_counter_ = 0;

Point::Point(const Vector3d& pos) :
    id_(point_counter_++),
    pos_(pos),
    normal_set_(false),
    n_obs_(0),
    v_pt_(NULL),
    last_published_ts_(0),
    last_projected_kf_id_(-1),
    type_(TYPE_UNKNOWN),
    n_failed_reproj_(0),
    n_succeeded_reproj_(0),
    last_structure_optim_(0)
{
    isBad_ = false;

    vPoint_ = NULL;
    nBA_ = 0;
}

Point::Point(const Vector3d& pos, Feature* ftr) :
    id_(point_counter_++),
    pos_(pos),
    normal_set_(false),
    n_obs_(1),
    v_pt_(NULL),
    last_published_ts_(0),
    last_projected_kf_id_(-1),
    type_(TYPE_UNKNOWN),
    n_failed_reproj_(0),
    n_succeeded_reproj_(0),
    last_structure_optim_(0)
{
    obs_.push_front(ftr);
    isBad_ = false;

    vPoint_ = NULL;

    nBA_ = 0;
}

Point::~Point()
{}

void Point::addFrameRef(Feature* ftr)
{
    obs_.push_front(ftr);
    ++n_obs_;
}

Feature* Point::findFrameRef(Frame* frame)
{
    for(auto it=obs_.begin(), ite=obs_.end(); it!=ite; ++it)
        if((*it)->frame == frame)
            return *it;
    return NULL;    // no keyframe found
}

bool Point::deleteFrameRef(Frame* frame)
{
    for(auto it=obs_.begin(), ite=obs_.end(); it!=ite; ++it)
        if((*it)->frame == frame)
        {
            obs_.erase(it);
            return true;
        }

    return false;
}

void Point::initNormal()
{
    assert(!obs_.empty());
    const Feature* ftr = obs_.back();
    assert(ftr->frame != NULL);
    normal_ = ftr->frame->T_f_w_.rotation_matrix().transpose()*(-ftr->f);
    normal_information_ = DiagonalMatrix<double,3,3>(pow(20/(pos_-ftr->frame->pos()).norm(),2), 1.0, 1.0);
    normal_set_ = true;
}

bool Point::getCloseViewObs(const Vector3d& framepos, Feature*& ftr) const
{
    // TODO: get frame with same point of view AND same pyramid level!
    Vector3d obs_dir(framepos - pos_); obs_dir.normalize();
    auto min_it=obs_.begin();
    double min_cos_angle = 0;
    for(auto it=obs_.begin(), ite=obs_.end(); it!=ite; ++it)
    {
        Vector3d dir((*it)->frame->pos() - pos_); dir.normalize();
        double cos_angle = obs_dir.dot(dir);
        if(cos_angle > min_cos_angle)
        {
            min_cos_angle = cos_angle;
            min_it = it;
        }
    }
    ftr = *min_it;
    if(min_cos_angle < 0.5) return false; // assume that observations larger than 60° are useless

    return true;
}

// #define POINT_OPTIMIZER_DEBUG

void Point::optimize(const size_t n_iter)
{
    // Vector3d old_point = pos_;
    // double chi2 = 0.0;
    // Matrix3d A;
    // Vector3d b;

    double old_idist = idist_;
    double chi2 = 0.0;
    double H=0,b=0;

    if(obs_.size() < 5) return;

    Frame* host = hostFeature_->frame;
    for(size_t i=0; i<n_iter; i++)
    {
        // A.setZero();
        // b.setZero();
        // double new_chi2 = 0.0;

        H=b=0;
        double new_chi2 = 0.0;

        // compute residuals
        for(auto it=obs_.begin(); it!=obs_.end(); ++it)
        {
            // Matrix23d J;
            // const Vector3d p_in_f((*it)->frame->T_f_w_ * pos_);
            // Point::jacobian_xyz2uv(p_in_f, (*it)->frame->T_f_w_.rotation_matrix(), J);
            // const Vector2d e(hso::project2d((*it)->f) - hso::project2d(p_in_f));

            Feature* ft = *it;
            if(ft->frame->id_ == host->id_) continue;

            Frame* target = ft->frame;
            SE3 Tth = target->T_f_w_ * host->T_f_w_.inverse();

            Vector3d pTarget = Tth * (hostFeature_->f*(1.0/idist_));
            Vector2d e(hso::project2d(ft->f) - hso::project2d(pTarget));    

            Vector2d J;
            jacobian_id2uv(pTarget, Tth, idist_, hostFeature_->f, J);


            if((*it)->type == Feature::EDGELET)
            {
                // double err_edge = (*it)->grad.transpose() * e;
                // new_chi2 += err_edge * err_edge;
                // A.noalias() += J.transpose()*(*it)->grad * (*it)->grad.transpose() * J;
                // b.noalias() -= J.transpose() *(*it)->grad * err_edge;

                double e_edge = (*it)->grad.transpose() * e;
                new_chi2 += e_edge * e_edge;

                double J_edge = (*it)->grad.transpose() * J;
                H += J_edge*J_edge;
                b -= J_edge*e_edge;
            }
            else
            {
                // new_chi2 += e.squaredNorm();
                // A.noalias() += J.transpose() * J;
                // b.noalias() -= J.transpose() * e;

                new_chi2 += e.squaredNorm();
                H += J.transpose() * J;
                b -= J.transpose() * e;
            }
        }

        // solve linear system
        // const Vector3d dp(A.ldlt().solve(b));
        const double id_step = b/H;

        // check if error increased
        if((i > 0 && new_chi2 > chi2) || (bool) std::isnan(id_step))
        {
            #ifdef POINT_OPTIMIZER_DEBUG
                cout << "it " << i << "\t FAILURE \t new_chi2 = " << new_chi2 << endl;
            #endif

            // pos_ = old_point; // roll-back
            idist_ = old_idist;
            break;
        }

        // update the model
        // Vector3d new_point = pos_ + dp;
        // old_point = pos_;
        // pos_ = new_point;
        // chi2 = new_chi2;

        double new_id = idist_+id_step;
        old_idist = idist_;
        idist_ = new_id;
        chi2 = new_chi2;

        #ifdef POINT_OPTIMIZER_DEBUG
            cout << "it " << i
            << "\t Success \t new_chi2 = " << new_chi2
            << "\t norm(b) = " << fabs(id_step)//hso::norm_max(b)
            << endl;
        #endif

        // stop when converged
        // if(hso::norm_max(dp) <= EPS) break;
        if(fabs(id_step) < 0.00001) break;
    }

    #ifdef POINT_OPTIMIZER_DEBUG
        cout << endl;
    #endif

    //update 3D position
    Vector3d pHost = hostFeature_->f*(1.0/idist_);
    pos_ = hostFeature_->frame->T_f_w_.inverse() * pHost;

}


void Point::optimizeLM(const size_t n_iter)
{
    double chi2 = 0.0;
    double rho = 0;
    double mu = 0.1;
    double nu = 2.0;
    bool stop = false;
    int n_trials = 0;
    Matrix3d A;
    Vector3d b;
    const int n_trials_max = 5;


    for(auto it=obs_.begin(); it!=obs_.end(); ++it)
    {
        const Vector3d p_in_f((*it)->frame->T_f_w_ * pos_);
        const Vector2d e(hso::project2d((*it)->f) - hso::project2d(p_in_f));

        chi2 += e.squaredNorm();
    }

    for(size_t iter = 0; iter < n_iter; iter++)
    {
        rho = 0;
        n_trials = 0;
        do
        {
            Vector3d new_pos;
            double new_chi2 = 0.0;
            A.setZero();
            b.setZero();

            for(auto it=obs_.begin(); it!=obs_.end(); ++it)
            {
                Matrix23d J;
                const Vector3d p_in_f((*it)->frame->T_f_w_ * pos_);
                Point::jacobian_xyz2uv(p_in_f, (*it)->frame->T_f_w_.rotation_matrix(), J);
                const Vector2d e(hso::project2d((*it)->f) - hso::project2d(p_in_f));

                A.noalias() += J.transpose() * J ;
                b.noalias() -= J.transpose() * e ;
            }

            A += (A.diagonal()*mu).asDiagonal();
            const Vector3d dp(A.ldlt().solve(b));

        if(!(bool)std::isnan((double)dp[0]))
        {
            new_pos = pos_ + dp;

            for(auto it=obs_.begin(); it!=obs_.end(); ++it)
            {
                Matrix23d J;
                const Vector3d p_in_f((*it)->frame->T_f_w_ * new_pos);
                const Vector2d e(hso::project2d((*it)->f) - hso::project2d(p_in_f));

                new_chi2 += e.squaredNorm();

            }
            rho = chi2 - new_chi2;
        }
        else
        {
            #ifdef POINT_OPTIMIZER_DEBUG
                cout << "Matrix is close to singular!" << endl;
                cout << "H = " << A << endl;
                cout << "Jres = " << b << endl;
            #endif

            rho = -1;
        }

        if(rho > 0)
        {
            pos_ = new_pos;
            chi2 = new_chi2;
            stop = hso::norm_max(dp) <= EPS;
            mu *= std::max(1./3., std::min(1.-std::pow(2*rho-1,3), 2./3.));
            nu = 2.;

            #ifdef POINT_OPTIMIZER_DEBUG
                cout << "It. " << iter
                << "\t Trial " << n_trials
                << "\t Success"
                << "\t new_chi2 = " << new_chi2
                << "\t mu = " << mu
                << "\t nu = " << nu
                << endl;
            #endif
        }
        else
        {
            mu *= nu;
            nu *= 2.;
            ++n_trials;
            if (n_trials >= n_trials_max) stop = true;

            #ifdef POINT_OPTIMIZER_DEBUG
                cout << "It. " << iter
                << "\t Trial " << n_trials
                << "\t Failure"
                << "\t new_chi2 = " << new_chi2
                << "\t mu = " << mu
                << "\t nu = " << nu
                << endl;
            #endif
        }

        }while(!(rho>0 || stop));

        if (stop) break;
    }

    #ifdef POINT_OPTIMIZER_DEBUG
        cout << "======================" << endl;
    #endif
}

// #define POINT_OPTIMIZER_DEBUG                        
void Point::optimizeID(const size_t n_iter)
{
    double oldEnergy = 0;
    double rho = 0;
    double mu = 0.1;
    double nu = 2.0;
    bool stop = false;
    int n_trials = 0;
    const int n_trials_max = 5;

    double H = 0;
    double b = 0;

    const float idistOld = idist_;

    // vector<SE3> Ts;
    Frame* host = hostFeature_->frame;
    for(auto it=obs_.begin(); it!=obs_.end(); ++it)
    {
        Feature* ft = *it;
        if(ft->frame->id_ == host->id_)
            continue;

        Frame* target = ft->frame;
        SE3 Tth = target->T_f_w_ * host->T_f_w_.inverse();
        // Ts.push_back(Tth);

        Vector3d pTarget = Tth * (hostFeature_->f*(1.0/idist_));
        Vector2d e(hso::project2d(ft->f) - hso::project2d(pTarget));

        oldEnergy += e.squaredNorm();
    }

    // //refresh position
    // Vector3d pHost = hostFeature_->f*(1.0/idist_);
    // pos_ = host->T_f_w_.inverse() * pHost;

    // if(oldEnergy == 0) return;
    assert(oldEnergy > 0);

    for(size_t iter = 0; iter < n_iter; iter++)
    {
        rho = 0;
        n_trials = 0;

        do
        {
            double newid;
            double newEnergy = 0.0;
            H = 0; b = 0;

            // size_t j = 0;
            for(auto it=obs_.begin(); it!=obs_.end(); ++it)
            {
                Feature* ft = *it;
                if(ft->frame->id_ == host->id_)
                    continue;

                Frame* target = ft->frame;
                SE3 Tth = target->T_f_w_ * host->T_f_w_.inverse();
                // SE3 Tth = Ts[j];
                // j++;

                Vector3d pTarget = Tth * (hostFeature_->f*(1.0/idist_));
                Vector2d e(hso::project2d(ft->f) - hso::project2d(pTarget));

                Vector2d Juvdd;
                jacobian_id2uv(pTarget, Tth, idist_, hostFeature_->f, Juvdd);
                H += Juvdd.transpose()*Juvdd;
                b -= Juvdd.transpose()*e;
            }

            H *= 1+mu;
            double step = (1.0/H)*b;

            if(!(bool)std::isnan(step))
            {
                newid = idist_+step;

                // size_t j = 0;
                for(auto it=obs_.begin(); it!=obs_.end(); ++it)
                {
                    Feature* ft = *it;
                    if(ft->frame->id_ == host->id_)
                        continue;

                    Frame* target = ft->frame;
                    SE3 Tth = target->T_f_w_ * host->T_f_w_.inverse();
                    // SE3 Tth = Ts[j];
                    // j++;

                    Vector3d pTarget = Tth * (hostFeature_->f*(1.0/newid));
                    Vector2d e(hso::project2d(ft->f) - hso::project2d(pTarget)); 
                    newEnergy += e.squaredNorm();
                }
                rho = oldEnergy - newEnergy;
            }
            else
            {
                #ifdef POINT_OPTIMIZER_DEBUG
                    cout << "Matrix is close to singular!" << endl;
                    cout << "H = " << H << endl;
                    cout << "b = " << b << endl;
                    cout << "Energy = " << oldEnergy << endl;
                #endif

                rho = -1;
            }

            if(rho > 0)
            {
                idist_ = newid;
                oldEnergy = newEnergy;
                stop = (step <= EPS);
                mu *= std::max(1./3., std::min(1.-std::pow(2*rho-1,3), 2./3.));
                nu = 2.;

                #ifdef POINT_OPTIMIZER_DEBUG
                    cout << "It. " << iter
                    << "\t Trial " << n_trials
                    << "\t Success"
                    << "\t new_chi2 = " << newEnergy
                    << "\t mu = " << mu
                    << "\t nu = " << nu
                    << endl;
                #endif
            }
            else
            {
                mu *= nu;
                nu *= 2.;
                ++n_trials;
                if (n_trials >= n_trials_max) stop = true;

                #ifdef POINT_OPTIMIZER_DEBUG
                    cout << "It. " << iter
                    << "\t Trial " << n_trials
                    << "\t Failure"
                    << "\t new_chi2 = " << newEnergy
                    << "\t mu = " << mu
                    << "\t nu = " << nu
                    << endl;
                #endif
            }   
        }while(!(rho>0 || stop));

        if(stop) break;
    }


    //update position
    Vector3d pHost = hostFeature_->f*(1.0/idist_);
    pos_ = hostFeature_->frame->T_f_w_.inverse() * pHost;


    #ifdef POINT_OPTIMIZER_DEBUG
        cout << "Before = " << idistOld << "\t" << "After = " << idist_ << endl;
    #endif

    #ifdef POINT_OPTIMIZER_DEBUG
        cout << "======================" << endl;
    #endif
}
} // namespace hso
