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

#include <algorithm>
#include <numeric>

#include <boost/bind.hpp>
#include <boost/math/distributions/normal.hpp>
#include <hso/global.h>
#include <hso/depth_filter.h>
#include <hso/frame.h>
#include <hso/point.h>
#include <hso/feature.h>
#include <hso/matcher.h>
#include <hso/config.h>
#include <hso/feature_detection.h>
#include <hso/IndexThreadReduce.h>
#include <hso/matcher.h>
#include <hso/feature_alignment.h>

#include "hso/vikit/robust_cost.h"
#include "hso/vikit/math_utils.h"

namespace hso {

int Seed::batch_counter = 0;
int Seed::seed_counter = 0;

Seed::Seed(Feature* ftr, float depth_mean, float depth_min, float converge_threshold) :
    batch_id(batch_counter),
    id(seed_counter++),
    ftr(ftr),
    a(10),
    b(10),
    mu(1.0/depth_mean),
    z_range(1.0/depth_min),
    sigma2(z_range*z_range/36),
    isValid(true),
    eplStart(Vector2i(0,0)),
    eplEnd(Vector2i(0,0)),
    haveReprojected(false),
    temp(NULL)
{
    vec_distance.push_back(depth_mean);
    vec_sigma2.push_back(sigma2);

    converge_thresh = converge_threshold;
}

DepthFilter::DepthFilter(
    feature_detection::FeatureExtractor* featureExtractor, callback_t seed_converged_cb) :
    featureExtractor_(featureExtractor),
    seed_converged_cb_(seed_converged_cb),
    seeds_updating_halt_(false),
    thread_(NULL),
    new_keyframe_set_(false),
    new_keyframe_min_depth_(0.0),
    new_keyframe_mean_depth_(0.0),
    px_error_angle_(-1)
{
    frame_prior_.resize(100000);

    threadReducer_ = new lsd_slam::IndexThreadReduce();

    runningStats_ = new RunningStats();
    n_update_last_ = 100;




    n_pre_update_ = 0;
    n_pre_try_ = 0;

    nPonits = 1;
    nSkipFrame = 0;


    nMeanConvergeFrame_ = 6;

    convergence_sigma2_thresh_ = 200;
}

DepthFilter::~DepthFilter()
{
    stopThread();
    HSO_INFO_STREAM("DepthFilter destructed.");
}

void DepthFilter::startThread()
{
    thread_ = new boost::thread(&DepthFilter::updateSeedsLoop, this);
}

void DepthFilter::stopThread()
{
    HSO_INFO_STREAM("DepthFilter stop thread invoked.");
    if(thread_ != NULL)
    {
        HSO_INFO_STREAM("DepthFilter interrupt and join thread... ");
        seeds_updating_halt_ = true;
        thread_->interrupt();
        thread_->join();
        thread_ = NULL;
    }

    delete threadReducer_;
    delete runningStats_;
}

void DepthFilter::addFrame(FramePtr frame)
{
    if(thread_ != NULL)
    {
        {
            lock_t lock(frame_queue_mut_);
            if(frame_queue_.size() > 2)
                frame_queue_.pop();
            frame_queue_.push(frame);
        }
        seeds_updating_halt_ = false;
        frame_queue_cond_.notify_one();
    }
    else
        updateSeeds(frame);
}

void DepthFilter::addKeyframe(FramePtr frame, double depth_mean, double depth_min, float converge_thresh)
{
    new_keyframe_min_depth_ = depth_min;
    new_keyframe_mean_depth_ = depth_mean;
    convergence_sigma2_thresh_ = converge_thresh;

    if(thread_ != NULL)
    {
        new_keyframe_ = frame;
        new_keyframe_set_ = true;
        seeds_updating_halt_ = true;
        frame_queue_cond_.notify_one();
    }
    else
        initializeSeeds(frame);
}

void DepthFilter::initializeSeeds(FramePtr frame)
{
    boost::unique_lock<boost::mutex> lock_c(detector_mut_);
    Features new_features;

    featureExtractor_->setExistingFeatures(frame->fts_);
    featureExtractor_->detect(frame.get(), 20, frame->gradMean_, new_features, frame->m_last_frame.get());

    lock_c.unlock();

    // initialize a seed for every new feature
    seeds_updating_halt_ = true;

    // by locking the updateSeeds function stops
    lock_t lock(seeds_mut_); 
    ++Seed::batch_counter;

    std::for_each(new_features.begin(), new_features.end(), [&](Feature* ftr)
    {
        Seed seed(ftr, new_keyframe_mean_depth_, new_keyframe_min_depth_, convergence_sigma2_thresh_);
        // seed.pre_frames.insert(seed.pre_frames.begin(), frame_prior_[Seed::batch_counter-1].begin(), frame_prior_[Seed::batch_counter-1].end());

        for(auto it = frame_prior_[Seed::batch_counter-1].begin(); it != frame_prior_[Seed::batch_counter-1].end(); ++it)
        {
            // if((*it)->isKeyframe()) break;
            seed.pre_frames.push_back(*it);
            // if(seed.pre_frames.size() > 1)
            //     assert(seed.pre_frames[0]->id_ > seed.pre_frames[1]->id_);
        }

        seeds_.push_back(seed);
    });

    // if(options_.verbose)
    //     HSO_INFO_STREAM("DepthFilter: Initialized "<<new_features.size()<<" new seeds");

    seeds_updating_halt_ = false;


    // release memory
    frame->finish();
}

void DepthFilter::removeKeyframe(FramePtr frame)
{
  seeds_updating_halt_ = true;
  lock_t lock(seeds_mut_);
  std::list<Seed>::iterator it=seeds_.begin();
  size_t n_removed = 0;
  while(it!=seeds_.end())
  {
    if(it->ftr->frame == frame.get())
    {
      it = seeds_.erase(it);
      ++n_removed;
    }
    else
      ++it;
  }
  seeds_updating_halt_ = false;
}

void DepthFilter::reset()
{
  seeds_updating_halt_ = true;
  {
    lock_t lock(seeds_mut_);
    seeds_.clear();
  }
  lock_t lock();
  while(!frame_queue_.empty())
    frame_queue_.pop();
  seeds_updating_halt_ = false;

  if(options_.verbose)
    HSO_INFO_STREAM("DepthFilter: RESET.");
}

void DepthFilter::updateSeedsLoop()
{
    while(!boost::this_thread::interruption_requested())
    {
        FramePtr frame;
        {
            lock_t lock(frame_queue_mut_);
            if(seeds_.empty())
            {
                while(frame_queue_.empty() && new_keyframe_set_ == false)
                    frame_queue_cond_.wait(lock);
            }
            else
            {
                std::list<Seed>::iterator it=seeds_.begin();
                // it--;

                while(frame_queue_.empty() && new_keyframe_set_ == false && it != seeds_.end())
                {
                    observeDepthWithPreviousFrameOnce(it);
                    it++;
                }
            }

            if(!frame_queue_.empty() || new_keyframe_set_)
            {
                if(new_keyframe_set_)
                {
                    new_keyframe_set_ = false;
                    seeds_updating_halt_ = false;
                    clearFrameQueue();
                    frame = new_keyframe_;
                }
                else
                {
                    frame = frame_queue_.front();
                    frame_queue_.pop();
                }
            }
            else
            {
                while(frame_queue_.empty() && new_keyframe_set_ == false)
                    frame_queue_cond_.wait(lock);

                if(new_keyframe_set_)
                {
                    new_keyframe_set_ = false;
                    seeds_updating_halt_ = false;
                    clearFrameQueue();
                    frame = new_keyframe_;
                }
                else
                {
                    frame = frame_queue_.front();
                    frame_queue_.pop();
                }
            }


            n_pre_update_ = 0;
            n_pre_try_ = 0;

            // if(new_keyframe_set_)
            // {
            //   new_keyframe_set_ = false;
            //   seeds_updating_halt_ = false;
            //   clearFrameQueue();
            //   frame = new_keyframe_;
            // }
            // else
            // {
            //   frame = frame_queue_.front();
            //   frame_queue_.pop();
            // }

        }

        updateSeeds(frame);

        if(frame->isKeyframe())
        {
            // propagateDepth(frame);

            initializeSeeds(frame);
        }
    }
}

void DepthFilter::updateSeeds(FramePtr frame)
{
    if(!frame->isKeyframe())
        frame_prior_[Seed::batch_counter].push_front(frame);
    else
        frame_prior_[Seed::batch_counter+1].push_front(frame);

    
    if(Seed::batch_counter > 5 && frame->isKeyframe())
    {
        list<FramePtr>::iterator it = frame_prior_[Seed::batch_counter-5].begin();
        while(it != frame_prior_[Seed::batch_counter-5].end())
        {
            Frame* dframe = (*it).get();
            if(!dframe->isKeyframe())
            {
                delete dframe;
                dframe = NULL;
            }
            ++it;
        }
    }

    active_frame_ = frame;
    // update only a limited number of seeds, because we don't have time to do it
    // for all the seeds in every frame!

    // size_t n_updates=0, n_failed_matches=0, n_seeds = seeds_.size();
    lock_t lock(seeds_mut_);
  
    if(this->px_error_angle_ == -1)
    {
        const double focal_length = frame->cam_->errorMultiplier2();
        double px_noise = 1.0;
        double px_error_angle = atan(px_noise/(2.0*focal_length))*2.0; 
        this->px_error_angle_ = px_error_angle;
    }

    std::list<Seed>::iterator it=seeds_.begin();
    while(it!=seeds_.end())
    {
        if(seeds_updating_halt_)
            return;

        // check if seed is not already too old
        if((Seed::batch_counter - it->batch_id) > options_.max_n_kfs)
        {
            assert(it->ftr->point == NULL); // TODO this should not happen anymore
            
            if(it->temp != NULL && it->haveReprojected)
                it->temp->seedStates_ = -1;
            else
            {
                delete it->ftr;
                it->ftr = NULL;
            }
            // seed_finish_cb_(it->temp, false);




            it->pre_frames.clear();
            it->optFrames_P.clear();
            it->optFrames_A.clear();


            it = seeds_.erase(it);
            continue;
        }

        it++;
    }

    observeDepth();

    it = seeds_.begin();
    while(it!=seeds_.end())
    {
        if(seeds_updating_halt_)
            return;

        if(sqrt(it->sigma2) < it->z_range/it->converge_thresh)
        {
            assert(it->ftr->point == NULL); // TODO this should not happen anymore

            

            bool isValid = true;
            if(activatePoint(*it, isValid)) 
                it->mu = it->opt_id;
            

            //depth check
            Vector3d pHost = it->ftr->f * (1.0/it->mu);
            if(it->mu < 1e-10 || pHost[2] < 1e-10)  isValid = false;

            if(!isValid)
            {   
                if(it->temp != NULL && it->haveReprojected)
                    it->temp->seedStates_ = -1;

                it = seeds_.erase(it);
                continue;
            }


            
            {
                // nPonits++;
                // nSkipFrame += it->vec_distance.size();
                // lock_t lock(mean_mutex_);
                // nMeanConvergeFrame_ = floor(float(nSkipFrame)/nPonits);

                if(m_v_n_converge.size() > (size_t)Config::maxFts())
                    m_v_n_converge.erase(m_v_n_converge.begin());

                m_v_n_converge.push_back(it->vec_distance.size());
            }


            
            Vector3d xyz_world = it->ftr->frame->T_f_w_.inverse() * pHost;
            Point* point = new Point(xyz_world, it->ftr);

            point->idist_ = it->mu;
            point->hostFeature_ = it->ftr;

            // const int L = it->ftr->level;
            // Vector2d pxL(it->ftr->px/(1<<L));
            point->color_ = it->ftr->frame->img_pyr_[0].at<uchar>((int)it->ftr->px[1], (int)it->ftr->px[0]);


            if(it->ftr->type == Feature::EDGELET)
                point->ftr_type_ = Point::FEATURE_EDGELET;
            else if(it->ftr->type == Feature::CORNER)
                point->ftr_type_ = Point::FEATURE_CORNER;
            else
                point->ftr_type_ = Point::FEATURE_GRADIENT;

            it->ftr->point = point;
            

            if(it->temp != NULL && it->haveReprojected)
                it->temp->seedStates_ = 1;
            else
                assert(it->temp == NULL && !it->haveReprojected);


            it->pre_frames.clear();
            it->optFrames_P.clear();
            it->optFrames_A.clear();

            // // FIXME it is not threadsafe to add a feature to the frame here.
            // if(active_frame_->isKeyframe() && it->last_update_frame == active_frame_)
            // {
            //     lock_t lock(m_converge_seed_mut);
            //     m_converge_seed.push_back(*it);
            // }
            // else
                seed_converged_cb_(point, it->sigma2); // put in candidate list


            it = seeds_.erase(it);
        }
        else if(!it->isValid)
        {
            HSO_WARN_STREAM("z_min is NaN");
            it = seeds_.erase(it);
        }
        else
            ++it;
    }

    lock_t lock_converge(mean_mutex_);
    if( m_v_n_converge.size() > size_t(0.5*Config::maxFts()) )
        nMeanConvergeFrame_ = std::accumulate(m_v_n_converge.begin(), m_v_n_converge.end(), 0) / m_v_n_converge.size();
    else
        nMeanConvergeFrame_ = 6;

}

void DepthFilter::clearFrameQueue()
{
  while(!frame_queue_.empty())
    frame_queue_.pop();
}

void DepthFilter::getSeedsCopy(const FramePtr& frame, std::list<Seed>& seeds)
{
  lock_t lock(seeds_mut_);
  for(std::list<Seed>::iterator it=seeds_.begin(); it!=seeds_.end(); ++it)
  {
    if (it->ftr->frame == frame.get())
      seeds.push_back(*it);
  }
}

#define UNZERO(val) (val < 0 ? (val > -1e-10 ? -1e-10 : val) : (val < 1e-10 ? 1e-10 : val))
void DepthFilter::updateSeed(const float x, const float tau2, Seed* seed)
{
    float id_var = seed->sigma2*1.01f;
    float w = tau2 / (tau2 + id_var);
    float new_idepth = (1-w)*x + w*seed->mu;
    seed->mu = UNZERO(new_idepth);
    id_var *= w;

    if(id_var < seed->sigma2) seed->sigma2 = id_var;
}

double DepthFilter::computeTau(
      const SE3& T_ref_cur,
      const Vector3d& f,
      const double z,
      const double px_error_angle)
{
    Vector3d t(T_ref_cur.translation());
    Vector3d a = f*z-t;
    double t_norm = t.norm();
    double a_norm = a.norm();
    double alpha = acos(f.dot(t)/t_norm); // dot product
    double beta = acos(a.dot(-t)/(t_norm*a_norm)); // dot product
    double beta_plus = beta + px_error_angle;
    double gamma_plus = PI-alpha-beta_plus; // triangle angles sum to PI
    double z_plus = t_norm*sin(beta_plus)/sin(gamma_plus); // law of sines
    return (z_plus - z); // tau
}

void DepthFilter::observeDepth()
{
  threadReducer_->reduce(boost::bind(&DepthFilter::observeDepthRow, this, _1, _2, _3), 0, (int)seeds_.size(), runningStats_, 10);

  runningStats_->n_seeds = seeds_.size();
  // if(options_.verbose)
  // {
  //   HSO_INFO_STREAM("DepthFilter:"
  //                   <<"  n_seeds:"<< runningStats_->n_seeds
  //                   <<"  n_updates:"<< runningStats_->n_updates 
  //                   <<"  n_fail_matches:"<< runningStats_->n_failed_matches
  //                   <<"  n_out_views:" << runningStats_->n_out_views);

  //    // HSO_INFO_STREAM("DepthFilter:"
  //    //                <<"  n_fail_lsd:"<< runningStats_->n_fail_lsd
  //    //                <<"  n_fail_triangulation:"<< runningStats_->n_fail_triangulation 
  //    //                <<"  n_fail_alignment:"<< runningStats_->n_fail_alignment
  //    //                <<"  n_fail_score:" << runningStats_->n_fail_score);
  // }
  n_update_last_ = runningStats_->n_updates;
  runningStats_->setZero();
}

void DepthFilter::observeDepthRow(int yMin, int yMax, RunningStats* stats)
{
    if(seeds_updating_halt_) return;

    std::list<Seed>::iterator it=seeds_.begin();
    for(int i = 0; i < yMin; ++i) it++;


    for(int idx = yMin; idx < yMax; ++idx, ++it)
    {
        if(seeds_updating_halt_) return;

        // check if point is visible in the current image
        SE3 T_ref_cur = it->ftr->frame->T_f_w_ * active_frame_->T_f_w_.inverse();

        Vector3d xyz_f = T_ref_cur.inverse()*(1.0/it->mu * it->ftr->f);
        if(xyz_f.z() < 0.0)  // behind the camera
        {
            stats->n_out_views++;
            it->is_update = false;
            continue;
        }

        if(!active_frame_->cam_->isInFrame(active_frame_->f2c(xyz_f).cast<int>()))  // point does not project in image
        {
            stats->n_out_views++;
            it->is_update = false;
            continue;
        }

        it->is_update = true;

        if(it->optFrames_A.size() < 15)
            it->optFrames_A.push_back(active_frame_);


        float z_inv_min = it->mu + 2*sqrt(it->sigma2);
        float z_inv_max = max(it->mu - 2*sqrt(it->sigma2), 0.00000001f);
        if(isnan(z_inv_min)) it->isValid = false;


        Matcher matcher;
        double z;
        int res = matcher.doLineStereo(*it->ftr->frame, 
                                       *active_frame_, 
                                       *it->ftr,
                                        1.0/z_inv_min, 
                                        1.0/it->mu, 
                                        1.0/z_inv_max, 
                                        z, 
                                        it->eplStart, 
                                        it->eplEnd);
        if(res != 1)
        {
            it->b++; // increase outlier probability when no match was found
            it->eplStart = Vector2i(0,0);
            it->eplEnd   = Vector2i(0,0);

            stats->n_failed_matches++;

            if(res == -1)
                stats->n_fail_lsd++;
            else if(res == -2)
                stats->n_fail_triangulation++;
            else if(res == -3)
                stats->n_fail_alignment++;
            else if(res == -4)
                stats->n_fail_score++;

            continue;
        }


        // compute tau
        double tau = computeTau(T_ref_cur, it->ftr->f, z, this->px_error_angle_);
        double tau_inverse = 0.5 * (1.0/max(0.0000001, z-tau) - 1.0/(z+tau));

        // update the estimate
        updateSeed(1./z, tau_inverse*tau_inverse, &*it);

        it->vec_distance.push_back(1.0/it->mu);
        it->vec_sigma2.push_back(it->sigma2);

        it->last_update_frame = active_frame_;
        it->last_matched_px = matcher.px_cur_;
        it->last_matched_level = matcher.search_level_;

        stats->n_updates++;

        if(active_frame_->isKeyframe())
        {
            boost::unique_lock<boost::mutex> lock(detector_mut_);
            featureExtractor_->setGridOccpuancy(matcher.px_cur_, it->ftr);
        }
    }
}

void DepthFilter::observeDepthWithPreviousFrameOnce(std::list<Seed>::iterator& ite)
{
    if(ite->pre_frames.empty() || this->px_error_angle_ == -1)
        return;

    // FramePtr preFrame = *(ite->pre_frames.end()-1);
    FramePtr preFrame = *(ite->pre_frames.begin());
    assert(preFrame->id_ < ite->ftr->frame->id_);

    n_pre_try_++;

    SE3 T_ref_cur = ite->ftr->frame->T_f_w_ * preFrame->T_f_w_.inverse();
    Vector3d xyz_f = T_ref_cur.inverse()*(1.0/ite->mu * ite->ftr->f);

    if(xyz_f.z() < 0.0) 
    {
        ite->pre_frames.erase(ite->pre_frames.begin());
        return;
    }
    if(!preFrame->cam_->isInFrame(preFrame->f2c(xyz_f).cast<int>()))
    {
        ite->pre_frames.erase(ite->pre_frames.begin());
        return;
    }

    if(ite->optFrames_P.size() < 15)
        ite->optFrames_P.push_back(preFrame);


    float z_inv_min = ite->mu + 2*sqrt(ite->sigma2);
    float z_inv_max = max(ite->mu - 2*sqrt(ite->sigma2), 0.00000001f);
    double z;

    if(!matcher_.findEpipolarMatchPrevious(*ite->ftr->frame, *preFrame, *ite->ftr, 1.0/ite->mu, 1.0/z_inv_min, 1.0/z_inv_max, z))
    {
        ite->pre_frames.erase(ite->pre_frames.begin());
        return;
    }

    n_pre_update_++;

    double tau = computeTau(T_ref_cur, ite->ftr->f, z, this->px_error_angle_);
    double tau_inverse = 0.5 * (1.0/max(0.0000001, z-tau) - 1.0/(z+tau));

    // update the estimate
    updateSeed(1./z, tau_inverse*tau_inverse, &*ite);

    ite->pre_frames.erase(ite->pre_frames.begin());
    
}

// #define ACTIVATE_DBUG  
bool DepthFilter::activatePoint(Seed& seed, bool& isValid)
{   
    seed.opt_id = seed.mu;

    const int halfPatchSize = 4;
    const int patchSize = halfPatchSize*2;
    const int patchArea = patchSize*patchSize;

    Frame* host = seed.ftr->frame;
    Vector3d pHost = seed.ftr->f*(1.0/seed.mu);

    vector< pair<FramePtr, Vector2d> > targets;
    targets.reserve(seed.optFrames_P.size()+seed.optFrames_A.size());
    for(size_t i = 0; i < seed.optFrames_P.size(); ++i)
    {
        FramePtr target = seed.optFrames_P[i];
        SE3 Tth = target->T_f_w_ * host->T_f_w_.inverse();
        Vector3d pTarget = Tth*pHost;
        if(pTarget[2] < 0.0001) continue;

        Vector2d px(target->cam_->world2cam(pTarget));
        if(!target->cam_->isInFrame(px.cast<int>(), 8)) 
            continue;

        targets.push_back(make_pair(target, px));
    }

    // cout << "pre size = " << targets.size() << endl;

    for(size_t i = 0; i < seed.optFrames_A.size(); ++i)
    {
        FramePtr target = seed.optFrames_A[i];
        SE3 Tth = target->T_f_w_ * host->T_f_w_.inverse();
        Vector3d pTarget = Tth*pHost;
        if(pTarget[2] < 0.0001) continue;

        Vector2d px(target->cam_->world2cam(pTarget));
        if(!target->cam_->isInFrame(px.cast<int>(), 8)) 
            continue;

        targets.push_back(make_pair(target, px));
    }

    float n_frame_thresh = nMeanConvergeFrame_*0.7;
    if(n_frame_thresh > 8)  n_frame_thresh = 8;
    if(n_frame_thresh < 3)  n_frame_thresh = 3;

    if(targets.size() < n_frame_thresh) return false;

    double distMean = 0;
    vector< pair<FramePtr, Vector2d> > targetResult; 
    targetResult.reserve(targets.size()); 
    vector<Vector2d> targetNormal; targetNormal.reserve(targets.size());
    for(size_t i = 0; i < targets.size(); ++i)
    {
        // cout << "---------------------" << endl;
        // cout << "before:   " << targets[i].second[0] << "  " << targets[i].second[1] << endl;
        Vector2d beforePx(targets[i].second);
        Matcher matcher;
        if(matcher.findMatchSeed(seed, *(targets[i].first.get()), targets[i].second, 0.65))
        {
            // cout << "after:   " << targets[i].second[0] << "  " << targets[i].second[1] << endl;
            // cout << "host Level = " << seed.ftr->level << "  Target Level = " << matcher.search_level_ << endl;

            Vector2d afterPx(targets[i].second);

            // cout << "optimize distance = " << (beforePx-afterPx).norm() << endl;
            // if((beforePx-afterPx).norm() > 3.0) continue;

            if(seed.ftr->type != Feature::EDGELET)
            {
                double err = (beforePx-afterPx).norm();
                err /= (1<<matcher.search_level_);
                distMean += err;
                // distMean += (beforePx-afterPx).norm() / (1<<matcher.search_level_);
                // distMean += ((beforePx/(1<<matcher.search_level_))-(afterPx/(1<<matcher.search_level_))).norm();
            }
            else
            {
                Vector2d normal(matcher.A_cur_ref_*seed.ftr->grad);
                normal.normalize();
                targetNormal.push_back(normal);
                double err = fabs(normal.transpose()*(beforePx-afterPx));
                err /= (1<<matcher.search_level_);
                distMean += err;
            }

            Vector3d f(targets[i].first->cam_->cam2world(targets[i].second));
            Vector2d obs(hso::project2d(f));
            targetResult.push_back(make_pair(targets[i].first, obs));
        }
    }

    if(targetResult.size() < n_frame_thresh) 
        return false;

    distMean /= targetResult.size();
    // cout << "mean distance = " << distMean << endl;
    if(seed.ftr->type != Feature::EDGELET && distMean > 3.2)
    {
        isValid = false;
        return false;
    }
    if(seed.ftr->type == Feature::EDGELET && distMean > 2.5)
    {
        isValid = false;
        return false;
    }
    isValid = true;

    if(seed.ftr->type != Feature::EDGELET && distMean > 2.5)
        return false;
    if(seed.ftr->type == Feature::EDGELET && distMean > 2.0)
        return false;

    #ifdef ACTIVATE_DBUG
        cout << "======================" << endl;
    #endif

    seedOptimizer(seed, targetResult, targetNormal);
    // seed.mu = seed.opt_id;

    return true;
}

void DepthFilter::seedOptimizer(
    Seed& seed, const vector<pair<FramePtr, Vector2d> >& targets, const vector<Vector2d>& normals)
{
    if(seed.ftr->type == Feature::EDGELET)
        assert(targets.size() == normals.size());


    double oldEnergy = 0.0, rho = 0, mu = 0.1, nu = 2.0;
    bool stop = false;
    int n_trials = 0;

    const int n_trials_max = 5;


    double old_id = seed.mu;
    vector<SE3> Tths; Tths.resize(targets.size());
    Vector3d pHost(seed.ftr->f * (1.0/old_id));

    vector<float> errors; errors.reserve(targets.size());
    for(size_t i = 0; i < targets.size(); ++i)
    {
        FramePtr target = targets[i].first;
        SE3 Tth = target->T_f_w_ * seed.ftr->frame->T_f_w_.inverse();
        Vector2d residual = targets[i].second-hso::project2d(Tth*pHost);

        if(seed.ftr->type == Feature::EDGELET)
            errors.push_back(fabs(normals[i].transpose()*residual));
        else
            errors.push_back(residual.norm());

        // Tths[i] = Tth;
    }

    hso::robust_cost::MADScaleEstimator mad_estimator;
    const double huberTH = mad_estimator.compute(errors);

    for(size_t i = 0; i < targets.size(); ++i)
    {
        FramePtr target = targets[i].first;
        SE3 Tth = target->T_f_w_ * seed.ftr->frame->T_f_w_.inverse();
        // SE3 Tth = Tths[i];
        Vector2d residual = targets[i].second-hso::project2d(Tth*pHost);

        if(seed.ftr->type == Feature::EDGELET)
        {
            double resEdgelet = normals[i].transpose()*residual;
            double hw = fabsf(resEdgelet) < huberTH ? 1 : huberTH / fabsf(resEdgelet);
            // double hw = weight_function.value(fabs(resEdgelet)/huberTH);

            // hw *= target->m_error_in_px;

            oldEnergy += resEdgelet*resEdgelet * hw;
        }
        else
        {
            double res_dist = residual.norm();
            double hw = res_dist < huberTH ? 1 : huberTH / res_dist;
            // double hw = weight_function.value(res_dist/huberTH);

            // hw *= target->m_error_in_px;

            oldEnergy += res_dist*res_dist * hw;
        }

        Tths[i] = Tth;
    }

    double H = 0, b = 0;
    for(int iter = 0; iter < 5; ++iter)
    {
        n_trials = 0;
        do
        {
            double new_id = old_id;
            double newEnergy = 0;
            H = b = 0;

            pHost = seed.ftr->f * (1.0/old_id);
            for(size_t i = 0; i < targets.size(); ++i)
            {
                FramePtr target = targets[i].first;
                SE3 Tth = Tths[i];

                Vector3d pTarget = Tth*pHost;
                Vector2d residual = targets[i].second-hso::project2d(pTarget);

                if(seed.ftr->type == Feature::EDGELET)
                {
                    double resEdgelet = normals[i].transpose()*residual;
                    double hw = fabsf(resEdgelet) < huberTH ? 1 : huberTH / fabsf(resEdgelet);
                    // double hw = weight_function.value(fabs(resEdgelet)/huberTH);

                    // hw *= target->m_error_in_px;

                    Vector2d Jxidd;
                    Point::jacobian_id2uv(pTarget, Tth, old_id, seed.ftr->f, Jxidd);
                    double JEdge = normals[i].transpose()*Jxidd;
                    H += JEdge*JEdge*hw;
                    b -= JEdge*resEdgelet*hw;
                }
                else
                {
                    double res_dist = residual.norm();
                    double hw = res_dist < huberTH ? 1 : huberTH / res_dist;
                    // double hw = weight_function.value(res_dist/huberTH);

                    // hw *= target->m_error_in_px;

                    Vector2d Jxidd;
                    Point::jacobian_id2uv(pTarget, Tth, old_id, seed.ftr->f, Jxidd);

                    // double Hdd = Jxidd.transpose()*Jxidd, Jres = Jxidd.transpose()*residual;
                    // H += Hdd * hw;
                    // b -= Jres * hw;

                    H += (Jxidd[0]*Jxidd[0] + Jxidd[1]*Jxidd[1])*hw;
                    b -= (Jxidd[0]*residual[0] + Jxidd[1]*residual[1])*hw;
                }
            }

            H *= 1.0+mu;
            double step = b/H;

            if(!(bool)std::isnan(step))
            {
                new_id = old_id+step;

                pHost = seed.ftr->f * (1.0/new_id);
                for(size_t i = 0; i < targets.size(); ++i)
                {
                    FramePtr target = targets[i].first;
                    SE3 Tth = Tths[i];
                    Vector2d residual = targets[i].second-hso::project2d(Tth*pHost);

                    if(seed.ftr->type == Feature::EDGELET)
                    {
                        double resEdgelet = normals[i].transpose()*residual;
                        double hw = fabsf(resEdgelet) < huberTH ? 1 : huberTH / fabsf(resEdgelet);
                        // double hw = weight_function.value(fabs(resEdgelet)/huberTH);

                        // hw *= target->m_error_in_px;

                        newEnergy += resEdgelet*resEdgelet * hw;
                    }
                    else
                    {
                        double res_dist = residual.norm();
                        double hw = res_dist < huberTH ? 1 : huberTH / res_dist;
                        // double hw = weight_function.value(res_dist/huberTH);

                        // hw *= target->m_error_in_px;

                        newEnergy += res_dist*res_dist * hw;
                    }
                }

                rho = oldEnergy - newEnergy;
            }
            else
            {
                #ifdef ACTIVATE_DBUG
                    cout << "Matrix is close to singular!" << endl;
                    cout << "H = " << H << endl;
                    cout << "b = " << b << endl;
                #endif

                rho = -1;
            }

            if(rho > 0)
            {
                #ifdef ACTIVATE_DBUG
                    if(seed.ftr->type == Feature::EDGELET)
                        cout<< "EDGELET:  ";
                    else
                        cout<< "CORNER:  ";
                    cout<< "It. " << iter
                        << "\t Trial " << n_trials
                        << "\t Succ"
                        << "\t old Energy = " << oldEnergy
                        << "\t new Energy = " << newEnergy
                        << "\t lambda = " << mu
                        << endl;
                #endif

                oldEnergy = newEnergy;

                old_id = new_id;
                
                seed.opt_id = new_id;

                stop = fabsf(step) < 0.00001*new_id;
                // stop = hso::norm_max(step) <= EPS;

                mu *= std::max(1./3., std::min(1.-std::pow(2*rho-1,3), 2./3.));
                nu = 2.;
            }
            else
            {
                mu *= nu;
                nu *= 2.;
                ++n_trials;
                if (n_trials >= n_trials_max) stop = true;

                #ifdef ACTIVATE_DBUG
                    if(seed.ftr->type == Feature::EDGELET)
                        cout<< "EDGELET:  ";
                    else
                        cout<< "CORNER:  ";
                    cout<< "It. " << iter
                        << "\t Trial " << n_trials
                        << "\t Fail"
                        << "\t old Energy = " << oldEnergy
                        << "\t new Energy = " << newEnergy
                        << "\t lambda = " << mu
                        << endl;
                #endif
            }
        }while(!(rho>0 || stop));

        if(stop) break;
    }
}


void DepthFilter::directPromotionFeature()
{
    lock_t lock(m_converge_seed_mut);

    list<Seed>::iterator seed = m_converge_seed.begin();
    while(seed != m_converge_seed.end())
    {
        // 我认为这里仅对于最近邻关键帧存在线程安全问题
        FramePtr obs_frame = seed->last_update_frame;
        assert(obs_frame->isKeyframe());

        Vector3d p_target = obs_frame->T_f_w_*seed->ftr->point->pos_;
        if(p_target[2] < 0.000001)
        {
            seed_converged_cb_(seed->ftr->point, seed->sigma2);
            seed = m_converge_seed.erase(seed);
            // assert(false);
            continue;
        }

        Feature* ftr = new Feature(obs_frame.get(), seed->last_matched_px, seed->last_matched_level);
        if(seed->ftr->type == Feature::CORNER)
            ftr->type = Feature::CORNER;
        else
            ftr->type = Feature::EDGELET;

        assert(seed->ftr->point->obs_.size() == 1);


        ftr->point = seed->ftr->point;
        ftr->point->addFrameRef(ftr);


        // 99.9999%  thread safe
        seed->ftr->frame->addFeature(seed->ftr);


        obs_frame->addFeature(ftr);


        seed = m_converge_seed.erase(seed);
    }
}


} // namespace hso
