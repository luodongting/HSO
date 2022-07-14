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

/**
 * modified file in ORB-SLAM (https://github.com/raulmur/ORB_SLAM)
 */

#include <hso/viewer.h>
#include <hso/frame_handler_mono.h>
#include <hso/map.h>
#include <hso/frame.h>
#include <hso/feature.h>
#include <hso/point.h>
#include <hso/depth_filter.h>
#include <pangolin/gl/gltext.h>

using namespace hso;


namespace hso {

Viewer::Viewer(hso::FrameHandlerMono* vo): _vo(vo)
{
    mbFinished = false;
    mViewpointX =  0;
    mViewpointY = -1.5;
    mViewpointZ =  -2;
    mViewpointF =  500;

    mKeyFrameSize = 0.05;
    mKeyFrameLineWidth = 1.0;
    mCameraSize = 0.08;
    mCameraLineWidth = 3.0;

    mPointSize = 3.0;
}

bool Viewer::CheckFinish()
{
    std::unique_lock<std::mutex> lock(mMutexFinish);
    return mbFinished;
}

void Viewer::SetFinish()
{
    std::unique_lock<std::mutex> lock(mMutexFinish);
    mbFinished = true;
}

void Viewer::DrawKeyFrames(const bool bDrawKF)
{
    hso::FramePtr lastframe = _vo->lastFrame();
    if(lastframe == NULL || lastframe->id_ == _drawedframeID)
    {
        //return;
    }
    else  // save new pose
    {
        _drawedframeID = lastframe->id_ ;
        _CurrentPoseTwc = lastframe->T_f_w_.inverse();
        _pos.push_back(_CurrentPoseTwc);
    }
    if(_pos.empty()) return;


    if(bDrawKF)
    {
        glPointSize(2);
        glBegin(GL_POINTS);
        glColor3f(1.0,0.0,0.0);
        // glLineWidth(1.6);
        // glBegin(GL_LINES);

        // glColor3f(1,0,0);
        // glBegin(GL_LINES);
        for(size_t i = 0; i<_pos.size();i++)
        {
            Sophus::SE3 Twc = _pos[i];
            glVertex3d(Twc.translation()[0], Twc.translation()[1], Twc.translation()[2]);
            // Sophus::SE3 TLast = _pos[i-1];
            // glVertex3d(TLast.translation()[0], TLast.translation()[1], TLast.translation()[2]);
            // Sophus::SE3 TCurrent = _pos[i];
            // glVertex3d(TCurrent.translation()[0], TCurrent.translation()[1], TCurrent.translation()[2]);
        }
        glEnd();
    }
}

void Viewer::DrawMapRegionPoints()
{
    glPointSize(mPointSize);
    glBegin(GL_POINTS);
    // glColor3f(0.4,0.4,0.4);


    for(auto kf = _vo->map_.keyframes_.begin(); kf != _vo->map_.keyframes_.end(); ++kf)
        for(auto& ft: (*kf)->fts_)
        {
            if(ft->point == NULL) continue;
            Eigen::Vector3d Pw = ft->point->pos_;
            // glColor3f(0.4,0.4,0.4);

            float color = float(ft->point->color_) / 255;
            if(color > 0.9) color = 0.9;

            
            // if(color < 0.2) color = 0.2;

            glColor3f(color,color,color);
            glVertex3f( Pw[0],Pw[1],Pw[2]);
        }
    glEnd();



}

void Viewer::DrawConstraints()
{
 


    set<Frame*> LocalMap = _vo->LocalMap_;
    Vector3d posCurrent(_vo->lastFrame()->pos());

    if(LocalMap.empty()) return;

    glLineWidth(2.5);
    glColor4f(0.0f,1.0f,0.0f,0.6f);
    glBegin(GL_LINES);
    for(set<Frame*>::iterator it = LocalMap.begin(); it != LocalMap.end(); ++it)
    {
        Frame* target = *it;
        if(target->id_ == _vo->lastFrame()->id_) continue;

        Vector3d posTarget(target->pos());
        glVertex3d(posCurrent[0], posCurrent[1], posCurrent[2]);
        glVertex3d(posTarget[0], posTarget[1], posTarget[2]);
    }
    glEnd();
}

void Viewer::DrawCurrentCamera(pangolin::OpenGlMatrix &Twc)
{
    const float &w = mCameraSize;
    const float h = w*0.75;
    const float z = w*0.6;

    glPushMatrix();

#ifdef HAVE_GLES
        glMultMatrixf(Twc.m);
#else
        glMultMatrixd(Twc.m);
#endif
    glLineWidth(mCameraLineWidth);
    glColor3f(0.0f,0.0f,1.0f);
    glBegin(GL_LINES);
    glVertex3f(0,0,0);
    glVertex3f(w,h,z);
    glVertex3f(0,0,0);
    glVertex3f(w,-h,z);
    glVertex3f(0,0,0);
    glVertex3f(-w,-h,z);
    glVertex3f(0,0,0);
    glVertex3f(-w,h,z);

    glVertex3f(w,h,z);
    glVertex3f(w,-h,z);

    glVertex3f(-w,h,z);
    glVertex3f(-w,-h,z);

    glVertex3f(-w,h,z);
    glVertex3f(w,h,z);

    glVertex3f(-w,-h,z);
    glVertex3f(w,-h,z);
    glEnd();

    glPopMatrix();
}

void Viewer::GetCurrentOpenGLCameraMatrix(pangolin::OpenGlMatrix &M)
{
    if(_drawedframeID != 0)  // we have new pose
    {
        Eigen::Matrix3d Rwc = _CurrentPoseTwc.rotation_matrix();
        Eigen::Vector3d twc = _CurrentPoseTwc.translation();

        M.m[0] = Rwc(0,0);
        M.m[1] = Rwc(1,0);
        M.m[2] = Rwc(2,0);
        M.m[3] = 0.0;

        M.m[4] = Rwc(0,1);
        M.m[5] = Rwc(1,1);
        M.m[6] = Rwc(2,1);
        M.m[7] = 0.0;

        M.m[8] = Rwc(0,2);
        M.m[9] = Rwc(1,2);
        M.m[10] = Rwc(2,2);
        M.m[11] = 0.0;

        M.m[12] = twc[0];
        M.m[13] = twc[1];
        M.m[14] = twc[2];
        M.m[15] = 1.0;
    }
    else
        M.SetIdentity();
}

void Viewer::run()
{
    mbFinished = false;
    pangolin::CreateWindowAndBind("HSO: Hybrid Sparse Visual Odometry", 1228, 801);

    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA,GL_ONE_MINUS_SRC_ALPHA);

    pangolin::CreatePanel("menu").SetBounds(0.0,1.0,0.0,pangolin::Attach::Pix(160));
    pangolin::Var<bool> menuFollowCamera("menu.Follow Camera",true,true);
    pangolin::Var<bool> menuShowKeyFrames("menu.Show Trajactory",true,true);
    pangolin::Var<bool> menuShowPoints("menu.Show Points",true,true);
    pangolin::Var<bool> menuShowConstrains("menu.Show Constrains",false,true);



    // Define Camera Render Object (for view / scene browsing)
    pangolin::OpenGlRenderState s_cam(
              pangolin::ProjectionMatrix(1024,768,mViewpointF,mViewpointF,512,389,0.1,1000),
              pangolin::ModelViewLookAt(mViewpointX,mViewpointY,mViewpointZ, 0,0,0,0.0,-1.0, 0.0)
              );

    // Add named OpenGL viewport to window and provide 3D Handler
    pangolin::View& d_cam = pangolin::CreateDisplay()
          .SetBounds(0.0, 1.0, pangolin::Attach::Pix(175), 1.0, -1024.0f/768.0f)
          .SetHandler(new pangolin::Handler3D(s_cam));

    pangolin::OpenGlMatrix Twc;
    Twc.SetIdentity();

    bool bFollow = true;
    while(!CheckFinish())
    {
        usleep(10000);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        GetCurrentOpenGLCameraMatrix(Twc);


        if(menuFollowCamera && bFollow)
        {
            s_cam.Follow(Twc);
        }
        else if(menuFollowCamera && !bFollow)
        {
            s_cam.SetModelViewMatrix(pangolin::ModelViewLookAt(mViewpointX,mViewpointY,mViewpointZ, 0,0,0,0.0,-1.0, 0.0));
            s_cam.Follow(Twc);
            bFollow = true;
        }
        else if(!menuFollowCamera && bFollow)
        {
            bFollow = false;
        }


        d_cam.Activate(s_cam);
        glClearColor(1.0f,1.0f,1.0f,0.5f);

        DrawCurrentCamera(Twc);

        DrawKeyFrames(menuShowKeyFrames);

        if(menuShowPoints) 
            DrawMapRegionPoints();

        if(menuShowConstrains)
            DrawConstraints();

        pangolin::FinishFrame();


    }

    pangolin::BindToContext("HSO");
    std::cout<<"pangolin close"<<std::endl;

}

}
