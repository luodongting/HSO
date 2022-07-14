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

#include <vector>
#include <string>
#include <fstream>
#include <iostream>

#include <opencv2/opencv.hpp>

#include <sophus/se3.h>

#include <boost/thread.hpp>

#include <hso/config.h>
#include <hso/frame_handler_mono.h>
#include <hso/frame_handler_base.h>
#include <hso/map.h>
#include <hso/frame.h>
#include <hso/feature.h>
#include <hso/point.h>
#include <hso/viewer.h>
#include <hso/depth_filter.h>
// #include "hso/PhotomatricCalibration.h"


#include "hso/camera.h"
#include "hso/ImageReader.h"



using namespace cv;
using namespace std;

const int G_MAX_RESOLUTION = 848*800;

string g_image_folder = "";
string g_stamp_folder = "None";
string g_calib_path   = "";
string g_result_name  = "KeyFrameTrajectory";

int g_start = 0;
int g_end = 60000;


void parseArgument(char* arg)
{
    int option;
    char buf[1000];

    if(1==sscanf(arg,"image=%s",buf))
    {
        g_image_folder = buf;
        printf("loading images from %s!\n", g_image_folder.c_str());
        return;
    }

    if(1==sscanf(arg,"calib=%s",buf))
    {
        g_calib_path = buf;
        printf("loading calibration from %s!\n", g_calib_path.c_str());
        return;
    }

    if(1==sscanf(arg,"times=%s",buf))
    {
        g_stamp_folder = buf;
        printf("loading timestamp from %s!\n", g_stamp_folder.c_str());
        return;
    }


    if(1==sscanf(arg,"name=%s",buf))
    {
        g_result_name = buf;
        printf("set result file name =  %s!\n", g_result_name.c_str());
        return;
    }


    if(1==sscanf(arg,"start=%d",&option))
    {
        g_start = option;
        printf("START AT %d!\n",g_start);
        return;
    }

    if(1==sscanf(arg,"end=%d",&option))
    {
        g_end = option;
        printf("END AT %d!\n",g_end);
        return;
    }
}


namespace hso {

class BenchmarkNode
{
public:
    hso::AbstractCamera* cam_;
    FrameHandlerMono* vo_;
    hso::Viewer* viewer_;
    boost::thread * viewer_thread_;

    BenchmarkNode();
    ~BenchmarkNode();
    void runFromFolder();
    void saveResult(bool stamp_valid);
};

BenchmarkNode::BenchmarkNode()
{
    // read calibration
    std::string calib_dir = g_calib_path;
    std::ifstream f_cam(calib_dir.c_str());
    if (!f_cam.good())
    {
        f_cam.close();
        printf("Camera calibration file not found, shutting down.\n");
        return;
    }

    std::string line_1;
    std::getline(f_cam, line_1);

    float ic[8];
    char camera_type[20];
    
    if(std::sscanf(line_1.c_str(), "%s %f %f %f %f %f %f %f %f",
       camera_type, &ic[0], &ic[1], &ic[2], &ic[3], &ic[4], &ic[5], &ic[6], &ic[7]) == 9)
    {
        if(camera_type[0] == 'P' || camera_type[0] == 'p')
        {
            std::string line_2;
            std::getline(f_cam, line_2);
            float wh[2]={0};
            assert(std::sscanf(line_2.c_str(), "%f %f", &wh[0], &wh[1]) == 2);

            int width_i=wh[0], height_i=wh[1];
            if(wh[0]*wh[1] > G_MAX_RESOLUTION + 0.00000001)
            {
                double resize_rate = sqrt(wh[0]*wh[1]/G_MAX_RESOLUTION);
                width_i  = int(wh[0]/resize_rate);
                height_i = int(wh[1]/resize_rate);
                resize_rate = sqrt(wh[0]*wh[1]/width_i*height_i);
                ic[0] /= resize_rate;
                ic[1] /= resize_rate;
                ic[2] /= resize_rate;
                ic[3] /= resize_rate;
            }

            cam_ = new hso::PinholeCamera(width_i, height_i, ic[0], ic[1], ic[2], ic[3], ic[4], ic[5], ic[6], ic[7]);

            cout << "Camera: " << "Pinhole\t" << "Width=" << wh[0] << "\tHeight=" << wh[1] << endl;
        }
        else if(camera_type[0] == 'E' || camera_type[0] == 'e')
        {
            std::string line_2;
            std::getline(f_cam, line_2);
            float wh[2]={0};
            assert(std::sscanf(line_2.c_str(), "%f %f", &wh[0], &wh[1]) == 2);

            int width_i=wh[0], height_i=wh[1];
            if(wh[0]*wh[1] > G_MAX_RESOLUTION + 0.00000001)
            {
                double resize_rate = sqrt(wh[0]*wh[1]/G_MAX_RESOLUTION);
                width_i  = int(wh[0]/resize_rate);
                height_i = int(wh[1]/resize_rate);
                resize_rate = sqrt(wh[0]*wh[1]/width_i*height_i);
                ic[0] /= resize_rate;
                ic[1] /= resize_rate;
                ic[2] /= resize_rate;
                ic[3] /= resize_rate;
            }

            cam_ = new hso::EquidistantCamera(width_i, height_i, ic[0], ic[1], ic[2], ic[3], ic[4], ic[5], ic[6], ic[7]);

            cout << "Camera: " << "Equidistant\t" << "Width=" << wh[0] << "\t" << "Height=" << wh[1] << endl;
        }
        else
        {
            printf("Camera type error.\n");
            f_cam.close();
            return;
        }
    }
    else if(std::sscanf(line_1.c_str(), "%s %f %f %f %f %f", camera_type, &ic[0], &ic[1], &ic[2], &ic[3], &ic[4]) == 6)
    {
        assert(camera_type[0] == 'F' || camera_type[0] == 'f');

        std::string line_2;
        std::getline(f_cam, line_2);
        float wh[2]={0};
        assert(std::sscanf(line_2.c_str(), "%f %f", &wh[0], &wh[1]) == 2);

        int width_i=wh[0], height_i=wh[1];
        if(wh[0]*wh[1] > G_MAX_RESOLUTION + 0.00000001)
        {
            double resize_rate = sqrt(wh[0]*wh[1]/G_MAX_RESOLUTION);
            width_i  = int(wh[0]/resize_rate);
            height_i = int(wh[1]/resize_rate);
            resize_rate = sqrt(wh[0]*wh[1]/width_i*height_i);

            if(ic[2] > 1 && ic[3] > 1)
            {
                ic[0] /= resize_rate;
                ic[1] /= resize_rate;
                ic[2] /= resize_rate;
                ic[3] /= resize_rate;
            }
        }

        std::string line_3;
        std::getline(f_cam, line_3);

        if(line_3 == "true")
            cam_ = new hso::FOVCamera(width_i, height_i, ic[0], ic[1], ic[2], ic[3], ic[4], true);
        else
            cam_ = new hso::FOVCamera(width_i, height_i, ic[0], ic[1], ic[2], ic[3], ic[4], false);

        cout << "Camera: " << "FOV\t" << "Width=" << wh[0] << "\t" << "Height=" << wh[1] << endl;
    }
    else
        printf("Camera file error.\n");

    f_cam.close();
}

BenchmarkNode::~BenchmarkNode()
{
    delete vo_;
    delete cam_;
    delete viewer_;
    delete viewer_thread_;
}


void BenchmarkNode::runFromFolder()
{
    hso::ImageReader image_reader(g_image_folder, cv::Size(cam_->width(), cam_->height()), g_stamp_folder);

    vo_ = new FrameHandlerMono(cam_, false);
    vo_->start();

    viewer_ = new hso::Viewer(vo_);
    viewer_thread_ = new boost::thread(&hso::Viewer::run, viewer_);
    viewer_thread_->detach();

    g_end = std::min(image_reader.getNumImages(), g_end);

    for(int img_id=g_start; img_id<g_end; ++img_id)
    {       
        cv::Mat image = image_reader.readImage(img_id);
        if(cam_->getUndistort()) cam_->undistortImage(image, image);

        if(image_reader.stampValid())
        {
            std::string time_stamp = image_reader.readStamp(img_id);

            // process frame
            vo_->addImage(image, img_id, &time_stamp);
        }
        else
            vo_->addImage(image, img_id);
        

        // display tracking quality
        cv::Mat tracking_img(image);
        cv::cvtColor(tracking_img, tracking_img, COLOR_GRAY2RGB);
        if(vo_->lastFrame() != NULL)
        {
            for(auto& ft: vo_->lastFrame()->fts_)
            {
                if(ft->point == NULL) continue;

                if(ft->type == hso::Feature::EDGELET)
                    cv::rectangle(tracking_img, cv::Point2f(ft->px.x()-3, ft->px.y()-3), cv::Point2f(ft->px.x()+3, ft->px.y()+3), cv::Scalar ( 0,255,255 ), cv::FILLED);
                else
                    cv::rectangle(tracking_img, cv::Point2f(ft->px.x()-3, ft->px.y()-3), cv::Point2f(ft->px.x()+3, ft->px.y()+3), cv::Scalar ( 0,255,0 ),   cv::FILLED);
            }
        }
        cv::imshow ("Tracking Image", tracking_img);
        cv::waitKey (1);
    }
    
    saveResult(image_reader.stampValid());
    // cv::waitKey (0);
}

void BenchmarkNode::saveResult(bool stamp_valid)
{
    // Trajectory
    std::ofstream okt("./result/"+g_result_name+".txt");
    for(auto it = vo_->map_.keyframes_.begin(); it != vo_->map_.keyframes_.end(); ++it)
    {
        SE3 Tinv = (*it)->T_f_w_.inverse();
        if(!stamp_valid)
            okt << (*it)->id_ << " ";
        else
            okt << (*it)->m_timestamp_s  << " ";

        okt << Tinv.translation()[0] << " " 
            << Tinv.translation()[1] << " " 
            << Tinv.translation()[2] << " "
            << Tinv.unit_quaternion().x() << " " 
            << Tinv.unit_quaternion().y() << " "
            << Tinv.unit_quaternion().z() << " "
            << Tinv.unit_quaternion().w() << endl;
    }
    okt.close();


}


} // namespace hso

int main(int argc, char** argv)
{
    if(argc < 2)
    {
        cerr << endl << "Minimal Usage: ./test_dataset  image=PATH_TO_IMAGE_FOLDER  calib=PATH_TO_CALIBRATION" << endl;
        return 1;
    }

    for(int i=1; i<argc; ++i) 
        parseArgument(argv[i]);

    hso::BenchmarkNode benchmark;
    benchmark.runFromFolder();

    printf("BenchmarkNode finished.\n");
    return 0;
}

