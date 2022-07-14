# HSO

This released code is the VO part of HSO based on [SVO](https://github.com/uzh-rpg/rpg_svo) and [PL-SVO](https://github.com/rubengooj/pl-svo).

**Authors:** Dongting Luo, Yan Zhuang and Sen Wang.

**Video:** https://youtu.be/AchJQ2u8K50  or  https://v.youku.com/v_show/id_XNDgyMjY4ODc4OA

## 0. Advertising
I am looking for a PhD position. If you think my work and research direction meet the needs of your Lab, please contact me (luo_dt@vip.163.com).

## 1. Related Publications

* Dongting Luo, Yan Zhuang and Sen Wang. **Hybrid Sparse Monocular Visual Odometry with Online Photometric Calibration**. In *International Journal of Robotics Research*

## 2. Required Dependencies

### boost
c++ Librairies (thread and system are needed). Install with
	
		sudo apt-get install libboost-all-dev

### Eigen3
Linear algebra.

		sudo apt-get install libeigen3-dev

### OpenCV 3.0 or later
Dowload and install instructions can be found at: http://opencv.org. We tested the code on OpenCV 3.2.0.

### Pangolin
Used for 3D visualization & the GUI. Dowload and install instructions can be found at: https://github.com/stevenlovegrove/Pangolin.
If you encounter difficulties in compiling the latest version of Pangolin, I put an old version in the *thirdparty* folder, which is easier to complie.

### Sophus, fast and g2o (Included in thirdparty folder)
The three libraries will be compiled automatically using script `build.sh`.

## 3. Build and Compile
We have tested the code on Ubuntu 14.04, 16.04, 18.04 and 20.04. A newer Ubuntu platform may also be easy to install.

		cd hso
		chmod +x build.sh
		./build.sh

This will create the executable `test_dataset` in the *bin* folder.

## 4. Run code

### EuRoC Dataset

1. Download the dataset from http://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets and uncompress it.
2. Go to the *test* folder and open the script `euroc_batch.sh`.
3. Change **pathDatasetEuroc** variable to point to the directory where the dataset has been uncompressed and **sequenceName** variables to the name of each sequence in your computer.
4. Execute the script: ./euroc_batch.sh
5. The result of keyframes trajectory will be saved in the *result* folder using TUM format.

### ICL-NUIM Dataset

1. Download the dataset from http://www.doc.ic.ac.uk/~ahanda/VaFRIC/iclnuim.html and uncompress it.
2. Go to the *test* folder and open the script `icl-nuim_batch.sh`.
3. Change **pathDatasetICL** variable to point to the directory where the dataset has been uncompressed and **sequece_name** variables to the name of each sequence in your computer.
4. Execute the script: ./icl-nuim_batch.sh
5. The result of keyframes trajectory will be saved in the *result* folder using TUM format.

### TUM MonoVO Dataset

1. Download the dataset from https://vision.in.tum.de/mono-dataset and uncompress it.
2. Uncompress the *images.zip* in each sequence.
3. Go to the *test* folder and open the script `tum_monoVO_batch.sh`.
4. Change **pathDatasetTUM** variable to point to the directory where the dataset has been uncompressed and **sequece_num** variables to the name of each sequence in your computer.
5. Execute the script: ./tum_monoVO_batch.sh
6. The result of keyframes trajectory will be saved in the *result* folder using TUM format.

### Your Sequence
The minimal usage is
	
		cd test
		./single_sequence.sh 
	
Change `pathImageFolder` in the script to the folder containing images, and `cameraCalibFile` to the geometric camera calibration file. HSO now support 3 types of camera model, please see below. 
The result will be saved in the *result* folder.

## 5. Camera Files Format

### Calibration File for pinhole model

	Pinhole fx fy cx cy k1 k2 p1 p2
	width height
	true / false

### Calibration File for FOV model

	FOV fx fy cx cy omega
	width height
	true / false

### Calibration File for Equidistant camera model

	EquiDistant fx fy cx cy k1 k2 r1 r2
	width height
	true / false

When the image is taken with a fisheye camera, then fill in "true" in the third line, otherwise "false". Some camera file examples are saved in ./test/cameras.

## 6. Command Line Argument
If you use "./bin/test_dataset" to run HSO, there are several commandline options you can use:

### Requried
- `image=X` where X is the folder containing images.

- `calib=X` where X is the geometric camera calibration file.

### Optional 
- `start=X`: Start at frame X. The default is 0. HSO’s monocular initialization is weak. When the initialization fails, we recommend skipping some beginning frames (e.g., start=450 in MH04).

- `end=X`  : End at frame X. The default is the number of the images the folder contains.

- `times=X`: The timestamp file of the images. Its number of lines must be equal to the number of images in the folder. HSO now support 4 types of timestamp file format, please see the file `ImageReader.cpp`. The default is "None".

- `name=X` : The file name of the result. The default is "KeyFrameTrajectory.txt".

## 7. License
The source code is released under a GPLv3 licence.

If you use HSO in your academic work, please cite:

      @article{Luo2022,
      title={Hybrid Sparse Monocular Visual Odometry with Online Photometric Calibration},
      author={Dongting Luo, Yan Zhuang and Sen Wang},
      journal={International Journal of Robotics Research},
      year={2022}
     }
