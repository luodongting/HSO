# run single
#-------------------------

#change it to your images folder
pathImageFolder='/media/datasets/euroc/Vicon_Room1_01/cam0/data' 


#change it to the corresponding camera file.
cameraCalibFile='./cameras/euroc.txt' 


./../bin/test_dataset image="$pathImageFolder" calib="$cameraCalibFile" #start=xxx end=xxx times=xxx name=xxx

