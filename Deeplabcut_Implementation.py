#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  7 16:14:42 2023

@author: hmudradi3
"""
import deeplabcut

path_config_file='/home/hmudradi3/Dropbox (GaTech)/DLC annotations/dlc_model-student-2023-07-26/config.yaml'
augmenter_type = "default"  # = imgaug!!
#options
augmenter_type2 = "scalecrop"

net="resnet_50"

newvideo='/home/hmudradi3/Downloads/0004_vid.mp4'
dest_folder='/home/hmudradi3/DLC_Results'

#deeplabcut.create_training_dataset(path_config_file, net_type=net, augmenter_type=augmenter_type)

#deeplabcut.train_network(path_config_file)

#deeplabcut.evaluate_network(path_config_file, plotting=True)

deeplabcut.analyze_videos(path_config_file,[newvideo], save_as_csv=True, destfolder=dest_folder)

deeplabcut.create_labeled_video(path_config_file, [newvideo], destfolder=dest_folder, save_frames=True)


