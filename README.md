# Computer Vision and Deeplabcut Implementations
## DeepLabCut Project Instructions
This README provides instructions on how to use the DeepLabCut script provided. The main script performs various tasks such as analyzing videos and creating labeled videos. The configuration file and video paths are specified within the script.

### Requirements
Ensure you have the following installed:

Python 3.x
DeepLabCut library
You can install DeepLabCut using pip:

'''
pip install deeplabcut
'''

### Files
> config.yaml: Configuration file for DeepLabCut. Located at /home/hmudradi3/Dropbox (GaTech)/DLC annotations/dlc_model-student-2023-07-26/config.yaml.
> 0004_vid.mp4: Video file to be analyzed. Located at /home/hmudradi3/Downloads/0004_vid.mp4.
> Script: The Python script provided for running DeepLabCut processes.

### Script Overview
The script includes lines for analyzing a video and creating a labeled video from the analyzed data. Additional functionality for creating training datasets, training the network, and evaluating the network is commented out.

### How to Use
1. Set Up Paths: Ensure the paths to the configuration file and video file are correct in the script.
2. Run Script: Execute the script to analyze the video and create labeled video frames.

'''
python script_name.py
'''

3. Review Results: The analyzed data and labeled video will be saved in the specified destination folder (/home/hmudradi3/DLC_Results).
