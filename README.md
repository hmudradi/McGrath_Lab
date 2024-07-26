# Project Overview(Computer Vision)
This project involves the analysis and comparison of classification metrics for gender prediction of fish using machine learning models. The data consists of predictions made by the YOLOV5 classifier on frames and videos. The goal is to evaluate and visualize the performance of these predictions using various metrics and to compare the performance on different subsets of the data (e.g., by track and gender).

## Files and Directories
>main_script.py: The main script containing the code for calculating metrics, grouping data by track, and visualizing results.

>data_frames: Directory containing the necessary CSV files used for analysis.

YOLOV5_Cls_Manual_Frame_exp.csv
YOLOV5_Cls_Manual_Videos_exp.csv
YOLOV5_Manual_Frames.csv
YOLOV5_Cls_Manual_Videos.csv
BioBoost_final_predictions.csv
results: Directory where the radar chart images are saved.
Prerequisites
Python 3.6+
Required Python packages:
pandas
numpy
matplotlib
Setup and Installation
Clone the repository:
sh
Copy code
git clone https://github.com/username/repository.git
Navigate to the project directory:
sh
Copy code
cd repository
Install the required packages:
sh
Copy code
pip install pandas numpy matplotlib
Usage
Prepare Data: Ensure that the CSV files are placed in the data_frames directory. The script expects specific file names as mentioned above.

Run the Script: Execute the main script to calculate metrics and generate visualizations.

sh
Copy code
python main_script.py
View Results: The radar charts will be displayed, and you can save them by uncommenting the plt.savefig lines in the create_radar_chart function.

Functions
Metrics
Class for calculating various performance metrics.

init(self, label, predict): Initializes the class with true labels and predictions.
get_metrics(self): Returns a dictionary of calculated metrics.
by_track(df, uid='uid', metric='acc')
Groups data by track and calculates specified metric for each track.

by_track_sex(df, uid='uid', metric='acc')
Groups data by track and gender, and calculates specified metric for each subset.

by_track_count(df, uid='uid', metric='acc')
Counts true positives, true negatives, false positives, and false negatives by track.

metrics_by_Tk_Id(df, flag=0, metric='acc')
Calculates specified metric for each track, optionally splitting by gender.

columns_to_dict(df, key_col, value_col)
Converts two columns of a DataFrame into a dictionary.

create_radar_chart(metrics_dict1, metrics_dict2, title)
Creates and displays a radar chart comparing two sets of metrics.

Data Preparation
The data frames are created by reading CSV files. Unique IDs and track IDs are generated for grouping and metric calculation. Ensure that the data frames have the required columns (label, prediction, uid, Tk_Id, etc.) before running the script.

Example Usage
To visualize the accuracy of gender prediction on different tracks, use the create_radar_chart function with the calculated metrics for manual frames and manual videos.

python
Copy code
create_radar_chart(updated_mv_f, updated_mf_f, "Female Track Acc")
create_radar_chart(updated_mv_m, updated_mf_m, "Male Track Acc")
create_radar_chart(mf, mv, "Frame bacc")
This will display radar charts comparing the metrics across different tracks and genders.





# Computer Vision and Deeplabcut Implementations
## DeepLabCut Project Instructions
This README provides instructions on how to use the DeepLabCut script provided. The main script performs various tasks such as analyzing videos and creating labeled videos. The configuration file and video paths are specified within the script.

### Requirements
Ensure you have the following installed:

Python 3.x
DeepLabCut library
You can install DeepLabCut using pip:

```
pip install deeplabcut
```

### Files
> config.yaml: Configuration file for DeepLabCut. Located at /home/hmudradi3/Dropbox (GaTech)/DLC annotations/dlc_model-student-2023-07-26/config.yaml.
> 0004_vid.mp4: Video file to be analyzed. Located at /home/hmudradi3/Downloads/0004_vid.mp4.
> Script: The Python script provided for running DeepLabCut processes.

### Script Overview
The script includes lines for analyzing a video and creating a labeled video from the analyzed data. Additional functionality for creating training datasets, training the network, and evaluating the network is commented out.

### How to Use
1. Set Up Paths: Ensure the paths to the configuration file and video file are correct in the script.
2. Run Script: Execute the script to analyze the video and create labeled video frames.

```
python script_name.py
```

3. Review Results: The analyzed data and labeled video will be saved in the specified destination folder (/home/hmudradi3/DLC_Results).
