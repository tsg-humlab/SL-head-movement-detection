# Sign-Language-Thesis

## What's in this repo?

This repository contains all code I used for my Data Science thesis. I've published the code here to promote transparency and reproducibility for my work. If you have any questions or remarks regarding, please don't hesitate to reach out to me on GitHub! I'm also happy to help with any problems you run into when using the code for yourself.

## Table of contents

 - [The thesis subject](#the-thesis-subject)
 - [How to use this code](#how-to-use-this-code)

## The thesis subject

As you probably gathered from the name, my thesis investigated sign language. Specifically Dutch sign language (NGT). Specifically, I implemented predictive models to automatically identify non-manuals (any movements/expressions relevant to the language that don't involve your hands) from video footage of people conversing in NGT. I used the (mostly) public CorpusNGT dataset for this (https://www.corpusngt.nl/).

While the code was developed with this specific goal in mind, you might find it useful as a pipeline for any task that builds models using video footage of people.

## How to use this code

Most of the codebase is built around working with an overview file. Which is a TSV file with a line for every video entry in the dataset you want to work with. Typically, you will need three things in order to run an experiment on this repo:

 - Access to the videos you want to work with (you can download CorpusNGT through the website)
 - An overview CSV file detailing the location of every video and label file
 - An output directory to store the results of the experiment in

To create label files and an overview CSV, you can use the create_overview.py script. This script expects a config.yml file to be present in the top directory of the project with the following fields:

 - overview: /path/to/store/overview.csv
 - media:
    - body_720: /path/to/videos
    - eaf: /path/to/eaf_files

This will then create a generic overview.csv with one line for every video and eaf file. You can split these into multiple videos if you so desire using video_splitter.py

You can then create the label files, for this project I use numpy arrays with 0's and 1's, where 1 indicates head-shakes and 0 is the background class. You can use the overview.csv with create_label_files.py to create and store these numpy arrays. You can then use split_dataset.py to separate the videos into N folds for cross validation and a test set.

The statistics directory contains code you might find useful for inspecting the quality of your data splits.

The models directory contains all code that used in the experiments, you don't have to run these directly.

Before you can run the experiments, you'll need to generate the pose detections and add the results to your overview file. You can do this with yolo_detector.py in the pose directory. Even though YOLOv8 nano is the smallest model of its kind, it still takes significant compute to generate the results over more than a handful of videos. I strongly recommend you run this script using a GPU (it will automatically make use of one if available).

You might find the analyze_results.py and review_conflicts.py scripts useful for inspecting if the correct person is identified at all times for your dataset.

Now you're finally ready to run the experiments! The validation directory contains cross_validatoin.py, where most experiments available through subcommands. evaluate_speakers.py can be used to aggregate results on a spaker-level and event_based.py was used to evaluate experiments on the event-level rather than the frame-level.
