# Automatic detection of shakes and nods in sign language

## What's in this repo?

This repository contains the code used for the training of a model that recognizes where head shakes and nods occur in videos of people communicating in sign language. We publish this code to promote transparency and reproducibility in research.  This readme will tell you how to make annotations from a video using a model we trained and provide. Further explanations on how to use the code to train a model by yourself can be found in the Github Wiki. Please do reach out to me if you have any questions!

This repo is an extension of the forked repo by @Casvanrijbroek who wrote his master thesis on detection of headshakes. Our main additions (apart from general code updates) were adding head nod detection and adding different types of models to train (LSTMs). The data was also extended, but that is not published in this repo. We also wrote a new annotation guide and added the scripts used to evaluate the inter-annotator agreement to this repo. 

## Relevance

In both signed and spoken languages, head shakes and head nods give meaning to an expressed message, whether that is to emphasize it, deny it, or otherwise. Automatic detection of these head movements in videos of signers can help facilitate both qualitative and quantitative research on head movements by replacing the highly time-consuming manual annotation step. 

## Data
To train the models, we used the public corpus for Dutch Sign Language: Corpus NGT (https://www.corpusngt.nl/). We also added annotations from the German Sign Language corpus DGS (https://www.sign-lang.uni-hamburg.de).

While the code was developed with this specific goal in mind, you might find it useful as a pipeline for any task that builds models using video footage of people.

## shake/nod prediction

The prediction can be run for a video of a single signer using this command:

Python run_shake_nod_predict --v video.mp4

The output is an annotated eaf file with one tier for head movements. 

Note: this project is ended while in construction. The models don't work as well yet as we had hoped. Feel welcome to improve on the methods we present here, retrain the models with your own data, or improve on this repo any other way. 
