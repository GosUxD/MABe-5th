Hello!

Below you can find a outline of how to reproduce my solution for the <MABe Challenge - Social Action Recognition in Mice> competition.
If you run into any trouble with the setup/code or have any questions please contact me at <gosuxd1@gmail.com>


# DATA SETUP
Data should be downloaded from Kaggle competition website and placed under /datamount/. Then preprocess the data to create the master skeleton using preprocess_skelet.py

# Example Training and Inference
`python train.py -C cfg_XXX` where XXX is the number of config
`python inference.py` (currently evalutes with same weights as final submission)
 
# Recreating 5th place submission:
Use checkpoints which are located under output/mabe-weights then do inference using the provided inference.py. This directory contains all of the necessary weights used for the final submission.

# Project Structure

- CONFIGS: Contains all of the configuration files for each model specifically. Some redundancy is present between files, however that makes the repository plug&play and training is a simple as changing the cfg argument during training.

- DATA: contains the datasets/dataloaders for different model families. 

- METRICS: implementation of the official metric used for evaluation score.

- MODELS: contains each model/model_family. The following models/model families are used for the final submission:
233 Family: weights 273;274;275;276 (4fold CV)
234 Family: weights 278;279;284;285 (4fold CV)
238 Family: weights 289;290;293;292 (4fold CV)
240 Family: weights 240;
242 Family: weights 242;
243 Family: weights 294;295;297 (3fold CV)
244 Family: weights 244;
245 Family: weights 264;266;267;269 (4fold CV)
256 Family: weights 256;

- OUTPUT: contains all of the checkpoints/weights ready for inference

- UTILS: contains various utils for seeding/post-processing

- preprocess_skelet.py preprocessing the data into .npy
- split_data.py helper script for fold splitting during training
- train.py main entrypoint for training
- inference.py inference script for predicting

# HARDWARE: (The following specs were used to create the original solution)
Ubuntu 22.04.3 LTS
CPU: i7-13700K (24 vCPUs)
2 x NVIDIA RTX 4090 (24GB each)
96GB RAM (2x32 GB + 2x64 GB)
1TB SSD

#SOFTWARE (python packages are detailed separately in `requirements.txt`):
python                    3.11.5
CUDA                      12.1
PyTorch                   2.1.0
