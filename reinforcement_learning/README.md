# Homework-3

## Setup and Installation (Local)

### Install MuJoCo

1. Download the MuJoCo version 2.1 binaries for
   [Linux](https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz) or
   [OSX](https://mujoco.org/download/mujoco210-macos-x86_64.tar.gz).
2. Extract the downloaded `mujoco210` directory into `~/.mujoco/mujoco210`.
3. Install mujoco-py: `pip install -U 'mujoco-py<2.2,>=2.1'`


### Setup environment

To set up the project environment, Use the `environment.yml` file. It contains the necessary dependencies and installation instructions.

    conda env create -f environment.yml
    
### Export paths variables

    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.mujoco/mujoco210/bin
    export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so

## Setup and Installation (Colab)

### Upload ipynb
Upload `CSE571_HW3.ipynb` to colab

### Upload other files
upload files listed below from the folder (or you can simply upload all files)
The `files` tab will be like
- sample_data (this one is default in colab)
- utils.py
- policy.py
- evaluate.py
- pytorch_utils.py
- expert_data.pkl
- expert_policy.pkl

## Running the assignment

### Policy Gradient
    python3 main.py --task policy_gradient
    
### Behavior Cloning
    python3 main.py --task behavior_cloning
    
### DAgger
    python3 main.py --task dagger
