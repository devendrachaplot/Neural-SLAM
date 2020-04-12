# Instructions

## Training
For training the complete Active Neural SLAM model on the Exploration task:
```
python main.py
```

### Specifying number of threads
The code runs multiple parallel threads for training. Each thread loads a scene on a GPU. The code automatically decides the total number of threads and number of threads on each GPU based on the available GPUs.

If you would like to not use the auto gpu config, you need to specify the following:
```
-n, --num_processes NUM_PROCESSES
--num_processes_per_gpu NUM_PROCESSES_PER_GPU
```
`NUM_PROCESSES_PER_GPU` will depend on your GPU memory, 11 works well of 16GB GPUs.
`NUM_PROCESSES` depends on the number of GPUs used for training and `NUM_PROCESSES_PER_GPU` such that 
```
NUM_PROCESSES <= min(NUM_PROCESSES_PER_GPU * number of GPUs, 72)
```
The Gibson training set consists of 72 scenes.

For example, on a 8 GPU system, with 16GB memory per GPU:
```
python main.py --auto_gpu_config 0 -n 72 --num_processes_per_gpu 11 --sim_gpu_id 1
```
Here, `sim_gpu_id = 1` specifies simulator threads to run from GPUs 1 onwards, and using GPU 0 only for pytorch model.
Each GPU from 1 to 6 will run 11 threads each, and GPU 7 will run 6 threads.

### Specifying log location, periodic model dumps
```
python main.py -d saved/ --exp_name exp1 --save_periodic 100000
```
The above will save the best model files and training log at `saved/models/exp1/` and save all models periodically every 100000 steps at `saved/dump/exp1/`. Each module will be saved in a separate file. 

### Specifying which modules to train and load
The Active Neural SLAM model consists of three independent modules: a Global Policy, a Local Policy
and a Neural SLAM Module. The model and code is modular, which means any subset of modules can be
trained. Specifying which modules need to trained using 
```
  --train_global TRAIN_GLOBAL
        0: Do not train the Global Policy
        1: Train the Global Policy (default: 1)
  --train_local TRAIN_LOCAL
        0: Do not train the Local Policy
        1: Train the Local Policy (default: 1)
  --train_slam TRAIN_SLAM
        0: Do not train the Neural SLAM Module
        1: Train the Neural SLAM Module (default: 1)
```
Each module can also be loaded independently using `--load_global`, `--load_local` and `--load_slam`
arguments.

### Using deterministic local policy
Instead of training the local policy, a deterministic local policy can be used which results in much
faster training and slightly lower final performance. Add `--use_deterministic_local 1` argument to
use the deterministic local policy.

### Hyper-parameters
Most of the default hyper-parameters should work fine. Some hyperparameters are set for training with 72 threads, which might need to be tuned when using fewer threads. Fewer threads lead to smaller batch size for Local and Global Policy. Consequently, their learning rates might need to be tuned:
```
--local_optimizer, (default='adam,lr=0.0001')
--global_lr, (default=2.5e-5)
```

### Specifying actuation and sensor noise
The code uses actuation and sensor noise based on models trained on real-world data. To turn off the
actuation and sensor noise use `--noisy_actions 0` and `--noisy_odometry 0`, respectively.

## Downloading pre-trained models
```
mkdir pretrained_models
wget -O pretrained_models/model_best.global http://www.cs.cmu.edu/~dchaplot/projects/active_neural_slam/model_best.global
wget -O pretrained_models/model_best.local http://www.cs.cmu.edu/~dchaplot/projects/active_neural_slam/model_best.local
wget -O pretrained_models/model_best.slam http://www.cs.cmu.edu/~dchaplot/projects/active_neural_slam/model_best.slam
```

## Evaluation

The following are instructions to evaluate on the Gibson val set.

### Converting datasets
To parallelize evaluation for speed, we provide a script to convert the Gibson val set into multi-threading format:
```
python scripts/convert_datasets.py 
```
The above will create a multi-thread version of the val set at `data/datasets/pointnav/gibson/v1//val_mt/`.

### Specifying number of threads and number of episodes
Specify number of threads for evaluation using `--num_processes` and number of evaluation episodes per thread using `--num_episodes`.
The Gibson val set consists of 14 scenes and 71 episodes per scene. Thus, we recommend using 14 threads for evaluation, and 71 episodes per thread.

For example, if you have 2 GPUs:
```
python  main.py --split val_mt --eval 1 \
--auto_gpu_config 0 -n 14 --num_episodes 71 --num_processes_per_gpu 7 \
--load_global pretrained_models/model_best.global --train_global 0 \
--load_local pretrained_models/model_best.local  --train_local 0 \
--load_slam pretrained_models/model_best.slam  --train_slam 0
```

### Specifying map sizes
The full and local map sizes are specified using the following arguments:
```
--map_size_cm (default: 2400) 
--global_downscaling (default:2)
```
The default arguments lead to full map size of `M = 480 (=24m)` and local map size of `G = 240 (=12m)`. 
Although the pre-trained models are trained with default map size for better training speed, they can be evaluated with larger full map size to improve performance.
The full map size can be increased to `M = 960 (=48m)` using `--map_size_cm 4800 --global_downscaling 4`:
```
python  main.py --split val_mt --eval 1 \
--auto_gpu_config 0 -n 14 --num_episodes 71 --num_processes_per_gpu 7 \
--map_size_cm 4800 --global_downscaling 4 \
--load_global pretrained_models/model_best.global --train_global 0 \
--load_local pretrained_models/model_best.local  --train_local 0 \
--load_slam pretrained_models/model_best.slam  --train_slam 0 
``` 

### Evaluating on small and large scenes
To evaluate on small and large scenes on the gibson set separately, create small and large splits using:
```
python scripts/convert_datasets.py --split_by_size 1
```
This will create two datasets at `data/datasets/pointnav/gibson/v1//val_mt_small/` (10 scenes) and `data/datasets/pointnav/gibson/v1//val_mt_large/` (4 scenes).
You can evaluate models on these datasets by changing `--split` and `-n` arguments in the above command to `val_mt_small` and `10` or `val_mt_large` and `4`. 

### Visualization and printing images
For visualizing the agent observations and predicted map and pose, add `-v 1` as an argument to the above command. This will require a display to be attached to the system.

To visualize on headless systems (without display), use `--print_images 1 -d results/ --exp_name exp1`. This will save the visualization images in `results/dump/exp1/episodes/`.

Both `-v 1` and `--print_images 1` can be used together to visualize and print images at the same time. 

## Troubleshooting
- `ImportError: cannot import name 'PILLOW_VERSION' from 'PIL'`. New Pillow version breaks torchvision, see the issue [here]( https://github.com/pytorch/vision/issues/1712). Downgrading the pillow version seems to work:
```
pip install "pillow<7"
```


## Tips
To silence habitat sim log add the following to your `~/.bashrc` (Linux) or `~/.bash_profile` (Mac) 
```
export GLOG_minloglevel=2
export MAGNUM_LOG="quiet"
```

