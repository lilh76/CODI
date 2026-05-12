# MA-MBTS

Multi-agent Model-based trajectory stitching is is a baseline used in our paper, which is an iterative data improvement strategy improved from Model-based trajectory stitching.

First, you need to run
```
pip install -r requirements.txt
```

You also need to prepare an H5 file as the dataset, which must include the indexes and corresponding data of 'obs', 'actions', 'reward', and 'terminated'. Then, modify the data loading paths in each of the py files accordingly.

Before running `TrajectoryStitching.py`, a forward model, inverse model and reward function must be pre-trained.  You need to run the `Train_VAE.py`,`Train_DM.py`,`Train_GAN_reward.py` by the following lines:
```
python3 Train_VAE.py --env_name "Halfcheetah" --env "halfcheetah-mediumexpert-v2" --diff "MedExp" 
```
```
python3 Train_DM.py --env_name "Halfcheetah" --env "halfcheetah-mediumexpert-v2" --diff "MedExp" 
```
```
python3 Train_GAN_reward.py --env_name "Halfcheetah" --env "halfcheetah-mediumexpert-v2" --diff "MedExp"
```

Now you can start MA-MBTS by running the following line (using HalfCheetah Medium Expert as an example)

```
python3 TrajectoryStitching.py --env_name "Halfcheetah" --env "halfcheetah-mediumexpert-v2" --diff "MedExp" --reward_function "WGAN"
```
