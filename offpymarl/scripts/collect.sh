# train behavior policies
offp
export CUDA_VISIBLE_DEVICES=4
python src/main.py --collect --config=qmix --env-config=gymma_collect \
--num_episodes_collected=10 \
--key="mpe:SimpleSpread-v0" --offline_data_quality=full --stop_winrate=0.01 \
--t_max=3000000 \
--time_limit=25 --save_model_interval=50000 \
--seed=1

# collect datasets
offp
export CUDA_VISIBLE_DEVICES=4
python src/main.py --collect --config=qmix --env-config=gymma_collect \
--num_episodes_collected=20000 \
--key="mpe:SimpleSpread-v0" --time_limit=25 \
--runner=episode_imbalance \
--checkpoint_path="" --evaluate=True \
--offline_data_quality=imbalance2_20k \
--n_random_agents=2
