git add . 
git commit -m "qmix"

# SMAC-v2
# CUDA_VISIBLE_DEVICES=3 python src/main.py --config=qmix --env-config=sc2_gen_zerg 

# SMAC-v1
CUDA_VISIBLE_DEVICES=2 python src/main.py --config=qmix --env-config=sc2 with env_args.map_name=2c_vs_64zg
