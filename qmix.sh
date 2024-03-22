git add . 
git commit -m "qmix"

# SMAC-v2
# CUDA_VISIBLE_DEVICES=3 python src/main.py --config=qmix --env-config=sc2_gen_zerg 

# SMAC-v1
CUDA_VISIBLE_DEVICES=1 python src/main.py --config=qmix --env-config=sc2 with env_args.map_name=5m_vs_6m
