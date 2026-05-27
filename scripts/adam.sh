# CUDA_VISIBLE_DEVICES=0 python run.py --config config/pda.yml --data.src_region NC --data.tgt_region TX\
#     --data.other_regions CA,OR,SC,MO,MN,PA,TN,VA --seed.right 10&
# CUDA_VISIBLE_DEVICES=1 python run.py --config config/pda.yml --acquisition.acquisition_function.reward sgd_score --data.src_region NC --data.tgt_region TX\
#     --data.other_regions CA,OR,SC,MO,MN,PA,TN,VA --seed.right 10&    
# CUDA_VISIBLE_DEVICES=2 python run.py --config config/oracle.yml --data.src_region NC --data.tgt_region TX\
#     --data.other_regions CA,OR,SC,MO,MN,PA,TN,VA --seed.right 10& 
# CUDA_VISIBLE_DEVICES=3 python run.py --config config/daa.yml --data.src_region NC --data.tgt_region TX\
#     --data.other_regions CA,OR,SC,MO,MN,PA,TN,VA --seed.right 10

# "src_region": "FL",
#         "tgt_region": "NY",
#         CA,OR,TX,SC,MN,PA,TN,VA
CUDA_VISIBLE_DEVICES=0 python run.py --config config/pda.yml --data.src_region FL --data.tgt_region NY\
    --data.other_regions CA,OR,TX,SC,MN,PA,TN,VA --seed.right 10&
CUDA_VISIBLE_DEVICES=1 python run.py --config config/pda.yml --acquisition.acquisition_function.reward sgd_score --data.src_region FL --data.tgt_region NY\
    --data.other_regions CA,OR,TX,SC,MN,PA,TN,VA --seed.right 10&    
CUDA_VISIBLE_DEVICES=2 python run.py --config config/oracle.yml --data.src_region FL --data.tgt_region NY\
    --data.other_regions CA,OR,TX,SC,MN,PA,TN,VA --seed.right 10& 
CUDA_VISIBLE_DEVICES=3 python run.py --config config/daa.yml --data.src_region FL --data.tgt_region NY\
    --data.other_regions CA,OR,TX,SC,MN,PA,TN,VA --seed.right 10