exp_name="gpt-4_synthetic_sim_2k_real_100_llama2"

collect_flag="
--maxAction 15
--batchSize 16
--data_path ./img_features/r2r_blip_DETR_vis2text
--history_first True
--exp_name $exp_name
"

checkpoint_path="./outputs/$exp_name/checkpoint-520/"

model_and_data="
  --model_name_or_path $checkpoint_path
  --data_split val_seen
"

torchrun --nproc_per_node=1 --master_port=2005 eval_agent.py $collect_flag $model_and_data

model_and_data="
  --model_name_or_path $checkpoint_path
  --data_split val_unseen
"

torchrun --nproc_per_node=1 --master_port=2006 eval_agent.py $collect_flag $model_and_data