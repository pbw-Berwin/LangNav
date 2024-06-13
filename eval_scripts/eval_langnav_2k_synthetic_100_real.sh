model_name="bpan/LangNav-Sim2k-Llama2"

evaluation_flags="
--maxAction 15
--batchSize 16
--data_path ./img_features/r2r_blip_DETR_vis2text
--history_first True
--model_name_or_path $model_name
"

data_splits=("val_seen" "val_unseen")

# Loop through each data split
for data_split in "${data_splits[@]}"
do
    echo "Evaluating on $data_split"
    # Adjust the master_port incrementally to avoid conflicts
    if [ "$data_split" = "val_seen" ]; then
        port=2005
    else
        port=2006
    fi

    # Run the evaluation command
    torchrun --nproc_per_node=1 --master_port=$port eval_agent.py --data_split $data_split $evaluation_flags
done
