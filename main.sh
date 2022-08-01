export CUDA_VISIBLE_DEVICES=1
data_dir=./sample_data
nmd=./new_modelzw
pof=./predictions.txt
log=./log.txt

#training
# NOTE: we have more options available, you can check our wiki for more information
python ./src/relation_extraction.py \
		--model_type bert \
		--data_format_mode 0 \
		--classification_scheme 1 \
		--pretrained_model bert-base-uncased \
		--data_dir $data_dir \
		--new_model_dir $nmd \
		--predict_output_file $pof \
		--overwrite_model_dir \
		--seed 13 \
		--max_seq_length 256 \
		--cache_data \
		--do_train \
		--do_lower_case \
		--train_batch_size 4 \
		--eval_batch_size 4 \
		--learning_rate 1e-5 \
		--num_train_epochs 3 \
		--gradient_accumulation_steps 1 \
		--do_warmup \
		--warmup_ratio 0.1 \
		--weight_decay 0 \
		--max_num_checkpoints 1 \
		--log_file $log \

#prediction
# we have to set data_dir, new_model_dir, model_type, log_file, and eval_batch_size, data_format_mode
python ./src/relation_extraction.py \
		--model_type bert \
		--data_format_mode 0 \
		--classification_scheme 1 \
		--pretrained_model bert-base-uncased \
		--data_dir $data_dir \
		--new_model_dir $nmd \
		--predict_output_file $pof \
		--overwrite_model_dir \
		--seed 13 \
		--max_seq_length 256 \
		--cache_data \
		--do_predict \
		--do_lower_case \
		--eval_batch_size 4 \
		--log_file $log \

#post processing
python src/data_processing/post_processing.py \
		--mode mul \
		--predict_result_file $pof \
		--entity_data_dir ./test_data_entity_only \
		--test_data_file ${data_dir}/test.tsv \
		--brat_result_output_dir ./brat_output