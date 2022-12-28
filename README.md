# Multi-scale Cell-based Layout Representation for Document Understanding
The code will be released soon. Any inquiries at mijung.a.kim@rakuten.com


# Model
- LayoutLM
    - Base: microsoft/layoutlm-base-uncased
    - Large: microsoft/layoutlm-base-uncased
- LayoutLMv2
  - Base: microsoft/layoutlmv2-base-uncased
  - Large: microsoft/layoutlmv2-large-uncased
- LayoutLMv3
  - Base: microsoft/layoutlmv3-base
  - Large: microsoft/layoutlmv3-large
***
# Dataset
- FUNSD
  - https://guillaumejaume.github.io/FUNSD/
- CORD
  - https://github.com/clovaai/cord
- RVL-CDIP
  - https://adamharley.com/rvl-cdip/
***
# Enviroment
## Container1
will be released soon.
***
# Results
***
# Commands
## Named-entity Recognition
### LayoutLMv1
- CODE DIR: layoutlmft/
- CONTAINER: Container1
- ENV: layoutlmft
```
python -m torch.distributed.launch --nproc_per_node=1 
--master_port 44398 
examples/run_funsd.py         
--model_name_or_path microsoft/layoutlm-base-uncased        
--output_dir output/test-ner3         
--do_train         
--do_predict         
--max_steps 5000         
--warmup_ratio 0.1         
--fp16   
--per_device_train_batch_size 4
```
### LayoutLMv2
- CODE DIR: layoutlmft/
- CONTAINER: Container1
- ENV: layoutlmft
```
python -m torch.distributed.launch --nproc_per_node=1 
--master_port 24398 examples/run_funsd.py         
--model_name_or_path microsoft/layoutlmv2-large-uncased         
--output_dir output/test-ner2         
--do_train         
--do_predict         
--max_steps 2000         
--warmup_ratio 0.1         
--fp16   
--overwrite_output_dir   
--per_device_train_batch_size 6
```
### LayoutLMv3
- CODE DIR: layoutlmv3
- CONTAINER: Container1
- ENV: v3
```
python -m torch.distributed.launch   
--nproc_per_node=1 
--master_port 4398 examples/run_funsd_cord.py   
--dataset_name funsd   
--do_train 
--do_eval   
--model_name_or_path microsoft/layoutlmv3-base   
--output_dir output/experimentID   
--segment_level_layout 1 
--visual_embed 1 
--input_size 224  
--max_steps 1000 
--save_steps -1 
--evaluation_strategy steps 
--eval_steps 1000   
--learning_rate 1e-5 
--per_device_train_batch_size 8 
--gradient_accumulation_steps 1   
--dataloader_num_workers 1   
--overwrite_output_dir  
--dataset_name cord
```
## Document Classification
### LayoutLMv1
- CODE DIR: layoutlm/deprecated/examples/classification
- CONTAINER: Container1
- ENV: v3

<!-- # CUDA_VISIBLE_DEVIC****ES=2  python run_classification.py  --data_dir  /root/dev/Datasets/RVL_CDIP                               --model_type layoutlm                               --model_name_or_path /root/dev/Models/layoutlm-base-uncased                                --output_dir output/shi-temp-asdfsadf                               --do_lower_case                               --max_seq_length 512                               --do_train                               --do_eval                               --num_train_epochs 40.0                               --logging_steps 5000                               --save_steps 5000                               --per_gpu_train_batch_size 16                               --per_gpu_eval_batch_size 16                               --evaluate_during_training                               --fp16
CUDA_VISIBLE_DEVICES=2  python run_classification.py  
--data_dir  /root/dev/Datasets/RVL_CDIP                               
--model_type layoutlm                               
--model_name_or_path /root/dev/Models/layoutlm-base-uncased                                
--output_dir output/shi-temp-asdfsadf                               
--do_lower_case                             
--max_seq_length 512                              
--do_train                              
--do_eval                               
--num_train_epochs 40.0                               
--logging_steps 5000                               
--save_steps 5000                               
--per_gpu_train_batch_size 16                               
--per_gpu_eval_batch_size 16                               
--evaluate_during_training                               
--fp16
-->
```
python run_classification.py  
--data_dir  /root/dev/Datasets/RVL_CDIP   
--model_type layoutlm                               
--model_name_or_path ~/dev/Models/LayoutLM/layoutlm-base-uncased   
--output_dir output/
--do_lower_case \
--max_seq_length 512 \
--do_train \
--do_eval \
--num_train_epochs 40.0 \
--logging_steps 5000 \
--save_steps 5000 \
--per_gpu_train_batch_size 16 \
--per_gpu_eval_batch_size 16 \
--evaluate_during_training \
--fp16 --data_level 1
```
### LayoutLMv3
- CODE DIR: layoutlm/deprecated/examples/classification
- CONTAINER: Container1
- ENV: v3
```
python run_classification.py  
--data_dir  /root/dev/Datasets/RVL_CDIP                               
--model_type v3                               
--model_name_or_path microsoft/layoutlmv3-base                                                            
--do_lower_case                               
--max_seq_length 512                               
--do_train                                                            
--num_train_epochs 40.0                               
--logging_steps 5000                               
--save_steps 5000                               
--per_gpu_train_batch_size 2                               
--per_gpu_eval_batch_size 2                               
--evaluate_during_training
```
***
# Reference
We use some codes from https://github.com/microsoft/unilm.