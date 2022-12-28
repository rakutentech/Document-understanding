# coding=utf-8

from __future__ import absolute_import, division, print_function
import sys
import argparse
import glob
import logging
import os
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
from transformers import (
    WEIGHTS_NAME,
    AdamW,
    BertConfig,
    BertForSequenceClassification,
    BertTokenizerFast,
    RobertaConfig,
    RobertaForSequenceClassification,
    RobertaTokenizer,
    get_linear_schedule_with_warmup,
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

from layoutlm import LayoutlmConfig, LayoutlmForSequenceClassification, LayoutLMv3ForSequenceClassification
from transformers import AutoModelForSequenceClassification

from layoutlm.data.rvl_cdip import CdipProcessor, load_and_cache_examples, RVL_CDIPDataset

try:
    from torch.utils.tensorboard import SummaryWriter
except:
    from tensorboardX import SummaryWriter


logger = logging.getLogger(__name__)

# ALL_MODELS = sum(
#     (
#         tuple(conf.pretrained_config_archive_map.keys())
#         for conf in ( RobertaConfig, LayoutlmConfig)
#     ),
#     (),
# )
ALL_MODELS = ""


MODEL_CLASSES = {
    "bert": (BertConfig, BertForSequenceClassification, BertTokenizerFast),
    "roberta": (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),
    "layoutlm": (LayoutlmConfig, LayoutlmForSequenceClassification, BertTokenizerFast),
    "v3" : (AutoConfig, AutoModelForSequenceClassification, AutoTokenizer),
}


def write_2_txt(filepath, content):
    if not os.path.exists(os.path.dirname(filepath)):
        os.makedirs(os.path.dirname(filepath))
    with open(filepath, "a+") as f:
        f.write(content+"\n")


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)
    # Set seed before initializing model.
    from transformers import set_seed as t_set_seed
    t_set_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def simple_accuracy(preds, labels):
    return (preds == labels).mean()

        
def train(args, train_dataset, model, tokenizer, data_level=1/1000):  # noqa C901
    """ Train the model """
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter(comment="_" + os.path.basename(args.output_dir))

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = (
        RandomSampler(train_dataset)
        if args.local_rank == -1
        else DistributedSampler(train_dataset)
    )
    train_dataloader = DataLoader(
        train_dataset, sampler=train_sampler, batch_size=args.train_batch_size
    )
    
    print("create eval_dataloader for evaluate in training")
    eval_dataset = load_and_cache_examples(args, tokenizer, mode="val", level=data_level)
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(
        eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size
    )
        
    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = (
            args.max_steps
            // (len(train_dataloader) // args.gradient_accumulation_steps)
            + 1
        )
    else:
        t_total = (
            len(train_dataloader)
            // args.gradient_accumulation_steps
            * args.num_train_epochs
        )
    print("t_total", t_total)

    
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(
        optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use fp16 training."
            )
        model, optimizer = amp.initialize(
            model, optimizer, opt_level=args.fp16_opt_level
        )

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            find_unused_parameters=True,
        )

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info(
        "  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size
    )
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps
        * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)
    
    best_acc = 0
    best_model_filepath = r""
    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(
        int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0]
    )
    print(train_iterator)
    if args.model_type in ["layoutlm", "v3"]:
        model_type = args.model_type
    
    for _ in train_iterator:
        epoch_iterator = tqdm(
            train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0]
        )
        for step, batch in enumerate(epoch_iterator):
            model.train()
            if args.model_type != model_type :
                batch = batch[:4]
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "labels": batch[3],
                "images" : batch[5],
            }
            if args.model_type == model_type:
                inputs["bbox"] = batch[4]
            inputs["token_type_ids"] = (
                batch[2] if args.model_type in ["bert", model_type] else None
            )  # RoBERTa don't use segment_ids
            outputs = model(**inputs)
            loss = outputs[
                0
            ]  # model outputs are always tuple in transformers (see doc)
#             print(loss)
            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    amp.master_params(optimizer), args.max_grad_norm
                )
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
#             print("args.gradient_accumulation_steps ", args.gradient_accumulation_steps )
#             print(" args.local_rank ", args.local_rank )
#             print("args.logging_steps", args.logging_steps)
#             print("args.save_steps", args.save_steps)
#             print("args.evaluate_during_training", args.evaluate_during_training)
#             print("global_step", global_step)



            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                if (
                    args.local_rank in [-1, 0]
                    and args.logging_steps > 0
                    and global_step % args.logging_steps == 0
                ):
                    # Log metrics
                    if (
                        args.local_rank == -1 and args.evaluate_during_training
                    ):  # Only evaluate when single GPU otherwise metrics may not average well
                        results = evaluate(args, model, tokenizer, "val", level=data_level, eval_dataloader=eval_dataloader)
                        for key, value in results.items():
                            tb_writer.add_scalar(
                                "eval_{}".format(key), value, global_step
                            )
                        print(results)
                        write_2_txt(args.log_file, f"{global_step} : {results}")
                        iter_acc = results ["acc"]
                        save_model = best_acc < iter_acc
                        if best_acc < iter_acc:
                            best_acc = iter_acc
#                     print("==============================")
#                     print(scheduler.get_lr()[0])
#                     print(scheduler.get_last_lr())
                    tb_writer.add_scalar("lr", scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar(
                        "loss",
                        (tr_loss - logging_loss) / args.logging_steps,
                        global_step,
                    )
                    logging_loss = tr_loss

                if (
                    args.local_rank in [-1, 0]
                    and args.save_steps > 0
                    and global_step % args.save_steps == 0 and save_model
                ):
                    # Save model checkpoint
                    output_dir = os.path.join(
                        args.output_dir, "checkpoint-best")
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = (
                        model.module if hasattr(model, "module") else model
                    )  # Take care of distributed/parallel training
                    model_to_save.save_pretrained(output_dir)
                    torch.save(args, os.path.join(output_dir, "training_args.bin"))
                    tokenizer.save_pretrained(output_dir)
                    logger.info("Saving model checkpoint to %s", output_dir)

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    if args.local_rank in [-1, 0]:
        tb_writer.close()
    write_2_txt(args.log_file, f"eval restult : {best_acc}")
    return global_step, tr_loss / global_step


def evaluate(args, model, tokenizer, mode, prefix="", level=1/1000, eval_dataloader=None):
    results = {}
    if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir)
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    logger.info("***** Running evaluation {} *****".format(prefix))
    
    if eval_dataloader is None:
        eval_dataset = load_and_cache_examples(args, tokenizer, mode=mode, level=level)

        # Note that DistributedSampler samples randomly
        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(
            eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size
        )
        logger.info("  Num examples = %d", len(eval_dataset))
        
    # Eval!
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None

    if args.model_type in ["layoutlm", "v3"]:
        model_type = args.model_type
        
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        if args.model_type != model_type:
            batch = batch[:4]
        batch = tuple(t.to(args.device) for t in batch)
        
        with torch.no_grad():
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "labels": batch[3],
                "images": batch[5],
            }
            if args.model_type == model_type:
                inputs["bbox"] = batch[4]
            inputs["token_type_ids"] = (
                batch[2] if args.model_type in ["bert", model_type] else None
            )  # RoBERTa don"t use segment_ids
            outputs = model(**inputs)
            tmp_eval_loss, logits = outputs[:2]
            
            eval_loss += tmp_eval_loss.mean().item()
        nb_eval_steps += 1
        if preds is None:
            preds = logits.detach().cpu().numpy()
            out_label_ids = inputs["labels"].detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(
                out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0
            )

    eval_loss = eval_loss / nb_eval_steps
    preds = np.argmax(preds, axis=1)
    
    result = {"acc": simple_accuracy(preds=preds, labels=out_label_ids)}
    results.update(result)

#     output_eval_file = os.path.join(
#         args.output_dir, prefix, "{}_results.txt".format(mode)
#     )
    output_eval_file = os.path.join(
        args.output_dir, "{}_results.txt".format(mode)
    )
    with open(output_eval_file, "w") as writer:
        logger.info("***** {} results {} *****".format(mode, prefix))
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
            writer.write("%s = %s\n" % (key, str(result[key])))

#     output_eval_file = os.path.join(
#         args.output_dir, prefix, "{}_figmpare.txt".format(mode)
#     )
    
    output_eval_file = os.path.join(
        args.output_dir, "{}_figmpare.txt".format(mode)
    )
    with open(output_eval_file, "w") as writer:
        for p, l in zip(preds, out_label_ids):
            writer.write("%s %s\n" % (p, l))
    return results


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
        required=True,
        help="The input data dir. Should contain the .tsv files (or other data files) for the task.",
    )
    parser.add_argument(
        "--model_type",
        default=None,
        type=str,
        required=True,
        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()),
    )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to pre-trained model or shortcut name selected in the list: "
        + ", ".join(ALL_MODELS),
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )

    ## Other parameters
    parser.add_argument(
        "--config_name",
        default="",
        type=str,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        default="",
        type=str,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--cache_dir",
        default="",
        type=str,
        help="Where do you want to store the pre-trained models downloaded from s3",
    )
    parser.add_argument(
        "--max_seq_length",
        default=512,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
        "than this will be truncated, sequences shorter will be padded.",
    )

    parser.add_argument(
        "--do_train", action="store_true", help="Whether to run training."
    )
    parser.add_argument(
        "--do_eval", action="store_true", help="Whether to run eval on the dev set."
    )
    parser.add_argument(
        "--do_test", action="store_true", help="Whether to run test on the test set."
    )
    parser.add_argument(
        "--evaluate_during_training",
        action="store_true",
        help="Rul evaluation during training at each logging step.",
    )
    parser.add_argument(
        "--do_lower_case",
        action="store_true",
        help="Set this flag if you are using an uncased model.",
    )

    parser.add_argument(
        "--per_gpu_train_batch_size",
        default=8,
        type=int,
        help="Batch size per GPU/CPU for training.",
    )
    parser.add_argument(
        "--per_gpu_eval_batch_size",
        default=8,
        type=int,
        help="Batch size per GPU/CPU for evaluation.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--learning_rate",
        default=5e-5,
        type=float,
        help="The initial learning rate for Adam.",
    )
    parser.add_argument(
        "--weight_decay", default=0.0, type=float, help="Weight deay if we apply some."
    )
    parser.add_argument(
        "--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer."
    )
    parser.add_argument(
        "--max_grad_norm", default=1.0, type=float, help="Max gradient norm."
    )
    parser.add_argument(
        "--num_train_epochs",
        default=3.0,
        type=float,
        help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument(
        "--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps."
    )

    parser.add_argument(
        "--logging_steps", type=int, default=50, help="Log every X updates steps."
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=50,
        help="Save checkpoint every X updates steps.",
    )
    parser.add_argument(
        "--eval_all_checkpoints",
        action="store_true",
        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number",
    )
    parser.add_argument(
        "--no_cuda", action="store_true", help="Avoid using CUDA when available"
    )
    parser.add_argument(
        "--overwrite_output_dir",
        action="store_true",
        help="Overwrite the content of the output directory",
    )
    parser.add_argument(
        "--overwrite_cache",
        action="store_true",
        help="Overwrite the cached training and evaluation sets",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="random seed for initialization"
    )

    parser.add_argument(
        "--tpu",
        action="store_true",
        help="Whether to run on the TPU defined in the environment variables",
    )

    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O1",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
        "See details at https://nvidia.github.io/apex/amp.html",
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="For distributed training: local_rank",
    )
    parser.add_argument(
        "--server_ip", type=str, default="", help="For distant debugging."
    )
    parser.add_argument(
        "--server_port", type=str, default="", help="For distant debugging."
    
    )
    parser.add_argument("--data_level", type=float, default=0.001, help="set number of samples")
    
    parser.add_argument("--pretrain",  action="store_true",  help="Use pretrain model ")
    parser.add_argument("--use_image",  action="store_true",  help="Use image data")
    
    parser.add_argument(
        "--sort_by", type=str, default="row", help="sort_by"
    )
    parser.add_argument("--sort_flag",  action="store_true",  help="sort_flag")
    parser.add_argument(
        "--PE_type",
        type=int,
        default=1,
        help="For distributed training: local_rank",
    )

    
    args = parser.parse_args()
    
    if (
        os.path.exists(args.output_dir)
        and os.listdir(args.output_dir)
        and args.do_train
        and not args.overwrite_output_dir
    ):
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir
            )
        )

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd

        print("Waiting for debugger attach")
        ptvsd.enable_attach(
            address=(args.server_ip, args.server_port), redirect_output=True
        )
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device(
            "cuda:0" if torch.cuda.is_available() and not args.no_cuda else "cpu"
        )
        if torch.cuda.is_available():
            torch.cuda.set_device(device)
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
        args.fp16,
    )

    # Set seed
    set_seed(args)
    # generate my log file
    my_log_file_path = os.path.join(args.output_dir, f"{int(time.time())}.txt")
    
    args.log_file = my_log_file_path
   
    processor = CdipProcessor()
    label_list = processor.get_labels()
    num_labels = len(label_list)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    
    if args.model_type == "v3":
        config = AutoConfig.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        num_labels=num_labels,
    )
    else:
        config = config_class.from_pretrained(
           args.config_name if args.config_name else args.model_name_or_path,
           num_labels=num_labels,
        )
    
    
#     config = AutoConfig.from_pretrained(pretrained_model_name_or_path="microsoft/layoutlmv3-base",num_labels=16)
#     config = AutoConfig.from_pretrained(pretrained_model_name_or_path="/root/dev/Models/LayoutLM/layoutlm-base-uncased", num_labels=16)
#     config.use_image = use_image
    
    

    config.data_level = args.data_level
    print("data_level", config.data_level)
    config.pretrain_flag = args.pretrain
    print("config.pretrain_flag", config.pretrain_flag)
    args.logging_steps = int(args.logging_steps * config.data_level * 8 / args.n_gpu)
    args.save_steps = int(args.save_steps * config.data_level* 8 / args.n_gpu)
    print("args.logging_steps", args.logging_steps)
    print("args.save_steps", args.save_steps)
    
#     config.sort_by = "col"
#     config.sort_flag = True
#     config.PE_type = 1
    config.sort_by = args.sort_by
    config.sort_flag = args.sort_flag
    config.PE_type = args.PE_type
    print(config.sort_by)
    print(config.sort_flag)
    print(config.PE_type)
    #   1:  [x1,y1,x3,y3,h,w]
    #   2:  [x1,y1,x3,y3,h,w,row,col]  # row and col share weights with h and col
    #   3:  [x1,y1,x3,y3,h,w,1,1]
    #   4:  [x1,y1,x3,y3,h,w,h,w]
    #   5:  [x1,y1,x3,y3,row,col]
    #   6:  [x1,y1,x3,y3,h,w,row,col]  # row and col don't share weights with h and col
    #   7:  [x1,y1,h,w,row,col]
    

    config.default_cell_id = [255, 255]
    config.default_scale = 0.005
    config.cell_embedding_dim = 709

    config.in_attn=False 
    config.norm_coor=False  # 让相邻的坐标使用相同的值，结果并没有太大的差别，暂时先放着
    config.cell_emb = False  # 使用cell emb或者不使用
    
    
    #     config.attention_type = "simple"  # 直接去掉，query，key, valude 的计算
    config.attention_type = "default"  # 默认的形式

    
    config.DA = None
    config.DA_eval = None
    config.DA_level = 0.5
    config.DA_bbox = False
    config.MS = None
    config.word_embedding_size = 512
    config.metric = 2   
    #     0 :default, based on word and token
    #     1 :only based on word
    #     2 :only based on toekn 
    
    if args.model_type == "v3":
        tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        tokenizer_file=None,  # avoid loading from a cached file of the pre-trained model in another machine
#         cache_dir=model_args.cache_dir,
        use_fast=True,
        add_prefix_space=True,
#         revision=args.model_revision,
#         use_auth_token=True if model_args.use_auth_token else None,
    )
    else:
        tokenizer = tokenizer_class.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        do_lower_case=args.do_lower_case,
    )
    
      #     print("create eval_dataloader for evaluate in training")
#     eval_dataset = load_and_cache_examples(args, tokenizer, mode="val", level=1)
#     args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
#     eval_sampler = SequentialSampler(eval_dataset)
#     eval_dataloader = DataLoader(
#         eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size
#     )

#     config.use_return_dict = True
    if config.pretrain_flag:
        model = model_class.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
        )
        print(f"load model from {args.model_name_or_path}")
    else:
        if args.model_type == "v3":
            model = LayoutLMv3ForSequenceClassification(config)
        elif args.model_type == "layoutlm":
            model = LayoutlmForSequenceClassification(config)
        else:
            raise RuntimeError("No such model type")

    print("log_file", args.log_file)
    write_2_txt(args.log_file, str(vars(args)))
    write_2_txt(args.log_file, str(config))
    
    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    model.to(args.device)

    logger.info("Training/evaluation parameters %s", args)

    # Training
    if args.do_train:
        train_dataset = load_and_cache_examples(args, tokenizer, mode="train", level=config.data_level)
        global_step, tr_loss = train(args, train_dataset, model, tokenizer, data_level=config.data_level)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    # Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        # Create output directory if needed
        if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(args.output_dir)

        logger.info("Saving model checkpoint to %s", args.output_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model_to_save = (
            model.module if hasattr(model, "module") else model
        )  # Take care of distributed/parallel training
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(args.output_dir, "training_args.bin"))

        # Load a trained model and vocabulary that you have fine-tuned
        model = model_class.from_pretrained(args.output_dir)
        tokenizer = tokenizer_class.from_pretrained(
            args.output_dir, do_lower_case=args.do_lower_case
        )
        model.to(args.device)

    # Evaluation
    results = {}
    if args.do_eval and args.local_rank in [-1, 0]:
        write_2_txt(args.log_file, f"Start Evaluation")
        
        tokenizer = tokenizer_class.from_pretrained(
            args.output_dir, do_lower_case=args.do_lower_case
        )
        checkpoints = [args.output_dir]
        
        if args.eval_all_checkpoints:
            checkpoints = list(
                os.path.dirname(c)
                for c in sorted(
                    glob.glob(args.output_dir + "/**/" + WEIGHTS_NAME, recursive=True)
                )
            )
            logging.getLogger("transformers.modeling_utils").setLevel(
                logging.WARN
            )  # Reduce logging

        checkpoints = [os.path.join(args.output_dir, "checkpoint-best")] 
        logger.info("Evaluate the following checkpoints: %s", checkpoints)

        for checkpoint in checkpoints:
            print("load checkpoint", checkpoint)
            global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
            prefix = (
                checkpoint.split("/")[-1]
                if checkpoint.find("checkpoint") != -1 and args.eval_all_checkpoints
                else ""
            )

            model = model_class.from_pretrained(checkpoint)
            model.to(args.device)
            
            result = evaluate(args, model, tokenizer, mode="val", prefix="val", level=config.data_level)
            result = dict(
                ("val_" + k + "_{}".format(global_step), v) for k, v in result.items()
            )
            results.update(result)
            print("evaluation result : " , result)
            write_2_txt(args.log_file, f"val result : {result}")
            
    if args.do_test and args.local_rank in [-1, 0]:
        
        write_2_txt(args.log_file, f"Start Testing")

        tokenizer = tokenizer_class.from_pretrained(
            args.output_dir, do_lower_case=args.do_lower_case
        )
        checkpoints = [args.output_dir]
        if args.eval_all_checkpoints:
            checkpoints = list(
                os.path.dirname(c)
                for c in sorted(
                    glob.glob(args.output_dir + "/**/" + WEIGHTS_NAME, recursive=True)
                )
            )
            logging.getLogger("transformers.modeling_utils").setLevel(
                logging.WARN
            )  # Reduce logging
        checkpoints = [os.path.join(args.output_dir, "checkpoint-best")] 
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
            prefix = (
                checkpoint.split("/")[-1]
                if checkpoint.find("checkpoint") != -1 and args.eval_all_checkpoints
                else ""
            )

            model = model_class.from_pretrained(checkpoint)
            model.to(args.device)
            result = evaluate(args, model, tokenizer, mode="test", prefix="test", level=config.data_level)
            result = dict(
                ("test_" + k + "_{}".format(global_step), v) for k, v in result.items()
            )
            write_2_txt(args.log_file, f"test result : {result}")
            
            results.update(result)
    print("cat ", args.log_file)
    
    return results


if __name__ == "__main__":
    sys.path.insert(0, '/root/dev/unilm/layoutlm/deprecated')

    main()
