from dagger_agent import DAggerAgent
import copy
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence

import torch
import transformers
import os
import random
import numpy as np
from lm_env import R2RBatch
from llm_api import llama_generate_api
import json
import io
from evaluation import Evaluation

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "</s>"
DEFAULT_UNK_TOKEN = "</s>"


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    maxAction: int = field(default=15)
    feedback: str = field(default="sample")
    name: str = field(default="test")
    policy_beta: float = field(default=0.8)
    lora_weights: str = field(default="")
    one_shot_eval: bool = field(default=False)
    
@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    traj_path: str = field(default=None, metadata={"help": "Path to the traj file."})
    fg_direction: bool = field(default=False, metadata={"help": "Whether to use fg_direction."})
    fg_feature: str = field(default="", metadata={"help": "which feature to use."})
    history_first: bool = field(default=False, metadata={"help": "Whether to use history_first."})
    dataset_debug: bool = field(default=False, metadata={"help": "Whether to use debug mode."})
    batchSize: int = field(default=1, metadata={"help": "Batch size."})
    model_max_length: int = field(
        default=4096,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    exp_seed: int = field(default=0, metadata={"help": "Random seed."})
    data_split: str = field(default="train", metadata={"help": "Which split to use."})
    exp_name: str = field(default="", metadata={"help": "Experiment name."})
    
def _make_r_io_base(f, mode: str):
    if not isinstance(f, io.IOBase):
        f = open(f, mode=mode)
    return f

def jload(f, mode="r"):
    """Load a .json file into a dictionary."""
    f = _make_r_io_base(f, mode)
    jdict = json.load(f)
    f.close()
    return jdict

def set_seed(seed: int):
    """Set seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    transformers.set_seed(seed)

def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg

def prepare_model_and_tokenizer(data_args, model_args):
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        device_map="auto",
        torch_dtype=torch.bfloat16        
    )
    
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        model_max_length=data_args.model_max_length,
        padding_side="left",
        use_fast=False,
    )

    if tokenizer.pad_token is None:
        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
            tokenizer=tokenizer,
            model=model,
        )
    
    if "llama" in model_args.model_name_or_path:
        tokenizer.add_special_tokens(
            {
                "eos_token": DEFAULT_EOS_TOKEN,
                "bos_token": DEFAULT_BOS_TOKEN,
                "unk_token": DEFAULT_UNK_TOKEN,
            }
        )

    tokenizer.padding_side = "left"
    return model, tokenizer

def prepare_feat_dict(data_args):
    logging.warning("Loading mp3d visual2text data...")
    feat_dict = {}
    scene_files = os.listdir(os.path.join(data_args.data_path))
    for scene_file in scene_files:
        if scene_file.endswith('.json'):
            scene_name = scene_file.split('.')[0]
            feat_dict[scene_name] = jload(os.path.join(data_args.data_path, scene_file))
    featurized_scans = set([key.split("_")[0] for key in list(feat_dict.keys())])
    return feat_dict, featurized_scans

def main():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments))
    model_args, data_args = parser.parse_args_into_dataclasses()

    # set seed for reproducibility
    set_seed(data_args.exp_seed)

    # prepare model and tokenizer
    model, tokenizer = prepare_model_and_tokenizer(data_args, model_args)
    llm_api = llama_generate_api(model, tokenizer)

    # prepare feat dict
    feat_dict, featurized_scans = prepare_feat_dict(data_args)

    splits = data_args.data_split.split('+')

    agent = DAggerAgent(
        llm=llm_api, 
        env=R2RBatch(batch_size=data_args.batchSize, splits=splits), 
        results_path="", 
        feat_dict=feat_dict,
        data_args=data_args,
        model_args=model_args,
    )

    iters = None

    if model_args.lora_weights != "" and model_args.lora_weights != "None":
        # env_name = model_args.lora_weights.split('/')[-2]
        save_path = os.path.join(model_args.lora_weights, 'sequantial_results')
    else:
        save_path = os.path.join(model_args.model_name_or_path, 'sequantial_results')
    
    if os.path.exists(os.path.join(save_path, '{}.json'.format(splits[0]))):
        print('Loading previous results from %s' % save_path)
        with open(os.path.join(save_path, '{}.json'.format(splits[0])), 'r') as f:
            result = json.load(f)
    else:    
        agent.test(iters=iters)
        result = agent.get_results()
        print('Saving results to %s' % save_path)

        if not os.path.exists(save_path):
            os.mkdir(save_path)

        with open(os.path.join(save_path, '{}.json'.format(splits[0])), 'w') as f:
            json.dump(result, f)

    evaluator = Evaluation(splits, featurized_scans)
    score_summary, _ = evaluator.score(result)
    loss_str = "Env name: %s" % splits[0]
    for metric,val in score_summary.items():
        loss_str += ', %s: %.4f' % (metric, val)
    print(loss_str)

if __name__ == '__main__':
    main()