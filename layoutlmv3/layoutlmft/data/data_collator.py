import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
import sys
from transformers import BatchEncoding, PreTrainedTokenizerBase
from transformers.data.data_collator import (
    DataCollatorMixin,
    _torch_collate_batch,
)
from transformers.file_utils import PaddingStrategy

from typing import NewType
InputDataClass = NewType("InputDataClass", Any)


@dataclass
class DataCollatorForKeyValueExtraction(DataCollatorMixin):
    """
    Data collator that will dynamically pad the inputs received, as well as the labels.
    Args:
        tokenizer (:class:`~transformers.PreTrainedTokenizer` or :class:`~transformers.PreTrainedTokenizerFast`):
            The tokenizer used for encoding the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.file_utils.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
        label_pad_token_id (:obj:`int`, `optional`, defaults to -100):
            The id to use when padding the labels (-100 will be automatically ignore by PyTorch loss functions).
    """

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    MS = None
    
    def __call__(self, features):
        label_name = "label" if "label" in features[0].keys() else "labels"
        labels = [feature[label_name] for feature in features] if label_name in features[0].keys() else None
#         print(features[0].keys())  #dict_keys(['attention_mask', 'bbox', 'images', 'input_ids', 'labels'])
        images = None
        if "images" in features[0]:
            images = torch.stack([torch.tensor(d.pop("images")) for d in features])
            IMAGE_LEN = int(images.shape[-1] / 16) * int(images.shape[-1] / 16) + 1
#             print(images.shape)  # batchsize, 3, 224, 224


        batch = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            # Conversion to tensors will fail if we have labels as they are not of the same length yet.
            return_tensors="pt" if labels is None else None,
        )
        
        has_bbox_input = "bbox" in features[0]
        has_position_input = "position_ids" in features[0]
        padding_idx=self.tokenizer.pad_token_id
        sequence_length = torch.tensor(batch["input_ids"]).shape[1]  # 412
        padding_side = self.tokenizer.padding_side
        TOKEN_LEN = 512 - sequence_length
        
        
        if images is not None:
            batch["images"] = images
            batch = {k: torch.tensor(v, dtype=torch.int64) if isinstance(v[0], list) and k == 'attention_mask' else v
                     for k, v in batch.items()}
            visual_attention_mask = torch.ones((len(batch['input_ids']), IMAGE_LEN), dtype=torch.long)
            if self.MS is None:
                batch["attention_mask"] = torch.cat([batch['attention_mask'], visual_attention_mask], dim=1)
            else:
                token_attention_mask = torch.ones((len(batch['input_ids']), TOKEN_LEN), dtype=torch.long)
                batch["attention_mask"] = torch.cat([batch['attention_mask'], token_attention_mask, visual_attention_mask], dim=1)

        if labels is None:
            return batch
        
        if self.MS is not None:
#             START : generate token_bboxes, token_labels, token_input_ids
            token_bboxes = []
            token_labels = []
            token_input_ids = []
            for bbox, labels_, input_ids in zip(batch["bbox"], batch["labels"],batch['input_ids']):
                token_bbox = []
                token_label = []
                token_input_id = []
                tmp_bbox_set = set()
                for bbox_, label_, input_id_ in zip(bbox, labels_, input_ids):
                    if bbox_ != [0,0,0,0] and tuple(bbox_) not in tmp_bbox_set:
                        tmp_bbox_set.add(tuple(bbox_))
                        token_bbox.append(bbox_)
                        token_label.append(label_)
                        token_input_id.append(input_id_)
                assert TOKEN_LEN > len(token_bbox)
                token_bboxes.append(token_bbox)
                token_labels.append(token_label)
                token_input_ids.append(token_input_id)
            token_labels = [label +  [self.label_pad_token_id] * (TOKEN_LEN - len(label)) for label in token_labels]
            token_input_ids = [input_ids +  [1] * (TOKEN_LEN - len(input_ids)) for input_ids in token_input_ids]
            token_bboxes= [bbox +  [[0,0,0,0]] * (TOKEN_LEN - len(bbox)) for bbox in token_bboxes]
            batch["input_ids"] = torch.cat([torch.Tensor(batch['input_ids']), torch.Tensor(token_input_ids)], dim=1).int()
#            END : generate token_bboxes, token_labels, token_input_ids



        if padding_side == "right":
#             print(has_bbox_input) # True
#             print(has_position_input)  # False
#             print(batch["bbox"][0])

            batch["labels"] = [label + [self.label_pad_token_id] * (sequence_length - len(label)) for label in labels]
#             print(len(batch['labels']), len(batch['labels'][0]))  # 2,100
#             print(len(token_labels), len(token_labels[0]))  # 2,412
            if self.MS is not None:
                batch["labels"] = torch.cat([torch.Tensor(batch['labels']), torch.Tensor(token_labels)], dim=1).int()
                
            if has_bbox_input:
                batch["bbox"] = [bbox + [[0, 0, 0, 0]] * (sequence_length - len(bbox)) for bbox in batch["bbox"]]
                if self.MS is not None:
                    batch["bbox"] = torch.cat([torch.Tensor(batch['bbox']), torch.Tensor(token_bboxes)], dim=1).int()
                
            if has_position_input:
                batch["position_ids"] = [position_id + [padding_idx] * (sequence_length - len(position_id))
                                          for position_id in batch["position_ids"]]

        else:
            batch["labels"] = [[self.label_pad_token_id] * (sequence_length - len(label)) + label for label in labels]
            if has_bbox_input:
                batch["bbox"] = [[[0, 0, 0, 0]] * (sequence_length - len(bbox)) + bbox for bbox in batch["bbox"]]
            if has_position_input:
                batch["position_ids"] = [[padding_idx] * (sequence_length - len(position_id))
                                          + position_id for position_id in batch["position_ids"]]

        batch = {k: torch.tensor(v, dtype=torch.int64) if isinstance(v[0], list) else v for k, v in batch.items()}
        
        
        if images is not None:
            visual_labels = torch.ones((len(batch['input_ids']), IMAGE_LEN), dtype=torch.long) * -100
            batch["labels"] = torch.cat([batch['labels'], visual_labels], dim=1)
        batch["cell"] = None
        return batch
