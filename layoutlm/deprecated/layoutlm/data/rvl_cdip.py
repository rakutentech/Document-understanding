print  # coding=utf-8
import copy
import json
import logging
import os
import re
from multiprocessing import Pool
import sys
import time

import torch
from lxml import html
from torch.utils.data import TensorDataset
from tqdm import tqdm
from transformers import DataProcessor
import traceback
from torch.utils.data import TensorDataset, Dataset
import numpy as np

from .image_preprocess import load_image

logger = logging.getLogger(__name__)


def get_text(node):
    textnodes = node.xpath(".//text()")
    s = "".join([text for text in textnodes])
    return re.sub(r"\s+", " ", s).strip()


def get_prop(node, name):
    title = node.get("title")
    props = title.split(";")
    for prop in props:
        (key, args) = prop.split(None, 1)
        args = args.strip('"')
        if key == name:
            return args
    return None


class DocExample(object):
    def __init__(self, guid, text_a, text_b=None, bbox=None, label=None, image_path=None):
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.bbox = bbox
        self.label = label
        self.image_path = image_path

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class CdipProcessor(DataProcessor):
    """Processor for the CDIP data set."""

    def worker(self, line):
        file, label = line.split()
        try:
            text, bbox = self.read_hocr_file(self.data_dir, file)
            return [text, bbox, label, os.path.join(self.data_dir, "images", file)]
        except Exception as e:
            traceback.print_exc()
            print(file)
            print(e)
            return None

    def get_examples(self, data_dir, mode, level=1 / 1000):
        self.data_dir = data_dir
        with open(os.path.join(data_dir, "labels", "{}.txt".format(mode))) as f:
            lines = f.readlines()

        lines = lines[:int(len(lines) * level)]
        print("num of samples", len(lines))
        examples = []
        with tqdm(lines, desc="get_examples function : Gettting {} examples".format(mode)) as t, Pool(48) as p:
            for example in p.imap(self.worker, lines):
                if example is not None:
                    examples.append(example)

                t.update()
        return self._create_examples(examples, mode)

    def _get_examples(self, data_dir, mode):
        with open(os.path.join(data_dir, "labels", "{}.txt".format(mode))) as f:
            lines = []
            for line in tqdm(f.readlines(), desc="_get_examples function : Gettting {} examples".format(mode)):
                file, label = line.split()
                text, bbox = self.read_hocr_file(data_dir, file)
                lines.append([text, bbox, label])
        return self._create_examples(lines, mode)

    def read_hocr_file(self, data_dir, file):
        #         hocr_file = os.path.join(data_dir, "images", file[:-4] + ".xml")
        hocr_file = os.path.join(data_dir, "images", file + ".hocr")
        text_buffer = []
        bbox_buffer = []
        try:
            doc = html.parse(hocr_file)
        except AssertionError:
            logger.warning(
                "%s is empty or its format is unacceptable. Skipped.", hocr_file
            )
            return [], []
        for page in doc.xpath("//*[@class='ocr_page']"):
            page_bbox = [int(x) for x in get_prop(page, "bbox").split()]
            width, height = page_bbox[2], page_bbox[3]
            for word in doc.xpath("//*[@class='ocrx_word']"):
                textnodes = word.xpath(".//text()")
                s = "".join([text for text in textnodes])
                text = re.sub(r"\s+", " ", s).strip()
                if text:
                    text_buffer.append(text)
                    bbox = [int(x) for x in get_prop(word, "bbox").split()]
                    bbox = [
                        bbox[0] / width,
                        bbox[1] / height,
                        bbox[2] / width,
                        bbox[3] / height,
                    ]
                    bbox = [int(x * 1000) for x in bbox]
                    bbox_buffer.append(bbox)
        return text_buffer, bbox_buffer

    def get_labels(self):
        return list(map(str, list(range(16))))

    def _create_examples(self, lines, mode):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (mode, i)
            text = line[0]
            bbox = line[1]
            label = line[2]
            image_path = line[3]
            examples.append(
                DocExample(guid=guid, text_a=text, text_b=None, bbox=bbox, label=label, image_path=image_path)
            )
        return examples


class DocFeature(object):
    def __init__(self, input_ids, bboxes, attention_mask, token_type_ids, label, image=None):
        assert (
                0 <= all(bboxes) <= 1000
        ), "Error with input bbox ({}): the coordinate value is not between 0 and 1000".format(
            bboxes
        )
        self.input_ids = input_ids
        self.bboxes = bboxes
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.label = label
        self.image = image

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


def convert_examples_to_features(
        examples,
        tokenizer,
        max_length=512,
        label_list=None,
        pad_on_left=False,
        pad_token="[PAD]",
        pad_token_id=0,
        pad_token_segment_id=0,
        mask_padding_with_zero=True,
        args=None,
        cached_features_file=None,
):
    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(tqdm(examples)):
        try:
            tokens = []
            bboxes = []
            image = load_image(example.image_path)

            if len(example.text_a) == 0:
                bboxes.append([0, 0, 0, 0])
                tokens.append(pad_token)

            for token, bbox in zip(example.text_a, example.bbox):
                sub_tokens = tokenizer.tokenize(token)
                for sub_token in sub_tokens:
                    bboxes.append(bbox)
                    tokens.append(sub_token)

            tokens = tokens[: max_length - 2]
            bboxes = bboxes[: max_length - 2]
            bboxes = [[0, 0, 0, 0]] + bboxes + [[1000, 1000, 1000, 1000]]

            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            input_ids = [tokenizer.cls_token_id] + input_ids + [tokenizer.sep_token_id]

            token_type_ids = [0] * len(input_ids)

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

            # Zero-pad up to the sequence length.
            padding_length = max_length - len(input_ids)
            if pad_on_left:
                input_ids = ([pad_token_id] * padding_length) + input_ids
                bboxes = ([[0, 0, 0, 0]] * padding_length) + bboxes
                attention_mask = (
                                         [0 if mask_padding_with_zero else 1] * padding_length
                                 ) + attention_mask
                token_type_ids = ([pad_token_segment_id] * padding_length) + token_type_ids
            else:
                input_ids = input_ids + ([pad_token_id] * padding_length)
                bboxes = bboxes + ([[0, 0, 0, 0]] * padding_length)
                attention_mask = attention_mask + (
                        [0 if mask_padding_with_zero else 1] * padding_length
                )
                token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)

        except Exception as e:
            traceback.print_exc()
            print(e)
            continue

        assert len(input_ids) == max_length, "Error with input length {} vs {}".format(
            len(input_ids), max_length
        )
        assert len(bboxes) == max_length, "Error with input length {} vs {}".format(
            len(bboxes), max_length
        )
        assert (
                len(attention_mask) == max_length
        ), "Error with input length {} vs {}".format(len(attention_mask), max_length)
        assert (
                len(token_type_ids) == max_length
        ), "Error with input length {} vs {}".format(len(token_type_ids), max_length)

        label = label_map[example.label]

        attention_mask.extend([1] * 197)

        features.append(
            DocFeature(
                input_ids=input_ids,
                bboxes=bboxes,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                label=label,
                image=image
            )
        )

    print("have loaded all image files")

    #     tmp_cached_features_file = f"{cached_features_file}-{i}"
    if args.local_rank in [-1, 0]:
        logger.info("Saving features into cached file %s", cached_features_file)
        torch.save(features, cached_features_file)

    return features


def load_and_cache_examples(args, tokenizer, mode="train", level=1 / 1000):
    if level == 1 and mode == "train":
        return RVL_CDIPDataset(args, tokenizer, mode=mode, level=level)

    if args.local_rank not in [-1, 0] and mode == "train":
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    processor = CdipProcessor()
    # Load data features from cache or dataset file
    cached_features_file = os.path.join(
        args.data_dir,
        "cached_{}_{}_{}".format(
            mode,
            list(filter(None, args.model_name_or_path.split("/"))).pop(),
            str(args.max_seq_length),
        ),
    )
    if level in [1, 0.1, 0.01, 0.001]:
        cached_features_file = os.path.join(args.data_dir, rf"{int(1 / level)}-{mode}-image")
        print("load features from ", cached_features_file)
    else:
        raise NotImplementedError("Not such data level")

    start_time = time.time()

    if (level == 0.1 and mode == "test") or (level == 1 and mode == "test"):
        features = []
        for i in range(10):
            tmp_cached_features_file = f"{cached_features_file}-{i}"
            if os.path.exists(tmp_cached_features_file) and not args.overwrite_cache:
                tmp_features = torch.load(tmp_cached_features_file)
                features.extend(tmp_features)
            else:
                raise RuntimeError(f"Can't find {tmp_cached_features_file}")
    else:
        if os.path.exists(cached_features_file) and not args.overwrite_cache:
            logger.info("Loading features from cached file %s", cached_features_file)
            features = torch.load(cached_features_file)

        else:
            logger.info("Creating features from dataset file at %s", args.data_dir)
            label_list = processor.get_labels()
            examples = processor.get_examples(args.data_dir, mode, level=level)
            print("have loaded all hocr files")
            features = convert_examples_to_features(
                examples,
                tokenizer,
                label_list=label_list,
                max_length=args.max_seq_length,
                pad_on_left=bool(args.model_type in ["xlnet"]),
                # pad on the left for xlnet
                pad_token=tokenizer.pad_token,
                pad_token_id=tokenizer.pad_token_id,
                pad_token_segment_id=4 if args.model_type in ["xlnet"] else 0,
                args=args,
                cached_features_file=cached_features_file
            )

    print(f"use {time.time() - start_time} seconds to generate features")
    if args.local_rank == 0 and mode == "train":
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_bboxes = torch.tensor([f.bboxes for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor(
        [f.attention_mask for f in features], dtype=torch.long
    )
    all_token_type_ids = torch.tensor(
        [f.token_type_ids for f in features], dtype=torch.long
    )

    all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
    #     print(f"{mode} labels" , all_labels)
    if level == 1 and mode == "train":
        all_image_ids = torch.tensor([f.image.numpy() for f in features], dtype=torch.float)
        dataset = TensorDataset(
            all_input_ids, all_attention_mask, all_token_type_ids, all_labels, all_bboxes, all_image_ids
        )
        return dataset
    else:
        all_images = torch.tensor(np.array([f.image.numpy() for f in features]), dtype=torch.float)

        dataset = TensorDataset(
            all_input_ids, all_attention_mask, all_token_type_ids, all_labels, all_bboxes, all_images
        )
        return dataset


def get_imagepaths(examples):
    imagepaths = [example.image_path for example in examples]
    return imagepaths


class RVL_CDIPDataset(Dataset):
    def __init__(self, args, tokenizer, mode="train", level=1):
        if args.local_rank not in [-1, 0] and mode == "train":
            torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

        processor = CdipProcessor()
        # Load data features from cache or dataset file
        cached_features_file = os.path.join(
            args.data_dir,
            "cached_{}_{}_{}".format(
                mode,
                list(filter(None, args.model_name_or_path.split("/"))).pop(),
                str(args.max_seq_length),
            ),
        )
        assert level == 1
        assert mode == "train"
        cached_features_file = os.path.join(args.data_dir, r"1-train-new")

        logger.info("Loading features from cached file %s", cached_features_file)
        start_time = time.time()
        features = torch.load(cached_features_file)
        print(f"use {time.time() - start_time} seconds to generate features")

        imagepaths = []
        with open(os.path.join(args.data_dir, "labels", "{}.txt".format(mode))) as f:
            lines = f.readlines()
        for line in lines:
            file, _ = line.split()
            imagepaths.append(os.path.join(args.data_dir, "images", file))

        self.image_paths = imagepaths

        if args.local_rank == 0 and mode == "train":
            torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache
        self.features = features
        # Convert to Tensors and build dataset
        self.all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        self.all_bboxes = torch.tensor([f.bboxes for f in features], dtype=torch.long)
        self.all_attention_mask = torch.tensor(
            [f.attention_mask for f in features], dtype=torch.long
        )
        self.all_token_type_ids = torch.tensor(
            [f.token_type_ids for f in features], dtype=torch.long
        )
        self.all_labels = torch.tensor([f.label for f in features], dtype=torch.long)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        image = load_image(self.image_paths[idx])
        return self.all_input_ids[idx], self.all_attention_mask[idx], \
            self.all_token_type_ids[idx], self.all_labels[idx], self.all_bboxes[idx], image
