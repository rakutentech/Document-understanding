import logging
import sys

import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
from transformers import BertConfig, BertModel, BertPreTrainedModel
# from transformers.modeling_bert import BertLayerNorm
#from transformers.models.bert.modeling_bert import BertLayerNorm
#BertLayerNorm = torch.nn.LayerNorm
from torch.nn import LayerNorm as BertLayerNorm
import numpy as np

logger = logging.getLogger(__name__)

LAYOUTLM_PRETRAINED_MODEL_ARCHIVE_MAP = {}

LAYOUTLM_PRETRAINED_CONFIG_ARCHIVE_MAP = {}




# find key point of every bbox
def get_keypoint(bbox):
    return bbox["boundingPoly"][0]

def intergrate(coos, threshold):
    new_coos = [coos[0]]
    for idx, coo in enumerate(coos[:-1]):
        if coos[idx+1] - coo > threshold:
            new_coos.append(coos[idx+1])
    return new_coos

def get_coo_2_cell_dict(coos, max_coo):
        
    cell_layout = [0 for i in range(max_coo)]
    if len(coos) == 1:
        cell_layout[coos[0]:] = [1 for i in range(max_coo-coos[0])]
        return cell_layout
    
    pre_value = 0
    for idx, value in enumerate(coos[1:]):
        value = int(value)
        for i in range(pre_value, value):
            cell_layout[i] = idx
        pre_value = value
    cell_layout[value:] = [idx+1 for i in range(max_coo-value)]

    return cell_layout
    
def get_bbox(path):
    with open(path) as f:
        annotation = json.load(f)
    return annotation['annotations']

def get_cell_coo(cell_coos,num_cell,max_coo):
    if len(cell_coos) ==1:
        return [cell_coos[0],1000]
    
    if num_cell > len(cell_coos)-2:
        start = cell_coos[num_cell]
        end = max_coo
    else:
        start =  cell_coos[num_cell]
        end =  cell_coos[num_cell+1]
    return [start, end]
    
    
def create_cell_layout_4_layoutlm(bboxes, max_x, max_y,scale=0.01):
    # Use key point the calulate number of rows and columns
    row_coos, col_coos=set(), set()
    cells = []
    cells_no = []
    for bbox in bboxes:
        key_point = bbox[:2]
        row_coos.add(key_point[1])
        col_coos.add(key_point[0])
        
    # sort x coordinates and y coordinates
    row_coos=list(row_coos)
    row_coos.sort()
    col_coos=list(col_coos)
    col_coos.sort()

    # intergrate coos
    row_coos = intergrate(row_coos, max_y*scale)
    col_coos = intergrate(col_coos, max_x*scale)
    row_coor_2_cell_dict = get_coo_2_cell_dict(row_coos, max_y)
    col_coor_2_cell_dict = get_coo_2_cell_dict(col_coos, max_x)
    cell_coordinates = []
    # give every bbox a cell number
    for bbox in bboxes:
        key_point = bbox[:2]
        row_cell = row_coor_2_cell_dict[int(key_point[1])]
        col_cell = col_coor_2_cell_dict[int(key_point[0])]
        
        y1, y3 = get_cell_coo(row_coos, row_cell, max_y)
        x1, x3 = get_cell_coo(col_coos, col_cell, max_x)
        cells.append([x1, y1, x3, y3])
        cells_no.append([row_cell, col_cell])
    return cells, cells_no




class LayoutlmConfig(BertConfig):
    pretrained_config_archive_map = LAYOUTLM_PRETRAINED_CONFIG_ARCHIVE_MAP
    model_type = "bert"

    def __init__(self, max_2d_position_embeddings=1024, **kwargs):
        super().__init__(**kwargs)
        self.max_2d_position_embeddings = max_2d_position_embeddings


class LayoutlmEmbeddings(nn.Module):
    def __init__(self, config):
        super(LayoutlmEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=0
        )
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size
        )
        self.x_position_embeddings = nn.Embedding(
            config.max_2d_position_embeddings, config.hidden_size
        )
        self.y_position_embeddings = nn.Embedding(
            config.max_2d_position_embeddings, config.hidden_size
        )
        self.h_position_embeddings = nn.Embedding(
            config.max_2d_position_embeddings, config.hidden_size
        )
        self.w_position_embeddings = nn.Embedding(
            config.max_2d_position_embeddings, config.hidden_size
        )
        self.token_type_embeddings = nn.Embedding(
            config.type_vocab_size, config.hidden_size
        )
        
        self.row_position_embeddings = nn.Embedding(config.max_2d_position_embeddings, config.hidden_size)
        self.col_position_embeddings = nn.Embedding(config.max_2d_position_embeddings, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(
        self,
        input_ids,
        bbox,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
        cell=None,
    ):
        seq_length = input_ids.size(1)
        if position_ids is None:
            position_ids = torch.arange(
                seq_length, dtype=torch.long, device=input_ids.device
            )
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        left_position_embeddings = self.x_position_embeddings(bbox[:, :, 0])
        upper_position_embeddings = self.y_position_embeddings(bbox[:, :, 1])
        if cell is None:
            right_position_embeddings = self.x_position_embeddings(bbox[:, :, 2])
            lower_position_embeddings = self.y_position_embeddings(bbox[:, :, 3])
        else:
            right_position_embeddings = self.row_position_embeddings(cell[:, :, 0])
            lower_position_embeddings = self.col_position_embeddings(cell[:, :, 1])
        h_position_embeddings = self.h_position_embeddings(
            bbox[:, :, 3] - bbox[:, :, 1]
        )
        w_position_embeddings = self.w_position_embeddings(
            bbox[:, :, 2] - bbox[:, :, 0]
        )
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = (
            words_embeddings
            + position_embeddings
            + left_position_embeddings
            + upper_position_embeddings
            + right_position_embeddings
            + lower_position_embeddings
            + h_position_embeddings
            + w_position_embeddings
            + token_type_embeddings
        )
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class LayoutlmModel(BertModel):

    config_class = LayoutlmConfig
    pretrained_model_archive_map = LAYOUTLM_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "bert"

    def __init__(self, config):
        super(LayoutlmModel, self).__init__(config)
        self.embeddings = LayoutlmEmbeddings(config)
        
        self.config.sort_flag = False
        self.config.sort_by = "row"
        self.config.PE_type = 7
        self.config.default_cell_id = [255, 255]
        self.config.default_scale = 0.005
        
        
        self.init_weights()

    def forward(
        self,
        input_ids,
        bbox,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
    ):

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        cells = None
        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(
            dtype=torch.float32
        )  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = (
                    head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                )
                head_mask = head_mask.expand(
                    self.config.num_hidden_layers, -1, -1, -1, -1
                )
            elif head_mask.dim() == 2:
                head_mask = (
                    head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
                )  # We can specify head_mask for each layer
            head_mask = head_mask.to(
                dtype=next(self.parameters()).dtype
            )  # switch to fload if need + fp16 compatibility
        else:
            head_mask = [None] * self.config.num_hidden_layers
        
        if self.config.sort_flag or self.config.PE_type in [2,5,6,7]:
            cell_inputs = []
            cell_coor_inputs = []
            cell_position_inputs = []
            for sample in bbox.cpu().numpy():
                sample = [[str(i) for i in j] for j in sample]
                sample = [",".join(i) for i in sample]
                sample_set = set(sample)
                sample_set.remove("0,0,0,0")
                new_bboxes = []
                cells = []
                cell_coors = []
                cell_positions = []
                for box in sample_set:
                    new_box = box.split(",")
                    new_box = [int(i) for i in new_box]
                    new_bboxes.append(new_box)

                cell_coordinates, cell_no = create_cell_layout_4_layoutlm(new_bboxes,1000,1000,scale=self.config.default_scale)

                cell_dict = {}
                cell_coor_dict = {}
                for idx, box in enumerate(sample_set):
                    cell_dict[box] = cell_no[idx] 
                    cell_coor_dict[box] = cell_coordinates[idx] 

                cell_dict["0,0,0,0"] = self.config.default_cell_id
                cell_coor_dict["0,0,0,0"] = [0,0,0,0]
#                 cell_dict["0,0,0,0"] = [1, 1]
                
                for i in sample:
                    cells.append(cell_dict[i])
                    cell_coors.append(cell_coor_dict[i])
#                     cells.append([1,1])
                    if self.config.sort_flag:
                        if self.config.sort_by == "col":
                            cell_positions.append(str(cell_dict[i][1]).zfill(3) + str(cell_dict[i][0]).zfill(3))
                        elif self.config.sort_by == "row":
                            cell_positions.append(str(cell_dict[i][0]).zfill(3) + str(cell_dict[i][1]).zfill(3))
                        else:
                            raise NotImplementedError("No Such sort_by, use --sort_by col or --sort_by row ")
                cell_inputs.append(cells)
                cell_coor_inputs.append(cell_coors)
                cell_position_inputs.append(cell_positions)
                

            cells = torch.from_numpy(np.array(cell_inputs)).cuda().long()
            cell_coors = torch.from_numpy(np.array(cell_coor_inputs)).cuda().int()
            cell_positions = np.array(cell_position_inputs)
            
        if self.config.sort_flag:
                new_idxs = []
                for i in range(bbox.shape[0]):
                    data = cell_positions[i]
                    idx = np.array( data[data!='10001000']).argsort(kind="mergesort")+1
                    length = len(idx)
                    new_idxs.append(idx)

                    input_ids[i,1:length+1] = input_ids[i,idx]
                    bbox[i,1:length+1,:] = bbox[i,idx,:]
                    token_type_ids[i,1:length+1] = token_type_ids[i,idx]
                
                new_idxs = np.array(new_idxs)
        
        
        embedding_output = self.embeddings(
            input_ids, bbox, position_ids=position_ids, token_type_ids=token_type_ids,cell=cells
        
        )
        encoder_outputs = self.encoder(
            embedding_output, extended_attention_mask, head_mask=head_mask
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output)

        outputs = (sequence_output, pooled_output) + encoder_outputs[
            1:
        ]  # add hidden_states and attentions if they are here
        return outputs  # sequence_output, pooled_output, (hidden_states), (attentions)


class LayoutlmForTokenClassification(BertPreTrainedModel):
    config_class = LayoutlmConfig
    pretrained_model_archive_map = LAYOUTLM_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "bert"

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.layoutlm = LayoutlmModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

    def forward(
        self,
        input_ids,
        bbox,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
    ):
        outputs = self.layoutlm(
            input_ids=input_ids,
            bbox=bbox,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
        )

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        outputs = (logits,) + outputs[
            2:
        ]  # add hidden states and attention if they are here
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)[active_loss]
                active_labels = labels.view(-1)[active_loss]
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), scores, (hidden_states), (attentions)


class LayoutlmForSequenceClassification(BertPreTrainedModel):
    config_class = LayoutlmConfig
    pretrained_model_archive_map = LAYOUTLM_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "bert"

    def __init__(self, config):
        super(LayoutlmForSequenceClassification, self).__init__(config)
        self.num_labels = config.num_labels

        self.bert = LayoutlmModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)

        self.init_weights()

    def forward(
        self,
        input_ids,
        bbox,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
    ):

        outputs = self.bert(
            input_ids=input_ids,
            bbox=bbox,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        outputs = (logits,) + outputs[
            2:
        ]  # add hidden states and attention if they are here

        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)
