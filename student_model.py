from abc import ABC
from copy import deepcopy

import torch
import torch.nn as nn
from torch.nn.init import xavier_normal_, xavier_uniform_
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from dict_hub import get_entity_dict, EntityDict, TripletDict

from dataclasses import dataclass
from transformers import AutoModel, AutoConfig
from triplet_mask import construct_mask
from transformers.models.bert.modeling_bert import BertConfig, BertPreTrainedModel, BertEncoder, BertModel


entity_dict: EntityDict = get_entity_dict()


class StudentModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        self.config = AutoConfig.from_pretrained(args.pretrained_model)

        self.log_inv_t = torch.nn.Parameter(torch.tensor(1.0 / args.t).log(), requires_grad=args.finetune_t)
        self.add_margin = args.additive_margin
        self.batch_size = args.batch_size
        # pre_batch: number of pre-batch used for negatives
        self.pre_batch = args.pre_batch
        # 定义pre_batch_vectors数量
        num_pre_batch_vectors = max(1, self.pre_batch) * self.batch_size

        random_vector = torch.randn(num_pre_batch_vectors, self.config.hidden_size)

        self.register_buffer("pre_batch_vectors",
                             nn.functional.normalize(random_vector, dim=1), 
                             persistent=False)
        
        self.offset = 0

        self.pre_batch_exs = [None for _ in range(num_pre_batch_vectors)]

 
        origin_bert = BertModel.from_pretrained("bert-base-uncased")
        teacher_config = origin_bert.config.to_dict()
        teacher_config["num_hidden_layers"] = args.student_bert_num_hidden_layers
        student_config = BertConfig.from_dict(teacher_config)
        self.teacher_layers = eval(args.teacher_layers)
        self.student_layers = eval(args.student_layers)
        self.hr_bert = type(origin_bert)(student_config)
        self.tail_bert = type(origin_bert)(student_config)

        self._distill_bert_weights(origin_bert, self.hr_bert)
        self._distill_bert_weights(origin_bert, self.tail_bert)

    

    def _encode(self, encoder, token_ids, mask, token_type_ids, mask_idx=None, mask_flag=True):
        outputs = encoder(input_ids=token_ids,
                          attention_mask=mask,
                          token_type_ids=token_type_ids,
                          return_dict=True)
        # print(0, mask)
        last_hidden_state = outputs.last_hidden_state
        cls_output = last_hidden_state[:, 0, :]
        # print(1, last_hidden_state.shape)
        cls_output = _pool_output(self.args.pooling, cls_output, mask, last_hidden_state)
        mask_output = torch.empty(cls_output.shape[0], cls_output.shape[1]).to(cls_output.device)
        if mask_flag:
            mask_output = torch.empty(cls_output.shape[0], cls_output.shape[1]).to(cls_output.device)
            mask_idx = mask_idx.tolist()
            mask_idx = [[x for x in sublist if x != -1] for sublist in mask_idx]
            for i, mask_index in enumerate(mask_idx):
                if len(mask_index) == 0:
                    mask_output[i] = torch.ones((1, 768))
                else:
                    a = last_hidden_state[i, mask_index, :].unsqueeze(0)
                    mask_output[i] = _pool_output("mean", a, mask[i, mask_index], a)
            return cls_output, mask_output
        return cls_output, mask_output


    def forward(self, hr_token_ids, hr_mask, hr_token_type_ids,
                tail_token_ids, tail_mask, tail_token_type_ids,
                head_token_ids, head_mask, head_token_type_ids,
                hr_token_mask, tail_token_mask, head_token_mask,
                student_hr_token_ids, student_tail_token_ids, student_head_token_ids,
                only_ent_embedding=False, **kwargs) -> dict:
        if only_ent_embedding:
            return self.predict_ent_embedding(tail_token_ids=student_tail_token_ids,
                                              tail_mask=tail_mask,
                                              tail_token_type_ids=tail_token_type_ids)

        hr_vector, hr_mask_output = self._encode(self.hr_bert,
                                 token_ids=student_hr_token_ids,
                                 mask=hr_mask,
                                 token_type_ids=hr_token_type_ids,
                                 mask_idx=hr_token_mask)

        tail_vector, tail_mask_output = self._encode(self.tail_bert,
                                   token_ids=student_tail_token_ids,
                                   mask=tail_mask,
                                   token_type_ids=tail_token_type_ids,
                                   mask_idx=tail_token_mask)

        head_vector, head_mask_output = self._encode(self.tail_bert,
                                   token_ids=student_head_token_ids,
                                   mask=head_mask,
                                   token_type_ids=head_token_type_ids,
                                   mask_idx=head_token_mask)

        # DataParallel only support tensor/dict
        return {'hr_vector': hr_vector,
                'tail_vector': tail_vector,
                'head_vector': head_vector,
                'hr_mask_output': hr_mask_output,
                'tail_mask_output' : tail_mask_output,
                'head_mask_output': head_mask_output}


    def compute_logits(self, output_dict: dict, batch_dict: dict) -> dict:
        hr_vector, tail_vector = output_dict['hr_vector'], output_dict['tail_vector']
        batch_size = hr_vector.size(0)
        labels = torch.arange(batch_size).to(hr_vector.device)

        logits = hr_vector.mm(tail_vector.t())

        if self.training:
            logits -= torch.zeros(logits.size()).fill_diagonal_(self.add_margin).to(logits.device)

        logits *= self.log_inv_t.exp()

        triplet_mask = batch_dict.get('triplet_mask', None)
        if triplet_mask is not None:
            logits.masked_fill_(~triplet_mask, -1e4)

        if self.pre_batch > 0 and self.training:
            pre_batch_logits = self._compute_pre_batch_logits(hr_vector, tail_vector, batch_dict)
            logits = torch.cat([logits, pre_batch_logits], dim=-1)

        if self.args.use_self_negative and self.training:
            head_vector = output_dict['head_vector']
            self_neg_logits = torch.sum(hr_vector * head_vector, dim=1) * self.log_inv_t.exp()
            self_negative_mask = batch_dict['self_negative_mask']
            self_neg_logits.masked_fill_(~self_negative_mask, -1e4)
            logits = torch.cat([logits, self_neg_logits.unsqueeze(1)], dim=-1)

        return {'logits': logits,
                'labels': labels,
                'inv_t': self.log_inv_t.detach().exp(),
                'hr_vector': hr_vector.detach(),
                'tail_vector': tail_vector.detach()}

    def _compute_pre_batch_logits(self, hr_vector: torch.tensor,
                                  tail_vector: torch.tensor,
                                  batch_dict: dict) -> torch.tensor:

        assert tail_vector.size(0) == self.batch_size

        batch_exs = batch_dict['batch_data']

        pre_batch_logits = hr_vector.mm(self.pre_batch_vectors.clone().t())
        pre_batch_logits *= self.log_inv_t.exp() * self.args.pre_batch_weight
        if self.pre_batch_exs[-1] is not None:
            pre_triplet_mask = construct_mask(batch_exs, self.pre_batch_exs).to(hr_vector.device)
            pre_batch_logits.masked_fill_(~pre_triplet_mask, -1e4)
        

        self.pre_batch_vectors[self.offset:(self.offset + self.batch_size)] = tail_vector.data.clone()
        self.pre_batch_exs[self.offset:(self.offset + self.batch_size)] = batch_exs
        self.offset = (self.offset + self.batch_size) % len(self.pre_batch_exs)

        return pre_batch_logits
    
    def _distill_bert_weights(
        self,
        teacher : nn.Module,
        student : nn.Module,
    ) -> None:
        """
        Recursively copies the weights of the (teacher) to the (student).
        This function is meant to be first called on a RobertaFor... model, but is then called on every children of that model recursively.
        The only part that's not fully copied is the encoder, of which only half is copied.
        """
        # If the part is an entire RoBERTa model or a RobertaFor..., unpack and iterate
        if isinstance(teacher, BertModel) or type(teacher).__name__.startswith('BertModel'):
            for teacher_part, student_part in zip(teacher.children(), student.children()):
                self._distill_bert_weights(teacher_part, student_part)
        # Else if the part is an encoder, copy one out of every layer
        elif isinstance(teacher, BertEncoder):
                teacher_encoding_layers = [layer for layer in next(teacher.children())]
                student_encoding_layers = [layer for layer in next(student.children())]
                for i in self.student_layers:
                    student_encoding_layers[i].load_state_dict(teacher_encoding_layers[self.teacher_layers[i]].state_dict())
        # Else the part is a head or something else, copy the state_dict
        else:
            student.load_state_dict(teacher.state_dict())

    @torch.no_grad()
    def predict_ent_embedding(self, tail_token_ids, tail_mask, tail_token_type_ids, **kwargs) -> dict:
        ent_vectors, _ = self._encode(self.tail_bert,
                                   token_ids=tail_token_ids,
                                   mask=tail_mask,
                                   token_type_ids=tail_token_type_ids,
                                   mask_flag=False)
        return {'ent_vectors': ent_vectors.detach()}

# cls_output = _pool_output(self.args.pooling, cls_output, mask, last_hidden_state)
def _pool_output(pooling: str,
                 cls_output: torch.tensor,
                 mask: torch.tensor,
                 last_hidden_state: torch.tensor) -> torch.tensor:
    if pooling == 'cls':
        output_vector = cls_output
    elif pooling == 'max':
        input_mask_expanded = mask.unsqueeze(-1).expand(last_hidden_state.size()).long()
        last_hidden_state[input_mask_expanded == 0] = -1e4
        output_vector = torch.max(last_hidden_state, 1)[0]
    elif pooling == 'mean':
        input_mask_expanded = mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-4)
        output_vector = sum_embeddings / sum_mask
    else:
        assert False, 'Unknown pooling mode: {}'.format(pooling)

    output_vector = nn.functional.normalize(output_vector, dim=1)
    return output_vector
