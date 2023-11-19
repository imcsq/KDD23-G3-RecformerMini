from typing import List, Union, Optional, Tuple
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn import CrossEntropyLoss

from transformers.models.longformer.modeling_longformer import (
    LongformerConfig,
    LongformerPreTrainedModel,
    LongformerEncoder,
    LongformerBaseModelOutputWithPooling,
    LongformerLMHead
)


class RecformerConfig(LongformerConfig):
    """ `RecformerConfig` inherits from `LongformerConfig` """
    def __init__(self, 
                attention_window: Union[List[int], int] = 64, 
                sep_token_id: int = 102,  # 2, # token ID: [PAD]: 0, [CLS]: 101, [SEP]: 102
                token_type_size: int = 4,  # <s>, key, value, <pad>
                max_token_num: int = 2048,
                max_item_embeddings: int = 32,  # 1 for <s>, 50 for items
                max_attr_num: int = 12,
                max_attr_length: int = 8,
                pooler_type: str = 'cls',
                temp: float = 0.05,
                mlm_weight: float = 0.1,
                item_num: int = 0,
                finetune_negative_sample_size: int = 0,
                **kwargs):

        # set up the inherited configuration parameters, `**kwargs` refers to any additional keyword arguments
        super().__init__(attention_window, sep_token_id, **kwargs)

        # set up the additional attributes that are specific to `RecformerConfig` 
        self.token_type_size = token_type_size
        self.max_token_num = max_token_num
        self.max_item_embeddings = max_item_embeddings
        self.max_attr_num = max_attr_num
        self.max_attr_length = max_attr_length
        self.pooler_type = pooler_type
        self.temp = temp
        self.mlm_weight = mlm_weight

        # fine-tuning config
        self.item_num = item_num
        self.finetune_negative_sample_size = finetune_negative_sample_size


class RecformerEmbeddings(nn.Module):
    """ Use BertEmbeddings + a tiny tweak for positional embeddings indexing """
    def __init__(self, config: RecformerConfig):
        super().__init__()

        self.padding_idx = config.pad_token_id

        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size, self.padding_idx)
        self.token_type_embeddings = nn.Embedding(config.token_type_size, config.hidden_size)
        self.item_position_embeddings = nn.Embedding(config.max_item_embeddings, config.hidden_size)

        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        
        
    def forward(self, input_ids=None, token_type_ids=None, position_ids=None, item_position_ids=None, inputs_embeds=None):
        if input_ids is None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            input_shape = input_ids.size()

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.create_position_ids_from_input_ids(input_ids).to(input_ids.device)

        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        word_embeddings = inputs_embeds
        position_embeddings = self.position_embeddings(position_ids)
        item_position_embeddings = self.item_position_embeddings(item_position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = word_embeddings + position_embeddings + token_type_embeddings + item_position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

    def create_position_ids_from_input_ids(self, input_ids):
        """ Replace non-padding symbols with their position numbers. Position numbers begin at padding_idx+1. Padding symbols are ignored. """
        mask = input_ids.ne(self.padding_idx).int()
        incremental_indices = torch.cumsum(mask, dim=1).type_as(mask) * mask
        return incremental_indices.long() + self.padding_idx   


class RecformerModel(LongformerPreTrainedModel):
    def __init__(self, config: RecformerConfig):
        super().__init__(config)
        self.config = config
        self.pooler_type = config.pooler_type  # 'cls'

        if isinstance(config.attention_window, int):
            assert config.attention_window > 0, f"`config.attention_window` should be positive, given {config.attention_window}"
            assert config.attention_window % 2 == 0, f"`config.attention_window` should be an even value, given {config.attention_window}"
            config.attention_window = [config.attention_window] * config.num_hidden_layers
        else:
            assert len(config.attention_window) == config.num_hidden_layers, f"`len(config.attention_window)` should equal to `config.num_hidden_layers`, given {len(config.attention_window)}, expected {config.num_hidden_layers}."

        self.embeddings = RecformerEmbeddings(config)
        self.encoder = LongformerEncoder(config)

        self.post_init()

    def pad_to_window_size(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, token_type_ids: torch.Tensor, position_ids: torch.Tensor, item_position_ids: torch.Tensor, inputs_embeds: torch.Tensor, pad_token_id: int):
        """Pad tokens and mask to work with implementation of Longformer self-attention."""
        if isinstance(self.config.attention_window, int):
            attention_window = self.config.attention_window
        else:
            attention_window = max(self.config.attention_window)
        assert attention_window % 2 == 0, f"`attention_window` should be an even value, given {attention_window}."
        
        if input_ids is None:
            input_shape = inputs_embeds.shape
        else:
            input_shape = input_ids.shape

        batch_size, seq_len = input_shape[:2]
        padding_len = (attention_window - seq_len % attention_window) % attention_window

        if padding_len > 0:
            if input_ids is not None:
                input_ids = nn.functional.pad(input_ids, (0, padding_len), value=pad_token_id)
            if position_ids is not None:
                position_ids = nn.functional.pad(position_ids, (0, padding_len), value=pad_token_id)
            if item_position_ids is not None:
                item_position_ids = nn.functional.pad(item_position_ids, (0, padding_len), value=pad_token_id)
            if inputs_embeds is not None:
                input_ids_padding = inputs_embeds.new_full((batch_size, padding_len), self.config.pad_token_id, dtype=torch.long)
                inputs_embeds_padding = self.embeddings(input_ids_padding)
                inputs_embeds = torch.cat([inputs_embeds, inputs_embeds_padding], dim=-2)

            attention_mask = nn.functional.pad(attention_mask, (0, padding_len), value=False)  # no attention on the padding tokens
            token_type_ids = nn.functional.pad(token_type_ids, (0, padding_len), value=0)  # pad with token_type_id = 0

        return padding_len, input_ids, attention_mask, token_type_ids, position_ids, item_position_ids, inputs_embeds

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        global_attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        item_position_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, LongformerBaseModelOutputWithPooling]:

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You must specify either input_ids or inputs_embeds")

        if input_ids is None:
            device = inputs_embeds.device
        else:
            device = input_ids.device

        if output_attentions is None:
            output_attentions = self.config.output_hidden_states
        else:
            output_attentions = output_attentions

        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)
        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if global_attention_mask is not None:
            if attention_mask is None:
                attention_mask = global_attention_mask + 1
            else:
                attention_mask = attention_mask * (global_attention_mask + 1)

        padding_len, input_ids, attention_mask, token_type_ids, position_ids, item_position_ids, inputs_embeds = self.pad_to_window_size(input_ids, attention_mask, token_type_ids, position_ids, item_position_ids, inputs_embeds, self.config.pad_token_id)

        # self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape, device)[:, 0, 0, :]

        embedding_output = self.embeddings(input_ids=input_ids, position_ids=position_ids, item_position_ids=item_position_ids, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds)
        encoder_outputs = self.encoder(embedding_output, attention_mask=extended_attention_mask, head_mask=head_mask, padding_len=padding_len, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        sequence_output = encoder_outputs[0]
        pooled_output = sequence_output[:, 0]  # `pooler_type` is hard-coded as 'cls'.

        if return_dict is None:
            return_dict = self.config.use_return_dict
        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return LongformerBaseModelOutputWithPooling(last_hidden_state=sequence_output, pooler_output=pooled_output, hidden_states=encoder_outputs.hidden_states, attentions=encoder_outputs.attentions, global_attentions=encoder_outputs.global_attentions)


class Similarity(nn.Module):
    """
    Dot product or cosine similarity
    """

    def __init__(self, config: RecformerConfig):
        super().__init__()
        self.temp = config.temp
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, x, y):
        return self.cos(x, y) / self.temp


@dataclass
class RecformerPretrainingOutput:
    misc: dict = None
    cl_correct_num: float = 0.0
    cl_total_num: float = 1e-5
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    global_attentions: Optional[Tuple[torch.FloatTensor]] = None


class RecformerForPretraining(LongformerPreTrainedModel):
    def __init__(self, config: RecformerConfig):
        super().__init__(config)

        self.longformer = RecformerModel(config)
        self.lm_head = LongformerLMHead(config)
        self.sim = Similarity(config)
        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids_a: Optional[torch.Tensor] = None,
        attention_mask_a: Optional[torch.Tensor] = None,
        global_attention_mask_a: Optional[torch.Tensor] = None,
        token_type_ids_a: Optional[torch.Tensor] = None,
        item_position_ids_a: Optional[torch.Tensor] = None,
        mlm_input_ids_a: Optional[torch.Tensor] = None,
        mlm_labels_a: Optional[torch.Tensor] = None,
        input_ids_b: Optional[torch.Tensor] = None,
        attention_mask_b: Optional[torch.Tensor] = None,
        global_attention_mask_b: Optional[torch.Tensor] = None,
        token_type_ids_b: Optional[torch.Tensor] = None,
        item_position_ids_b: Optional[torch.Tensor] = None,
        mlm_input_ids_b: Optional[torch.Tensor] = None,
        mlm_labels_b: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        batch_size = input_ids_a.size(0)

        outputs_a = self.longformer(
            input_ids_a,
            attention_mask=attention_mask_a,
            global_attention_mask=global_attention_mask_a,
            head_mask=head_mask,
            token_type_ids=token_type_ids_a,
            position_ids=position_ids,
            item_position_ids=item_position_ids_a,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )
        outputs_b = self.longformer(
            input_ids_b,
            attention_mask=attention_mask_b,
            global_attention_mask=global_attention_mask_b,
            head_mask=head_mask,
            token_type_ids=token_type_ids_b,
            position_ids=position_ids,
            item_position_ids=item_position_ids_b,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )

        # MLM auxiliary objective
        mlm_outputs_a = None
        if mlm_input_ids_a is not None:
            mlm_outputs_a = self.longformer(
                mlm_input_ids_a,
                attention_mask=attention_mask_a,
                global_attention_mask=global_attention_mask_a,
                head_mask=head_mask,
                token_type_ids=token_type_ids_a,
                position_ids=position_ids,
                item_position_ids=item_position_ids_a,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=True,
            )

        mlm_outputs_b = None
        if mlm_input_ids_b is not None:
            mlm_outputs_b = self.longformer(
                mlm_input_ids_b,
                attention_mask=attention_mask_b,
                global_attention_mask=global_attention_mask_b,
                head_mask=head_mask,
                token_type_ids=token_type_ids_b,
                position_ids=position_ids,
                item_position_ids=item_position_ids_b,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=True,
            )

        z1 = outputs_a.pooler_output  # (bs*num_sent, hidden_size)
        z2 = outputs_b.pooler_output  # (bs*num_sent, hidden_size)

        # Gather all embeddings if using distributed training
        if dist.is_initialized() and self.training:
            
            # Dummy vectors for allgather
            z1_list = [torch.zeros_like(z1) for _ in range(dist.get_world_size())]
            z2_list = [torch.zeros_like(z2) for _ in range(dist.get_world_size())]
            # Allgather
            dist.all_gather(tensor_list=z1_list, tensor=z1.contiguous())
            dist.all_gather(tensor_list=z2_list, tensor=z2.contiguous())

            # Since allgather results do not have gradients, we replace the
            # current process's corresponding embeddings with original tensors
            z1_list[dist.get_rank()] = z1
            z2_list[dist.get_rank()] = z2
            # Get full batch embeddings: (bs x N, hidden)
            z1 = torch.cat(z1_list, 0)
            z2 = torch.cat(z2_list, 0)

        cos_sim = self.sim(z1.unsqueeze(1), z2.unsqueeze(0))
        labels = torch.arange(cos_sim.size(0)).long().to(cos_sim.device)

        loss_fct = CrossEntropyLoss()
        loss = loss_fct(cos_sim, labels)
        correct_num = (torch.argmax(cos_sim, 1) == labels).sum()

        if mlm_outputs_a is not None and mlm_labels_a is not None:
            mlm_labels_a = mlm_labels_a.view(-1, mlm_labels_a.size(-1))
            prediction_scores_a = self.lm_head(mlm_outputs_a.last_hidden_state)
            masked_lm_loss_a = loss_fct(prediction_scores_a.view(-1, self.config.vocab_size), mlm_labels_a.view(-1))
            loss = loss + self.config.mlm_weight * masked_lm_loss_a

        
        if mlm_outputs_b is not None and mlm_labels_b is not None:
            mlm_labels_b = mlm_labels_b.view(-1, mlm_labels_b.size(-1))
            prediction_scores_b = self.lm_head(mlm_outputs_b.last_hidden_state)
            masked_lm_loss_b = loss_fct(prediction_scores_b.view(-1, self.config.vocab_size), mlm_labels_b.view(-1))
            loss = loss + self.config.mlm_weight * masked_lm_loss_b

        return RecformerPretrainingOutput(
            misc={"mlm_loss": masked_lm_loss_a + masked_lm_loss_b},
            loss=loss,
            logits=cos_sim,
            cl_correct_num=correct_num,
            cl_total_num=batch_size,
            hidden_states=outputs_a.hidden_states,
            attentions=outputs_a.attentions,
            global_attentions=outputs_a.global_attentions,
        )


class RecformerForSeqRec(LongformerPreTrainedModel):
    def __init__(self, config: RecformerConfig):
        super().__init__(config)

        self.longformer = RecformerModel(config)
        self.sim = Similarity(config)
        # Initialize weights and apply final processing
        self.post_init()

    def init_item_embedding(self, embeddings: Optional[torch.Tensor] = None):
        self.item_embedding = nn.Embedding(num_embeddings=self.config.item_num, embedding_dim=self.config.hidden_size)
        if embeddings is not None:
            self.item_embedding = nn.Embedding.from_pretrained(embeddings, freeze=True)
            print('Initalize item embeddings from vectors.')

    def similarity_score(self, pooler_output, candidates=None):
        if candidates is None:
            candidate_embeddings = self.item_embedding.weight.unsqueeze(0) # (1, num_items, hidden_size)
        else:
            candidate_embeddings = self.item_embedding(candidates) # (batch_size, candidates, hidden_size)
        pooler_output = pooler_output.unsqueeze(1) # (batch_size, 1, hidden_size)
        return self.sim(pooler_output, candidate_embeddings)

    def forward(self,
                input_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                global_attention_mask: Optional[torch.Tensor] = None,
                head_mask: Optional[torch.Tensor] = None,
                token_type_ids: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.Tensor] = None,
                item_position_ids: Optional[torch.Tensor] = None,
                inputs_embeds: Optional[torch.Tensor] = None,
                output_attentions: Optional[bool] = None,
                output_hidden_states: Optional[bool] = None,
                return_dict: Optional[bool] = None,
                candidates: Optional[torch.Tensor] = None, # candidate item ids
                labels: Optional[torch.Tensor] = None, # target item ids
                ):

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        batch_size = input_ids.size(0)

        outputs = self.longformer(
            input_ids,
            attention_mask=attention_mask,
            global_attention_mask=global_attention_mask,
            head_mask=head_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            item_position_ids=item_position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )

        pooler_output = outputs.pooler_output # (bs, hidden_size)

        if labels is None:
            return self.similarity_score(pooler_output, candidates)

        loss_fct = CrossEntropyLoss()

        if self.config.finetune_negative_sample_size<=0: ## using full softmax
            logits = self.similarity_score(pooler_output)
            loss = loss_fct(logits, labels)

        else:  ## using sampled softmax
            candidates = torch.cat((labels.unsqueeze(-1), torch.randint(0, self.config.item_num, size=(batch_size, self.config.finetune_negative_sample_size)).to(labels.device)), dim=-1)
            logits = self.similarity_score(pooler_output, candidates)
            target = torch.zeros_like(labels, device=labels.device)
            loss = loss_fct(logits, target)

        return loss
