import torch
from diffusers.models.attention_dispatch import dispatch_attention_fn
from diffusers.models.transformers.transformer_flux import _get_qkv_projections
from diffusers.models.transformers.transformer_sd3 import Attention

from processor.common import scaled_dot_product_attention


class SD3AttnWithAttentionWeightsProcessor:
    _attention_backend = None

    def __init__(
        self, start_target_length: int = 77, target_length_of_key: int = 512, save_memory: bool = True
    ):
        if not hasattr(torch.nn.functional, "scaled_dot_product_attention"):
            raise ImportError(
                f"{self.__class__.__name__} requires PyTorch 2.0. Please upgrade your pytorch version."
            )
        self.attention_weights = None
        self.start_target_length = start_target_length
        self.target_length_of_key = target_length_of_key
        self.save_memory = save_memory

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        attention_mask: torch.FloatTensor | None = None,
        *args,
        **kwargs,
    ) -> torch.FloatTensor:
        query, key, value, encoder_query, encoder_key, encoder_value = _get_qkv_projections(
            attn, hidden_states, encoder_hidden_states
        )

        query = query.unflatten(-1, (attn.heads, -1))
        key = key.unflatten(-1, (attn.heads, -1))
        value = value.unflatten(-1, (attn.heads, -1))

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # `context` projections.
        if attn.added_kv_proj_dim is not None:
            encoder_query = encoder_query.unflatten(-1, (attn.heads, -1))
            encoder_key = encoder_key.unflatten(-1, (attn.heads, -1))
            encoder_value = encoder_value.unflatten(-1, (attn.heads, -1))

            if attn.norm_added_q is not None:
                encoder_query = attn.norm_added_q(encoder_query)
            if attn.norm_added_k is not None:
                encoder_key = attn.norm_added_k(encoder_key)

            # changing the order here
            query = torch.cat([encoder_query, query], dim=1)
            key = torch.cat([encoder_key, key], dim=1)
            value = torch.cat([encoder_value, value], dim=1)

        if not self.save_memory:
            hidden_states, self.attention_weights = scaled_dot_product_attention(
                query, key, value, attn_mask=attention_mask, is_causal=False
            )
        else:
            length_of_key = key.shape[1]
            eye_matrix_of_key = (
                torch.eye(length_of_key, device=value.device, dtype=value.dtype)
                .unsqueeze(1)
                .unsqueeze(0)
                .repeat(value.shape[0], 1, value.shape[2], 1)
            )
            eye_matrix_of_key = eye_matrix_of_key[
                :, :, :, : self.start_target_length + self.target_length_of_key
            ]
            concat_value = torch.cat([value, eye_matrix_of_key], dim=-1)
            hidden_states = dispatch_attention_fn(
                query, key, concat_value, attn_mask=attention_mask, backend=self._attention_backend
            )
            hidden_states, attention_weights = hidden_states.split(
                [value.shape[-1], self.start_target_length + self.target_length_of_key], dim=-1
            )
            self.attention_weights = attention_weights.permute(0, 2, 1, 3).mean(dim=1)[
                :,
                self.start_target_length + self.target_length_of_key :,
                self.start_target_length : self.target_length_of_key + self.start_target_length,
            ]

        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.to(query.dtype)

        if encoder_hidden_states is not None:
            # Split the attention outputs.
            encoder_hidden_states, hidden_states = hidden_states.split_with_sizes(
                [encoder_hidden_states.shape[1], hidden_states.shape[1] - encoder_hidden_states.shape[1]],
                dim=1,
            )
            if not attn.context_pre_only:
                encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if encoder_hidden_states is not None:
            return hidden_states, encoder_hidden_states
        else:
            return hidden_states


class SD3AttnWithAttentionWeightsProcessorReverse:
    _attention_backend = None

    def __init__(
        self, start_target_length: int = 77, target_length_of_key: int = 512, save_memory: bool = True
    ):
        if not hasattr(torch.nn.functional, "scaled_dot_product_attention"):
            raise ImportError(
                f"{self.__class__.__name__} requires PyTorch 2.0. Please upgrade your pytorch version."
            )
        self.attention_weights = None
        self.start_target_length = start_target_length
        self.target_length_of_key = target_length_of_key
        self.save_memory = save_memory

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        attention_mask: torch.FloatTensor | None = None,
        *args,
        **kwargs,
    ) -> torch.FloatTensor:
        image_seq_length = hidden_states.shape[1]
        query, key, value, encoder_query, encoder_key, encoder_value = _get_qkv_projections(
            attn, hidden_states, encoder_hidden_states
        )

        query = query.unflatten(-1, (attn.heads, -1))
        key = key.unflatten(-1, (attn.heads, -1))
        value = value.unflatten(-1, (attn.heads, -1))

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # `context` projections.
        if attn.added_kv_proj_dim is not None:
            encoder_query = encoder_query.unflatten(-1, (attn.heads, -1))
            encoder_key = encoder_key.unflatten(-1, (attn.heads, -1))
            encoder_value = encoder_value.unflatten(-1, (attn.heads, -1))

            if attn.norm_added_q is not None:
                encoder_query = attn.norm_added_q(encoder_query)
            if attn.norm_added_k is not None:
                encoder_key = attn.norm_added_k(encoder_key)

            # changing the order here
            query = torch.cat([query, encoder_query], dim=1)
            key = torch.cat([key, encoder_key], dim=1)
            value = torch.cat([value, encoder_value], dim=1)

        if not self.save_memory:
            hidden_states, self.attention_weights = scaled_dot_product_attention(
                query, key, value, attn_mask=attention_mask, is_causal=False
            )
        else:
            length_of_key = key.shape[1]
            eye_matrix_of_key = (
                torch.eye(length_of_key, device=value.device, dtype=value.dtype)
                .unsqueeze(1)
                .unsqueeze(0)
                .repeat(value.shape[0], 1, value.shape[2], 1)
            )
            eye_matrix_of_key = eye_matrix_of_key[:, :, :, :image_seq_length]
            concat_value = torch.cat([value, eye_matrix_of_key], dim=-1)
            hidden_states = dispatch_attention_fn(
                query, key, concat_value, attn_mask=attention_mask, backend=self._attention_backend
            )
            hidden_states, attention_weights = hidden_states.split(
                [value.shape[-1], image_seq_length], dim=-1
            )
            self.attention_weights = attention_weights.permute(0, 2, 1, 3).mean(dim=1)[
                :,
                image_seq_length + self.start_target_length :,
            ]

        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.to(query.dtype)

        if encoder_hidden_states is not None:
            # Split the attention outputs.
            hidden_states, encoder_hidden_states = hidden_states.split_with_sizes(
                [hidden_states.shape[1] - encoder_hidden_states.shape[1], encoder_hidden_states.shape[1]],
                dim=1,
            )
            if not attn.context_pre_only:
                encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if encoder_hidden_states is not None:
            return hidden_states, encoder_hidden_states
        else:
            return hidden_states
