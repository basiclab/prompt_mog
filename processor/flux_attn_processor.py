import torch
from diffusers.models.attention_dispatch import dispatch_attention_fn
from diffusers.models.embeddings import apply_rotary_emb
from diffusers.models.transformers.transformer_flux import (
    FluxAttention,
    _get_qkv_projections,
)

from processor.common import scaled_dot_product_attention


class FluxAttnWithAttentionWeightsProcessor:
    _attention_backend = None

    def __init__(self, target_length_of_key: int = 512, save_memory: bool = True):
        if not hasattr(torch.nn.functional, "scaled_dot_product_attention"):
            raise ImportError(
                f"{self.__class__.__name__} requires PyTorch 2.0. Please upgrade your pytorch version."
            )
        self.attention_weights = None
        self.target_length_of_key = target_length_of_key
        self.save_memory = save_memory

    def __call__(
        self,
        attn: FluxAttention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        image_rotary_emb: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        query, key, value, encoder_query, encoder_key, encoder_value = _get_qkv_projections(
            attn, hidden_states, encoder_hidden_states
        )

        query = query.unflatten(-1, (attn.heads, -1))
        key = key.unflatten(-1, (attn.heads, -1))
        value = value.unflatten(-1, (attn.heads, -1))

        query = attn.norm_q(query)
        key = attn.norm_k(key)

        if attn.added_kv_proj_dim is not None:
            encoder_query = encoder_query.unflatten(-1, (attn.heads, -1))
            encoder_key = encoder_key.unflatten(-1, (attn.heads, -1))
            encoder_value = encoder_value.unflatten(-1, (attn.heads, -1))

            encoder_query = attn.norm_added_q(encoder_query)
            encoder_key = attn.norm_added_k(encoder_key)

            query = torch.cat([encoder_query, query], dim=1)
            key = torch.cat([encoder_key, key], dim=1)
            value = torch.cat([encoder_value, value], dim=1)

        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb, sequence_dim=1)
            key = apply_rotary_emb(key, image_rotary_emb, sequence_dim=1)

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
            eye_matrix_of_key = eye_matrix_of_key[:, :, :, : self.target_length_of_key]
            concat_value = torch.cat([value, eye_matrix_of_key], dim=-1)
            hidden_states = dispatch_attention_fn(
                query, key, concat_value, attn_mask=attention_mask, backend=self._attention_backend
            )
            hidden_states, attention_weights = hidden_states.split(
                [value.shape[-1], self.target_length_of_key], dim=-1
            )
            self.attention_weights = attention_weights.permute(0, 2, 1, 3).mean(dim=1)[
                :, self.target_length_of_key :
            ]

        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.to(query.dtype)

        if encoder_hidden_states is not None:
            encoder_hidden_states, hidden_states = hidden_states.split_with_sizes(
                [encoder_hidden_states.shape[1], hidden_states.shape[1] - encoder_hidden_states.shape[1]],
                dim=1,
            )
            hidden_states = attn.to_out[0](hidden_states)
            hidden_states = attn.to_out[1](hidden_states)
            encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

            return hidden_states, encoder_hidden_states
        else:
            return hidden_states


class FluxAttnWithAttentionWeightsProcessorReverse:
    _attention_backend = None

    def __init__(self, target_length_of_key: int = 512, save_memory: bool = True):
        if not hasattr(torch.nn.functional, "scaled_dot_product_attention"):
            raise ImportError(
                f"{self.__class__.__name__} requires PyTorch 2.0. Please upgrade your pytorch version."
            )
        self.attention_weights = None
        self.target_length_of_key = target_length_of_key
        self.save_memory = save_memory

    def __call__(
        self,
        attn: FluxAttention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        image_rotary_emb: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        query, key, value, encoder_query, encoder_key, encoder_value = _get_qkv_projections(
            attn, hidden_states, encoder_hidden_states
        )

        query = query.unflatten(-1, (attn.heads, -1))
        key = key.unflatten(-1, (attn.heads, -1))
        value = value.unflatten(-1, (attn.heads, -1))

        query = attn.norm_q(query)
        key = attn.norm_k(key)

        if attn.added_kv_proj_dim is not None:
            encoder_query = encoder_query.unflatten(-1, (attn.heads, -1))
            encoder_key = encoder_key.unflatten(-1, (attn.heads, -1))
            encoder_value = encoder_value.unflatten(-1, (attn.heads, -1))

            encoder_query = attn.norm_added_q(encoder_query)
            encoder_key = attn.norm_added_k(encoder_key)

            query = torch.cat([encoder_query, query], dim=1)
            key = torch.cat([encoder_key, key], dim=1)
            value = torch.cat([encoder_value, value], dim=1)

        img_seq_length = query.shape[1] - self.target_length_of_key

        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb, sequence_dim=1)
            key = apply_rotary_emb(key, image_rotary_emb, sequence_dim=1)

        # to restore the correct order due to `image_rotary_emb`
        encoder_query, query = query.split([self.target_length_of_key, img_seq_length], dim=1)
        encoder_key, key = key.split([self.target_length_of_key, img_seq_length], dim=1)
        encoder_value, value = value.split([self.target_length_of_key, img_seq_length], dim=1)
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
            eye_matrix_of_key = eye_matrix_of_key[:, :, :, :img_seq_length]
            concat_value = torch.cat([value, eye_matrix_of_key], dim=-1)
            hidden_states = dispatch_attention_fn(
                query, key, concat_value, attn_mask=attention_mask, backend=self._attention_backend
            )
            hidden_states, attention_weights = hidden_states.split([value.shape[-1], img_seq_length], dim=-1)
            self.attention_weights = attention_weights.permute(0, 2, 1, 3).mean(dim=1)[:, img_seq_length:]

        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.to(query.dtype)

        if encoder_hidden_states is not None:
            hidden_states, encoder_hidden_states = hidden_states.split(
                [img_seq_length, encoder_hidden_states.shape[1]], dim=1
            )
            hidden_states = attn.to_out[0](hidden_states)
            hidden_states = attn.to_out[1](hidden_states)
            encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

            return hidden_states, encoder_hidden_states
        else:
            return hidden_states
