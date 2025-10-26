import torch
from diffusers.models.attention_dispatch import dispatch_attention_fn
from diffusers.models.embeddings import apply_rotary_emb
from diffusers.models.transformers.transformer_cogview4 import Attention

from processor.common import scaled_dot_product_attention


class CogViewWithAttentionWeightsProcessor:
    _attention_backend = None

    def __init__(self, target_length_of_key: int = 512, save_memory: bool = True):
        if not hasattr(torch.nn.functional, "scaled_dot_product_attention"):
            raise ImportError(
                f"{self.__class__.__name__} requires PyTorch 2.0. Please upgrade your pytorch version."
            )
        self.attention_weights = None
        self.target_length_of_key = target_length_of_key  # not used in CogView4
        self.save_memory = save_memory
        self.count = 0  # used to save attention weights only for even steps (due to classifier-free guidance)

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        image_rotary_emb: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        dtype = encoder_hidden_states.dtype

        batch_size, text_seq_length, _ = encoder_hidden_states.shape
        batch_size, image_seq_length, _ = hidden_states.shape
        # hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)
        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

        # 1. QKV projections
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        query = query.unflatten(2, (attn.heads, -1))
        key = key.unflatten(2, (attn.heads, -1))
        value = value.unflatten(2, (attn.heads, -1))

        # 2. QK normalization
        if attn.norm_q is not None:
            query = attn.norm_q(query).to(dtype=dtype)
        if attn.norm_k is not None:
            key = attn.norm_k(key).to(dtype=dtype)

        # 3. Rotational positional embeddings applied to latent stream
        if image_rotary_emb is not None:
            query[:, text_seq_length:, :, :] = apply_rotary_emb(
                query[:, text_seq_length:, :, :],
                image_rotary_emb,
                use_real_unbind_dim=-2,
                sequence_dim=1,
            )
            key[:, text_seq_length:, :, :] = apply_rotary_emb(
                key[:, text_seq_length:, :, :],
                image_rotary_emb,
                use_real_unbind_dim=-2,
                sequence_dim=1,
            )

        # 4. Attention
        if attention_mask is not None:
            text_attn_mask = attention_mask
            assert text_attn_mask.dim() == 2, (
                "the shape of text_attn_mask should be (batch_size, text_seq_length)"
            )
            text_attn_mask = text_attn_mask.float().to(query.device)
            mix_attn_mask = torch.ones((batch_size, text_seq_length + image_seq_length), device=query.device)
            mix_attn_mask[:, :text_seq_length] = text_attn_mask
            mix_attn_mask = mix_attn_mask.unsqueeze(2)
            attn_mask_matrix = mix_attn_mask @ mix_attn_mask.transpose(1, 2)
            attention_mask = (attn_mask_matrix > 0).unsqueeze(1).to(query.dtype)

        if self.count % 2 != 0:
            hidden_states = dispatch_attention_fn(
                query, key, value, attn_mask=attention_mask, backend=self._attention_backend
            )
        else:
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
                eye_matrix_of_key = eye_matrix_of_key[:, :, :, :text_seq_length]
                concat_value = torch.cat([value, eye_matrix_of_key], dim=-1)
                hidden_states = dispatch_attention_fn(
                    query, key, concat_value, attn_mask=attention_mask, backend=self._attention_backend
                )
                hidden_states, attention_weights = hidden_states.split(
                    [value.shape[-1], text_seq_length], dim=-1
                )
                self.attention_weights = attention_weights.permute(0, 2, 1, 3).mean(dim=1)[
                    :, text_seq_length:
                ]
        self.count += 1

        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.type_as(query)

        # 5. Output projection
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        encoder_hidden_states, hidden_states = hidden_states.split(
            [text_seq_length, hidden_states.size(1) - text_seq_length], dim=1
        )
        return hidden_states, encoder_hidden_states


class CogViewWithAttentionWeightsProcessorReverse:
    _attention_backend = None

    def __init__(self, target_length_of_key: int = 512, save_memory: bool = True):
        if not hasattr(torch.nn.functional, "scaled_dot_product_attention"):
            raise ImportError(
                f"{self.__class__.__name__} requires PyTorch 2.0. Please upgrade your pytorch version."
            )
        self.attention_weights = None
        self.target_length_of_key = target_length_of_key  # not used in CogView4
        self.save_memory = save_memory
        self.count = 0  # used to save attention weights only for even steps (due to classifier-free guidance)

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        image_rotary_emb: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        dtype = encoder_hidden_states.dtype

        batch_size, text_seq_length, _ = encoder_hidden_states.shape
        batch_size, image_seq_length, _ = hidden_states.shape
        # hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)
        hidden_states = torch.cat([hidden_states, encoder_hidden_states], dim=1)

        # 1. QKV projections
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        query = query.unflatten(2, (attn.heads, -1))
        key = key.unflatten(2, (attn.heads, -1))
        value = value.unflatten(2, (attn.heads, -1))

        # 2. QK normalization
        if attn.norm_q is not None:
            query = attn.norm_q(query).to(dtype=dtype)
        if attn.norm_k is not None:
            key = attn.norm_k(key).to(dtype=dtype)

        # 3. Rotational positional embeddings applied to latent stream
        if image_rotary_emb is not None:
            query[:, :image_seq_length, :, :] = apply_rotary_emb(
                query[:, :image_seq_length, :, :],
                image_rotary_emb,
                use_real_unbind_dim=-2,
                sequence_dim=1,
            )
            key[:, :image_seq_length, :, :] = apply_rotary_emb(
                key[:, :image_seq_length, :, :],
                image_rotary_emb,
                use_real_unbind_dim=-2,
                sequence_dim=1,
            )

        # 4. Attention
        if attention_mask is not None:
            text_attn_mask = attention_mask
            assert text_attn_mask.dim() == 2, (
                "the shape of text_attn_mask should be (batch_size, text_seq_length)"
            )
            text_attn_mask = text_attn_mask.float().to(query.device)
            mix_attn_mask = torch.ones((batch_size, text_seq_length + image_seq_length), device=query.device)
            mix_attn_mask[:, image_seq_length:] = text_attn_mask
            mix_attn_mask = mix_attn_mask.unsqueeze(2)
            attn_mask_matrix = mix_attn_mask @ mix_attn_mask.transpose(1, 2)
            attention_mask = (attn_mask_matrix > 0).unsqueeze(1).to(query.dtype)

        if self.count % 2 != 0:
            hidden_states = dispatch_attention_fn(
                query, key, value, attn_mask=attention_mask, backend=self._attention_backend
            )
        else:
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
                    :, image_seq_length:
                ]
        self.count += 1

        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.type_as(query)

        # 5. Output projection
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        hidden_states, encoder_hidden_states = hidden_states.split([image_seq_length, text_seq_length], dim=1)
        return hidden_states, encoder_hidden_states
