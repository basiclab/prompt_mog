import torch
from diffusers.models.attention_dispatch import dispatch_attention_fn
from diffusers.models.transformers.transformer_flux import _get_qkv_projections
from diffusers.models.transformers.transformer_qwenimage import Attention, apply_rotary_emb_qwen

from processor.common import scaled_dot_product_attention


class QwenAttnWithAttentionWeightsProcessor:
    _attention_backend = None

    def __init__(self, target_length_of_key: int = 512, save_memory: bool = True):
        if not hasattr(torch.nn.functional, "scaled_dot_product_attention"):
            raise ImportError(
                f"{self.__class__.__name__} requires PyTorch 2.0. Please upgrade your pytorch version."
            )
        self.attention_weights = None
        self.target_length_of_key = target_length_of_key  # not used in QwenImage
        self.save_memory = save_memory
        self.count = 0  # used to save attention weights only for even steps (due to classifier-free guidance)

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,  # Image stream
        encoder_hidden_states: torch.FloatTensor = None,  # Text stream
        encoder_hidden_states_mask: torch.FloatTensor = None,
        attention_mask: torch.FloatTensor | None = None,
        image_rotary_emb: torch.Tensor | None = None,
    ) -> torch.FloatTensor:
        seq_txt = encoder_hidden_states.shape[1]

        query, key, value, encoder_query, encoder_key, encoder_value = _get_qkv_projections(
            attn, hidden_states, encoder_hidden_states
        )
        query = query.unflatten(-1, (attn.heads, -1))
        key = key.unflatten(-1, (attn.heads, -1))
        value = value.unflatten(-1, (attn.heads, -1))
        encoder_query = encoder_query.unflatten(-1, (attn.heads, -1))
        encoder_key = encoder_key.unflatten(-1, (attn.heads, -1))
        encoder_value = encoder_value.unflatten(-1, (attn.heads, -1))

        # Apply QK normalization
        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)
        if attn.norm_added_q is not None:
            encoder_query = attn.norm_added_q(encoder_query)
        if attn.norm_added_k is not None:
            encoder_key = attn.norm_added_k(encoder_key)

        # Apply RoPE
        if image_rotary_emb is not None:
            img_freqs, txt_freqs = image_rotary_emb
            query = apply_rotary_emb_qwen(query, img_freqs, use_real=False)
            key = apply_rotary_emb_qwen(key, img_freqs, use_real=False)
            encoder_query = apply_rotary_emb_qwen(encoder_query, txt_freqs, use_real=False)
            encoder_key = apply_rotary_emb_qwen(encoder_key, txt_freqs, use_real=False)

        # Concatenate for joint attention
        # Order: [text, image]
        query = torch.cat([encoder_query, query], dim=1)
        key = torch.cat([encoder_key, key], dim=1)
        value = torch.cat([encoder_value, value], dim=1)

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
                eye_matrix_of_key = eye_matrix_of_key[:, :, :, :seq_txt]
                concat_value = torch.cat([value, eye_matrix_of_key], dim=-1)
                hidden_states = dispatch_attention_fn(
                    query, key, concat_value, attn_mask=attention_mask, backend=self._attention_backend
                )
                hidden_states, attention_weights = hidden_states.split([value.shape[-1], seq_txt], dim=-1)
                self.attention_weights = attention_weights.permute(0, 2, 1, 3).mean(dim=1)[:, seq_txt:]
        self.count += 1

        # Reshape back
        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.to(query.dtype)

        # Split attention outputs back
        txt_attn_output = hidden_states[:, :seq_txt, :]  # Text part
        img_attn_output = hidden_states[:, seq_txt:, :]  # Image part

        # Apply output projections
        img_attn_output = attn.to_out[0](img_attn_output)
        if len(attn.to_out) > 1:
            img_attn_output = attn.to_out[1](img_attn_output)  # dropout

        txt_attn_output = attn.to_add_out(txt_attn_output)

        return img_attn_output, txt_attn_output


class QwenAttnWithAttentionWeightsProcessorReverse:
    _attention_backend = None

    def __init__(self, target_length_of_key: int = 512, save_memory: bool = True):
        if not hasattr(torch.nn.functional, "scaled_dot_product_attention"):
            raise ImportError(
                f"{self.__class__.__name__} requires PyTorch 2.0. Please upgrade your pytorch version."
            )
        self.attention_weights = None
        self.target_length_of_key = target_length_of_key  # not used in QwenImage
        self.save_memory = save_memory
        self.count = 0  # used to save attention weights only for even steps (due to classifier-free guidance)

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,  # Image stream
        encoder_hidden_states: torch.FloatTensor = None,  # Text stream
        encoder_hidden_states_mask: torch.FloatTensor = None,
        attention_mask: torch.FloatTensor | None = None,
        image_rotary_emb: torch.Tensor | None = None,
    ) -> torch.FloatTensor:
        seq_txt = encoder_hidden_states.shape[1]
        seq_img = hidden_states.shape[1]

        query, key, value, encoder_query, encoder_key, encoder_value = _get_qkv_projections(
            attn, hidden_states, encoder_hidden_states
        )
        query = query.unflatten(-1, (attn.heads, -1))
        key = key.unflatten(-1, (attn.heads, -1))
        value = value.unflatten(-1, (attn.heads, -1))
        encoder_query = encoder_query.unflatten(-1, (attn.heads, -1))
        encoder_key = encoder_key.unflatten(-1, (attn.heads, -1))
        encoder_value = encoder_value.unflatten(-1, (attn.heads, -1))

        # Apply QK normalization
        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)
        if attn.norm_added_q is not None:
            encoder_query = attn.norm_added_q(encoder_query)
        if attn.norm_added_k is not None:
            encoder_key = attn.norm_added_k(encoder_key)

        # Apply RoPE
        if image_rotary_emb is not None:
            img_freqs, txt_freqs = image_rotary_emb
            query = apply_rotary_emb_qwen(query, img_freqs, use_real=False)
            key = apply_rotary_emb_qwen(key, img_freqs, use_real=False)
            encoder_query = apply_rotary_emb_qwen(encoder_query, txt_freqs, use_real=False)
            encoder_key = apply_rotary_emb_qwen(encoder_key, txt_freqs, use_real=False)

        # Concatenate for joint attention
        # Order: [image, text]
        query = torch.cat([query, encoder_query], dim=1)
        key = torch.cat([key, encoder_key], dim=1)
        value = torch.cat([value, encoder_value], dim=1)

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
                eye_matrix_of_key = eye_matrix_of_key[:, :, :, :seq_img]
                concat_value = torch.cat([value, eye_matrix_of_key], dim=-1)
                hidden_states = dispatch_attention_fn(
                    query, key, concat_value, attn_mask=attention_mask, backend=self._attention_backend
                )
                hidden_states, attention_weights = hidden_states.split([value.shape[-1], seq_img], dim=-1)
                self.attention_weights = attention_weights.permute(0, 2, 1, 3).mean(dim=1)[:, seq_img:]
        self.count += 1

        # Reshape back
        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.to(query.dtype)

        # Split attention outputs back
        txt_attn_output = hidden_states[:, seq_img:, :]  # Text part
        img_attn_output = hidden_states[:, :seq_img, :]  # Image part

        # Apply output projections
        img_attn_output = attn.to_out[0](img_attn_output)
        if len(attn.to_out) > 1:
            img_attn_output = attn.to_out[1](img_attn_output)  # dropout

        txt_attn_output = attn.to_add_out(txt_attn_output)

        return img_attn_output, txt_attn_output
