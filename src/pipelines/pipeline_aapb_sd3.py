# Copyright 2025 Stability AI, The HuggingFace Team and The InstantX Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Adaptive Auxiliary Prompt Blending (AAPB) pipeline for Stable Diffusion 3.

Implements the closed-form adaptive coefficient gamma*_t (Eq. 13) that
dynamically balances target and anchor prompt contributions at each
denoising step via Tweedie-based score alignment.
"""

import random
import numpy as np

import inspect
from typing import Any, Callable, Dict, List, Optional, Union

import torch
from transformers import (
    CLIPTextModelWithProjection,
    CLIPTokenizer,
    T5EncoderModel,
    T5TokenizerFast,
)

from diffusers.image_processor import VaeImageProcessor
from diffusers.loaders import FromSingleFileMixin, SD3LoraLoaderMixin
from diffusers.models.autoencoders import AutoencoderKL
from diffusers.models.transformers import SD3Transformer2DModel
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.utils import is_torch_xla_available, logging, replace_example_docstring
from diffusers.utils.torch_utils import randn_tensor
from diffusers import DiffusionPipeline
from diffusers.pipelines.stable_diffusion_3 import StableDiffusion3PipelineOutput


if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False


logger = logging.get_logger(__name__)

EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> pipe = AAPBDiffusion3Pipeline.from_pretrained(
        ...     "stabilityai/stable-diffusion-3-medium", revision="refs/pr/26"
        ... ).to("cuda")
        >>> image = pipe(
        ...     r2f_prompts={"r2f_prompt": [["A hairy animal", "A hairy frog"]]},
        ...     seed=42,
        ... ).images[0]
        >>> image.save("aapb_output.png")
        ```
"""


def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed.")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(
            inspect.signature(scheduler.set_timesteps).parameters.keys()
        )
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(
            inspect.signature(scheduler.set_timesteps).parameters.keys()
        )
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


class AAPBDiffusion3Pipeline(
    DiffusionPipeline, SD3LoraLoaderMixin, FromSingleFileMixin
):
    r"""
    AAPB (Adaptive Auxiliary Prompt Blending) pipeline for SD3.

    At each denoising step, computes three score function evaluations:
      - s_θ(x_t)           : unconditional
      - s_θ(x_t, c̃_T)     : target (rare) conditioned
      - s_θ(x_t, c̃_A)     : anchor (frequent) conditioned

    Then applies Eq. 13 to compute γ*_t and Eq. 8 for the blended guidance:
      s̃(x_t; w, γ_t) = U + w * ((1-γ_t)*T + γ_t*F - U)
    """

    model_cpu_offload_seq = (
        "text_encoder->text_encoder_2->text_encoder_3->transformer->vae"
    )
    _optional_components = []
    _callback_tensor_inputs = [
        "latents",
        "prompt_embeds",
        "negative_prompt_embeds",
        "negative_pooled_prompt_embeds",
    ]

    def __init__(
        self,
        transformer: SD3Transformer2DModel,
        scheduler: FlowMatchEulerDiscreteScheduler,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModelWithProjection,
        tokenizer: CLIPTokenizer,
        text_encoder_2: CLIPTextModelWithProjection,
        tokenizer_2: CLIPTokenizer,
        text_encoder_3: T5EncoderModel,
        tokenizer_3: T5TokenizerFast,
    ):
        super().__init__()

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            text_encoder_2=text_encoder_2,
            text_encoder_3=text_encoder_3,
            tokenizer=tokenizer,
            tokenizer_2=tokenizer_2,
            tokenizer_3=tokenizer_3,
            transformer=transformer,
            scheduler=scheduler,
        )
        self.vae_scale_factor = (
            2 ** (len(self.vae.config.block_out_channels) - 1)
            if hasattr(self, "vae") and self.vae is not None
            else 8
        )
        self.image_processor = VaeImageProcessor(
            vae_scale_factor=self.vae_scale_factor
        )
        self.tokenizer_max_length = (
            self.tokenizer.model_max_length
            if hasattr(self, "tokenizer") and self.tokenizer is not None
            else 77
        )
        self.default_sample_size = (
            self.transformer.config.sample_size
            if hasattr(self, "transformer") and self.transformer is not None
            else 128
        )

    def _get_t5_prompt_embeds(
        self,
        prompt: Union[str, List[str]] = None,
        num_images_per_prompt: int = 1,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        device = device or self._execution_device
        dtype = dtype or self.text_encoder.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        if self.text_encoder_3 is None:
            return torch.zeros(
                (
                    batch_size,
                    self.tokenizer_max_length,
                    self.transformer.config.joint_attention_dim,
                ),
                device=device,
                dtype=dtype,
            )

        text_inputs = self.tokenizer_3(
            prompt,
            padding="max_length",
            max_length=self.tokenizer_max_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        prompt_embeds = self.text_encoder_3(text_input_ids.to(device))[0]

        dtype = self.text_encoder_3.dtype
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(
            batch_size * num_images_per_prompt, seq_len, -1
        )

        return prompt_embeds

    def _get_clip_prompt_embeds(
        self,
        prompt: Union[str, List[str]],
        num_images_per_prompt: int = 1,
        device: Optional[torch.device] = None,
        clip_skip: Optional[int] = None,
        clip_model_index: int = 0,
    ):
        device = device or self._execution_device

        clip_tokenizers = [self.tokenizer, self.tokenizer_2]
        clip_text_encoders = [self.text_encoder, self.text_encoder_2]

        tokenizer = clip_tokenizers[clip_model_index]
        text_encoder = clip_text_encoders[clip_model_index]

        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer_max_length,
            truncation=True,
            return_tensors="pt",
        )

        text_input_ids = text_inputs.input_ids
        prompt_embeds = text_encoder(
            text_input_ids.to(device), output_hidden_states=True
        )
        pooled_prompt_embeds = prompt_embeds[0]

        if clip_skip is None:
            prompt_embeds = prompt_embeds.hidden_states[-2]
        else:
            prompt_embeds = prompt_embeds.hidden_states[-(clip_skip + 2)]

        prompt_embeds = prompt_embeds.to(dtype=self.text_encoder.dtype, device=device)

        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(
            batch_size * num_images_per_prompt, seq_len, -1
        )

        pooled_prompt_embeds = pooled_prompt_embeds.repeat(
            1, num_images_per_prompt, 1
        )
        pooled_prompt_embeds = pooled_prompt_embeds.view(
            batch_size * num_images_per_prompt, -1
        )

        return prompt_embeds, pooled_prompt_embeds

    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        prompt_2: Union[str, List[str]],
        prompt_3: Union[str, List[str]],
        device: Optional[torch.device] = None,
        num_images_per_prompt: int = 1,
        do_classifier_free_guidance: bool = True,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt_2: Optional[Union[str, List[str]]] = None,
        negative_prompt_3: Optional[Union[str, List[str]]] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        clip_skip: Optional[int] = None,
    ):
        device = device or self._execution_device

        prompt = [prompt] if isinstance(prompt, str) else prompt
        if prompt is not None:
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            prompt_2 = prompt_2 or prompt
            prompt_2 = [prompt_2] if isinstance(prompt_2, str) else prompt_2

            prompt_3 = prompt_3 or prompt
            prompt_3 = [prompt_3] if isinstance(prompt_3, str) else prompt_3

            prompt_embed, pooled_prompt_embed = self._get_clip_prompt_embeds(
                prompt=prompt,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
                clip_skip=clip_skip,
                clip_model_index=0,
            )
            prompt_2_embed, pooled_prompt_2_embed = self._get_clip_prompt_embeds(
                prompt=prompt_2,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
                clip_skip=clip_skip,
                clip_model_index=1,
            )
            clip_prompt_embeds = torch.cat([prompt_embed, prompt_2_embed], dim=-1)

            t5_prompt_embed = self._get_t5_prompt_embeds(
                prompt=prompt_3,
                num_images_per_prompt=num_images_per_prompt,
                device=device,
            )

            clip_prompt_embeds = torch.nn.functional.pad(
                clip_prompt_embeds,
                (0, t5_prompt_embed.shape[-1] - clip_prompt_embeds.shape[-1]),
            )

            prompt_embeds = torch.cat(
                [clip_prompt_embeds, t5_prompt_embed], dim=-2
            )
            pooled_prompt_embeds = torch.cat(
                [pooled_prompt_embed, pooled_prompt_2_embed], dim=-1
            )

        if do_classifier_free_guidance and negative_prompt_embeds is None:
            negative_prompt = negative_prompt or ""
            negative_prompt_2 = negative_prompt_2 or negative_prompt
            negative_prompt_3 = negative_prompt_3 or negative_prompt

            negative_prompt = (
                batch_size * [negative_prompt]
                if isinstance(negative_prompt, str)
                else negative_prompt
            )
            negative_prompt_2 = (
                batch_size * [negative_prompt_2]
                if isinstance(negative_prompt_2, str)
                else negative_prompt_2
            )
            negative_prompt_3 = (
                batch_size * [negative_prompt_3]
                if isinstance(negative_prompt_3, str)
                else negative_prompt_3
            )

            negative_prompt_embed, negative_pooled_prompt_embed = (
                self._get_clip_prompt_embeds(
                    negative_prompt,
                    device=device,
                    num_images_per_prompt=num_images_per_prompt,
                    clip_skip=None,
                    clip_model_index=0,
                )
            )
            negative_prompt_2_embed, negative_pooled_prompt_2_embed = (
                self._get_clip_prompt_embeds(
                    negative_prompt_2,
                    device=device,
                    num_images_per_prompt=num_images_per_prompt,
                    clip_skip=None,
                    clip_model_index=1,
                )
            )
            negative_clip_prompt_embeds = torch.cat(
                [negative_prompt_embed, negative_prompt_2_embed], dim=-1
            )

            t5_negative_prompt_embed = self._get_t5_prompt_embeds(
                prompt=negative_prompt_3,
                num_images_per_prompt=num_images_per_prompt,
                device=device,
            )

            negative_clip_prompt_embeds = torch.nn.functional.pad(
                negative_clip_prompt_embeds,
                (
                    0,
                    t5_negative_prompt_embed.shape[-1]
                    - negative_clip_prompt_embeds.shape[-1],
                ),
            )

            negative_prompt_embeds = torch.cat(
                [negative_clip_prompt_embeds, t5_negative_prompt_embed], dim=-2
            )
            negative_pooled_prompt_embeds = torch.cat(
                [negative_pooled_prompt_embed, negative_pooled_prompt_2_embed],
                dim=-1,
            )

        return (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        )

    def prepare_latents(
        self,
        batch_size,
        num_channels_latents,
        height,
        width,
        dtype,
        device,
        generator,
        latents=None,
    ):
        if latents is not None:
            return latents.to(device=device, dtype=dtype)

        shape = (
            batch_size,
            num_channels_latents,
            int(height) // self.vae_scale_factor,
            int(width) // self.vae_scale_factor,
        )

        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        latents = randn_tensor(
            shape, generator=generator, device=device, dtype=dtype
        )

        return latents

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def clip_skip(self):
        return self._clip_skip

    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale > 1

    @property
    def joint_attention_kwargs(self):
        return self._joint_attention_kwargs

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def interrupt(self):
        return self._interrupt

    @staticmethod
    def _fix_seed(seed=42):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms = True

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        r2f_prompts: Dict[str, Any] = None,
        batch_size: Optional[int] = 1,
        seed: Optional[int] = None,
        prompt_2: Optional[Union[str, List[str]]] = None,
        prompt_3: Optional[Union[str, List[str]]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        timesteps: List[int] = None,
        guidance_scale: float = 7.0,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt_2: Optional[Union[str, List[str]]] = None,
        negative_prompt_3: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        clip_skip: Optional[int] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        gamma_t: Optional[float] = None,
    ):
        r"""
        AAPB generation with adaptive auxiliary prompt blending.

        Args:
            r2f_prompts: Dict with key "r2f_prompt" containing a list of
                [anchor_prompt, target_prompt] pairs. The anchor (index 0)
                is the frequent/semantically related prompt; the target
                (index -1) is the rare concept prompt.
            gamma_t: If provided, uses a fixed blending coefficient instead
                of the adaptive closed-form solution (Eq. 13). Range [0, 1].
                None (default) uses the adaptive method.
            guidance_scale: CFG scale w. Default 7.0 for rare concept generation.
            seed: Random seed for reproducibility.
            num_inference_steps: Number of denoising steps T. Default 50.

        Examples:

        Returns:
            StableDiffusion3PipelineOutput with generated images.
        """

        # 1. Setup
        self._guidance_scale = guidance_scale
        self._clip_skip = clip_skip
        self._joint_attention_kwargs = joint_attention_kwargs
        self._interrupt = False

        # Extract prompt list: [anchor_prompt, target_prompt]
        prompt_list = r2f_prompts["r2f_prompt"][0]

        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor

        if seed is not None and seed > 0:
            self._fix_seed(seed=seed)

        # 2. Define call parameters
        device = self._execution_device

        # 3. Encode all prompts (anchor at index 0, target at index -1)
        prompt_embeds_list = []
        pooled_prompt_embeds_list = []
        for i, prompt in enumerate(prompt_list):
            (
                prompt_embeds,
                negative_prompt_embeds,
                pooled_prompt_embeds,
                negative_pooled_prompt_embeds,
            ) = self.encode_prompt(
                prompt=prompt,
                prompt_2=prompt_2,
                prompt_3=prompt_3,
                negative_prompt=negative_prompt,
                negative_prompt_2=negative_prompt_2,
                negative_prompt_3=negative_prompt_3,
                do_classifier_free_guidance=self.do_classifier_free_guidance,
                prompt_embeds=None,
                negative_prompt_embeds=None,
                pooled_prompt_embeds=None,
                negative_pooled_prompt_embeds=None,
                device=device,
                clip_skip=self.clip_skip,
                num_images_per_prompt=num_images_per_prompt,
            )

            if self.do_classifier_free_guidance:
                prompt_embeds = torch.cat(
                    [negative_prompt_embeds, prompt_embeds], dim=0
                )
                pooled_prompt_embeds = torch.cat(
                    [negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0
                )

            if i == 0:
                # 4. Prepare timesteps
                timesteps, num_inference_steps = retrieve_timesteps(
                    self.scheduler, num_inference_steps, device, timesteps
                )
                num_warmup_steps = max(
                    len(timesteps) - num_inference_steps * self.scheduler.order, 0
                )
                self._num_timesteps = len(timesteps)

                # 5. Prepare latent variables
                num_channels_latents = self.transformer.config.in_channels
                latents = self.prepare_latents(
                    batch_size * num_images_per_prompt,
                    num_channels_latents,
                    height,
                    width,
                    prompt_embeds.dtype,
                    device,
                    generator,
                    latents,
                )

            prompt_embeds_list.append(prompt_embeds)
            pooled_prompt_embeds_list.append(pooled_prompt_embeds)

        # 6. Denoising loop with AAPB
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                if self.do_classifier_free_guidance:
                    # Extract conditional/unconditional parts
                    # target (rare) is at index -1, anchor (frequent) at index 0
                    rare_neg_embeds, rare_pos_embeds = prompt_embeds_list[
                        -1
                    ].chunk(2)
                    freq_neg_embeds, freq_pos_embeds = prompt_embeds_list[
                        0
                    ].chunk(2)

                    rare_neg_pooled, rare_pos_pooled = pooled_prompt_embeds_list[
                        -1
                    ].chunk(2)
                    freq_neg_pooled, freq_pos_pooled = pooled_prompt_embeds_list[
                        0
                    ].chunk(2)

                    # Single batched transformer call: [uncond, target_cond, anchor_cond]
                    batch_latent_input = torch.cat(
                        [latents, latents, latents], dim=0
                    )
                    batch_timestep = t.expand(batch_latent_input.shape[0])

                    batch_prompt_embeds = torch.cat(
                        [rare_neg_embeds, rare_pos_embeds, freq_pos_embeds],
                        dim=0,
                    )
                    batch_pooled_embeds = torch.cat(
                        [rare_neg_pooled, rare_pos_pooled, freq_pos_pooled],
                        dim=0,
                    )

                    batch_noise_pred = self.transformer(
                        hidden_states=batch_latent_input,
                        timestep=batch_timestep,
                        encoder_hidden_states=batch_prompt_embeds,
                        pooled_projections=batch_pooled_embeds,
                        joint_attention_kwargs=self.joint_attention_kwargs,
                        return_dict=False,
                    )[0]

                    noise_pred_uncond, noise_pred_rare, noise_pred_freq = (
                        batch_noise_pred.chunk(3)
                    )

                    # --- AAPB: Adaptive gamma*_t (Eq. 13) ---
                    w = float(self.guidance_scale)
                    U = noise_pred_uncond.float()
                    T = noise_pred_rare.float()  # s_θ(x_t, c̃_T)
                    F = noise_pred_freq.float()  # s_θ(x_t, c̃_A)

                    if gamma_t is not None:
                        # Fixed interpolation mode
                        S_cond = (1.0 - gamma_t) * T + gamma_t * F
                        noise_pred = (U + w * (S_cond - U)).to(
                            noise_pred_uncond.dtype
                        )
                    else:
                        # Adaptive coefficient (Eq. 13):
                        # C = (1-w)(U - T),  D = w(F - T)
                        # gamma*_t = -<C, D> / ||D||^2
                        C = (1.0 - w) * (U - T)
                        D = w * (F - T)
                        den = (D * D).sum()

                        if den <= 1e-12:
                            # Degenerate case: anchor ≈ target, fallback to standard CFG
                            noise_pred = (U + w * (T - U)).to(
                                noise_pred_uncond.dtype
                            )
                        else:
                            num = (C * D).sum()
                            gamma_star = torch.clamp(-num / den, 0.01, 0.99)

                            S_cond = (1.0 - gamma_star) * T + gamma_star * F
                            noise_pred = (U + w * (S_cond - U)).to(
                                noise_pred_uncond.dtype
                            )
                else:
                    # Non-CFG path (guidance_scale <= 1)
                    batch_latent_input = torch.cat([latents, latents], dim=0)
                    batch_timestep = t.expand(batch_latent_input.shape[0])

                    batch_prompt_embeds = torch.cat(
                        [prompt_embeds_list[-1], prompt_embeds_list[0]], dim=0
                    )
                    batch_pooled_embeds = torch.cat(
                        [
                            pooled_prompt_embeds_list[-1],
                            pooled_prompt_embeds_list[0],
                        ],
                        dim=0,
                    )

                    batch_noise_pred = self.transformer(
                        hidden_states=batch_latent_input,
                        timestep=batch_timestep,
                        encoder_hidden_states=batch_prompt_embeds,
                        pooled_projections=batch_pooled_embeds,
                        joint_attention_kwargs=self.joint_attention_kwargs,
                        return_dict=False,
                    )[0]

                    noise_pred_rare, noise_pred_freq = batch_noise_pred.chunk(2)

                    if gamma_t is not None:
                        noise_pred = (
                            (1.0 - gamma_t) * noise_pred_rare.float()
                            + gamma_t * noise_pred_freq.float()
                        ).to(noise_pred_rare.dtype)
                    else:
                        noise_pred = noise_pred_rare

                # Scheduler step
                latents_dtype = latents.dtype
                latents = self.scheduler.step(
                    noise_pred, t, latents, return_dict=False
                )[0]

                if latents.dtype != latents_dtype:
                    if torch.backends.mps.is_available():
                        latents = latents.to(latents_dtype)

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(
                        self, i, t, callback_kwargs
                    )
                    latents = callback_outputs.pop("latents", latents)

                if i == len(timesteps) - 1 or (
                    (i + 1) > num_warmup_steps
                    and (i + 1) % self.scheduler.order == 0
                ):
                    progress_bar.update()

                if XLA_AVAILABLE:
                    xm.mark_step()

        # 7. Decode latents
        if output_type == "latent":
            image = latents
        else:
            latents = (
                latents / self.vae.config.scaling_factor
            ) + self.vae.config.shift_factor
            image = self.vae.decode(latents, return_dict=False)[0]
            image = self.image_processor.postprocess(image, output_type=output_type)

        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        return StableDiffusion3PipelineOutput(images=image)
