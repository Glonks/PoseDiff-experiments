import random
import torch
import torch.nn as nn
import functools

from torchvision import models

from components import *

from typing import Optional, Union


class PoseDiffModel(nn.Module):
    def __init__(self, config: dict):
        super().__init__()

        # TODO: Figure out sane parameter config architecture
        self.input_image_shape: tuple[int, int, int] = config['image_shape']
        self.keypoints_shape: tuple[int, int, int] = config['keypoints_shape']

        cfg_config = config['classifier_free_guidance']
        self.enable_cfg: bool = cfg_config['enable']
        self.discard_conditioning_probability: float = cfg_config['discard_conditioning_probability']

        self.device: str = config['device']

        # Setup components
        self.diffuser, self.timesteps = self.get_diffuser(config['diffuser'])

        timestep_embedder_config = config['timestep_embedder']
        self.timestep_embedder = TimestepEmbedder(
            timestep_embedder_config['encoding_dim'],
            timestep_embedder_config['hidden_dim'],
            timestep_embedder_config['output_dim'],
            activation=eval(timestep_embedder_config['activation'])
        )

        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)  # Imagenet V2 weights
        self.visual_feature_extractor = nn.Sequential(*list(resnet.children())[:-1])

        visual_feature_extractor_output_shape = self.get_extracted_visual_features_shape(self.input_image_shape)

        self.features_to_timestep_projector = nn.Linear(
            visual_feature_extractor_output_shape[1], timestep_embedder_config['output_dim']
        )

        condition_embedder_config = config['condition_embedder']
        self.condition_embedder = nn.Sequential(
            nn.Linear(2 * timestep_embedder_config['output_dim'], condition_embedder_config['hidden_dim']),
            eval(condition_embedder_config['activation']),
            nn.Linear(condition_embedder_config['hidden_dim'], condition_embedder_config['output_dim'])
        )

        # TODO: figure out keypoint_dim from the dataset
        unet_config = config['unet']
        self.unet = ConditionalUNet(
            keypoint_dim,
            condition_embedder_config['output_dim'],
            unet_config['hidden_dims'],
            eval(unet_config['activation'])
        )

        # Null features for CFG
        self.null_features = nn.Parameter(torch.randn(visual_feature_extractor_output_shape[1]))

        self.to(self.device)

    @staticmethod
    def get_diffuser(config: dict) -> tuple[Union[DDPMDiffuser, ...], int]:
        sampler = config['sampler']
        timesteps = config['timesteps']

        if sampler == 'ddpm':
            diffuser = DDPMDiffuser(timesteps)
        elif sampler == 'ddim':
            raise NotImplementedError()
        elif sampler == 'dpm++':
            raise NotImplementedError()
        else:
            raise ValueError(f'Unknown diffuser: {sampler}')

        return diffuser, timesteps

    @torch.no_grad()
    def get_extracted_visual_features_shape(
            self,
            input_shape: tuple[int, int, int]
    ) -> torch.Size:
        sample_input = torch.randn(1, *input_shape)

        feature = self.visual_feature_extractor(sample_input)

        return feature.shape

    def extract_visual_features(
            self,
            image: torch.Tensor,
            discard_conditioning: bool = False
    ) -> torch.Tensor:
        """
        Either return the extracted features from input images or null features expanded
        to batch_size if using CGF.
        """
        if discard_conditioning:
            batch_size = image.shape[0]
            features = self.null_features[None, :].expand(batch_size, -1)
        else:
            features = (
                self.visual_feature_extractor(image)  # [batch_size, 2048, 1, 1]
                .squeeze(-1)                          # [batch_size, 2048, 1]
                .squeeze(-1)                          # [batch_size, 2048]
            )

        return features

    def forward_training(
            self,
            image: torch.Tensor,
            keypoints: torch.Tensor
    ) -> torch.Tensor:
        """
        The forward call during training.
        """
        batch_size = image.shape[0]

        # Grab features (maybe with CFG)
        # TODO: change this to per sample CFG sampling
        discard_conditioning = (random.random() < self.discard_conditioning_probability) and self.enable_cfg
        features = self.extract_visual_features(image, discard_conditioning=discard_conditioning)
        features_projected = self.features_to_timestep_projector(features)

        # Sample timesteps
        timesteps = torch.randint(0, self.timesteps, (batch_size,), device=self.device)
        timestep_embedding = self.timestep_embedder(timesteps)

        # Compute condition
        condition_vector = torch.cat([features_projected, timestep_embedding], dim=-1)
        condition_embedding = self.condition_embedder(condition_vector)

        # Diffusion: forward process
        noise = torch.randn_like(keypoints)
        noisy_keypoints = self.diffuser.q_sample(keypoints, timesteps, noise)

        # Get noise prediction
        predicted_noise = self.unet(noisy_keypoints, timesteps, condition_embedding)

        return predicted_noise

    @torch.no_grad()
    def forward_evaluation(
            self,
            image: torch.Tensor
    ) -> torch.Tensor:
        """
        The forward call during evaluation.
        """
        batch_size = image.shape[0]

        features = self.extract_visual_features(image, discard_conditioning=False)
        features_projected = self.features_to_timestep_projector(features)

        # Diffusion: full reverse process
        x_t = torch.randn((batch_size, *self.keypoints_shape), device=self.device)
        for timestep in reversed(range(self.timesteps)):
            timestep = torch.full((batch_size,), timestep, device=self.device, dtype=torch.long)
            timestep_embedding = self.timestep_embedder(timestep)

            condition_vector = torch.cat([features_projected, timestep_embedding], dim=-1)
            condition_embedding = self.condition_embedder(condition_vector)

            # TODO: fix p_sample prototype to be more generic
            x_t = self.diffuser.p_sample(self.unet, x_t, timestep, condition=condition_embedding)

        return x_t

    def forward(
            self,
            image: torch.Tensor,
            keypoints: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Returns predicted noise if in training mode and the actual keypoints produced by the
        full reverse diffusion process if in evaluation mode.
        """
        if self.training:
            if keypoints is None:
                raise ValueError('keypoints need to be provided during training')

            return self.forward_training(image, keypoints)
        else:
            return self.forward_evaluation(image)
