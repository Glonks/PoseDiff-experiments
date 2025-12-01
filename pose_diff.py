import torch
import torch.nn as nn

from torchvision import models

from .components import *

from typing import Optional, Union


class PoseDiffModel(nn.Module):
    def __init__(self, config: dict):
        super().__init__()

        # TODO: Figure out sane parameter config architecture
        self.input_image_shape: tuple[int, int, int] = config['image_shape']
        self.num_keypoints: int = config['num_keypoints']
        self.keypoints_shape: tuple[int, int, int] = config['keypoints_shape']
        self.enable_flow: bool = config['enable_flow']
        cfg_config = config['classifier_free_guidance']
        self.enable_cfg: bool = cfg_config['enable']
        self.discard_conditioning_probability: float = cfg_config['discard_conditioning_probability']
        self.guidance_strength: float = cfg_config['strength']

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
        for param in self.visual_feature_extractor.parameters():
            param.requires_grad = False
        visual_feature_extractor_output_shape = self.get_extracted_visual_features_shape(self.input_image_shape)
        feature_dim = visual_feature_extractor_output_shape[1]
        if self.enable_flow:
            print("Optical Flow Stream ENABLED.")
            # Flow Feature Extractor
            resnet_flow = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            self.flow_feature_extractor = nn.Sequential(*list(resnet_flow.children())[:-1])
            
            # Cross Attention Module
            self.cross_attention = nn.MultiheadAttention(
                embed_dim=feature_dim,
                num_heads=8,
                batch_first=True
            )
            self.norm = nn.LayerNorm(feature_dim)
        else:
            print("Optical Flow Stream DISABLED. Using RGB only.")
            self.flow_feature_extractor = None
            self.cross_attention = None
            self.norm = None

        self.features_to_timestep_projector = nn.Linear(
            visual_feature_extractor_output_shape[1], timestep_embedder_config['output_dim']
        )

        condition_embedder_config = config['condition_embedder']
        self.condition_embedder = nn.Sequential(
            nn.Linear(2 * timestep_embedder_config['output_dim'], condition_embedder_config['hidden_dim']),
            eval(condition_embedder_config['activation'])(),
            nn.Linear(condition_embedder_config['hidden_dim'], condition_embedder_config['output_dim'])
        )

        # TODO: figure out keypoint_dim from the dataset
        unet_config = config['unet']
        self.unet = ConditionalUNet(
            self.num_keypoints * 3,  # TODO: check if this is correct
            condition_embedder_config['output_dim'],
            unet_config['hidden_dims'],
            eval(unet_config['activation'])
        )
        if self.enable_cfg:
            self.unet = CFGWrapper(self.unet, self.guidance_strength)

        # Null features for CFG
        self.null_features = nn.Parameter(torch.randn(visual_feature_extractor_output_shape[1]))

        self.to(self.device)

    def get_diffuser(self, config: dict) -> tuple[Union[DDPMDiffuser, DPMPPDiffuser], int]:
        sampler = config['sampler']
        timesteps = config['timesteps']

        if sampler == 'ddpm':
            diffuser = DDPMDiffuser(timesteps)
        elif sampler == 'dpm++':
            diffuser = DPMPPDiffuser(timesteps)
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
    def extract_flow_features(self, flow: torch.Tensor) -> torch.Tensor:
            """
            Input: (Batch, Time, 3, H, W)
            Output: (Batch, Time, Feature_Dim)
            """
            b, t, c, h, w = flow.shape
            # Batch Folding: Merge Batch and Time
            flow_folded = flow.view(b * t, c, h, w)
            
            # Extract features: (B*T, 2048, 1, 1) -> (B*T, 2048)
            features = self.flow_feature_extractor(flow_folded).squeeze(-1).squeeze(-1)
        
            # Unfold: Restore Batch and Time dimensions
            return features.view(b, t, -1)
    def extract_visual_features(
            self,
            image: torch.Tensor
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """
        Extract features from input images.

        If CFG is enabled, we also grab the null features and produced a mixed batch of
        features for training.
        """
        real_features = (
            self.visual_feature_extractor(image)
            .squeeze(-1)
            .squeeze(-1)
        )

        if self.enable_cfg:
            batch_size = image.shape[0]

            null_features = (
                self.null_features
                .unsqueeze(0)
                .expand(batch_size, -1)
            )

            if self.training:
                discard_mask = (
                    torch.rand(batch_size, device=self.device) < self.discard_conditioning_probability
                ).unsqueeze(-1)

                features = torch.where(discard_mask, null_features, real_features)

                return features

            else:
                return real_features, null_features

        else:
            return real_features
    def fuse_visual_and_flow(self, visual_features: torch.Tensor, flow_features: torch.Tensor) -> torch.Tensor:
            """
            Applies Cross Attention:
            - Query: Visual Features (RGB)
            - Key/Value: Flow Features (Motion History)
            """
            # visual_features: (Batch, 2048)
            # flow_features:   (Batch, Time, 2048)

            # Prepare Query: Needs sequence dim -> (Batch, 1, 2048)
            query = visual_features.unsqueeze(1)

            # Cross Attention
            # attn_output: (Batch, 1, 2048)
            attn_output, _ = self.cross_attention(query=query, key=flow_features, value=flow_features)

            fused = self.norm(query + attn_output)

            return fused.squeeze(1) ##remove that time dimension
    def forward_training(
            self,
            image: torch.Tensor,
            flow: Optional[torch.Tensor],
            keypoints: torch.Tensor,
            noise: torch.Tensor
    ) -> torch.Tensor:
        """
        The forward call during training.
        """
        batch_size = image.shape[0]

        # Grab features (maybe with CFG)
        features = self.extract_visual_features(image)
        if self.enable_flow:
            if flow is None:
                raise ValueError("'enable_flow=True', but no flow data was passed!")
            
            flow_feat = self.extract_flow_features(flow)
            
            # Fuse the RGB (features) with the Flow
            features = self.fuse_visual_and_flow(features, flow_feat)
        features_projected = self.features_to_timestep_projector(features)

        # Sample timesteps
        timesteps = torch.randint(0, self.timesteps, (batch_size,), device=self.device)
        timestep_embedding = self.timestep_embedder(timesteps)

        # Compute condition
        condition_vector = torch.cat([features_projected, timestep_embedding], dim=-1)
        condition_embedding = self.condition_embedder(condition_vector)

        # Diffusion: forward process
        # noise = torch.randn_like(keypoints)
        keypoints = keypoints.reshape(batch_size, -1)  # TODO: remove this once the dataset gives flat keypoints
        noise = noise.reshape(batch_size, -1)  # TODO: remove this when the above is done
        noisy_keypoints = self.diffuser.q_sample(keypoints, timesteps, noise)

        # Get noise prediction
        predicted_noise = self.unet(noisy_keypoints, timesteps, condition=condition_embedding)

        return predicted_noise

    @torch.no_grad()
    def forward_evaluation(
            self,
            image: torch.Tensor,
            flow: torch.Tensor
    ) -> torch.Tensor:
        """
        The forward call during evaluation.
        """
        batch_size = image.shape[0]

        # Diffusion: full reverse process
        if self.enable_cfg:
            real_features, null_features = self.extract_visual_features(image)
            if self.enable_flow:
                if flow is None:
                    raise ValueError("Model config has 'enable_flow=True', but no flow data was passed!")
                
                flow_feat = self.extract_flow_features(flow)
                real_features = self.fuse_visual_and_flow(real_features, flow_feat)
                null_features = self.fuse_visual_and_flow(null_features, flow_feat)
            real_features_projected = self.features_to_timestep_projector(real_features)
            null_features_projected = self.features_to_timestep_projector(null_features)
           
            x_t = torch.randn(batch_size, self.num_keypoints * 3, device=self.device)
            for timestep in reversed(range(self.timesteps)):
                timestep = torch.full((batch_size,), timestep, device=self.device, dtype=torch.long)
                timestep_embedding = self.timestep_embedder(timestep)

                real_condition_vector = torch.cat([real_features_projected, timestep_embedding], dim=-1)
                real_condition_embedding = self.condition_embedder(real_condition_vector)

                null_condition_vector = torch.cat([null_features_projected, timestep_embedding], dim=-1)
                null_condition_embedding = self.condition_embedder(null_condition_vector)

                x_t = self.diffuser.p_sample(
                    self.unet,
                    x_t,
                    timestep,
                    real_condition=real_condition_embedding,
                    null_condition=null_condition_embedding
                )

            return x_t

        else:
            features = self.extract_visual_features(image)
            if self.enable_flow:
                if flow is None:
                    raise ValueError("'enable_flow=True', but no flow data was passed!")
                flow_feat = self.extract_flow_features(flow)
                features = self.fuse_visual_and_flow(features, flow_feat)
            features_projected = self.features_to_timestep_projector(features)

            x_t = torch.randn((batch_size, self.num_keypoints * 3), device=self.device)
            for timestep in reversed(range(self.timesteps)):
                timestep = torch.full((batch_size,), timestep, device=self.device, dtype=torch.long)
                timestep_embedding = self.timestep_embedder(timestep)

                condition_vector = torch.cat([features_projected, timestep_embedding], dim=-1)
                condition_embedding = self.condition_embedder(condition_vector)

                x_t = self.diffuser.p_sample(self.unet, x_t, timestep, condition=condition_embedding)

            return x_t

    def forward(
            self,
            image: torch.Tensor,
            flow: Optional[torch.Tensor] = None, ## expected shape (B, T, 3, H, W)
            keypoints: Optional[torch.Tensor] = None,
            noise: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Returns predicted noise if in training mode and the actual keypoints produced by the
        full reverse diffusion process if in evaluation mode.
        """
        if self.training:
            if keypoints is None:
                raise ValueError('keypoints need to be provided during training')
            if noise is None:
                raise ValueError('noise needs to be provided during training')

            return self.forward_training(image,flow,keypoints, noise)
        else:
            return self.forward_evaluation(image,flow)
