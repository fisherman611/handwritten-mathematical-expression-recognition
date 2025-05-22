import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from dataclasses import dataclass
from typing import Optional, Tuple, List, Union
import math

@dataclass
class CANConfig:
    """Configuration for CAN model."""
    GROWTH_RATE: int = 32
    BLOCK_CONFIG: Tuple[int, int, int, int] = (6, 12, 24, 16)
    NUM_INIT_FEATURES: int = 64
    HIDDEN_SIZE: int = 256
    EMBEDDING_DIM: int = 256
    MAX_SEQ_LEN: int = 1024
    BEAM_WIDTH: int = 5
    LAMBDA_COUNT: float = 0.01
    MAX_LEN: int = 200

CONFIG = CANConfig()

class DenseBlock(nn.Module):
    """DenseNet block with concatenated feature layers."""
    def __init__(self, in_channels: int, growth_rate: int, num_layers: int):
        super().__init__()
        self.layers = nn.ModuleList([
            self._make_layer(in_channels + i * growth_rate, growth_rate)
            for i in range(num_layers)
        ])

    def _make_layer(self, in_channels: int, growth_rate: int) -> nn.Sequential:
        return nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, 4 * growth_rate, kernel_size=1, bias=False),
            nn.BatchNorm2d(4 * growth_rate),
            nn.ReLU(inplace=True),
            nn.Conv2d(4 * growth_rate, growth_rate, kernel_size=3, padding=1, bias=False)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = [x]
        for layer in self.layers:
            features.append(layer(torch.cat(features, dim=1)))
        return torch.cat(features, dim=1)

class TransitionLayer(nn.Module):
    """Transition layer for downsampling between DenseBlocks."""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.transition = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.transition(x)

class DenseNetBackbone(nn.Module):
    """DenseNet backbone for feature extraction."""
    def __init__(self, growth_rate: int = CONFIG.GROWTH_RATE,
                 block_config: Tuple[int, ...] = CONFIG.BLOCK_CONFIG,
                 num_init_features: int = CONFIG.NUM_INIT_FEATURES):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, num_init_features, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(num_init_features),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = DenseBlock(num_features, growth_rate, num_layers)
            self.features.add_module(f'denseblock{i+1}', block)
            num_features += growth_rate * num_layers
            if i < len(block_config) - 1:
                trans = TransitionLayer(num_features, num_features // 2)
                self.features.add_module(f'transition{i+1}', trans)
                num_features = num_features // 2

        self.features.add_module('norm5', nn.BatchNorm2d(num_features))
        self.features.add_module('relu5', nn.ReLU(inplace=True))
        self.out_channels = num_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)

class DenseNetFeatureExtractor(nn.Module):
    """DenseNet feature extractor with pretrained weights."""
    def __init__(self, densenet_model: nn.Module, out_channels: int = 684):
        super().__init__()
        self.conv0 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.conv0.weight.data = densenet_model.features.conv0.weight.data.mean(dim=1, keepdim=True)
        self.features = densenet_model.features
        self.final_conv = nn.Conv2d(1024, out_channels, kernel_size=1)
        self.final_bn = nn.BatchNorm2d(out_channels)
        self.final_relu = nn.ReLU(inplace=True)
        self.out_channels = out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv0(x)
        x = self.features.norm0(x)
        x = self.features.relu0(x)
        x = self.features.pool0(x)
        x = self.features.denseblock1(x)
        x = self.features.transition1(x)
        x = self.features.denseblock2(x)
        x = self.features.transition2(x)
        x = self.features.denseblock3(x)
        x = self.features.transition3(x)
        x = self.features.denseblock4(x)
        x = self.features.norm5(x)
        x = self.final_conv(x)
        x = self.final_bn(x)
        x = self.final_relu(x)
        return x

class BasicBlock(nn.Module):
    """Basic ResNet block."""
    expansion = 1
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential() if stride == 1 and in_channels == out_channels else nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + self.shortcut(x))

class Bottleneck(nn.Module):
    """Bottleneck ResNet block."""
    expansion = 4
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.shortcut = nn.Sequential() if stride == 1 and in_channels == out_channels * self.expansion else nn.Sequential(
            nn.Conv2d(in_channels, out_channels * self.expansion, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_channels * self.expansion)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        return self.relu(out + self.shortcut(x))

class ResNetBackbone(nn.Module):
    """ResNet backbone for feature extraction."""
    def __init__(self, block_type: str = 'bottleneck', layers: List[int] = [3, 4, 6, 3],
                 num_init_features: int = CONFIG.NUM_INIT_FEATURES):
        super().__init__()
        block = Bottleneck if block_type == 'bottleneck' else BasicBlock
        expansion = block.expansion

        self.conv1 = nn.Conv2d(1, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(num_init_features)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, num_init_features, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 64 * expansion, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 128 * expansion, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 256 * expansion, 512, layers[3], stride=2)

        self.final_conv = nn.Conv2d(512 * expansion, 684, kernel_size=1)
        self.final_bn = nn.BatchNorm2d(684)
        self.final_relu = nn.ReLU(inplace=True)
        self.out_channels = 684

        self._initialize_weights()

    def _make_layer(self, block: nn.Module, in_channels: int, out_channels: int, num_blocks: int, stride: int) -> nn.Sequential:
        layers = [block(in_channels, out_channels, stride)]
        for _ in range(1, num_blocks):
            layers.append(block(out_channels * block.expansion, out_channels))
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return self.final_relu(self.final_bn(self.final_conv(x)))

class ResNetFeatureExtractor(nn.Module):
    """ResNet feature extractor with pretrained weights."""
    def __init__(self, resnet_model: nn.Module, out_channels: int = 684):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.conv1.weight.data = resnet_model.conv1.weight.data.sum(dim=1, keepdim=True)
        self.bn1 = resnet_model.bn1
        self.relu = resnet_model.relu
        self.maxpool = resnet_model.maxpool
        self.layer1 = resnet_model.layer1
        self.layer2 = resnet_model.layer2
        self.layer3 = resnet_model.layer3
        self.layer4 = resnet_model.layer4
        self.final_conv = nn.Conv2d(2048, out_channels, kernel_size=1)
        self.final_bn = nn.BatchNorm2d(out_channels)
        self.final_relu = nn.ReLU(inplace=True)
        self.out_channels = out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return self.final_relu(self.final_bn(self.final_conv(x)))

class ChannelAttention(nn.Module):
    """Channel-wise attention mechanism."""
    def __init__(self, in_channels: int, ratio: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // ratio, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // ratio, in_channels, kernel_size=1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return self.sigmoid(avg_out + max_out)

class MSCM(nn.Module):
    """Multi-Scale Counting Module."""
    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.attention1 = ChannelAttention(256)
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=5, padding=2),
            nn.ReLU(inplace=True)
        )
        self.attention2 = ChannelAttention(256)
        self.conv_reduce = nn.Conv2d(512, num_classes, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        out1 = self.branch1(x) * self.attention1(self.branch1(x))
        out2 = self.branch2(x) * self.attention2(self.branch2(x))
        concat_features = torch.cat([out1, out2], dim=1)
        count_map = self.sigmoid(self.conv_reduce(concat_features))
        count_vector = count_map.sum(dim=(2, 3))
        return count_map, count_vector

class PositionalEncoding(nn.Module):
    """Positional encoding for attention decoder."""
    def __init__(self, d_model: int, max_seq_len: int = CONFIG.MAX_SEQ_LEN):
        super().__init__()
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        self.d_model = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, h, w, _ = x.shape
        seq_len = h * w
        if seq_len > self.pe.size(0):
            device = x.device
            extended_pe = torch.zeros(seq_len, self.d_model, device=device)
            position = torch.arange(0, seq_len, dtype=torch.float, device=device).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, self.d_model, 2, device=device).float() * (-math.log(10000.0) / self.d_model))
            extended_pe[:, 0::2] = torch.sin(position * div_term)
            extended_pe[:, 1::2] = torch.cos(position * div_term)
            pos_encoding = extended_pe.view(h, w, -1)
        else:
            pos_encoding = self.pe[:seq_len].view(h, w, -1)
        return pos_encoding.unsqueeze(0).expand(b, -1, -1, -1)

class CCAD(nn.Module):
    """Counting-Combined Attentional Decoder."""
    def __init__(self, input_channels: int, hidden_size: int, embedding_dim: int, num_classes: int, use_coverage: bool = True):
        super().__init__()
        self.hidden_size = hidden_size
        self.embedding_dim = embedding_dim
        self.use_coverage = use_coverage

        self.feature_proj = nn.Conv2d(input_channels, hidden_size * 2, kernel_size=1)
        self.pos_encoder = PositionalEncoding(hidden_size * 2)
        self.embedding = nn.Embedding(num_classes, embedding_dim)
        self.gru = nn.GRUCell(embedding_dim + hidden_size + num_classes, hidden_size)
        self.attention_w = nn.Linear(hidden_size * 2, hidden_size)
        self.attention_v = nn.Linear(hidden_size, 1)
        self.coverage_proj = nn.Linear(1, hidden_size) if use_coverage else None
        self.out = nn.Linear(hidden_size * 2 + num_classes, num_classes)

    def forward(self, feature_map: torch.Tensor, count_vector: torch.Tensor, target: Optional[torch.Tensor] = None,
                teacher_forcing_ratio: float = 0.5, max_len: int = CONFIG.MAX_LEN) -> torch.Tensor:
        batch_size = feature_map.size(0)
        device = feature_map.device

        projected_features = self.feature_proj(feature_map).permute(0, 2, 3, 1)
        projected_features = projected_features + self.pos_encoder(projected_features)
        projected_features = projected_features.view(batch_size, -1, self.hidden_size * 2)

        h_t = torch.zeros(batch_size, self.hidden_size, device=device)
        coverage = torch.zeros(batch_size, projected_features.size(1), 1, device=device) if self.use_coverage else None
        y_t_1 = torch.ones(batch_size, dtype=torch.long, device=device)

        max_len = target.size(1) if target is not None else max_len
        outputs = torch.zeros(batch_size, max_len, self.out.out_features, device=device)

        for t in range(max_len):
            embedded = self.embedding(y_t_1)
            attention_input = self.attention_w(projected_features)

            if self.use_coverage and self.coverage_proj is not None:
                coverage_input = self.coverage_proj(coverage)
                attention_input = attention_input + coverage_input

            attention_input = torch.tanh(attention_input + h_t.unsqueeze(1))
            alpha_t = F.softmax(self.attention_v(attention_input).squeeze(-1), dim=1)

            if self.use_coverage:
                coverage = coverage + alpha_t.unsqueeze(-1)

            context = torch.bmm(alpha_t.unsqueeze(1), projected_features).squeeze(1)[:, :self.hidden_size]
            gru_input = torch.cat([embedded, context, count_vector], dim=1)
            h_t = self.gru(gru_input, h_t)
            outputs[:, t] = self.out(torch.cat([h_t, context, count_vector], dim=1))

            y_t_1 = target[:, t] if target is not None and torch.rand(1).item() < teacher_forcing_ratio else outputs[:, t].argmax(1)

        return outputs

class CAN(nn.Module):
    """Counting-Aware Network for handwritten mathematical expression recognition."""
    def __init__(self, num_classes: int, backbone: Optional[nn.Module] = None,
                 hidden_size: int = CONFIG.HIDDEN_SIZE, embedding_dim: int = CONFIG.EMBEDDING_DIM, use_coverage: bool = True):
        super().__init__()
        self.backbone = backbone or DenseNetBackbone()
        self.mscm = MSCM(self.backbone.out_channels, num_classes)
        self.decoder = CCAD(self.backbone.out_channels, hidden_size, embedding_dim, num_classes, use_coverage)
        self.hidden_size = hidden_size
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.use_coverage = use_coverage

    def forward(self, x: torch.Tensor, target: Optional[torch.Tensor] = None,
                teacher_forcing_ratio: float = 0.5) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone(x)
        count_map, count_vector = self.mscm(features)
        outputs = self.decoder(features, count_vector, target, teacher_forcing_ratio)
        return outputs, count_vector

    def calculate_loss(self, outputs: torch.Tensor, targets: torch.Tensor, count_vectors: torch.Tensor,
                       count_targets: torch.Tensor, lambda_count: float = CONFIG.LAMBDA_COUNT) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        L_cls = F.cross_entropy(outputs.view(-1, outputs.size(-1)), targets.view(-1))
        L_counting = F.smooth_l1_loss(count_vectors, count_targets)
        total_loss = L_cls + lambda_count * L_counting
        return total_loss, L_cls, L_counting

    def recognize(self, images: torch.Tensor, max_length: int = 150, start_token: Optional[int] = None,
                  end_token: Optional[int] = None, beam_width: int = CONFIG.BEAM_WIDTH) -> Tuple[List[int], List[torch.Tensor]]:
        if images.size(0) != 1:
            raise ValueError("Beam search supports batch_size=1 only")

        device = images.device
        visual_features = self.backbone(images)
        _, count_vector = self.mscm(visual_features)
        projected_features = self.decoder.feature_proj(visual_features).permute(0, 2, 3, 1)
        projected_features = projected_features + self.decoder.pos_encoder(projected_features)
        projected_features = projected_features.view(1, -1, self.decoder.hidden_size * 2)
        H_W = projected_features.size(1)

        beam_sequences = [torch.tensor([start_token], device=device)] * beam_width
        beam_scores = torch.zeros(beam_width, device=device)
        h_t = torch.zeros(beam_width, self.hidden_size, device=device)
        coverage = torch.zeros(beam_width, H_W, device=device) if self.use_coverage else None
        all_attention_weights = []

        for _ in range(max_length):
            current_tokens = torch.tensor([seq[-1] for seq in beam_sequences], device=device)
            embedded = self.decoder.embedding(current_tokens)
            attention_input = self.decoder.attention_w(projected_features.expand(beam_width, -1, -1))

            if self.use_coverage and self.decoder.coverage_proj is not None:
                coverage_input = self.decoder.coverage_proj(coverage.unsqueeze(-1))
                attention_input = attention_input + coverage_input

            attention_input = torch.tanh(attention_input + h_t.unsqueeze(1))
            alpha_t = F.softmax(self.decoder.attention_v(attention_input).squeeze(-1), dim=1)
            all_attention_weights.append(alpha_t.detach())

            if self.use_coverage:
                coverage = coverage + alpha_t

            context = torch.bmm(alpha_t.unsqueeze(1), projected_features.expand(beam_width, -1, -1)).squeeze(1)[:, :self.hidden_size]
            gru_input = torch.cat([embedded, context, count_vector.expand(beam_width, -1)], dim=1)
            h_t = self.decoder.gru(gru_input, h_t)
            output = self.decoder.out(torch.cat([h_t, context, count_vector.expand(beam_width, -1)], dim=1))
            scores = F.log_softmax(output, dim=1)

            new_beam_scores = beam_scores.unsqueeze(1) + scores
            topk_scores, topk_indices = new_beam_scores.view(-1).topk(beam_width)
            beam_indices = topk_indices // self.num_classes
            token_indices = topk_indices % self.num_classes

            new_beam_sequences, new_h_t, new_coverage = [], [], [] if self.use_coverage else None
            for i in range(beam_width):
                prev_beam_idx = beam_indices[i].item()
                token = token_indices[i].item()
                new_beam_sequences.append(torch.cat([beam_sequences[prev_beam_idx], torch.tensor([token], device=device)]))
                new_h_t.append(h_t[prev_beam_idx])
                if self.use_coverage:
                    new_coverage.append(coverage[prev_beam_idx])

            beam_sequences = new_beam_sequences
            beam_scores = topk_scores
            h_t = torch.stack(new_h_t)
            if self.use_coverage:
                coverage = torch.stack(new_coverage)

        best_idx = beam_scores.argmax()
        best_sequence = beam_sequences[best_idx].tolist()[1:] if best_sequence[0] == start_token else best_sequence
        if end_token in best_sequence:
            best_sequence = best_sequence[:best_sequence.index(end_token)]

        return best_sequence, all_attention_weights

def create_can_model(num_classes: int, pretrained_backbone: bool = False, backbone_type: str = 'densenet') -> CAN:
    """Create CAN model with specified backbone."""
    if backbone_type == 'densenet':
        backbone = DenseNetFeatureExtractor(models.densenet121(pretrained=pretrained_backbone)) if pretrained_backbone else DenseNetBackbone()
    elif backbone_type == 'resnet':
        backbone = ResNetFeatureExtractor(models.resnet50(pretrained=pretrained_backbone)) if pretrained_backbone else ResNetBackbone()
    else:
        raise ValueError(f"Unknown backbone type: {backbone_type}")

    return CAN(num_classes=num_classes, backbone=backbone)

if __name__ == "__main__":
    model = create_can_model(num_classes=101)
    input_image = torch.randn(4, 1, 128, 384)
    target = torch.randint(0, 101, (4, 50))
    outputs, count_vectors = model(input_image, target)
    print(f"Outputs shape: {outputs.shape}")
    print(f"Count vectors shape: {count_vectors.shape}")