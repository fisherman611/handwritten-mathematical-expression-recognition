import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import math


class DenseBlock(nn.Module):
    """
    Basic DenseNet block
    """
    def __init__(self, in_channels, growth_rate, num_layers):
        super(DenseBlock, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(self._make_layer(in_channels + i * growth_rate, growth_rate))
    
    def _make_layer(self, in_channels, growth_rate):
        layer = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, 4 * growth_rate, kernel_size=1, bias=False),
            nn.BatchNorm2d(4 * growth_rate),
            nn.ReLU(inplace=True),
            nn.Conv2d(4 * growth_rate, growth_rate, kernel_size=3, padding=1, bias=False)
        )
        return layer
    
    def forward(self, x):
        features = [x]
        for layer in self.layers:
            new_feature = layer(torch.cat(features, dim=1))
            features.append(new_feature)
        return torch.cat(features, dim=1)


class TransitionLayer(nn.Module):
    """
    Transition layer between DenseBlocks
    """
    def __init__(self, in_channels, out_channels):
        super(TransitionLayer, self).__init__()
        self.transition = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )
    
    def forward(self, x):
        return self.transition(x)


class DenseNetBackbone(nn.Module):
    """
    DenseNet backbone for CAN
    """
    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16), num_init_features=64):
        super(DenseNetBackbone, self).__init__()
        
        # Initial layer
        self.features = nn.Sequential(
            nn.Conv2d(1, num_init_features, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(num_init_features),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        # DenseBlocks
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = DenseBlock(num_features, growth_rate, num_layers)
            self.features.add_module(f'denseblock{i+1}', block)
            num_features = num_features + growth_rate * num_layers
            if i != len(block_config) - 1:
                trans = TransitionLayer(num_features, num_features // 2)
                self.features.add_module(f'transition{i+1}', trans)
                num_features = num_features // 2
        
        # Final processing
        self.features.add_module('norm5', nn.BatchNorm2d(num_features))
        self.features.add_module('relu5', nn.ReLU(inplace=True))
        
        self.out_channels = num_features  # 684 (with default configuration)
    
    def forward(self, x):
        return self.features(x)


class ChannelAttention(nn.Module):
    """
    Channel-wise attention mechanism
    """
    def __init__(self, in_channels, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // ratio, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // ratio, in_channels, kernel_size=1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class MSCM(nn.Module):
    """
    Multi-Scale Counting Module
    """
    def __init__(self, in_channels, num_classes):
        super(MSCM, self).__init__()
        
        # Branch 1: 3x3 kernel
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.attention1 = ChannelAttention(256)
        
        # Branch 2: 5x5 kernel
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=5, padding=2),
            nn.ReLU(inplace=True)
        )
        self.attention2 = ChannelAttention(256)
        
        # 1x1 Conv layer to reduce channels and create counting map
        self.conv_reduce = nn.Conv2d(512, num_classes, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # Process branch 1
        out1 = self.branch1(x)
        out1 = out1 * self.attention1(out1)
        
        # Process branch 2
        out2 = self.branch2(x)
        out2 = out2 * self.attention2(out2)
        
        # Concatenate features from both branches
        concat_features = torch.cat([out1, out2], dim=1)  # Shape: B x 512 x H x W
        
        # Create counting map
        count_map = self.sigmoid(self.conv_reduce(concat_features))  # Shape: B x C x H x W
        
        # Apply sum-pooling to create 1D counting vector
        # Sum over the entire feature map along height and width
        count_vector = torch.sum(count_map, dim=(2, 3))  # Shape: B x C
        
        return count_map, count_vector


class PositionalEncoding(nn.Module):
    """
    Positional encoding for attention decoder
    """
    def __init__(self, d_model, max_seq_len=1024):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        
        # Create positional encoding matrix
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        # x shape: B x H x W x d_model
        b, h, w, _ = x.shape
        
        # Ensure we have enough positional encodings for the feature map size
        if h*w > self.pe.size(0):       #type: ignore
            # Dynamically extend positional encodings if needed
            device = self.pe.device
            extended_pe = torch.zeros(h*w, self.d_model, device=device)                                                             #type: ignore
            position = torch.arange(0, h*w, dtype=torch.float, device=device).unsqueeze(1)                                          #type: ignore
            div_term = torch.exp(torch.arange(0, self.d_model, 2, device=device).float() * (-math.log(10000.0) / self.d_model))     #type: ignore
            
            extended_pe[:, 0::2] = torch.sin(position * div_term)
            extended_pe[:, 1::2] = torch.cos(position * div_term)
            
            pos_encoding = extended_pe.view(h, w, -1)
        else:
            # Use pre-computed positional encodings
            pos_encoding = self.pe[:h*w].view(h, w, -1)                                                                             #type: ignore
              
        pos_encoding = pos_encoding.unsqueeze(0).expand(b, -1, -1, -1)  # B x H x W x d_model
        return pos_encoding


class CCAD(nn.Module):
    """
    Counting-Combined Attentional Decoder
    """
    def __init__(self, input_channels, hidden_size, embedding_dim, num_classes, use_coverage=True):
        super(CCAD, self).__init__()
        
        self.hidden_size = hidden_size
        self.embedding_dim = embedding_dim
        self.use_coverage = use_coverage
        
        # Input layer to reduce feature map
        self.feature_proj = nn.Conv2d(input_channels, hidden_size * 2, kernel_size=1)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(hidden_size * 2)
        
        # Embedding layer for output symbols
        self.embedding = nn.Embedding(num_classes, embedding_dim)
        
        # GRU cell
        self.gru = nn.GRUCell(embedding_dim + hidden_size + num_classes, hidden_size)
        
        # Attention
        self.attention_w = nn.Linear(hidden_size * 2, hidden_size)
        self.attention_v = nn.Linear(hidden_size, 1)
        if use_coverage:
            self.coverage_proj = nn.Linear(1, hidden_size)
        
        # Output layer
        self.out = nn.Linear(hidden_size + hidden_size + num_classes, num_classes)
    
    def forward(self, feature_map, count_vector, target=None, teacher_forcing_ratio=0.5, max_len=200):
        batch_size = feature_map.size(0)
        device = feature_map.device
        
        # Transform feature map
        projected_features = self.feature_proj(feature_map)  # B x 2*hidden_size x H x W
        H, W = projected_features.size(2), projected_features.size(3)
        
        # Reshape feature map to B x H*W x 2*hidden_size
        projected_features = projected_features.permute(0, 2, 3, 1).contiguous()  # B x H x W x 2*hidden_size
        
        # Add positional encoding
        pos_encoding = self.pos_encoder(projected_features)  # B x H x W x 2*hidden_size
        projected_features = projected_features + pos_encoding
        
        # Reshape for attention processing
        projected_features = projected_features.view(batch_size, H*W, -1)  # B x H*W x 2*hidden_size
        
        # Initialize initial hidden state
        h_t = torch.zeros(batch_size, self.hidden_size, device=device)
        
        # Initialize coverage attention if used
        if self.use_coverage:
            coverage = torch.zeros(batch_size, H*W, 1, device=device)
        
        # First <SOS> token
        y_t_1 = torch.ones(batch_size, dtype=torch.long, device=device)
        
        # Prepare target sequence if provided
        if target is not None:
            max_len = target.size(1)
        
        # Array to store predictions
        outputs = torch.zeros(batch_size, max_len, self.embedding.num_embeddings, device=device)
        
        for t in range(max_len):
            # Apply embedding to the previous symbol
            embedded = self.embedding(y_t_1)  # B x embedding_dim
            
            # Compute attention
            attention_input = self.attention_w(projected_features)  # B x H*W x hidden_size
            
            # Add coverage attention if used
            if self.use_coverage:
                coverage_input = self.coverage_proj(coverage.float())                        #type: ignore
                attention_input = attention_input + coverage_input
            
            # Add hidden state to attention
            h_expanded = h_t.unsqueeze(1).expand(-1, H*W, -1)  # B x H*W x hidden_size
            attention_input = torch.tanh(attention_input + h_expanded)
            
            # Compute attention weights
            e_t = self.attention_v(attention_input).squeeze(-1)  # B x H*W
            alpha_t = F.softmax(e_t, dim=1)  # B x H*W
            
            # Update coverage if used
            if self.use_coverage:
                coverage = coverage + alpha_t.unsqueeze(-1)                                  #type: ignore
            
            # Compute context vector
            alpha_t = alpha_t.unsqueeze(1)  # B x 1 x H*W
            context = torch.bmm(alpha_t, projected_features).squeeze(1)  # B x 2*hidden_size
            context = context[:, :self.hidden_size]  # Take the first half as context vector
            
            # Combine embedding, context vector, and count vector
            gru_input = torch.cat([embedded, context, count_vector], dim=1)
            
            # Update hidden state
            h_t = self.gru(gru_input, h_t)
            
            # Predict output symbol
            output = self.out(torch.cat([h_t, context, count_vector], dim=1))
            outputs[:, t] = output
            
            # Decide the next input symbol
            if target is not None and torch.rand(1).item() < teacher_forcing_ratio:
                y_t_1 = target[:, t]
            else:
                # Greedy decoding
                _, y_t_1 = output.max(1)
        
        return outputs


class CAN(nn.Module):
    """
    Counting-Aware Network for handwritten mathematical expression recognition
    """
    def __init__(self, num_classes, backbone=None, hidden_size=256, embedding_dim=256, use_coverage=True):
        super(CAN, self).__init__()
        
        # Backbone
        if backbone is None:
            self.backbone = DenseNetBackbone()
        else:
            self.backbone = backbone
        backbone_channels = self.backbone.out_channels
        
        # Multi-Scale Counting Module
        self.mscm = MSCM(backbone_channels, num_classes)
        
        # Counting-Combined Attentional Decoder
        self.decoder = CCAD(
            input_channels=backbone_channels,
            hidden_size=hidden_size,
            embedding_dim=embedding_dim,
            num_classes=num_classes,
            use_coverage=use_coverage
        )
        
        # Save parameters for later use
        self.hidden_size = hidden_size
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.use_coverage = use_coverage
    
    def init_hidden_state(self, visual_features):
        """
        Initialize hidden state and cell state for LSTM
        
        Args:
            visual_features: Visual features from backbone
            
        Returns:
            h, c: Initial hidden and cell states
        """
        batch_size = visual_features.size(0)
        device = visual_features.device
        
        # Initialize hidden state with zeros
        h = torch.zeros(1, batch_size, self.hidden_size, device=device)
        c = torch.zeros(1, batch_size, self.hidden_size, device=device)
        
        return h, c

    def forward(self, x, target=None, teacher_forcing_ratio=0.5):
        # Extract features from backbone
        features = self.backbone(x)
        
        # Compute count map and count vector from MSCM
        count_map, count_vector = self.mscm(features)
        
        # Decode with CCAD
        outputs = self.decoder(features, count_vector, target, teacher_forcing_ratio)
        
        return outputs, count_vector
    
    def calculate_loss(self, outputs, targets, count_vectors, count_targets, lambda_count=0.01):
        """
        Compute the combined loss function for CAN
        
        Args:
            outputs: Predicted output sequence from decoder
            targets: Actual target sequence
            count_vectors: Predicted count vector
            count_targets: Actual target count vector
            lambda_count: Weight for counting loss
        
        Returns:
            Total loss: L = L_cls + λ * L_counting
        """
        # Loss for decoder (cross entropy)
        L_cls = F.cross_entropy(outputs.view(-1, outputs.size(-1)), targets.view(-1))
        
        # Loss for counting (MSE)
        L_counting = F.mse_loss(count_vectors, count_targets)
        
        # Total loss
        total_loss = L_cls + lambda_count * L_counting
        
        return total_loss, L_cls, L_counting

    def recognize(self, images, max_length=150, start_token=None, end_token=None, beam_width=5):
        """
        Recognize the handwritten expression using beam search (batch_size=1 only).
        
        Args:
            images: Input image tensor, shape (1, channels, height, width)
            max_length: Maximum length of the output sequence
            start_token: Start token index
            end_token: End token index
            beam_width: Beam width for beam search
            
        Returns:
            best_sequence: List of token indices
            attention_weights: List of attention weights for visualization
        """
        if images.size(0) != 1:
            raise ValueError("Beam search is implemented only for batch_size=1")
        
        device = images.device
        
        # Encode the image
        visual_features = self.backbone(images)
        
        # Get count vector
        _, count_vector = self.mscm(visual_features)
        
        # Prepare feature map for decoder
        projected_features = self.decoder.feature_proj(visual_features)  # (1, 2*hidden_size, H, W)
        H, W = projected_features.size(2), projected_features.size(3)
        projected_features = projected_features.permute(0, 2, 3, 1).contiguous()  # (1, H, W, 2*hidden_size)
        pos_encoding = self.decoder.pos_encoder(projected_features)  # (1, H, W, 2*hidden_size)
        projected_features = projected_features + pos_encoding  # (1, H, W, 2*hidden_size)
        projected_features = projected_features.view(1, H*W, -1)  # (1, H*W, 2*hidden_size)
        
        # Initialize beams
        beam_sequences = [torch.tensor([start_token], device=device)] * beam_width  # List of (seq_len) tensors
        beam_scores = torch.zeros(beam_width, device=device)  # (beam_width)
        h_t = torch.zeros(beam_width, self.hidden_size, device=device)  # (beam_width, hidden_size)
        if self.use_coverage:
            coverage = torch.zeros(beam_width, H*W, device=device)  # (beam_width, H*W)
        
        all_attention_weights = []
        
        for step in range(max_length):
            # Get current tokens for all beams
            current_tokens = torch.tensor([seq[-1] for seq in beam_sequences], device=device)  # (beam_width)
            
            # Apply embedding
            embedded = self.decoder.embedding(current_tokens)  # (beam_width, embedding_dim)
            
            # Compute attention for each beam
            attention_input = self.decoder.attention_w(projected_features.expand(beam_width, -1, -1))  # (beam_width, H*W, hidden_size)
            if self.use_coverage:
                coverage_input = self.decoder.coverage_proj(coverage.unsqueeze(-1))  # (beam_width, H*W, hidden_size)            #type: ignore
                attention_input = attention_input + coverage_input
            
            h_expanded = h_t.unsqueeze(1).expand(-1, H*W, -1)  # (beam_width, H*W, hidden_size)
            attention_input = torch.tanh(attention_input + h_expanded)
            
            e_t = self.decoder.attention_v(attention_input).squeeze(-1)  # (beam_width, H*W)
            alpha_t = F.softmax(e_t, dim=1)  # (beam_width, H*W)
            
            all_attention_weights.append(alpha_t.detach())
            
            if self.use_coverage:
                coverage = coverage + alpha_t            #type: ignore
            
            context = torch.bmm(alpha_t.unsqueeze(1), projected_features.expand(beam_width, -1, -1)).squeeze(1)  # (beam_width, 2*hidden_size)
            context = context[:, :self.hidden_size]  # (beam_width, hidden_size)
            
            # Expand count_vector to (beam_width, num_classes)
            count_vector_expanded = count_vector.expand(beam_width, -1)  # (beam_width, num_classes)
            
            gru_input = torch.cat([embedded, context, count_vector_expanded], dim=1)  # (beam_width, embedding_dim + hidden_size + num_classes)
            
            h_t = self.decoder.gru(gru_input, h_t)  # (beam_width, hidden_size)
            
            output = self.decoder.out(torch.cat([h_t, context, count_vector_expanded], dim=1))  # (beam_width, num_classes)
            scores = F.log_softmax(output, dim=1)  # (beam_width, num_classes)
            
            # Compute new scores for all beam-token combinations
            new_beam_scores = beam_scores.unsqueeze(1) + scores  # (beam_width, num_classes)
            new_beam_scores_flat = new_beam_scores.view(-1)  # (beam_width * num_classes)
            
            # Select top beam_width scores and indices
            topk_scores, topk_indices = new_beam_scores_flat.topk(beam_width)
            
            # Determine which beam and token each top score corresponds to
            beam_indices = topk_indices // self.num_classes  # (beam_width)
            token_indices = topk_indices % self.num_classes  # (beam_width)
            
            # Create new beam sequences and states
            new_beam_sequences = []
            new_h_t = []
            if self.use_coverage:
                new_coverage = []
            for i in range(beam_width):
                prev_beam_idx = beam_indices[i].item()
                token = token_indices[i].item()
                new_seq = torch.cat([beam_sequences[prev_beam_idx], torch.tensor([token], device=device)])           #type: ignore
                new_beam_sequences.append(new_seq)
                new_h_t.append(h_t[prev_beam_idx])
                if self.use_coverage:
                    new_coverage.append(coverage[prev_beam_idx])           #type: ignore
            
            # Update beams
            beam_sequences = new_beam_sequences
            beam_scores = topk_scores
            h_t = torch.stack(new_h_t)
            if self.use_coverage:
                coverage = torch.stack(new_coverage)           #type: ignore
        
        # Select the sequence with the highest score
        best_idx = beam_scores.argmax()
        best_sequence = beam_sequences[best_idx].tolist()
        
        # Remove <start> and stop at <end>
        if best_sequence[0] == start_token:
            best_sequence = best_sequence[1:]
        if end_token in best_sequence:
            end_idx = best_sequence.index(end_token)
            best_sequence = best_sequence[:end_idx]
        
        return best_sequence, all_attention_weights


# Use the model
def create_can_model(num_classes, pretrained_backbone=False):
    """
    Create CAN model
    
    Args:
        num_classes: Number of symbol classes
        pretrained_backbone: Whether to use a pretrained backbone
    
    Returns:
        CAN model
    """
    # Create backbone
    if pretrained_backbone:
        backbone = models.densenet121(pretrained=True)
        # Adjust input layer for single-channel input images
        backbone.features.conv0 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    else:
        backbone = DenseNetBackbone()
    
    # Create model
    model = CAN(
        num_classes=num_classes,
        backbone=backbone,
        hidden_size=256,
        embedding_dim=256,
        use_coverage=True
    )
    
    return model


# # Example usage
# if __name__ == "__main__":
#     # Create CAN model with 101 symbol classes (example)
#     num_classes = 101  # Number of symbol classes + special tokens like <SOS>, <EOS>
#     model = create_can_model(num_classes)
    
#     # Create dummy input data
#     batch_size = 4
#     input_image = torch.randn(batch_size, 1, 128, 384)  # B x C x H x W
#     target = torch.randint(0, num_classes, (batch_size, 50))  # B x max_len
    
#     # Forward pass
#     outputs, count_vectors = model(input_image, target)
    
#     # Print output shapes
#     print(f"Outputs shape: {outputs.shape}")  # B x max_len x num_classes
#     print(f"Count vectors shape: {count_vectors.shape}")  # B x num_classes