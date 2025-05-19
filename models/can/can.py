import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import math


class DenseBlock(nn.Module):
    """
    Khối DenseNet cơ bản
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
    Lớp chuyển tiếp giữa các khối DenseBlock
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
    Backbone DenseNet cho CAN
    """
    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16), num_init_features=64):
        super(DenseNetBackbone, self).__init__()
        
        # Lớp đầu tiên
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
        
        # Các xử lý cuối cùng
        self.features.add_module('norm5', nn.BatchNorm2d(num_features))
        self.features.add_module('relu5', nn.ReLU(inplace=True))
        
        self.out_channels = num_features  # 684 (với cấu hình mặc định)
    
    def forward(self, x):
        return self.features(x)


class ChannelAttention(nn.Module):
    """
    Cơ chế attention theo kênh
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
        
        # Nhánh 1: Kernel 3x3
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.attention1 = ChannelAttention(256)
        
        # Nhánh 2: Kernel 5x5
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=5, padding=2),
            nn.ReLU(inplace=True)
        )
        self.attention2 = ChannelAttention(256)
        
        # Lớp Conv 1x1 để giảm kênh và tạo bản đồ đếm
        self.conv_reduce = nn.Conv2d(512, num_classes, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # Xử lý nhánh 1
        out1 = self.branch1(x)
        out1 = out1 * self.attention1(out1)
        
        # Xử lý nhánh 2
        out2 = self.branch2(x)
        out2 = out2 * self.attention2(out2)
        
        # Nối các đặc trưng từ hai nhánh
        concat_features = torch.cat([out1, out2], dim=1)  # Shape: B x 512 x H x W
        
        # Tạo bản đồ đếm
        count_map = self.sigmoid(self.conv_reduce(concat_features))  # Shape: B x C x H x W
        
        # Áp dụng sum-pooling để tạo vector đếm 1D
        # Sum trên toàn bộ bản đồ đặc trưng theo chiều cao và chiều rộng
        count_vector = torch.sum(count_map, dim=(2, 3))  # Shape: B x C
        
        return count_map, count_vector


class PositionalEncoding(nn.Module):
    """
    Positional encoding cho attention decoder
    """
    def __init__(self, d_model, max_seq_len=1024):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        
        # Tạo positional encoding matrix
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
        if h*w > self.pe.size(0):        #type: ignore
            # Dynamically extend positional encodings if needed
            device = self.pe.device
            extended_pe = torch.zeros(h*w, self.d_model, device=device)   #type: ignore
            position = torch.arange(0, h*w, dtype=torch.float, device=device).unsqueeze(1)           #type: ignore
            div_term = torch.exp(torch.arange(0, self.d_model, 2, device=device).float() * (-math.log(10000.0) / self.d_model))        #type: ignore
            
            extended_pe[:, 0::2] = torch.sin(position * div_term)
            extended_pe[:, 1::2] = torch.cos(position * div_term)
            
            pos_encoding = extended_pe.view(h, w, -1)
        else:
            # Use pre-computed positional encodings
            pos_encoding = self.pe[:h*w].view(h, w, -1)       #type: ignore
              
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
        
        # Lớp đầu vào để thu hẹp bản đồ đặc trưng
        self.feature_proj = nn.Conv2d(input_channels, hidden_size * 2, kernel_size=1)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(hidden_size * 2)
        
        # Embedding layer cho các ký hiệu đầu ra
        self.embedding = nn.Embedding(num_classes, embedding_dim)
        
        # GRU cell
        self.gru = nn.GRUCell(embedding_dim + hidden_size + num_classes, hidden_size)
        
        # Attention
        self.attention_w = nn.Linear(hidden_size * 2, hidden_size)
        self.attention_v = nn.Linear(hidden_size, 1)
        if use_coverage:
            self.coverage_proj = nn.Linear(1, hidden_size)
        
        # Lớp đầu ra
        self.out = nn.Linear(hidden_size + hidden_size + num_classes, num_classes)
    
    def forward(self, feature_map, count_vector, target=None, teacher_forcing_ratio=0.5, max_len=200):
        batch_size = feature_map.size(0)
        device = feature_map.device
        
        # Biến đổi bản đồ đặc trưng
        projected_features = self.feature_proj(feature_map)  # B x 2*hidden_size x H x W
        H, W = projected_features.size(2), projected_features.size(3)
        
        # Chuyển đổi bản đồ đặc trưng thành B x H*W x 2*hidden_size
        projected_features = projected_features.permute(0, 2, 3, 1).contiguous()  # B x H x W x 2*hidden_size
        
        # Thêm positional encoding
        pos_encoding = self.pos_encoder(projected_features)  # B x H x W x 2*hidden_size
        projected_features = projected_features + pos_encoding
        
        # Reshape để xử lý attention
        projected_features = projected_features.view(batch_size, H*W, -1)  # B x H*W x 2*hidden_size
        
        # Khởi tạo trạng thái ẩn ban đầu
        h_t = torch.zeros(batch_size, self.hidden_size, device=device)
        
        # Khởi tạo coverage attention nếu sử dụng
        if self.use_coverage:
            coverage = torch.zeros(batch_size, H*W, 1, device=device)
        
        # Token <SOS> đầu tiên
        y_t_1 = torch.ones(batch_size, dtype=torch.long, device=device)
        
        # Chuẩn bị target sequence nếu được cung cấp
        if target is not None:
            max_len = target.size(1)
        
        # Mảng để lưu các dự đoán
        outputs = torch.zeros(batch_size, max_len, self.embedding.num_embeddings, device=device)
        
        for t in range(max_len):
            # Áp dụng embedding cho ký hiệu trước đó
            embedded = self.embedding(y_t_1)  # B x embedding_dim
            
            # Tính toán attention
            attention_input = self.attention_w(projected_features)  # B x H*W x hidden_size
            
            # Thêm coverage attention nếu sử dụng
            if self.use_coverage:
                coverage_input = self.coverage_proj(coverage.float())  # B x H*W x hidden_size        #type:ignore
                attention_input = attention_input + coverage_input
            
            # Thêm trạng thái ẩn vào attention
            h_expanded = h_t.unsqueeze(1).expand(-1, H*W, -1)  # B x H*W x hidden_size
            attention_input = torch.tanh(attention_input + h_expanded)
            
            # Tính attention weights
            e_t = self.attention_v(attention_input).squeeze(-1)  # B x H*W
            alpha_t = F.softmax(e_t, dim=1)  # B x H*W
            
            # Cập nhật coverage nếu sử dụng
            if self.use_coverage:
                coverage = coverage + alpha_t.unsqueeze(-1)        #type: ignore
            
            # Tính context vector
            alpha_t = alpha_t.unsqueeze(1)  # B x 1 x H*W
            context = torch.bmm(alpha_t, projected_features).squeeze(1)  # B x 2*hidden_size
            context = context[:, :self.hidden_size]  # Lấy nửa đầu làm context vector
            
            # Kết hợp embedding, context vector và count vector
            gru_input = torch.cat([embedded, context, count_vector], dim=1)
            
            # Cập nhật trạng thái ẩn
            h_t = self.gru(gru_input, h_t)
            
            # Dự đoán ký hiệu đầu ra
            output = self.out(torch.cat([h_t, context, count_vector], dim=1))
            outputs[:, t] = output
            
            # Quyết định ký hiệu đầu vào tiếp theo
            if target is not None and torch.rand(1).item() < teacher_forcing_ratio:
                y_t_1 = target[:, t]
            else:
                # Greedy decoding
                _, y_t_1 = output.max(1)
        
        return outputs


class CAN(nn.Module):
    """
    Counting-Aware Network cho nhận dạng biểu thức toán học viết tay
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
    
    def forward(self, x, target=None, teacher_forcing_ratio=0.5):
        # Trích xuất đặc trưng từ backbone
        features = self.backbone(x)
        
        # Tính toán count map và count vector từ MSCM
        count_map, count_vector = self.mscm(features)
        
        # Decoding với CCAD
        outputs = self.decoder(features, count_vector, target, teacher_forcing_ratio)
        
        return outputs, count_vector
    
    def calculate_loss(self, outputs, targets, count_vectors, count_targets, lambda_count=0.01):
        """
        Tính toán tổng hợp loss function cho CAN
        
        Args:
            outputs: Dự đoán chuỗi đầu ra từ decoder
            targets: Chuỗi mục tiêu thực tế
            count_vectors: Vector đếm dự đoán
            count_targets: Vector đếm mục tiêu thực tế
            lambda_count: Trọng số cho loss đếm
        
        Returns:
            Tổng loss: L = L_cls + λ * L_counting
        """
        # Loss cho decoder (cross entropy)
        L_cls = F.cross_entropy(outputs.view(-1, outputs.size(-1)), targets.view(-1))
        
        # Loss cho đếm (MSE)
        L_counting = F.mse_loss(count_vectors, count_targets)
        
        # Tổng loss
        total_loss = L_cls + lambda_count * L_counting
        
        return total_loss, L_cls, L_counting


# Sử dụng mô hình
def create_can_model(num_classes, pretrained_backbone=False):
    """
    Tạo mô hình CAN
    
    Args:
        num_classes: Số lượng lớp ký hiệu
        pretrained_backbone: Có sử dụng pretrained backbone hay không
    
    Returns:
        Mô hình CAN
    """
    # Tạo backbone
    if pretrained_backbone:
        backbone = models.densenet121(pretrained=True)
        # Điều chỉnh lớp đầu vào cho ảnh đầu vào 1 kênh
        backbone.features.conv0 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    else:
        backbone = DenseNetBackbone()
    
    # Tạo mô hình
    model = CAN(
        num_classes=num_classes,
        backbone=backbone,
        hidden_size=256,
        embedding_dim=256,
        use_coverage=True
    )
    
    return model


# Ví dụ sử dụng
if __name__ == "__main__":
    # Tạo mô hình CAN với 101 lớp ký hiệu (ví dụ)
    num_classes = 101  # Số lượng lớp ký hiệu + token đặc biệt như <SOS>, <EOS>
    model = create_can_model(num_classes)
    
    # Tạo dữ liệu đầu vào giả lập
    batch_size = 4
    input_image = torch.randn(batch_size, 1, 128, 384)  # B x C x H x W
    target = torch.randint(0, num_classes, (batch_size, 50))  # B x max_len
    
    # Forward pass
    outputs, count_vectors = model(input_image, target)
    
    # In ra kích thước đầu ra
    print(f"Outputs shape: {outputs.shape}")  # B x max_len x num_classes
    print(f"Count vectors shape: {count_vectors.shape}")  # B x num_classes