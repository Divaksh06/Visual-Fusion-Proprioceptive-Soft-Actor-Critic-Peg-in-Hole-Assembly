"""Cross-Modal Attention Module"""

import torch
import torch.nn as nn

class CrossModalAttention(nn.Module):
    """Bidirectional cross-modal attention between vision and force"""
    def __init__(self, feature_dim=128, num_heads=4, dropout=0.1):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        
        self.v2f_attention = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        self.f2v_attention = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        self.norm_v = nn.LayerNorm(feature_dim)
        self.norm_f = nn.LayerNorm(feature_dim)
        
        self.ffn = nn.Sequential(
            nn.Linear(feature_dim * 2, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, feature_dim * 2),
        )
        
        self.final_norm = nn.LayerNorm(feature_dim * 2)
    
    def forward(self, vision_features, force_features):
        """
        Args:
            vision_features: [batch, feature_dim]
            force_features: [batch, feature_dim]
        Returns:
            fused_features: [batch, feature_dim * 2]
            attention_weights: dict
        """
        φ_v = vision_features.unsqueeze(1)
        φ_f = force_features.unsqueeze(1)
        
        v2f_out, v2f_weights = self.v2f_attention(
            query=φ_v, key=φ_f, value=φ_f
        )
        φ_v_enhanced = self.norm_v(φ_v + v2f_out)
        
        f2v_out, f2v_weights = self.f2v_attention(
            query=φ_f, key=φ_v, value=φ_v
        )
        φ_f_enhanced = self.norm_f(φ_f + f2v_out)
        
        φ_fused = torch.cat([
            φ_v_enhanced.squeeze(1),
            φ_f_enhanced.squeeze(1)
        ], dim=-1)
        
        φ_out = self.final_norm(φ_fused + self.ffn(φ_fused))
        
        attention_weights = {
            'v2f': v2f_weights.squeeze(1),
            'f2v': f2v_weights.squeeze(1)
        }
        
        return φ_out, attention_weights
