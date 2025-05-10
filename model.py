from transformers import AutoModel, AutoConfig
import torch
import torch.nn as nn
import torch.nn.functional as F

class EnhancedLightAttentionModel(nn.Module):
    def __init__(self, pretrained_model_name, num_classes, dropout_rate=0.3):
        super(EnhancedLightAttentionModel, self).__init__()
        
        # 加载预训练模型
        self.roberta = AutoModel.from_pretrained(pretrained_model_name)
        self.hidden_size = self.roberta.config.hidden_size
        
        # 注意力机制用于关注重要词
        self.attention = nn.Linear(self.hidden_size, 1)
        
        # 添加注意力缩放因子，使注意力分布更加sharp
        self.attention_scale = nn.Parameter(torch.tensor(1.0))
        
        # 添加Layer Normalization层
        self.layer_norm = nn.LayerNorm(self.hidden_size)
        
        # 分类器前的MLP
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dense = nn.Linear(self.hidden_size, self.hidden_size)
        self.activation = nn.GELU()  # 使用GELU激活函数
        self.dropout2 = nn.Dropout(dropout_rate * 0.5)  # 第二个dropout率略低
        
        # 分类器
        self.classifier = nn.Linear(self.hidden_size, num_classes)
        
        # 存储注意力权重以便可视化
        self.attention_weights = None
        
    def forward(self, input_ids, attention_mask=None, token_type_ids=None, output_attentions=False):
        outputs = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        
        sequence_output = outputs.last_hidden_state
        cls_output = sequence_output[:, 0, :]  # 获取[CLS]表示用于残差连接
        
        # 应用注意力权重
        attention_scores = self.attention(sequence_output)
        attention_scores = attention_scores.squeeze(-1)
        
        # 使用可学习的缩放因子调整注意力分布的锐度
        attention_scores = attention_scores * self.attention_scale
        
        # 掩码处理
        if attention_mask is not None:
            attention_scores = attention_scores.masked_fill(attention_mask == 0, -1e9)
        
        attention_weights = F.softmax(attention_scores, dim=1)
        
        # 存储注意力权重以便后续可视化
        if output_attentions:
            self.attention_weights = attention_weights
        
        # 加权平均
        weighted_output = torch.bmm(attention_weights.unsqueeze(1), sequence_output)
        weighted_output = weighted_output.squeeze(1)
        
        # 添加残差连接 - 融合[CLS]和注意力表示
        fused_output = weighted_output + cls_output
        
        # 应用Layer Normalization
        normalized_output = self.layer_norm(fused_output)
        
        # MLP层
        x = self.dropout1(normalized_output)
        x = self.dense(x)
        x = self.activation(x)
        x = self.dropout2(x)
        
        # 分类
        logits = self.classifier(x)
        
        if output_attentions:
            return logits, self.attention_weights
        return logits