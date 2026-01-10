import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class CriticalValueAwareLoss(nn.Module):
    """
    针对危急值检测的复合损失函数
    重点惩罚将危急值误判为非危急值的情况（假阴性）
    """
    def __init__(self, 
                 pos_weight=5.0,  # 正样本（危急值）权重
                 gamma=2.0,       # Focal Loss 的 gamma 参数
                 alpha=0.75,      # 正样本的权重系数
                 penalty_factor=10.0,  # 假阴性的额外惩罚因子
                 smooth_epsilon=1e-6,  # 数值稳定性的epsilon
                 use_focal=True,
                 use_penalty=True):
        super(CriticalValueAwareLoss, self).__init__()
        
        # 基础交叉熵损失（带类别权重）
        self.base_weight = torch.tensor([1.0, pos_weight])  # [负类权重, 正类权重]
        self.base_loss = nn.CrossEntropyLoss(weight=self.base_weight)
        
        # Focal Loss 参数
        self.gamma = gamma
        self.alpha = alpha
        self.use_focal = use_focal
        
        # 假阴性惩罚参数
        self.penalty_factor = penalty_factor
        self.use_penalty = use_penalty
        
        self.smooth_epsilon = smooth_epsilon
        
    def focal_loss(self, inputs, targets):
        """
        Focal Loss：降低易分类样本的权重，聚焦于难分类样本
        """
        # 获取预测概率
        probabilities = F.softmax(inputs, dim=1)
        
        # 获取目标类别的概率
        targets_one_hot = F.one_hot(targets, num_classes=inputs.size(1)).float()
        pt = torch.sum(probabilities * targets_one_hot, dim=1)
        
        # 计算focal weight
        focal_weight = (1 - pt) ** self.gamma
        
        # 计算加权交叉熵
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        focal_loss = self.alpha * focal_weight * ce_loss
        
        return focal_loss.mean()
    
    def false_negative_penalty(self, inputs, targets):
        """
        专门惩罚假阴性（危急值被预测为非危急值）
        """
        batch_size = inputs.size(0)
        
        # 获取预测结果
        predictions = torch.argmax(inputs, dim=1)
        
        # 创建掩码：真实标签为危急值（1）但预测为非危急值（0）
        fn_mask = (targets == 1) & (predictions == 0)
        
        if not torch.any(fn_mask):
            return torch.tensor(0.0, device=inputs.device)
        
        # 获取假阴性样本的预测置信度
        fn_probs = F.softmax(inputs[fn_mask], dim=1)
        
        # 计算假阴性惩罚：
        # 1. 正类（危急值）的预测概率
        positive_probs = fn_probs[:, 1]
        
        # 2. 计算惩罚项：正类概率越低，惩罚越大
        # 使用指数惩罚：当正类概率接近0时，惩罚非常大
        penalty = torch.exp(-self.penalty_factor * positive_probs)
        
        return penalty.mean()
    
    def asymmetric_loss(self, inputs, targets):
        """
        非对称损失：对假阴性和假阳性给予不同惩罚
        """
        probabilities = F.softmax(inputs, dim=1)
        
        # 获取正类和负类的概率
        positive_prob = probabilities[:, 1]  # 危急值的概率
        negative_prob = probabilities[:, 0]  # 非危急值的概率
        
        # 创建掩码
        positive_mask = (targets == 1).float()
        negative_mask = (targets == 0).float()
        
        # 不同情况的不同惩罚系数
        # 假阴性：危急值被预测为非危急值 - 高惩罚
        fn_loss = -torch.log(positive_prob + self.smooth_epsilon) * positive_mask
        
        # 假阳性：非危急值被预测为危急值 - 中等惩罚
        fp_loss = -torch.log(negative_prob + self.smooth_epsilon) * negative_mask * 0.5
        
        # 真阳性：正确预测为危急值 - 正常
        tp_loss = -torch.log(positive_prob + self.smooth_epsilon) * positive_mask * 0.1
        
        # 真阴性：正确预测为非危急值 - 正常
        tn_loss = -torch.log(negative_prob + self.smooth_epsilon) * negative_mask * 0.1
        
        # 组合损失
        loss = fn_loss * 5.0 + fp_loss * 2.0 + tp_loss + tn_loss
        
        return loss.mean()
    
    def confidence_aware_loss(self, inputs, targets):
        """
        置信度感知损失：对低置信度的危急值预测给予额外关注
        """
        probabilities = F.softmax(inputs, dim=1)
        positive_prob = probabilities[:, 1]
        
        # 获取正类样本
        positive_mask = (targets == 1)
        
        if not torch.any(positive_mask):
            return torch.tensor(0.0, device=inputs.device)
        
        positive_probs = positive_prob[positive_mask]
        
        # 计算置信度权重：置信度越低，权重越高
        confidence_weights = 1.0 / (positive_probs + self.smooth_epsilon)
        confidence_weights = torch.clamp(confidence_weights, 1.0, 10.0)
        
        # 计算加权的交叉熵损失
        ce_loss = F.cross_entropy(inputs[positive_mask], targets[positive_mask], reduction='none')
        weighted_loss = confidence_weights * ce_loss
        
        return weighted_loss.mean()
    
    def forward(self, inputs, targets):
        """
        组合损失函数
        """
        # 基础损失
        if self.use_focal:
            base_loss_value = self.focal_loss(inputs, targets)
        else:
            base_loss_value = self.base_loss(inputs, targets)
        
        total_loss = base_loss_value
        
        # 添加假阴性惩罚
        if self.use_penalty:
            fn_penalty = self.false_negative_penalty(inputs, targets)
            total_loss = total_loss + fn_penalty
        
        # 添加非对称损失（可选）
        asymmetric_loss_value = self.asymmetric_loss(inputs, targets)
        total_loss = total_loss + asymmetric_loss_value * 0.3
        
        # 添加置信度感知损失（可选）
        confidence_loss_value = self.confidence_aware_loss(inputs, targets)
        total_loss = total_loss + confidence_loss_value * 0.2
        
        return total_loss


class CriticalValueLossWithMetrics(CriticalValueAwareLoss):
    """
    带指标监控的损失函数，可以记录假阴性率等关键指标
    """
    def __init__(self, **kwargs):
        super(CriticalValueLossWithMetrics, self).__init__(**kwargs)
        
    def forward_with_metrics(self, inputs, targets):
        """
        返回损失值以及关键指标
        """
        # 计算损失
        loss = self.forward(inputs, targets)
        
        # 计算预测结果
        probabilities = F.softmax(inputs, dim=1)
        predictions = torch.argmax(probabilities, dim=1)
        
        # 计算各种指标
        metrics = {}
        
        # 假阴性（False Negative）：实际是危急值但预测为非危急值
        fn_mask = (targets == 1) & (predictions == 0)
        metrics['false_negative'] = torch.sum(fn_mask).item()
        
        # 真阳性（True Positive）：正确预测为危急值
        tp_mask = (targets == 1) & (predictions == 1)
        metrics['true_positive'] = torch.sum(tp_mask).item()
        
        # 假阳性（False Positive）：实际是非危急值但预测为危急值
        fp_mask = (targets == 0) & (predictions == 1)
        metrics['false_positive'] = torch.sum(fp_mask).item()
        
        # 真阴性（True Negative）：正确预测为非危急值
        tn_mask = (targets == 0) & (predictions == 0)
        metrics['true_negative'] = torch.sum(tn_mask).item()
        
        # 计算假阴性率（False Negative Rate）
        total_positive = torch.sum(targets == 1).item()
        if total_positive > 0:
            metrics['false_negative_rate'] = metrics['false_negative'] / total_positive
        else:
            metrics['false_negative_rate'] = 0.0
        
        # 计算危急值的平均置信度
        positive_mask = (targets == 1)
        if torch.any(positive_mask):
            positive_probs = probabilities[positive_mask, 1]
            metrics['avg_positive_confidence'] = torch.mean(positive_probs).item()
        else:
            metrics['avg_positive_confidence'] = 0.0
        
        # 计算假阴性样本的平均置信度
        if metrics['false_negative'] > 0:
            fn_probs = probabilities[fn_mask, 1]
            metrics['avg_fn_confidence'] = torch.mean(fn_probs).item()
        else:
            metrics['avg_fn_confidence'] = 0.0
        
        return loss, metrics


# 使用示例
def create_critical_value_loss(pos_weight=5.0, use_focal=True, use_penalty=True):
    """
    创建针对危急值检测的损失函数
    """
    return CriticalValueLossWithMetrics(
        pos_weight=pos_weight,
        gamma=2.0,
        alpha=0.75,
        penalty_factor=10.0,
        use_focal=use_focal,
        use_penalty=use_penalty
    )


# 训练时使用的包装函数
def train_epoch_with_critical_focus(model, dataloader, criterion, optimizer, device):
    """
    使用危急值关注损失进行训练
    """
    model.train()
    total_loss = 0.0
    total_fn = 0  # 假阴性计数
    total_tp = 0  # 真阳性计数
    
    for batch_idx, (data, targets) in enumerate(dataloader):
        data, targets = data.to(device), targets.to(device)
        
        # 前向传播
        outputs = model(data)
        
        # 计算损失和指标
        if isinstance(criterion, CriticalValueLossWithMetrics):
            loss, metrics = criterion.forward_with_metrics(outputs, targets)
            total_fn += metrics['false_negative']
            total_tp += metrics['true_positive']
        else:
            loss = criterion(outputs, targets)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # 打印假阴性信息（每50个batch）
        if batch_idx % 50 == 0 and isinstance(criterion, CriticalValueLossWithMetrics):
            print(f'Batch {batch_idx}: Loss = {loss.item():.4f}, '
                  f'FN = {metrics["false_negative"]}, '
                  f'TP = {metrics["true_positive"]}, '
                  f'FNR = {metrics["false_negative_rate"]:.4f}')
    
    avg_loss = total_loss / len(dataloader)
    fn_rate = total_fn / (total_fn + total_tp + 1e-10)  # 假阴性率
    
    return avg_loss, fn_rate


# 动态调整损失权重（可选）
class DynamicCriticalLoss(CriticalValueAwareLoss):
    """
    动态调整权重的损失函数，根据历史假阴性率调整惩罚
    """
    def __init__(self, **kwargs):
        super(DynamicCriticalLoss, self).__init__(**kwargs)
        self.fn_history = []
        self.window_size = 10
        
    def update_penalty(self, current_fn_rate):
        """
        根据当前假阴性率动态调整惩罚因子
        """
        self.fn_history.append(current_fn_rate)
        if len(self.fn_history) > self.window_size:
            self.fn_history.pop(0)
        
        # 计算平均假阴性率
        avg_fn_rate = np.mean(self.fn_history) if self.fn_history else 0.0
        
        # 动态调整惩罚因子：假阴性率越高，惩罚越大
        if avg_fn_rate > 0.2:  # 如果假阴性率超过20%
            self.penalty_factor = min(20.0, self.penalty_factor * 1.1)
        elif avg_fn_rate < 0.05:  # 如果假阴性率低于5%
            self.penalty_factor = max(5.0, self.penalty_factor * 0.9)
        
        return self.penalty_factor