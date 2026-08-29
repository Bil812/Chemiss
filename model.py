# -*- coding: utf-8 -*-
"""
model.py
========
化学棋 AlphaZero 风格神经网络：策略 + 价值双头网络。

输入:  [17, 8, 8] 棋盘特征张量
输出:  策略 logits [ACTION_SPACE_SIZE]  +  价值标量 [-1, 1]（执子方胜率）

结构:  输入卷积(17->64) + 3×残差块(64 通道) + 策略头 + 价值头。
轻量级，适配 3060 Laptop 6GB。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from chemis_env import ACTION_SPACE_SIZE
import numpy as np


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        res = F.relu(self.bn1(self.conv1(x)))
        res = self.bn2(self.conv2(res))
        return F.relu(x + res)


class ChemisNet(nn.Module):
    def __init__(self, action_space_size=ACTION_SPACE_SIZE, num_channels=64, num_res_blocks=3):
        super().__init__()
        self.action_space_size = action_space_size
        self.num_channels = num_channels

        self.conv_input = nn.Conv2d(17, num_channels, kernel_size=3, padding=1, bias=False)
        self.bn_input = nn.BatchNorm2d(num_channels)

        self.res_blocks = nn.Sequential(*[ResidualBlock(num_channels) for _ in range(num_res_blocks)])

        # 策略头
        self.policy_conv = nn.Conv2d(num_channels, 2, kernel_size=1, bias=False)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * 8 * 8, action_space_size)

        # 价值头
        self.value_conv = nn.Conv2d(num_channels, 1, kernel_size=1, bias=False)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(1 * 8 * 8, 128)
        self.value_fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.bn_input(self.conv_input(x)))
        x = self.res_blocks(x)

        # 策略头
        p = F.relu(self.policy_bn(self.policy_conv(x)))
        p = p.view(p.size(0), -1)
        p = self.policy_fc(p)

        # 价值头
        v = F.relu(self.value_bn(self.value_conv(x)))
        v = v.view(v.size(0), -1)
        v = F.relu(self.value_fc1(v))
        v = torch.tanh(self.value_fc2(v))
        return p, v


# ---------------------------------------------------------------------------
# 导出权重（扁平化 JSON 数组），供 HTML 端纯 JS 推理重建
# ---------------------------------------------------------------------------
# PyTorch state_dict() 的迭代顺序是固定的，按注册顺序：
#   0  conv_input.weight        [64, 17, 3, 3]
#   1  bn_input.weight          [64]
#   2  bn_input.bias            [64]
#   3  bn_input.running_mean    [64]
#   4  bn_input.running_var     [64]
#   5  res_blocks.0.conv1.weight ...
#   ... 每个残差块 4 个 conv weight + 4 组 bn 参数 ...
#   之后策略头 / 价值头。
# 为便于纯 JS 重建，导出时每层保留 shape，最终 JSON 为扁平数组。
import json


def layer_shapes(net: ChemisNet):
    """返回 state_dict 顺序对应的 (name, shape) 列表。"""
    return [(k, tuple(v.shape)) for k, v in net.state_dict().items()]


def export_weights_flat(net: ChemisNet):
    """返回扁平化后的 (flat_list, shapes) 字节列表。

    weights: 按 state_dict 顺序拼接的 float 列表。
    shapes:  与 net.state_dict() 顺序一致，每项 [numel, shape]。
    """
    flat = []
    shapes = []
    for k, v in net.state_dict().items():
        if k.endswith('num_batches_tracked'):
            continue  # 推理不需要，剔除以保持 JS 重建精简
        shape = tuple(v.shape)
        numel = v.numel()
        flat.extend([float(x) for x in v.flatten().tolist()])
        shapes.append({'name': k, 'shape': shape, 'numel': numel})
    return flat, shapes


def save_model_weights(net: ChemisNet, path='model_weights.json'):
    flat, shapes = export_weights_flat(net)
    data = {
        'action_space_size': net.action_space_size,
        'input_channels': 17,
        'board_size': 8,
        'num_channels': net.num_channels,
        'num_res_blocks': 3,
        'shapes': shapes,
        'flat': flat,
    }
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, separators=(',', ':'))
    return path


def export_weights_b64(net: ChemisNet):
    """权重扁平化为 base64 的 float32 字节（比 JSON 小数小很多），配 shapes。"""
    import base64 as _b64
    flat, shapes = export_weights_flat(net)
    arr = np.array(flat, dtype=np.float32)
    b64 = _b64.b64encode(arr.tobytes()).decode('ascii')
    return {
        'action_space_size': net.action_space_size,
        'input_channels': 17,
        'board_size': 8,
        'num_channels': net.num_channels,
        'num_res_blocks': 3,
        'shapes': shapes,
        'flat_b64': b64,
        'flat_len': arr.size,
    }


def save_model_weights_js(net: ChemisNet, path='model_weights.js'):
    """导出成 window.__CHEMISS_ML_JSON__ = {...} 的 JS 文件，配合 <script src>。
    这样用 file:// 直接打开页面也能无后端加载权重（不再是 fetch，不触发 CORS）。"""
    data = export_weights_b64(net)
    body = json.dumps(data, separators=(',', ':'))
    with open(path, 'w', encoding='utf-8') as f:
        f.write('window.__CHEMISS_ML_JSON__ = ' + body + ';\n')
    return path
