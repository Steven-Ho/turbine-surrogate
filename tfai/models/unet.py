import torch
import torch.nn as nn
import torch.nn.functional as F
# Imports `Dict` from `typing` for use in this module.
from typing import Dict
# from thop import profile as thop_profile

class UNet(nn.Module):
    def __init__(self, in_channels=1, n_classes=1, img_size=(128, 128, 128), model_scale="base"):
        super().__init__()
        # Stores the `model_scale` on the instance so other methods can reuse it.
        self.model_scale = model_scale
        # 自适应padding
        self.img_size_padded = adaptive_pad(img_size)
        # 根据model_scale设置相应的参数
        if model_scale == "small":
            base_c = 32
        # Checks `model_scale == "base"` before entering this conditional branch.
        elif model_scale == "base":
            base_c = 64
        elif model_scale == "large":
            base_c = 128
        # Continues the surrounding statement or collection definition.
        else:
            raise ValueError(f"Invalid model_scale: {model_scale}. Expected 'small', 'base', or 'large'.")
        # 编码器部分：逐步下采样和特征提取
        self.enc1 = self.conv_block(in_channels, base_c)      # 第一层编码
        self.enc2 = self.conv_block(base_c, base_c*2)         # 第二层编码
        # Stores the `enc3` on the instance so other methods can reuse it.
        self.enc3 = self.conv_block(base_c*2, base_c*4)       # 第三层编码
        self.enc4 = self.conv_block(base_c*4, base_c*8)       # 第四层编码
        self.pool = nn.MaxPool3d(2)                           # 下采样操作，空间尺寸减半
        # 中心部分：最深层特征提取
        self.center = self.conv_block(base_c*8, base_c*16)    # 中心层
        # 解码器部分：逐步上采样和特征融合
        # Stores the `up4` on the instance so other methods can reuse it.
        self.up4 = nn.ConvTranspose3d(base_c*16, base_c*8, 2, stride=2) # 上采样
        self.dec4 = self.conv_block(base_c*16, base_c*8)                 # 解码，融合中心和编码器第四层特征
        self.up3 = nn.ConvTranspose3d(base_c*8, base_c*4, 2, stride=2)
        self.dec3 = self.conv_block(base_c*8, base_c*4)
        # Stores the `up2` on the instance so other methods can reuse it.
        self.up2 = nn.ConvTranspose3d(base_c*4, base_c*2, 2, stride=2)
        self.dec2 = self.conv_block(base_c*4, base_c*2)
        self.up1 = nn.ConvTranspose3d(base_c*2, base_c, 2, stride=2)
        self.dec1 = self.conv_block(base_c*2, base_c)
        # 最终输出层：1x1卷积映射为类别数或回归通道
        # Stores the `final` on the instance so other methods can reuse it.
        self.final = nn.Conv3d(base_c, n_classes, 1)

    def conv_block(self, in_c, out_c):
        # 两次卷积+BN+ReLU，提升特征表达能力
        return nn.Sequential(
            nn.Conv3d(in_c, out_c, 3, padding=1),
            # Completes the multi-line `nn.BatchNorm3d` arguments or elements.
            nn.BatchNorm3d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_c, out_c, 3, padding=1),
            nn.BatchNorm3d(out_c),
            # Completes the multi-line `nn.ReLU` arguments or elements.
            nn.ReLU(inplace=True)
        )

    def center_crop_or_pad(self, enc_feat, target_shape):
        # 跳跃连接时，裁剪或填充特征图以匹配目标尺寸
        # Support both 2D feature maps (4D tensor) and 3D feature maps (5D tensor)
        if enc_feat.dim() == 4:
            # N, C, H, W
            # Stores the `(_, _, h, w)` for use by the remaining steps in this flow.
            _, _, h, w = enc_feat.shape
            th, tw = target_shape
            dh = h - th
            dw = w - tw
            # crop：如果特征图比目标大，则居中裁剪
            # Checks `dh > 0` before entering this conditional branch.
            if dh > 0:
                enc_feat = enc_feat[:, :, dh//2:dh//2+th, :]
            if dw > 0:
                enc_feat = enc_feat[:, :, :, dw//2:dw//2+tw]
            # pad：如果特征图比目标小，则填充
            # Checks `dh < 0 or dw < 0` before entering this conditional branch.
            if dh < 0 or dw < 0:
                pad = [0, max(0, -dw), 0, max(0, -dh)]  # left, right, top, bottom
                enc_feat = F.pad(enc_feat, pad)
            return enc_feat

        # Checks `enc_feat.dim() == 5` before entering this conditional branch.
        elif enc_feat.dim() == 5:
            # N, C, D, H, W
            _, _, d, h, w = enc_feat.shape
            td, th, tw = target_shape
            dd = d - td
            # Derives the `dh` from the current values for downstream processing.
            dh = h - th
            dw = w - tw
            # crop：如果特征图比目标大，则居中裁剪
            if dd > 0:
                enc_feat = enc_feat[:, :, dd//2:dd//2+td, :, :]
            # Checks `dh > 0` before entering this conditional branch.
            if dh > 0:
                enc_feat = enc_feat[:, :, :, dh//2:dh//2+th, :]
            if dw > 0:
                enc_feat = enc_feat[:, :, :, :, dw//2:dw//2+tw]
            # pad：如果特征图比目标小，则填充
            # Checks `dd < 0 or dh < 0 or dw < 0` before entering this conditional branch.
            if dd < 0 or dh < 0 or dw < 0:
                # F.pad pad format: (W_left, W_right, H_left, H_right, D_left, D_right)
                pad = [
                    0, max(0, -dw),
                    0, max(0, -dh),
                    # Completes the multi-line `max` arguments or elements.
                    0, max(0, -dd)
                ]
                enc_feat = F.pad(enc_feat, pad)
            return enc_feat

        # Continues the surrounding statement or collection definition.
        else:
            # Unexpected dimensionality; return as-is
            return enc_feat

    def forward(self, x):
        input_shape = x.shape[2:]  # 原始输入尺寸，用于最后插值
        # 自适应padding
        # Stores the `x` for use by the remaining steps in this flow.
        x = pad_to_shape(x, self.img_size_padded)
        # 编码器：逐步提取特征并下采样
        e1 = self.enc1(x)                  # 第一层编码
        e2 = self.enc2(self.pool(e1))      # 第二层编码
        e3 = self.enc3(self.pool(e2))      # 第三层编码
        # Stores the `e4` for use by the remaining steps in this flow.
        e4 = self.enc4(self.pool(e3))      # 第四层编码
        # 中心部分
        center = self.center(self.pool(e4))# 最深层特征
        # 解码器 + 跳跃连接，逐步上采样并融合编码器特征
        d4 = self.up4(center)
        e4_adj = self.center_crop_or_pad(e4, d4.shape[2:])  # 尺寸对齐
        # Stores the `d4` for use by the remaining steps in this flow.
        d4 = torch.cat([d4, e4_adj], dim=1)
        d4 = self.dec4(d4)
        d3 = self.up3(d4)
        e3_adj = self.center_crop_or_pad(e3, d3.shape[2:])
        # Stores the `d3` for use by the remaining steps in this flow.
        d3 = torch.cat([d3, e3_adj], dim=1)
        d3 = self.dec3(d3)
        d2 = self.up2(d3)
        e2_adj = self.center_crop_or_pad(e2, d2.shape[2:])
        # Stores the `d2` for use by the remaining steps in this flow.
        d2 = torch.cat([d2, e2_adj], dim=1)
        d2 = self.dec2(d2)
        d1 = self.up1(d2)
        e1_adj = self.center_crop_or_pad(e1, d1.shape[2:])
        # Stores the `d1` for use by the remaining steps in this flow.
        d1 = torch.cat([d1, e1_adj], dim=1)
        d1 = self.dec1(d1)
        # 输出层
        out = self.final(d1)
        #插值回padding尺寸
        out = F.interpolate(out, size=self.img_size_padded, mode='trilinear', align_corners=True)
        # 裁剪回原始输入尺寸
        # Stores the `out` for use by the remaining steps in this flow.
        out = crop_to_original(out, input_shape)
        return out
    


def adaptive_pad(img_size):
    """
    保证输入尺寸至少为16
    Args:
        img_size: tuple(int, int, int) - (D, H, W)
    Returns:
        img_size_padded: tuple(int, int, int) - (pD, pH, pW)
    """
    
    D, H, W = img_size
    # Initializes the `dims` as a collection for values from this processing stage.
    dims = [D, H, W]
    img_size_padded = []
    for dim in dims:
        if dim < 16:
            # Stores the `padded` for use by the remaining steps in this flow.
            padded = 16
        else:
            padded = dim
        
        img_size_padded.append(padded)
    
    # Returns `tuple(img_size_padded)` to the caller for the next stage.
    return tuple(img_size_padded)

def pad_to_shape(t, target):
        _, _, d, h, w = t.shape
        td, th, tw = target
        # Validates `d > td or h > th or w > tw` and raises `ValueError` when the input cannot be processed safely.
        if d > td or h > th or w > tw:
            raise ValueError(f"输入体尺寸 {t.shape} 大于目标尺寸 {target}")
        pad = (0, tw - w, 0, th - h, 0, td - d)
        return F.pad(t, pad)

# 将模型输出裁剪回原始输入的空间尺寸。
def crop_to_original(output, original_shape):
    """
    将模型输出裁剪回原始输入的空间尺寸。
    
    Args:
        output: torch.Tensor, shape (B, C, D_pad, H_pad, W_pad) - 模型输出（padding后的尺寸）
        original_shape: tuple(int, int, int) - 原始输入的空间尺寸 (D, H, W)
        
    Returns:
        output_cropped: torch.Tensor, shape (B, C, D, H, W) - 裁剪后的输出
    """
    D, H, W = original_shape
    return output[..., :D, :H, :W]

'''if __name__ == "__main__":
    print("\n=== 测试自适应UNet ===")
    
    # 测试不同尺寸的输入
    test_shapes = [(200,60,32),(201,61,1)]

    for shape in test_shapes:
        
        x_test = torch.randn(1, 6, *shape)
        model = UNet(in_channels=6,img_size=shape,n_classes=3,model_scale="small")
        #print(model) 
        print(f"Testing shape: {shape}")
        with torch.no_grad():
            y_test = model(x_test)
            macs, thop_params = thop_profile(model, inputs=(x_test,), verbose=False)
        print(f"Input: {x_test.shape}, Output: {y_test.shape}")
        print(f"FLOPs: {2*macs/1e9:.1f} G, Params: {thop_params/1e6:.1f}M")
        del model'''