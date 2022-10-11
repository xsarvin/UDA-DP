import torch.nn as nn
import torch
import torch.nn.functional as F


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)

    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


class Attention_2_branches(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.attn = None

    def forward(self, x1, x2, use_attn=True, Only_self_attention_branch=False):
        B, N, C = x1.shape
        if Only_self_attention_branch:
            qkv1 = self.qkv(x1).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q1, k1, v1 = qkv1[0], qkv1[1], qkv1[2]

            attn1 = (q1 @ k1.transpose(-2, -1)) * self.scale
            attn1 = attn1.softmax(dim=-1)
            self.attn = attn1
            attn1 = self.attn_drop(attn1)

            x1 = (attn1 @ v1) if use_attn else v1

            x1 = x1.transpose(1, 2).reshape(B, N, C)
            x1 = self.proj(x1)
            x1 = self.proj_drop(x1)
            x2 = None
        else:
            qkv = self.qkv(x1).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

            qkv2 = self.qkv(x2).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q2, k2, v2 = qkv2[0], qkv2[1], qkv2[2]  # make torchscript happy (cannot use tensor as tuple)

            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn2 = (q @ k2.transpose(-2, -1)) * self.scale

            attn = attn.softmax(dim=-1)
            attn2 = attn2.softmax(dim=-1)
            self.attn = attn
            attn = self.attn_drop(attn)
            attn2 = self.attn_drop(attn2)

            x1 = (attn @ v) if use_attn else v
            x2 = (attn2 @ v2) if use_attn else v2

            x1 = x1.transpose(1, 2).reshape(B, N, C)
            x1 = self.proj(x1)
            x1 = self.proj_drop(x1)

            x2 = x2.transpose(1, 2).reshape(B, N, C)
            x2 = self.proj(x2)
            x2 = self.proj_drop(x2)

        return x1, x2


class dual_channel_attention_branches(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention_2_branches(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x1, x2, use_cross=False, use_attn=True, domain_norm=False,
                Only_self_attention_branch=False):
        if Only_self_attention_branch:
            xa_attn, _  = self.attn(self.norm1(x1), None, Only_self_attention_branch=Only_self_attention_branch)
            xa = x1 + self.drop_path(xa_attn)
            xa = xa + self.drop_path(self.mlp(self.norm2(xa)))
            xb = None
        else:
            xa_attn, xb_attn = self.attn(self.norm1(x1), self.norm1(x2),
                                          Only_self_attention_branch=Only_self_attention_branch)
            xa = x1 + self.drop_path(xa_attn)
            xa = xa + self.drop_path(self.mlp(self.norm2(xa)))

            xb = x2 + self.drop_path(xb_attn)
            xb = xb + self.drop_path(self.mlp(self.norm2(xb)))

        return xa, xb


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class uda(nn.Module):
    def __init__(self, params, mlp_ratio=4, attn_drop_rate=0.,
                 drop_path_rate=0., qkv_bias=False, qk_scale=None, norm_layer=nn.LayerNorm):
        super(uda, self).__init__()
        self.embedding_dim = params['EMBED_DIM']
        self.embedding = nn.Embedding(params['DICT_SIZE'], params['EMBED_DIM'])
        self.relu_conv = nn.ReLU()

        self.pos_drop = nn.Dropout(p=params["drop_rate"])
        self.soft = nn.Softmax(dim=-1)
        self.num_classes = 2
        self.cls_token = nn.Parameter(torch.zeros(1, 1, params['EMBED_DIM']))
        self.norm = norm_layer(self.embedding_dim)
        self.sigmoid = nn.Sigmoid()
        self.BatchNorm = nn.BatchNorm1d(self.embedding_dim)
        self.BatchNorm.bias.requires_grad_(False)
        self.classifier = nn.Linear(self.embedding_dim, self.num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)
        self.BatchNorm.apply(weights_init_kaiming)
        self.max_length = params["max_token"]
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, params["depth"])]
        self.blocks = nn.ModuleList([
            dual_channel_attention_branches(
                dim=params["EMBED_DIM"], num_heads=params["num_head"], mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=params["drop_rate"], attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(params["depth"])])

    def forward(self, x, x2=None, domain_norm=False, state=None, Only_self_attention_branch=False):
        if not self.training:
            Only_self_attention_branch = True
        if state == "pretraining":
            B = x.shape[0]
            x = self.embedding(x[:, :self.max_length])
            cls_tokens = self.cls_token.expand(B, -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)
            x1 = self.pos_drop(x)
            for i, blk in enumerate(self.blocks):
                x1, x2 = blk(x1, x1, use_cross=False,
                             domain_norm=domain_norm,
                             Only_self_attention_branch=Only_self_attention_branch)

            x1 = self.norm(x1)
            global_feat1 = x1[:, 0]
            feat1 = self.BatchNorm(global_feat1)
            cls_score1 = self.soft(self.classifier(feat1))

            return (feat1, cls_score1), (None, None)

        else:
            if Only_self_attention_branch:

                x = self.embedding(x[:, :self.max_length])

                B = x.shape[0]
                cls_tokens = self.cls_token.expand(B, -1, -1)
                x = torch.cat((cls_tokens, x), dim=1)

                x1 = self.pos_drop(x)

                for i, blk in enumerate(self.blocks):
                    x1, x2 = blk(x1, x1, use_cross=False,
                                 domain_norm=domain_norm,
                                 Only_self_attention_branch=Only_self_attention_branch)

                x1 = self.norm(x1)
                global_feat = x1[:, 0]
                feat = self.BatchNorm(global_feat)
                cls_score = self.soft(self.classifier(feat))
                return (feat, cls_score), (None, None)
            else:
                x1 = self.embedding(x[:, :self.max_length])
                x2 = self.embedding(x2[:, :self.max_length])

                B = x.shape[0]
                cls_tokens = self.cls_token.expand(B, -1, -1)
                x1 = torch.cat((cls_tokens, x1), dim=1)
                x2 = torch.cat((cls_tokens, x2), dim=1)
                x1 = self.pos_drop(x1)
                x2 = self.pos_drop(x2)
                for i, blk in enumerate(self.blocks):
                    x1, x2 = blk(x1, x2, use_cross=False,
                                 domain_norm=domain_norm,
                                 Only_self_attention_branch=Only_self_attention_branch)
                x1 = self.norm(x1)
                x2 = self.norm(x2)

                global_feat, global_feat2 = x1[:, 0], x2[:, 0]

                feat = self.BatchNorm(global_feat)
                feat2 = self.BatchNorm(global_feat2)

                cls_score = self.soft(self.classifier(feat))
                cls_score2 = self.soft(self.classifier(feat2))

                return (feat, cls_score), (feat2, cls_score2)
