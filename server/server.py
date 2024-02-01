import torch
import torch.nn as nn
import torch.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import json
from IPython.display import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
import torchvision
from scipy import ndimage
from torchvision.transforms import v2
import config

new_size = 448
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(1)
torch.cuda.manual_seed(1)
np.random.seed(1)



def pad_img(img):
    """
    Pad image to square using pytorch, to the right or bottom
    """
    h, w = img.shape[-2:]
    if h > w:
        pad = h - w
        img = transforms.Pad((0, 0, pad, 0))(img)
    elif w > h:
        pad = w - h
        img = transforms.Pad((0, 0, 0, pad))(img)
    return img

def resize_img(img, size):
    """
    Resize image to square using pytorch
    """
    img = transforms.Resize(size, antialias=True)(img)
    return img

def make_it_cv2(img):
    img = img.permute(1, 2, 0)
    img = np.array(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img


def pre_process_image(img):
    if img.shape[0] == 1:
        img = img.repeat(3, 1, 1)
    elif img.shape[0] != 3:
        img = img[:3]
    h, w = img.shape[-2:]
    max_shape = max(h, w)
    ratio = new_size / max_shape
    img = pad_img(img)
    img = resize_img(img, (new_size, new_size))
    img = img/255.
    img = img.float()
    return img, ratio


def get_bbox_resized(prebb, ratio):
    bb = prebb.copy()
    bb[0] = bb[0] * ratio
    bb[1] = bb[1] * ratio
    bb[2] = bb[2] * ratio
    bb[3] = bb[3] * ratio
    return bb
def get_middle_point(bb):
    x = (bb[0] + bb[2])/2
    y = (bb[1] + bb[3])/2
    return [x, y]


def get_batch(b):
    output = []
    b_iter = iter(dataset)
    for i in range(b):
        output.append(next(b_iter))
    return output

dropout = 0.0
factor = 1

class VggBlock(nn.Module):
    def __init__(self, in_ch, out_ch, maxpool=True):
        super(VggBlock, self).__init__()
        block = []
        for i in range(len(in_ch)):
            block.append(nn.Conv2d(in_ch[i], out_ch[i], kernel_size=3, padding=1, bias=False))
            block.append(nn.BatchNorm2d(out_ch[i]))
            block.append(nn.LeakyReLU(0.1, inplace=True))
            block.append(nn.Dropout2d(dropout, inplace=False))
        if maxpool:
            block.append(nn.MaxPool2d(kernel_size=2, stride=2))
        
        block_modules = nn.ModuleList(block)
        self.block = nn.Sequential(*block_modules)

    def forward(self, x):
        x = self.block(x)
        return x
    
class Vgg(nn.Module):
    def __init__(self):
        super(Vgg, self).__init__()
        self.conv1 = VggBlock([3, int(64*factor)], [int(64*factor), int(64*factor)])
        self.conv2 = VggBlock([int(64*factor), int(128*factor)], [int(128*factor), int(128*factor)])
        self.conv3 = VggBlock([int(128*factor), int(256*factor), int(256*factor)], [int(256*factor), int(256*factor), int(256*factor)])
        self.conv4 = VggBlock([int(256*factor), int(512*factor), int(512*factor)], [int(512*factor), int(512*factor), int(512*factor)])
        self.conv5 = VggBlock([int(512*factor), int(512*factor), int(512*factor)], [int(512*factor), int(512*factor), int(512*factor)])
        self.conv6 = VggBlock([int(512*factor), int(512*factor), int(512*factor)], [int(512*factor), int(512*factor), int(512*factor)], False)
        
    def forward(self, x):
        x = self.conv1(x)
        x1 = self.conv2(x)
        x2 = self.conv3(x1)
        x3 = self.conv4(x2)
        x4 = self.conv5(x3)
        x5 = self.conv6(x4)
        return x1, x2, x3, x4, x5
    
class Upconv(nn.Module):
    def __init__(self,in_ch, out_ch, upsample=True):
        super(Upconv, self).__init__()
        self.upsample = upsample
        layer = []
        layer.append(nn.Conv2d(in_ch, out_ch*2, kernel_size=1, padding=0, bias=False))
        layer.append(nn.BatchNorm2d(out_ch*2))
        layer.append(nn.LeakyReLU(0.1, inplace=True))
        layer.append(nn.Dropout2d(dropout, inplace=False))
        layer.append(nn.Conv2d(out_ch*2, out_ch, kernel_size=3, padding=1, bias=False))
        layer.append(nn.BatchNorm2d(out_ch))
        layer.append(nn.LeakyReLU(0.1, inplace=True))
        layer.append(nn.Dropout2d(dropout, inplace=False))
        
        layer_modules = nn.ModuleList(layer)
        self.layer = nn.Sequential(*layer_modules)
        
    def forward(self, x):
        if self.upsample:
            size = x.size()[2:]
            size = (size[0]*2, size[1]*2)
            x = nn.functional.interpolate(x, size=size, mode='bilinear', align_corners=False)
        x = self.layer(x)
        return x


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.vgg = Vgg()
        self.upconv1 = Upconv(int(1024*factor), int(256*factor))
        self.upconv2 = Upconv(int(768*factor), int(128*factor))
        self.upconv3 = Upconv(int(384*factor), int(64*factor))
        self.upconv4 = Upconv(int(192*factor), int(64*factor), True)
        self.conv_last = nn.Sequential(
            nn.Conv2d(int(64*factor), int(64*factor), kernel_size=3, padding=1), nn.BatchNorm2d(int(64*factor)), nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(int(64*factor), int(64*factor), kernel_size=3, padding=1), nn.BatchNorm2d(int(64*factor)), nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(int(64*factor), int(32*factor), kernel_size=3, padding=1), nn.BatchNorm2d(int(32*factor)), nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(int(32*factor), int(32*factor), kernel_size=1), nn.BatchNorm2d(int(32*factor)), nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(int(32*factor), 1, kernel_size=1)
        )
        

    def forward(self, x):
        vgg_outputs = self.vgg(x)
        y1 = torch.cat([vgg_outputs[3], vgg_outputs[4]], dim=1)
        y1 = self.upconv1(y1)
        
        y2 = torch.cat([y1, vgg_outputs[2]], dim=1)
        y2 = self.upconv2(y2)
        
        y3 = torch.cat([y2, vgg_outputs[1]], dim=1)
        y3 = self.upconv3(y3)
        
        y4 = torch.cat([y3, vgg_outputs[0]], dim=1)
        y4 = self.upconv4(y4)
        
        y = self.conv_last(y4)
        y = y.view(-1, new_size//2, new_size//2)

        return y

detection_model = Model()
detection_model = nn.DataParallel(detection_model, device_ids=[0])
checkpoint = torch.load(config.detection_model_path, map_location=torch.device('cpu'))
detection_model.load_state_dict(checkpoint)
for parameter in detection_model.parameters():
    parameter.requires_grad = False
detection_model.to(device)


def get_smaller_parts(seg, bb, t):
    y1, y2, x1, x2 = bb
    d_b = abs(y1 - y2) * abs(x1 - x2)
    bb_out = [bb]
    
    if d_b < 50:
        return bb_out
    
    croped = seg[y1:y2, x1:x2].copy()
    detected = croped[croped > t]
    # croped = (croped - croped.min())/(croped.max() - croped.min())
    d_b = np.sum(detected)
    max_parts = 1
    while t < 1:
        t += 0.01
        binary_mask = croped > t
        
        labeled_mask, num_labels = ndimage.label(binary_mask)
        if max_parts == 1:
            d_b = np.sum(croped[binary_mask])
        if num_labels <= max_parts:
            continue
        max_parts = max(max_parts, num_labels)
        bounding_boxes_ = ndimage.find_objects(labeled_mask)
        current_bbs = []
        for label, bbox in enumerate(bounding_boxes_):
            if bbox is not None:
                y1_, y2_, x1_, x2_ = bbox[0].start, bbox[0].stop, bbox[1].start, bbox[1].stop
                current_bbs.append((y1_ + y1, y2_ + y1, x1_ + x1, x2_ + x1))
                

        size = np.sum(croped[binary_mask])
                
        if size/d_b > 0.5:
            bb_out = current_bbs
            # d_b = size
            # max_parts = num_labels
        else:
            break
        
    return bb_out


def from_Tensor2BB(segmentation_maps, ratios, threshold=0.5):
    batches = []
    for i in range(len(segmentation_maps)):

        current_seg = segmentation_maps[i]
        current_seg = (current_seg - current_seg.min()) / (current_seg.max() - current_seg.min())
        
        binary_mask = current_seg > threshold

        labeled_mask, num_labels = ndimage.label(binary_mask)

        bounding_boxes = ndimage.find_objects(labeled_mask)
        bbs = []

        for label, bbox in enumerate(bounding_boxes, start=1):
            if bbox is not None:
                y1, y2, x1, x2 = bbox[0].start, bbox[0].stop, bbox[1].start, bbox[1].stop

                parts = get_smaller_parts(current_seg, (y1, y2, x1, x2), threshold)
                for part in parts:
                    y1, y2, x1, x2 = part
                    bbs.append((0, int(x1*2/ratios[i]), int(y1*2/ratios[i]), int(x2*2/ratios[i]), int(y2*2/ratios[i])))
                    
#                 bbs.append((0, int(x1*2/ratios[i]), int(y1*2/ratios[i]), int(x2*2/ratios[i]), int(y2*2/ratios[i])))
        batches.append(bbs)

    return batches

dropout = 0.4

def get_positional_embeddings(sequence_length, d):
    result = torch.ones(sequence_length, d)
    for i in range(sequence_length):
        for j in range(d):
            result[i][j] = (
                np.sin(i / (10000 ** (j / d)))
                if j % 2 == 0
                else np.cos(i / (10000 ** ((j - 1) / d)))
            )
    return result

class Patches(nn.Module):
    def __init__(self, ph, pw, in_ch):
        super(Patches, self).__init__()
        self.ph = ph
        self.pw = pw
        self.ln = nn.LayerNorm(ph*pw*in_ch)
        
    def forward(self, x):
        b, c, h, w = x.shape
        nh, nw = h // self.ph, w // self.pw
        x = torch.reshape(x, (b, c, nh, self.ph, nw, self.pw))
        x = torch.permute(x, (0, 2, 4, 3, 5, 1))
        x = torch.reshape(x, [b, nh * nw, self.ph * self.pw * c])
        x = self.ln(x)
        return x
    
class CharacterEmbedding(nn.Module):
    def __init__(self, vocab_size, block_size, n_embed):
        super(CharacterEmbedding, self).__init__()
        self.block_size = block_size
        self.token_embed_tab = nn.Embedding(vocab_size, n_embed)
        self.register_buffer(
                    "positional_embeddings",
                    get_positional_embeddings(block_size, n_embed),
                    persistent=False,
                )
        
    def forward(self, x):
        # x shape: (batch, block_size)
        B, T = x.shape
        x = self.token_embed_tab(x) # shape: (batch, block_size, n_embed)
        position_embed = self.positional_embeddings[:T, :]
        return x + position_embed.repeat(x.shape[0], 1, 1)
    
class PatchEmbedding(nn.Module):
    def __init__(self, n_patches, patch_size, n_embed, add_patches=0):
        super(PatchEmbedding, self).__init__()
        self.n_patches = n_patches
        self.token_embed_tab = nn.Linear(patch_size, n_embed)
        self.positional_embeddings = nn.Parameter(torch.randn(1, n_patches + add_patches + 1, n_embed)*0.02)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, n_embed))
        self.dropout = nn.Dropout(0.)
        if add_patches > 0:
            self.add_patches = nn.Parameter(torch.zeros(1, add_patches, n_embed))
        else:
            self.add_patches = None
        
    def forward(self, x):
        B = x.shape[0]
        x = self.token_embed_tab(x)
        cls_tok = self.cls_token.expand([B, -1, -1])
        x = torch.cat([cls_tok, x], dim=1)
        if self.add_patches != None:
            new_patches = self.add_patches.expand([B, -1, -1])
            x = torch.cat([x, new_patches], dim=1)
        x = x + self.positional_embeddings.repeat(B, 1, 1)
        x = self.dropout(x)
        return x
    
class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, block_size, n_embed, apply_mask, cross=False):
        super(MultiHeadAttention, self).__init__()
        self.single_head_size = n_embed // n_heads
        self.n_heads = n_heads
        self.out = nn.Linear(n_embed, n_embed)
        self.dropout = nn.Dropout(dropout)
        self.apply_mask = apply_mask
        self.attn_weights = None
        if apply_mask:
            self.register_buffer('mask', torch.tril(torch.ones(block_size, block_size)))
        self.cross = cross
        if cross:
            self.kv = nn.Linear(n_embed, 2*n_embed, bias=False)
            self.q = nn.Linear(n_embed, n_embed, bias=False)
        else:
            self.qkv = nn.Linear(n_embed, 3*n_embed, bias=False)
        
    def forward(self, x, kv=None):
        B, T, C = x.shape
        if self.cross:
            k, v = self.kv(kv).reshape(B, kv.shape[1], 2, self.n_heads, self.single_head_size).permute(2, 0, 3, 1, 4).unbind(0) # (2, B, n_heads, T, head_dim)
            q = self.q(x).reshape(B, T, self.n_heads, self.single_head_size).permute(0, 2, 1, 3)
        else:
            q, k, v = self.qkv(x).reshape(B, T, 3, self.n_heads, self.single_head_size).permute(2, 0, 3, 1, 4).unbind(0)
        q = q * (self.single_head_size ** -0.5)
        attn = q @ k.transpose(-2, -1)
        if self.apply_mask:
            attn = attn.masked_fill(self.mask[:T, :T] == 0, float('-inf'))
        attn = attn.softmax(dim=-1)
        self.attn_weights = attn.detach().cpu().numpy()
        x = attn @ v
        x = x.transpose(1, 2).reshape(B, T, C)
        x = self.dropout(self.out(x))
        return x
    
class FeedForward(nn.Module):
    def __init__(self, n_embed):
        super(FeedForward, self).__init__()
        layers = [nn.Linear(n_embed, 4*n_embed),
                  nn.GELU(),
                  nn.Linear(4*n_embed, n_embed),
                  nn.Dropout(dropout)]
        layer_modules = nn.ModuleList(layers)
        self.net = nn.Sequential(*layer_modules)
        
    def forward(self, x):
        return self.net(x)

class DropPath(nn.Module):
    def __init__(self, drop_prob=0.15):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        
    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        output = x.div(keep_prob) * random_tensor
        return output
    
class EncoderBlock(nn.Module):
    def __init__(self, n_heads, block_size, n_embed):
        super(EncoderBlock, self).__init__()
        self.self_attention = MultiHeadAttention(n_heads, block_size, n_embed, False)
        self.fw = FeedForward(n_embed)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)
        self.layer_scale1 = nn.Parameter(1e-5 * torch.ones(n_embed))
        self.layer_scale2 = nn.Parameter(1e-5 * torch.ones(n_embed))
        self.drop_path = DropPath()
        
    def forward(self, x):
        x = x + self.drop_path(self.layer_scale1*self.self_attention(self.ln1(x)))
        x = x + self.drop_path(self.layer_scale2*self.fw(self.ln2(x)))
        return x
    
class DecoderBlock(nn.Module):
    def __init__(self, n_heads, block_size, n_embed):
        super(DecoderBlock, self).__init__()
        self.masked_self_attention = MultiHeadAttention(n_heads, block_size, n_embed, True)
        self.cross_attention = MultiHeadAttention(n_heads, block_size, n_embed, False, True)
        self.fw = FeedForward(n_embed)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)
        self.ln3 = nn.LayerNorm(n_embed)
        layer_scale = False
        self.layer_scale1 = nn.Parameter(1e-5 * torch.ones(n_embed)) if layer_scale else nn.Identity()
        self.layer_scale2 = nn.Parameter(1e-5 * torch.ones(n_embed)) if layer_scale else nn.Identity()
        self.layer_scale3 = nn.Parameter(1e-5 * torch.ones(n_embed)) if layer_scale else nn.Identity()
        self.drop_path = DropPath()
        
    def forward(self, x):
        x, kv = x[0], x[1]
        x = x + self.drop_path(self.layer_scale1(self.masked_self_attention(self.ln1(x))))
        x = x + self.drop_path(self.layer_scale2(self.cross_attention(self.ln2(x), kv)))
        x = x + self.drop_path(self.layer_scale3(self.fw(self.ln3(x))))
        return (x, kv)
    
class Encoder(nn.Module):
    def __init__(self, N, n_heads, block_size, n_embed):
        super(Encoder, self).__init__()
        self.blocks = [EncoderBlock(n_heads, block_size, n_embed) for _ in range(N)]
        self.blocks = nn.ModuleList(self.blocks)
        self.blocks = nn.Sequential(*self.blocks)
        
    def forward(self, x):
        return self.blocks(x)
    
class Decoder(nn.Module):
    def __init__(self, N, n_heads, block_size, n_embed):
        super(Decoder, self).__init__()
        self.blocks = [DecoderBlock(n_heads, block_size, n_embed) for _ in range(N)]
        self.blocks = nn.ModuleList(self.blocks)
        self.blocks = nn.Sequential(*self.blocks)
    
    def forward(self, x, kv):
        return self.blocks((x, kv))
    
class TrOcr(nn.Module):
    def __init__(self):
        super(TrOcr, self).__init__()
        n_embed = 384
        L_en = 12
        L_de = 4
        n_heads = 12
        patch_h = 16
        patch_w = 16
        h = 128
        w = 32
        ch = 1
        self.seq_len = 27
        n_patches = w*h//(patch_h*patch_w)
        patch_size = w*h*ch//n_patches
        add_patches = 0
        self.to_patches = Patches(patch_h, patch_w, ch)
        self.patch_embedding = PatchEmbedding(n_patches, patch_size, n_embed, add_patches)
        self.token_embedding = CharacterEmbedding(len(vocab), self.seq_len, n_embed)
        self.encoder = Encoder(L_en, n_heads, n_patches, n_embed)
        self.decoder = Decoder(L_de, n_heads, self.seq_len, n_embed)
        self.ln = nn.LayerNorm(n_embed)
        self.last_layer = nn.Linear(n_embed, len(vocab))
        
    def forward(self, x, de_input):
        en_output = self.to_patches(x)
        en_output = self.patch_embedding(en_output)
        en_output = self.encoder(en_output)
        de_input = self.token_embedding(de_input)
        de_output = self.decoder(de_input, en_output)
        output = self.ln(de_output[0])
        output = output[:, :self.seq_len]
        B, T, E = output.shape
        output = output.reshape(B*T, E)
        output = self.last_layer(output).view(B, T, len(vocab))
        return output

# # %%
# chars = set()
# for img_idx in range(len(img_ids)):
#     img_anns = bounding_boxes['imgToAnns'][img_ids[img_idx][:-4]]
#     for idx in range(len(img_anns)):
#         ann = bounding_boxes['anns'][img_anns[idx]]
#         for char in ann['utf8_string']:
#             chars.add(char)
# pad_token = '█'
# start_token = '├'
# end_token = '┤'
# chars.add(pad_token)
# chars.add(start_token)
# chars.add(end_token)
# vocab = sorted(list(chars))
# print(len(vocab))
# print(vocab)

vocab = ['!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';', '<', '=', '>', '?', '@', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '[', '\\', ']', '^', '_', '`', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '{', '|', '}', '~', '¡', '¢', '£', '¥', '§', '©', '«', '¬', '®', '°', '±', '²', '³', 'µ', '·', '¹', 'º', '»', '¼', '½', '¾', '¿', 'À', 'Á', 'Â', 'Ã', 'Ä', 'Å', 'Æ', 'Ç', 'È', 'É', 'Ê', 'Ë', 'Ì', 'Í', 'Î', 'Ï', 'Ñ', 'Ò', 'Ó', 'Ô', 'Õ', 'Ö', '×', 'Ø', 'Ù', 'Ú', 'Û', 'Ü', 'Ý', 'ß', 'à', 'á', 'â', 'ã', 'ä', 'å', 'æ', 'ç', 'è', 'é', 'ê', 'ë', 'ì', 'í', 'î', 'ï', 'ð', 'ñ', 'ò', 'ó', 'ô', 'õ', 'ö', '÷', 'ø', 'ù', 'ú', 'û', 'ü', 'ý', 'þ', 'ÿ', 'Ā', 'ā', 'Ă', 'ă', 'Ą', 'ą', 'Ć', 'ć', 'Ĉ', 'ĉ', 'ċ', 'Č', 'č', 'Đ', 'Ē', 'ē', 'Ĕ', 'ĕ', 'Ė', 'ė', 'Ę', 'ę', 'Ě', 'ě', 'ĝ', 'Ğ', 'ğ', 'ġ', 'Ģ', 'Ĩ', 'ĩ', 'Ī', 'ī', 'ĭ', 'İ', 'ı', 'Ĺ', 'ĺ', 'Ľ', 'Ł', 'Ń', 'ń', 'Ň', 'ň', 'ŉ', 'ŋ', 'Ō', 'ō', 'Ŏ', 'ŏ', 'Ő', 'ő', 'Œ', 'œ', 'Ŕ', 'ŕ', 'ŗ', 'Ř', 'ř', 'Ś', 'ś', 'Ş', 'ş', 'Š', 'š', 'Ţ', 'ţ', 'Ť', 'Ŧ', 'ŧ', 'ũ', 'Ū', 'ū', 'ŭ', 'Ů', 'ů', 'Ű', 'ű', 'ŵ', 'Ŷ', 'ŷ', 'Ź', 'ź', 'Ż', 'ż', 'Ž', 'ž', 'Ɓ', 'ƃ', 'Ɔ', 'Ƈ', 'ƈ', 'Ɖ', 'ƌ', 'Ɛ', 'ƒ', 'Ɵ', 'Ơ', 'ơ', 'Ƭ', 'Ư', 'ư', 'Ƶ', 'ǎ', 'Ǐ', 'ǐ', 'Ǒ', 'ǒ', 'ǔ', 'Ǧ', 'Ǫ', 'ǫ', 'Ǭ', 'ǵ', 'ǹ', 'ǿ', 'Ȃ', 'Ȅ', 'Ȇ', 'ȇ', 'Ȋ', 'ȋ', 'Ȍ', 'ȍ', 'Ȏ', 'ȏ', 'ȑ', 'ȓ', 'ȕ', 'ȗ', 'Ș', 'ș', 'Ț', 'ț', 'Ȧ', 'ȧ', 'Ȩ', 'ȩ', 'Ȯ', 'ȯ', 'ȳ', 'ə', 'ɮ', 'ʃ', 'ʊ', 'ʌ', 'ʰ', 'ʲ', 'ʳ', '˂', '˃', '˄', '˅', 'ˉ', 'ˍ', 'ˏ', '˙', '˚', 'ˡ', 'ˢ', 'ˣ', '̓', '̩', 'Δ', 'Θ', 'Λ', 'Ξ', 'Π', 'Ρ', 'Σ', 'Φ', 'Ω', 'ί', 'α', 'β', 'γ', 'δ', 'ε', 'ζ', 'η', 'θ', 'ι', 'κ', 'λ', 'μ', 'ν', 'ξ', 'ο', 'π', 'ρ', 'ς', 'σ', 'τ', 'υ', 'φ', 'ω', 'ϋ', 'а', 'б', 'в', 'г', 'д', 'е', 'ж', 'и', 'й', 'к', 'л', 'м', 'н', 'о', 'п', 'р', 'с', 'т', 'у', 'ъ', 'ы', 'ь', 'э', 'я', 'ײ', '،', 'ب', 'ر', 'ل', '٢', '٣', 'ڮ', 'ڽ', 'ۊ', '฿', '።', 'ᐟ', 'ᴎ', 'ᴐ', 'ᴙ', 'ᴣ', 'ᴭ', 'ᴰ', 'ᴱ', 'ᴴ', 'ᴵ', 'ᴸ', 'ᴹ', 'ᴺ', 'ᴼ', 'ᴾ', 'ᴿ', 'ᵀ', 'ᵃ', 'ᵇ', 'ᵈ', 'ᵉ', 'ᵍ', 'ᵏ', 'ᵐ', 'ᵒ', 'ᵖ', 'ᵗ', 'ᵛ', 'ᶜ', 'ᶠ', 'ᶦ', 'ᶲ', 'ᶻ', 'ᶾ', 'Ḃ', 'ḉ', 'Ḍ', 'Ḏ', 'ḕ', 'Ḣ', 'ḻ', 'ṁ', 'Ṅ', 'ṅ', 'Ṓ', 'Ṙ', 'Ṛ', 'Ṡ', 'Ṣ', 'ṣ', 'ṫ', 'ẏ', 'Ạ', 'Ả', 'ả', 'Ấ', 'Ầ', 'ầ', 'Ẩ', 'Ậ', 'Ắ', 'Ặ', 'Ế', 'ế', 'Ề', 'ề', 'Ể', 'Ễ', 'Ệ', 'ọ', 'Ỏ', 'Ố', 'ố', 'Ồ', 'ồ', 'Ộ', 'Ớ', 'ớ', 'ờ', 'Ợ', 'Ứ', 'ό', 'Ᾱ', 'ῑ', 'ῡ', '\u200b', '\u200e', '‑', '–', '—', '―', '‖', '‘', '’', '‛', '“', '”', '„', '‟', '†', '‡', '•', '‣', '․', '…', '‧', '′', '‵', '‹', '›', '⁎', '⁰', '⁴', '⁵', '⁶', '⁷', '⁸', '⁹', '⁺', '⁻', '⁼', '⁽', '⁾', 'ⁿ', '₀', '₁', '₂', '₃', '₄', '₅', '₋', 'ₒ', 'ₓ', 'ₖ', 'ₘ', 'ₙ', 'ₜ', '₤', '€', '₮', '₵', '₹', '₿', '℃', 'ℇ', '℉', 'ℑ', 'ℒ', '№', '℗', '℘', '™', 'Ω', 'ℨ', 'Å', '℮', 'ℱ', '⅓', '⅔', '⅛', 'Ⅰ', 'Ⅱ', 'Ⅲ', 'Ⅳ', 'Ⅴ', 'Ⅵ', 'Ⅶ', 'Ⅷ', 'Ⅸ', 'Ⅹ', 'Ⅺ', 'Ⅻ', '←', '↑', '→', '↓', '↰', '↱', '↲', '↳', '↵', '↶', '↷', '↹', '↺', '↻', '↼', '⇅', '⇆', '⇈', '⇐', '⇑', '⇒', '⇓', '⇗', '⇜', '⇝', '⇤', '⇥', '⇦', '⇧', '⇨', '⇩', '⇪', '⇫', '⇵', '⇾', '∁', '∃', '∅', '∆', '∇', '∈', '∉', '∊', '∎', '∑', '−', '∓', '∖', '∘', '∙', '√', '∝', '∞', '∟', '∣', '∥', '∧', '∨', '∩', '∫', '∴', '∷', '∼', '∽', '∾', '≀', '≈', '≋', '≠', '≡', '≣', '≤', '≥', '≪', '≫', '≶', '≷', '⊂', '⊘', '⊙', '⊚', '⊝', '⊡', '⊢', '⊣', '⊤', '⊥', '⊲', '⊳', '⊼', '⊿', '⋄', '⋅', '⋆', '⋏', '⋙', '⋮', '⋯', '⌀', '⌃', '⌄', '⌘', '⌙', '⌣', '〈', '〉', '⌫', '⌴', '⌾', '⍙', '⎯', '⎵', '⏘', '␣', '①', '②', '③', '④', '⑤', '⑥', 'Ⓐ', 'Ⓒ', 'Ⓓ', 'Ⓖ', 'Ⓛ', 'Ⓡ', 'Ⓣ', 'Ⓤ', 'Ⓦ', 'ⓒ', 'ⓔ', 'ⓘ', '─', '━', '│', '┆', '┌', '┍', '┏', '┒', '┓', '└', '├', '┤', '╎', '═', '╥', '╲', '╳', '╵', '╻', '▇', '█', '▎', '■', '□', '▢', '▬', '▯', '▰', '▲', '△', '▴', '▵', '▷', '▸', '▹', '►', '▼', '▽', '▾', '▿', '◁', '◂', '◃', '◄', '◆', '◉', '◊', '○', '●', '◡', '◦', '◬', '◯', '★', '☆', '☐', '☒', '☓', '☰', '♡', '♲', '⚯', '✓', '✕', '✚', '✞', '✧', '✯', '✰', '✱', '✽', '❐', '❑', '❖', '❘', '❚', '❜', '❝', '❶', '➀', '➁', '➂', '➃', '➄', '➅', '➊', '➋', '➌', '➍', '➎', '➔', '➙', '➜', '➝', '➞', '➦', '➧', '➨', '➩', '➮', '➯', '➸', '⟐', '⟨', '⟩', '⟲', '⟳', '⟵', '⟶', '⟷', '⟹', '⟺', '⤣', '⤤', '⤦', '⤶', '⤷', '⤸', '⤹', '⤺', '⤻', '⤾', '⤿', '⦁', '⦸', '⧋', '⧐', '⧓', '⧫', '⬁', '⬂', '⬈', '⬉', '⬊', '⬋', '⬌', '⬍', '⬏', '⬑', '⬤', '⬸', '⮌', '⮠', 'ⱽ', '⸬', '、', '。', '〈', '〉', '《', '》', '「', '」', '〤', '〱', '・', '卐', 'ꝋ', '꞉', '꞊', '︱', '︺', '︽', '︾', '︿', '﹀', '﹘', '）', '，', '－', '：', '＝', '～', '￠', '￡', '￥', '�', '𝔀', '𝕲', '𝕵', '𝖅', '𝖓', '🠄', '🠅', '🠆', '🠇', '🠈', '🡨']

recognition_model = TrOcr()
recognition_model = nn.DataParallel(recognition_model, device_ids=[0])
checkpoint = torch.load(config.recognition_model_path, map_location=torch.device('cpu'))
recognition_model.load_state_dict(checkpoint)
for parameter in recognition_model.parameters():
    parameter.requires_grad = False
recognition_model.to(device)

chtoi = {ch: i for i, ch in enumerate(vocab)}
itoch = {i: ch for i, ch in enumerate(vocab)}
pad_token = '█'
start_token = '├'
end_token = '┤'

def encode(s: str):
    encoded = []
    for c in s:
        encoded.append(chtoi[c])
    return encoded

def decode(d: list):
    decoded = ""
    for i in d:
        if type(i) == torch.Tensor:
            decoded += itoch[i.item()]
        else:
            decoded += itoch[i]
    return decoded

pad_token_en = encode(pad_token)[0]


def pre_process_text_box(img):
    img = transforms.Resize((32, 128), antialias=True)(img)
    img = v2.Grayscale()(img)
    img = img / 255.
    return img

def trim_pad(s):
    idx = s.index(end_token, 1)
    return s[1:idx]

def get_text_bb(img, bbs, t=0.0):
    img = img.permute(1, 2, 0).numpy()
    recognition_model.eval()
    B = len(bbs)
    i_end_token = encode(end_token)[0]
    croped_ls = []
    for i in range(B):
        _, x1, y1, x2, y2 = bbs[i]
        croped = img[y1:y2, x1:x2]
        croped = torch.tensor(croped)
        croped = croped.permute(2, 0, 1)
        croped = pre_process_text_box(croped)
        croped = croped.view(1, 32, 128)
        croped_ls.append(croped)
    features = torch.cat(croped_ls, axis=0)
    features = features.to(device)
    
    current_token = [start_token for _ in range(B)]
    current_de_string = torch.tensor(encode(start_token)[0], dtype=torch.long).view(1, 1).expand(B, -1)
    features = features.view(B, 1, 32, 128)
    idx = 0
    confidences = [0 for _ in range(B)]
    n_characters = [0 for _ in range(B)]
    remaining_words_to_predict = [i for i in range(B)]
    while idx < 27:
        current_de_string.to(device)
        with torch.autocast('cuda'):
            predictions = recognition_model.forward(features, current_de_string)
        predictions = nn.functional.softmax(predictions, dim=-1)
        current_token = torch.argmax(predictions, dim=-1)[:, idx].cpu()
            
        for j in range(B):
            if current_token[i] == i_end_token:
                if j in remaining_words_to_predict:
                    remaining_words_to_predict.remove(j)
            
        first_token_val = torch.max(predictions, dim=-1)
        confidences = [confidences[i] + first_token_val[0][i][0] if i in remaining_words_to_predict else confidences[i] for i in range(B)]
        n_characters = [n_characters[i] + 1 if i in remaining_words_to_predict else n_characters[i] for i in range(B)]
        
        
        
        current_de_string = torch.cat([current_de_string, current_token.view(B, 1)], dim=-1)
        idx += 1
    confidences = [confidences[i]/n_characters[i] for i in range(B)]
    output = [(bbs[i], trim_pad(decode(current_de_string[i]))) for i in range(B) if confidences[i] > t]
    return output
    

def get_bbs(raw_images):
    if len(raw_images.shape) == 3:
        raw_images = [raw_images]
    detection_model.eval()
    # raw_images = get_batch(50)
    test_features = [pre_process_image(raw_images[i]) for i in range(len(raw_images))]
    ratios = [tf[1] for tf in test_features]
    test_features = torch.cat([tf[0].view(1, tf[0].shape[0], tf[0].shape[1], tf[0].shape[2]) for tf in test_features], axis=0)
    if len(test_features.shape) == 3:
        test_features = test_features.view(1, test_features.shape[0], test_features.shape[1], test_features.shape[2])
    test_features = test_features.to(device)
    with torch.autocast('cuda'):
        predictions = detection_model.forward(test_features)

    predictions_bb = from_Tensor2BB(predictions.detach().cpu().numpy(), ratios, 0.7)
    # min_shape = min(raw_images[0].shape[1], raw_images[0].shape[2])
    max_shape = max(raw_images[0].shape[1], raw_images[0].shape[2])
    predictions = resize_img(predictions, max_shape)
    predictions = predictions[:, :raw_images[0].shape[1], :raw_images[0].shape[2]]
    return predictions, predictions_bb

def from_jpg_toTensor(img_path):
    return read_image(img_path)

