
# Swin Transformer: Hierarchical Vision Transformer using Shifted Windows

## 0. Abstract
- This new vision transformer capably serves as a general-purpose backbone for computer vision
- The auhtor proposes hierarchical transformer whose representation is computed with shifted windows for solving main challenges of previous transformer which are **large variations in the scale of visual entities** and **hight resolution of pixels in images**.
- Shifted windowing scheme brings greater efficiency by **limiting self-attention computation to non-overlapping local windows**.

## 1. Introduction
<p align="center">
  <img width="400" height="400" src="https://user-images.githubusercontent.com/32179857/124716739-5fe97080-df3f-11eb-9783-6df7c0eecdaa.png">
</p>

- Modeling for computer visions and for NLP has taken a different path in the past. Along with the tremendous success of transformer in NLP domain, researchers started to investigate its adaptation to computer vision.
- The authors argue that significant challenges in transferring its high performance to the visual domain can be explained by difference of two modalities.

    1. Unlike word tokens that serve as the basic elements of processng in NLP tasks, visual elements vary substantially in scale.
    2. various vision tasks such as semantic segmentation require dense prediction at the pixel level, which can computationally expensive quadratically to image size.

- To overcome these issues, the authors propose a general-purpose Transformer backbone, called Swin Transformer.
<p align="center">
  <img width="400" height="400" src="https://user-images.githubusercontent.com/32179857/124723889-721add00-df46-11eb-9c75-3bd32d1c2075.png">
</p>

#### Characteristics


- Hierarchical representation by starting from small-sized patches and gradually merging neighboring patches in deeper layers.
   --> this can leverage advanced techniques for dense prediction such as FPN or U-Net.
- Key design for Swin Transformer is its **shift of window partition** between consecutive self-attention layers.



## 2. Related work
- **Self-attention based backbone architectures**
    - In these works, the self-attention is computed within a local window of each pixel to expedite optimization, and they achieve slightly better acc.
    - However, costly memory access causes significantly larger latency.
    - ![image](https://user-images.githubusercontent.com/32179857/125021485-b4aef780-e0b5-11eb-8f4e-f1078fda2128.png)
    (https://stats.stackexchange.com/questions/421935/what-exactly-are-keys-queries-and-values-in-attention-mechanisms)
    
- **Self-attention/Transformers to complement CNNs**
    - Self-attention layers can complement backbones or head networks by providing the capability to encode distant dependencies or heterogeneous interactions.
    - Recently, encoder-decoder design in Transformer has been applied for OD nd Seg. tasks.
    - [Self-attention Transformer Encoder](https://theaisummer.com/transformer/#self-attention-the-transformer-encoder)
- **Transformer based vision backbones**
    - Vision Transformer (ViT) and its follow-ups.
    - ViT directly applies a Transformer architecture on non-overlapping medium-sized image patches for classification.
    - ViT performs well but requries large dataset.
    - DeiT improves ViT so that it can use ImageNet-1K too. 
    - Left drawback here is lack of 'general purpose' network.
- **Rise of the best speed-accuracy tradeoff architecture, Swin Transformer.**

## 3. Method
#### 3.1 Overall Architecture
<p align="center">
  <img width="1200" height="400" src="https://user-images.githubusercontent.com/32179857/124725841-2b2de700-df48-11eb-9cca-325a8d775488.png">
</p>

1. It first splits an input RGB image into non-overlapping patches b a patch splitting module, like ViT.
    (Each patch is treated as 'Token' and its feature is set as a concat. of the raw pixel RGB values.)
2. Several Transformer blocks with modified self-attention (Swin Transformer blocks) are applied on these patch tokens.
    (This block maintain the number of tokens as (H/4 x W/4), and referred as **'Stage1'**.
3. To produce hierarchical representation, the number of token is reduced by patch merging layers as the network gets deeper.
    (This layer concats neighboring 2x2 patches into one patch, making the number of channels double which is 'Down sampling'. As the network goes deeper,
    since the feature map size which pass each stage gets smaller, and it can play a role as 'hierarchical representation' such as FPN, UNet)
4. It continues each stage with a same process, with different amount of iterations ([2,2,6,2])
- Patch partitioning and Patch merging, Linear embedding
    - patch partition & patch merging

        ![image](https://user-images.githubusercontent.com/32179857/125008196-fa12fb00-e09c-11eb-90b7-76adf2dd83ca.png)
    - Linear embedding

        ![image](https://user-images.githubusercontent.com/32179857/125008344-4c541c00-e09d-11eb-8811-dd194d5c8dc0.png)


- Swin Transformer block
    - Swin Transformer block is built by replacing the standard multi-head self attention (MSA) module by a module based on shifted windows, with other layers kept same.
    - It consists of a shifted window based MSA module, followed by a 2-layer MLP(multi layer perceptron) with GELU non-linearity in between.
    - [Multi-head Attention](https://theaisummer.com/transformer/#the-core-building-block-multi-head-attention-and-parallel-implementation)
    
        ![image](https://user-images.githubusercontent.com/32179857/125007362-fbdbbf00-e09a-11eb-8cd0-4d7cb65f9be4.png)
        ![image](https://user-images.githubusercontent.com/32179857/125007379-06965400-e09b-11eb-8c47-23c051108261.png)

#### 3.2 Shifted Window based Self-Attention
- The standard Transformer and its adaptation for image classification both conduct global self-attention, where the relationships between a token and all other tokens are computed. --> this leads to quadratic complexity with respect to the number of tokens (not suitable for vision tasks due to high resolution)

##### Self attention with non-overlapping windows
- The authors propose efficient self-attention within local windows
- computational complexity
    - Suppose each window contains M x M patches, 
        - global MSA: 

        ![image](https://user-images.githubusercontent.com/32179857/125007742-e31fd900-e09b-11eb-98c4-7a37f2234784.png)
    - the former(standard Transformer) is quadratic to patch number hw
    - the latter in linear when M is fixed.
    - since the patch size M is much smalled than image resolution h,w, the computational cost is also much cheaper.
##### Shifted window partitioning in successive blocks
- The window-based self-attention module lacks connections across the windows
- To solve this, the authors propose a shifted window partitioning approach which alternates between two partitioning configurations in consecutive Swin Transformer blocks
    - 1st module uses reguar window partitioning
    - 2nd module adopts a windowing configuration that is shifted from that of the preceding layer by displacing the windows by (M/2, M/2) pixels

        ![image](https://user-images.githubusercontent.com/32179857/125009292-39424b80-e09f-11eb-846c-b594347f951a.png)

#### 3.3 Architect Variants
```
• Swin-T: C = 96, layer numbers = {2, 2, 6, 2}
• Swin-S: C = 96, layer numbers ={2, 2, 18, 2}
• Swin-B: C = 128, layer numbers ={2, 2, 18, 2}
• Swin-L: C = 192, layer numbers ={2, 2, 18, 2}
```

## Experiments
- experiments conducted with ImageNet-1K image classification / COCO object detection, ADE20K semantic segmentation.
<p align="center">
  <img width="800" height="600" src="https://user-images.githubusercontent.com/32179857/124850442-94ab0580-dfdb-11eb-950c-2a694030c4da.png">
</p>

1. Classification
   - AdamW optimizer
   - 300 epoch
   - 20 epoch learning rate warmup
   - batch 1024
   - ImageNet-22K pretrained


2. Object Detection
    - Multi-scale training
    - AdamW optimizer
    - batch 16
   
   
### Ablation Study
<p align="center">
  <img width="800" height="500" src="https://user-images.githubusercontent.com/32179857/124851005-a8a33700-dfdc-11eb-9593-195bf6a19c75.png">
</p>

&nbsp;
&nbsp;

## Code Analysis
- source code from [Official repo.](https://github.com/microsoft/Swin-Transformer)

#### Swin Transformer
```python
class SwinTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=1000,
                 embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                 window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, **kwargs):
        super().__init__()
		
        # ...
        
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
            
        # ...

        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                               input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                 patches_resolution[1] // (2 ** i_layer)),
                               depth=depths[i_layer],
                               num_heads=num_heads[i_layer],
                               window_size=window_size,
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias, qk_scale=qk_scale,
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               norm_layer=norm_layer,
                               downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                               use_checkpoint=use_checkpoint)
            self.layers.append(layer)
        
        self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
		
        # ...
    
    # ...

    def forward_features(self, x):
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)  # B L C
        x = self.avgpool(x.transpose(1, 2))  # B C 1
        x = torch.flatten(x, 1)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x
	
    # ...
```

Overall process or forward is,
> PatchEmbed module -> BasicLayer modules(nn.ModuleList) -> norm&avgpool -> nn.Linear

&nbsp;&nbsp;

#### PatchEmbed (Patch Partition + Linear Embedding)

```python
class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding
    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x

    # ...
```
We can see reformatting process here.
The dimension starts from (B, 3, 224, 224) and converted through (B, 96, 56, 56) by self.proj(x) --> (B, 96, 56x56) by flatten(2) --> (B, 56x56, 96) by .transpose(1,2)

> (B, 3, 224, 224) -> self.proj(x).flatten(2).transpose(1, 2) -> (B, 56x56, 96)

&nbsp;&nbsp;

#### Swin Transformer Block
```python
class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        # ...
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False):

        super().__init__()
        # ...

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x

    # ...
```
- This is composed with the depth number of SwinTransformerBlock and downsampling process.
Every odd SwinTransformerBlock setup shift_size as window_size//2, and downsample is the process of Patch Merging

&nbsp;

```python
class SwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block.
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        # ...
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

    # ...
```

&nbsp;

Main module of the paper
> (B, 56x56, 96) -> (B, 56, 56, 96) -> windows_partition

```python
def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows
```
> (B, 56, 56, 96) -> (B, 8, 7, 8, 7, 96) -> (B, 8, 8, 7, 7, 96) -> (Bx8x8, 7, 7, 96)

Back to SiwnTransofrmerBlock module, each window in (Bx8x8, 7, 7, 96) tensor should be 2-dimensional tensor to process self-attention.
So it becomes (Bx8x8, 7x7, 96) and compute self-attention.
The output from self-attention has the same shape as input due to the characteristic of self-attention. Thus, the shape is recoverd to (B, 56x56, 96) with window_reverse function.

&nbsp;

Window partitioning and shift process can be done with 'torch.roll' function easilly.
```python
#torch.roll examples
>>> x = torch.tensor(
    [
        [1, 2, 3, 4],
        [5, 6, 7, 8]
    ]
)

>>> x = torch.roll(x, 1)
>>> x
tensor(
    [8, 1, 2, 3],
    [4, 5, 6, 7]
)
```

#### Patch Merging
```python
class PatchMerging(nn.Module):
    r""" Patch Merging Layer.
    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        # ...

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x

    # ...
```

![image](https://user-images.githubusercontent.com/32179857/125012606-6d206f80-e0a5-11eb-92d7-856a1030a30a.png)
> (B, 56x56, 96) -> (B, 56, 56, 96) -> (B, 56/2, 56/2, 96x4) -> (B, 56/2, 56/2, 96x2)
- Patch merged (downsampled) tensor is used as an input of next BasicLayer.
- The tensor after all the BasicLayer is loss-computed after being a 'class-number-fit tensor' with norm/avgpool/flatten/nn.Linear.

&nbsp;
&nbsp;

## Training / Inference process
- This description is for the source from [Official repo.](https://github.com/microsoft/Swin-Transformer).

#### Setup
- Installation of mmdetection
    - https://github.com/open-mmlab/mmdetection/blob/master/docs/get_started.md
- For 30xx series
    - based on Docker envrionment 
       ```bash
       ARG PYTORCH="1.7.1"
       ARG CUDA="11.0"
       ARG CUDNN="8"

       FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

       ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0+PTX"
       ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
       ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"

       RUN apt-get update && apt-get install -y ffmpeg libsm6 libxext6 git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 \
           && apt-get clean \
           && rm -rf /var/lib/apt/lists/*

       # Install MMCV
       RUN pip3 install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu110/torch1.7.1/index.html
       ```
    - After generating docker image, in the container do,
        > git clone https://github.com/open-mmlab/mmcv.git

        > cd mmcv

        > MMCV_WITH_OPS=1 MMCV_CUDA_ARGS='-gencode=arch=compute_80,code=sm_80' pip install -e . # package mmcv-full will be installed after this step

        > cd ..

        > git clone https://github.com/open-mmlab/mmdetection.git

        > cd mmdetection

        > pip install -r requirements/build.txt

        > MMCV_WITH_OPS=1 MMCV_CUDA_ARGS='-gencode=arch=compute_80,code=sm_80' pip install -e . # or "python setup.py develop"


#### Classification
- For classification task, the model is based on [timm](https://pypi.org/project/timm/) pacakge.
- With the formatted directory structure and annotations, we can train / test the model easily.
- Refer: https://github.com/microsoft/Swin-Transformer/blob/main/get_started.md

#### Detection
- For Detection task, the model is based on [mmdetection](https://github.com/open-mmlab/mmdetection) toolbox
- Swin Trasnformer for Object Detection: https://github.com/SwinTransformer/Swin-Transformer-Object-Detection
- Due to the mmdetection, all the model / dataset structures are setup with **config** files.
- There exists **__base__** configs for the dataset / models, and they are imported at other specific model configs.
- **For Swin, it does not support single GPU training script so run with tools/train_dist.sh with #gpu = 1 parameter.**

#### For Custom Dataset
- There are things to modify / add to train / test with custom dataset
  1. Add custom dataset script.py at mmdet/datasts by duplicating coco.py or other format. 
  2. Edit CLASSES with the class name (ex: ('fire',))
     
     2.1. if only 1 class, must need to add comma after 'class'
  3. Add config file for model option. duplicate scripts at configs/swin/
    
     3.1. need to change num_classes to the amount of you want to train
  5. Add coco_instance_custom.py or other format by referring coco_instance.py at configs/__base__/datasets/coco_instance.py


### References
- https://byeongjo-kim.tistory.com/36
- https://www.youtube.com/watch?v=AA621UofTUA
- https://mlfromscratch.com/activation-functions-explained/#/
- https://wikidocs.net/31379
- https://visionhong.tistory.com/31
- https://github.com/microsoft/Swin-Transformer
- https://xzcodes.github.io/posts/paper-review-swin-transformer
