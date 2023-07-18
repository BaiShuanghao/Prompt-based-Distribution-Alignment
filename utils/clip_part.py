import copy

import clip
from clip import clip

import torch
import torch.nn as nn
from torch.nn import functional as F


def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url, cfg.MODEL.BACKBONE.PATH)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())

    return model


class Simple_TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.token_embedding = clip_model.token_embedding
        self.positional_embedding = clip_model.positional_embedding
        self.transformer = clip_model.transformer
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, text):
        x = self.token_embedding(text).type(self.dtype)
        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
        return x


class TextEncoder(nn.Module):
    def __init__(self, cfg, clip_model, prompt_learner=None):
        super().__init__()
        self.positional_embedding = clip_model.positional_embedding
        self.transformer = clip_model.transformer
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype
        
        trainer = cfg.TRAINER.NAME.split('_')[0].upper()
        self.tp = cfg.TRAINER[trainer].TP if hasattr(cfg.TRAINER[trainer], 'TP') else False
        self.deep = cfg.TRAINER[trainer].T_DEEP if hasattr(cfg.TRAINER[trainer], 'T_DEEP') else False
        self.num_tokens = cfg.TRAINER[trainer].NUM_TOKENS if hasattr(cfg.TRAINER[trainer], 'NUM_TOKENS') else 10
        self.location = cfg.TRAINER[trainer].LOCATION if hasattr(cfg.TRAINER[trainer], 'LOCATION') else 'middle'
        self.deep_layer = cfg.TRAINER[trainer].DEEP_LAYERS if hasattr(cfg.TRAINER[trainer], 'DEEP_LAYERS') else None
        self.num_layer = cfg.MODEL.NUM_LAYER if hasattr(cfg.MODEL, 'NUM_LAYERS') else 12  
        
        dropout = cfg.TRAINER[trainer].prompt_dropout if hasattr(cfg.TRAINER[trainer], 'prompt_dropout') else 0.0
        self.prompt_dropout = nn.Dropout(dropout)
        
        self.enbale_adpater = cfg.TRAINER[trainer].ENABLE_ADAPTER if hasattr(cfg.TRAINER[trainer], 'ENABLE_ADAPTER') else False
        if self.enbale_adpater:
            self.adapter = prompt_learner.adapter
            self.deep_adapter = prompt_learner.deep_adapter
       
    def forward(self, prompts, tokenized_prompts, deep_prompts=None):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        if self.enbale_adpater or self.deep:
            x = self.transformer_deep(x, deep_prompts)
        else:
            x = self.transformer(x) 
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x
    
    def transformer_deep(self, x, deep_x):
        if self.deep:
            if deep_x.dim == 2:
                if self.deep_layer == None:
                    deep_x = deep_x.unsueeze(0).expand(self.num_layer - 1, -1, -1)  # all layers exsit prompt
                else:
                    deep_x = deep_x.unsueeze(0).expand(self.deep_layer[1] - self.deep_layer[0] + 1, -1, -1) # only specified layers exsit prompt
            
        for i in range(self.num_layer):
            if i == 0:
                if self.enbale_adpater:     # adapter is not activate
                    x = self.resblocks_adapter(i, x)
                else:
                    x = self.transformer.resblocks[i](x)
                    
            else:
                if self.deep:
                    if self.deep_layer == None:   # only specified layers exsit prompt
                        if i <= deep_x.shape[0]:   
                            deep_ctx_i =  self.prompt_dropout(deep_x[i-1])
                            deep_ctx = deep_ctx_i.expand(x.shape[1], -1, -1).permute(1, 0, 2).half()

                            if self.location == "middle":
                                x = torch.cat((x[:1, :, :], deep_ctx, x[(1+self.num_tokens):, :, :]), dim=0)
                            else:   # 'last'
                                prefix = x[0: x.shape[0] - self.num_tokens, :, :]
                                x = torch.cat([prefix, deep_ctx], dim=0)
                    else: # all layers exsit prompt
                        j = 0
                        if i in range(self.deep_layer[0], self.deep_layer[1]+1):
                            deep_ctx_i =  self.prompt_dropout(deep_x[j])
                            j = j + 1
                            deep_ctx = deep_ctx_i.expand(x.shape[1], -1, -1).permute(1, 0, 2).half()

                            if self.location == "middle":
                                x = torch.cat((x[:1, :, :], deep_ctx, x[(1+self.num_tokens):, :, :]), dim=0)
                            else:   # 'last'
                                prefix = x[0: x.shape[0] - self.num_tokens, :, :]
                                x = torch.cat([prefix, deep_ctx], dim=0)
                    
                if self.enbale_adpater:
                    x = self.resblocks_adapter(i, x)  
                else:
                    x = self.transformer.resblocks[i](x)
                    
        return x
    
    def resblocks_adapter(self, i, x):
        attn = self.transformer.resblocks[i].attn
        ln_1 = self.transformer.resblocks[i].ln_1
        mlp = self.transformer.resblocks[i].mlp
        ln_2 = self.transformer.resblocks[i].ln_2
        attn_mask = self.transformer.resblocks[i].attn_mask
        
        def attention(x, attn, attn_mask):
            attn_mask = attn_mask.to(dtype=x.dtype, device=x.device) if attn_mask is not None else None
            return attn(x, x, x, need_weights=False, attn_mask=attn_mask)[0]

        x = x + attention(ln_1(x), attn, attn_mask)
        residual = self.adapter(x) * 0.5
        x = x + residual + mlp(ln_2(x))
        return x
    
          
class ImageEncoder_Conv(nn.Module):
    def __init__(self, cfg, clip_model, prompt_learner=None):
        super().__init__()
        self.conv1, self.bn1, self.relu1 = clip_model.visual.conv1, clip_model.visual.bn1, clip_model.visual.relu1
        self.conv2, self.bn2, self.relu2 = clip_model.visual.conv2, clip_model.visual.bn2, clip_model.visual.relu2
        self.conv3, self.bn3, self.relu3 = clip_model.visual.conv3, clip_model.visual.bn3, clip_model.visual.relu3
        self.avgpool = clip_model.visual.avgpool

        self.layer1 = clip_model.visual.layer1
        self.layer2 = clip_model.visual.layer2
        self.layer3 = clip_model.visual.layer3
        self.layer4 = clip_model.visual.layer4
        self.attnpool = clip_model.visual.attnpool
        
        trainer = cfg.TRAINER.NAME.split('_')[0].upper()
        self.prompt_learner = prompt_learner
        self.dim = clip_model.text_projection.shape[1]

    def forward(self, x, vctx=None, deep_vctx=None, return_feat=False):
        def stem(x):
            for conv, bn, relu in [(self.conv1, self.bn1, self.relu1), (self.conv2, self.bn2, self.relu2), (self.conv3, self.bn3, self.relu3)]:
                x = relu(bn(conv(x)))
            x = self.avgpool(x)
            return x

        x = x.type(self.conv1.weight.dtype)
        x = stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)  # [B, C, H, W] = [B, 2048, 7, 7]
        if return_feat:
            Fs = x.permute(0, 2, 3, 1).view(x.shape[0], -1, x.shape[1]) # [B, 49, 2048]
            Fs = F.adaptive_avg_pool1d(Fs, self.dim) # [B, 49, 1024]
        x = self.attnpool(x)    # [B, 1024]

        if return_feat:
            return x, Fs
        
        return x
    

class ImageEncoder_Trans(nn.Module):
    def __init__(self, cfg, clip_model, prompt_learner=None):
        super().__init__()
        self.conv1 = clip_model.visual.conv1
        self.class_embedding = clip_model.visual.class_embedding
        self.positional_embedding = clip_model.visual.positional_embedding
        self.ln_pre = clip_model.visual.ln_pre
        self.transformer = clip_model.visual.transformer
        self.ln_post = clip_model.visual.ln_post

        if clip_model.visual.proj is not None:
            self.proj = clip_model.visual.proj
        
        trainer = cfg.TRAINER.NAME.split('_')[0].upper()
        self.vp = cfg.TRAINER[trainer].VP if hasattr(cfg.TRAINER[trainer], 'VP') else False
        self.deep = cfg.TRAINER[trainer].V_DEEP if hasattr(cfg.TRAINER[trainer], 'V_DEEP') else False
        self.num_tokens = cfg.TRAINER[trainer].NUM_TOKENS if hasattr(cfg.TRAINER[trainer], 'NUM_TOKENS') else 10
        self.location = cfg.TRAINER[trainer].LOCATION if hasattr(cfg.TRAINER[trainer], 'LOCATION') else 'middle'
        self.deep_layer = cfg.TRAINER[trainer].DEEP_LAYERS if hasattr(cfg.TRAINER[trainer], 'DEEP_LAYERS') else None
        self.enable_attn = cfg.TRAINER[trainer].ENABLE_ATTN if hasattr(cfg.TRAINER[trainer], 'ENABLE_ATTN') else None
        self.num_layer = cfg.MODEL.NUM_LAYER if hasattr(cfg.MODEL, 'NUM_LAYERS') else 12
        
        dropout = cfg.TRAINER[trainer].prompt_dropout if hasattr(cfg.TRAINER[trainer], 'prompt_dropout') else 0.0
        self.prompt_dropout = nn.Dropout(dropout)
        
        self.prompt_learner = prompt_learner
        
        self.dim = clip_model.text_projection.shape[1]

    def forward(self, x, vctx=None, deep_vctx=None, return_feat=None):
        x = self.conv1(x)  # shape = [*, width, grid, grid] = [B, 768, 14, 14]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width] = [B, 196, 768]
        x = torch.cat(  # shape = [*, grid ** 2 + 1, width]
                [
                    self.class_embedding.to(x.dtype) + torch.zeros(
                    x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x
                ], 
                dim=1,
            )  
        x = x + self.positional_embedding.to(x.dtype)   # image embedding [B, 197, 768]

        if self.vp and vctx != None:
            x = self.incorporate_prompt(x, vctx)    # [B, 197+num_token, 768]

        x = self.ln_pre(x)      # [B, 197+num_token, 768]

        x = x.permute(1, 0, 2)  # NLD -> LND [197+num_token, B, 768]
        if not self.deep or deep_vctx == None:
            x = self.transformer(x) # [197+num_token, B, 768]
        else:
            x = self.transformer_deep(x, deep_vctx)
        x = x.permute(1, 0, 2)  # LND -> NLD    [B, 197+num_tokens, 768]
        
        if return_feat:
            Fs = x
            Fs = F.adaptive_avg_pool1d(Fs, self.dim) # [B, 197+num, 512]
        x = self.ln_post(x[:, 0, :])    # [B, 768]
    
        if self.proj is not None:
            x = x @ self.proj   # [B, 512]

        if return_feat:
            return x, Fs
        
        return x

    def transformer_deep(self, x, deep_x):
        if deep_x.dim == 2:
            if self.deep_layer == None:
                deep_x = deep_x.expand(self.num_layer - 1, -1, -1)
            else:
                deep_x = deep_x.expand(self.deep_layer[1] - self.deep_layer[0] + 1, -1, -1)
            
        for i in range(self.num_layer):
            if i == 0:
                x = self.transformer.resblocks[i](x)
            else:
                if self.deep_layer == None:
                    if i <= deep_x.shape[0]:
                        deep_ctx_i =  self.prompt_dropout(deep_x[i-1])
                        deep_ctx = deep_ctx_i.expand(x.shape[1], -1, -1).permute(1, 0, 2).half()

                        if self.location == "middle":
                            x = torch.cat((x[:1, :, :], deep_ctx, x[(1+self.num_tokens):, :, :]), dim=0)
                        else:   # 'last'
                            prefix = x[0: x.shape[0] - self.num_tokens, :, :]
                            x = torch.cat([prefix, deep_ctx], dim=0)
                else:
                    j = 0
                    if i in range(self.deep_layer[0], self.deep_layer[1]+1):
                        deep_ctx_i =  self.prompt_dropout(deep_x[j])
                        j = j + 1
                        deep_ctx = deep_ctx_i.expand(x.shape[1], -1, -1).permute(1, 0, 2).half()

                        if self.location == "middle":
                            x = torch.cat((x[:1, :, :], deep_ctx, x[(1+self.num_tokens):, :, :]), dim=0)
                        else:   # 'last'
                            prefix = x[0: x.shape[0] - self.num_tokens, :, :]
                            x = torch.cat([prefix, deep_ctx], dim=0)
                         
                x = self.transformer.resblocks[i](x)

        return x
    
    def incorporate_prompt(self, x, vctx):
        # combine prompt embeddings with image-patch embeddings
        if self.location == "middle":
            x = torch.cat((
                    x[:, :1, :],
                    self.prompt_dropout(vctx).expand(x.shape[0], -1, -1).half(),
                    x[:, 1:, :]
                ), dim=1)   # (batch_size, cls_token + n_prompt + n_patches, hidden_dim)
        else:
            visual_ctx = self.prompt_dropout(vctx).expand(x.shape[0], -1, -1).half()
            x = torch.cat([x, visual_ctx], dim=1)
        
        return x    # [B, 197 + num_token, 768]


