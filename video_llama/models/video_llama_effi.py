import logging
import random

import torch
from torch.cuda.amp import autocast as autocast
import torch.nn as nn

from video_llama.common.registry import registry
from video_llama.models.blip2 import Blip2Base, disabled_train
from video_llama.models.modeling_llama import LlamaForCausalLM
# from video_llama.models.Qformer import BertEncoder
from video_llama.common.registry import registry

from transformers import LlamaTokenizer,BertConfig
# from transformers.models.bert.modeling_bert import BertEncoder
import einops
import copy
from video_llama.models.Qformer import BertConfig, BertLMHeadModel
from video_llama.models.ImageBind.models.imagebind_model import ImageBindModel,ModalityType
from video_llama.models.ImageBind.models import imagebind_model
# from flamingo_pytorch import PerceiverResampler
import torch.nn.functional as F

from transformers.activations import ACT2FN
from transformers.modeling_utils import Conv1D
import numpy as np
import yaml
import os

class MLP(nn.Module):
    def __init__(self, n_state, nx, n_out, activation = 'relu'):  # in MLP: n_state=3072 (4 * n_embd)
        super().__init__()
        self.c_fc = Conv1D(n_state, nx)
        self.c_proj = nn.Linear(nx, n_out)
        #self.c_proj = Conv1D(nx, n_out)
        self.act = ACT2FN[activation]
        self.act2 = ACT2FN['sigmoid']
        #self.dropout = nn.Dropout(config.resid_pdrop)

    def forward(self, x):
        h = self.act(self.c_fc(x))
        h2 = self.c_proj(h)
        return self.act2(h2)
        #return self.dropout(h2)

class MLPs(nn.Module):
    def __init__(self, num_mlps, hidden_size):
        super(MLPs, self).__init__()

        self.mlps = nn.ModuleList([MLP(hidden_size, hidden_size, hidden_size) for _ in range(num_mlps)])

    def forward(self, x):
        for mlp in self.mlps:
            x = mlp(x)
        return x

@registry.register_model("video_llama_effi")
class VideoLLAMA_effi(Blip2Base):
    """
    BLIP2 GPT-LLAMA model.
    """

    PRETRAINED_MODEL_CONFIG_DICT = {
        "pretrain_vicuna": "configs/models/video_llama.yaml",
        "pretrain_llama_v2": "configs/models/video_llama.yaml",
    }

    @classmethod
    def init_video_Qformer(cls, num_query_token, vision_width,num_hidden_layers =2):
        encoder_config = BertConfig.from_pretrained("bert-base-uncased")
        encoder_config.num_hidden_layers = num_hidden_layers
        encoder_config.encoder_width = vision_width
        # insert cross-attention layer every other block
        encoder_config.add_cross_attention = True
        encoder_config.cross_attention_freq = 1
        encoder_config.query_length = num_query_token
        Qformer = BertLMHeadModel(config=encoder_config)
        query_tokens = nn.Parameter(
            torch.zeros(1, num_query_token, encoder_config.hidden_size)
        )
        query_tokens.data.normal_(mean=0.0, std=encoder_config.initializer_range)
        return Qformer, query_tokens

    def __init__(
        self,
        vit_model="eva_clip_g",
        q_former_model="https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/blip2_pretrained_flant5xxl.pth",
        img_size=224,
        drop_path_rate=0,
        use_grad_checkpoint=False,
        vit_precision="fp16",
        freeze_vit=True,
        freeze_qformer=True,
        freeze_light_predictor = False, 
        memory_merge = "",
        num_query_token=32,
        llama_model="",
        prompt_path="",
        prompt_template="",
        max_txt_len=32,
        end_sym='\n',
        low_resource=False,  # use 8 bit and put vit in cpu
        device_8bit=0,  # the device of 8bit model should be set when loading and cannot be changed anymore.

        frozen_llama_proj=True,
        frozen_video_Qformer=True,
        frozen_audio_Qformer=True,

        llama_proj_model='',
        fusion_header_type= "seqTransf",
        max_frame_pos= 32,
        fusion_head_layers = 2,
        num_video_query_token = 32,
        num_audio_query_token = 8,
        imagebind_ckpt_path = '/mnt/workspace/ckpt',
        equip_audio_branch = True, 
        
        light_encoder_lnum = 10,
        delta_token_num = 5,
        replace_video_qformer = False,

        num_thin_patch_token = 32,
        pick_delta_token = "",

        full_layer_num = 20,
        light_loss_weight = 1.0,
        light_weight = "",
        light_type = "",
        light2full = False, 
        full_gap=8
    ):
        super().__init__()

        self.tokenizer = self.init_tokenizer()
        self.low_resource = low_resource

        print('Loading VIT')
        self.visual_encoder, self.ln_vision = self.init_vision_encoder(
            vit_model, img_size, drop_path_rate, use_grad_checkpoint, vit_precision
        )
        self.ln_vision_shallow = copy.deepcopy(self.ln_vision)
        if freeze_vit:
            for name, param in self.visual_encoder.named_parameters():
                param.requires_grad = False
            self.visual_encoder = self.visual_encoder.eval()
            # self.visual_encoder.train = disabled_train   #by jsm
            for name, param in self.ln_vision.named_parameters():
                param.requires_grad = False
            self.ln_vision = self.ln_vision.eval()
            self.ln_vision.train = disabled_train
            logging.info("freeze vision encoder")
        print('Loading VIT Done')
        '''
        for name, module in self.named_modules():
            module.to("cuda:1")
        '''
            
        print('Loading Q-Former')
        self.Qformer, self.query_tokens = self.init_Qformer(
            num_query_token, self.visual_encoder.num_features
        )
        self.Qformer.cls = None
        self.Qformer.bert.embeddings.word_embeddings = None
        self.Qformer.bert.embeddings.position_embeddings = None
        for layer in self.Qformer.bert.encoder.layer:
            layer.output = None
            layer.intermediate = None
        self.load_from_pretrained(url_or_filename=q_former_model)

        self.light_encoder_lnum = light_encoder_lnum
        if light_weight == "qformer":
            self.light_predictor = self.Qformer
        else:
            self.light_predictor = copy.deepcopy(self.Qformer)

        self.memory_merge = memory_merge
        self.pick_delta_token = pick_delta_token
        self.full_layer_num = full_layer_num
        self.light_type = light_type
        self.light2full = light2full

        if self.light_type:
            for param in self.visual_encoder.import_head.parameters():
                param.requires_grad = True
            self.visual_encoder.import_head = self.visual_encoder.import_head.float()
            self.visual_encoder = self.visual_encoder.train()
            self.visual_encoder.train = disabled_train

        #预测头
        hidden_dim = 768  #self.video_encoder.qformer.config.hidden_size
        self.weight_head = nn.Linear(hidden_dim, 1)
        self.seg_head = MLP(hidden_dim, hidden_dim, 1)
        #self.seg_loss = nn.CrossEntropyLoss()
        self.seg_loss = nn.BCEWithLogitsLoss()
 
        self.delta_token_num = delta_token_num
        self.light_loss_weight = light_loss_weight

        if self.memory_merge == "predict_weight":
            self.merge_weight_predictor = nn.Linear(hidden_dim, 1)

        if freeze_qformer:
            for name, param in self.Qformer.named_parameters():
                param.requires_grad = False
            self.Qformer = self.Qformer.eval()
            self.Qformer.train = disabled_train
            self.query_tokens.requires_grad = False
            logging.info("freeze Qformer")
        logging.info('Loading Q-Former Done')

        if freeze_light_predictor and light_weight!="qformer":
            for name, param in self.light_predictor.named_parameters():
                param.requires_grad = False
            self.light_predictor = self.light_predictor.eval()
            self.light_predictor.train = disabled_train
            logging.info("freeze light predictor")

            for name, param in self.ln_vision_shallow.named_parameters():
                param.requires_grad = False
            self.ln_vision_shallow = self.ln_vision_shallow.eval()
            self.ln_vision_shallow.train = disabled_train
        logging.info('Loading light predictor Done')
        
        logging.info('Loading LLAMA Tokenizer')
        self.llama_tokenizer = LlamaTokenizer.from_pretrained(llama_model, use_fast=False)
        if self.llama_tokenizer.pad_token is None:
            self.llama_tokenizer.pad_token = self.llama_tokenizer.unk_token 
        DEFAULT_IMAGE_PATCH_TOKEN = '<ImageHere>'
        DEFAULT_AUDIO_PATCH_TOKEN = '<AudioHere>'
        self.llama_tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
        self.llama_tokenizer.add_tokens([DEFAULT_AUDIO_PATCH_TOKEN], special_tokens=True)
        
        self.IMAGE_PATCH_TOKEN_ID = self.llama_tokenizer.get_vocab()[DEFAULT_IMAGE_PATCH_TOKEN]
        self.AUDIO_PATCH_TOKEN_ID = self.llama_tokenizer.get_vocab()[DEFAULT_AUDIO_PATCH_TOKEN]

        logging.info('Loading LLAMA Model')
        if self.low_resource:
            self.llama_model = LlamaForCausalLM.from_pretrained(
                llama_model,
                torch_dtype=torch.bfloat16,
                load_in_8bit=True,
                device_map={'': device_8bit}
            )
        else:
            self.llama_model = LlamaForCausalLM.from_pretrained(
                llama_model,
                torch_dtype=torch.bfloat16,
            )

        for name, param in self.llama_model.named_parameters():
            param.requires_grad = False
        logging.info('Loading LLAMA Done')
        # self.llama_model = nn.Linear(
        #     self.llama_model.config.hidden_size,
        #     self.llama_model.config.hidden_size
        # )


        logging.info('Loading LLAMA proj')
        self.llama_proj = nn.Linear(
            self.Qformer.config.hidden_size, self.llama_model.config.hidden_size
        )
        if llama_proj_model:
            print("load llama proj weight: {}".format(llama_proj_model))
            llama_proj_weight = torch.load(llama_proj_model, map_location="cpu")
            msg = self.load_state_dict(llama_proj_weight['model'], strict=False)

        if frozen_llama_proj:
            #  todo frozen  llama_proj
            for name, param in self.llama_proj.named_parameters():
                param.requires_grad = False
            logging.info('LLAMA proj is frozen')
        else:
            for name, param in self.llama_proj.named_parameters():
                param.requires_grad = True
            logging.info('LLAMA proj is not frozen')

        logging.info('Loading llama_proj Done')

        self.max_txt_len = max_txt_len
        self.end_sym = end_sym


        if prompt_path:
            with open(prompt_path, 'r') as f:
                raw_prompts = f.read().splitlines()
            filted_prompts = [raw_prompt for raw_prompt in raw_prompts if "<ImageHere>" in raw_prompt]
            self.prompt_list = [prompt_template.format(p) for p in filted_prompts]
            print('Load {} training prompts'.format(len(self.prompt_list)))
            print('Prompt Example \n{}'.format(random.choice(self.prompt_list)))
        else:
            self.prompt_list = []

        self.video_frame_position_embedding = nn.Embedding(max_frame_pos, self.Qformer.config.hidden_size)

        #try MLP replace video_Qformer
        self.replace_video_qformer = replace_video_qformer
        if replace_video_qformer:
            self.video_MLP = MLPs(2, hidden_dim)
            self.num_video_query_token = num_video_query_token
        else:
            self.num_video_query_token = num_video_query_token
            self.video_Qformer,self.video_query_tokens = self.init_video_Qformer(num_query_token = num_video_query_token,\
                vision_width=self.Qformer.config.hidden_size, num_hidden_layers =2)

            self.video_Qformer.cls = None
            self.video_Qformer.bert.embeddings.word_embeddings = None
            self.video_Qformer.bert.embeddings.position_embeddings = None
            for layer in self.video_Qformer.bert.encoder.layer:
                layer.output = None
                layer.intermediate = None


            if frozen_video_Qformer:
                #  todo frozen  llama_proj
                for name, param in self.video_Qformer.named_parameters():
                    param.requires_grad = False
                for name, param in self.video_frame_position_embedding.named_parameters():
                    param.requires_grad = False
                self.video_query_tokens.requires_grad = False
                
                logging.info('video_Qformer is frozen')
            else:
                for name, param in self.video_Qformer.named_parameters():
                    param.requires_grad = True
                for name, param in self.video_frame_position_embedding.named_parameters():
                    param.requires_grad = True
                self.video_query_tokens.requires_grad = True
                logging.info('video_Qformer is not frozen')

        if frozen_video_Qformer and (not frozen_audio_Qformer):
            self.train_flag = 1 # 只训练audio_Qformer
        elif not(frozen_video_Qformer) and frozen_audio_Qformer:
            self.train_flag = 0 # 训练video_Qformer
        elif not(frozen_video_Qformer) and not(frozen_audio_Qformer):
            self.train_flag = 2 # video_Qformer and AL trained
        else:
            self.train_flag = 0  #3

        if equip_audio_branch:
            print (f'Initializing audio encoder from {imagebind_ckpt_path} ...')
            self.audio_encoder,self.audio_hidden_size = \
                imagebind_model.imagebind_huge()
            self.audio_encoder.load_state_dict(torch.load("{}/imagebind_huge.pth".format(imagebind_ckpt_path)))
            # free vision encoder
            for name, param in self.audio_encoder.named_parameters():
                param.requires_grad = False
            self.audio_encoder.eval()
            print ('audio encoder initialized.')
            
            self.num_audio_query_token = num_audio_query_token
            self.audio_Qformer,self.audio_query_tokens = self.init_video_Qformer(num_query_token = self.num_audio_query_token,\
                vision_width=self.audio_hidden_size, num_hidden_layers =2)
            self.audio_Qformer.cls = None
            self.audio_Qformer.bert.embeddings.word_embeddings = None
            self.audio_Qformer.bert.embeddings.position_embeddings = None
            for layer in self.audio_Qformer.bert.encoder.layer:
                layer.output = None
                layer.intermediate = None
            self.audio_llama_proj = nn.Linear(
                self.audio_Qformer.config.hidden_size, self.llama_model.config.hidden_size
            )
            self.audio_position_embedding = nn.Embedding(8, self.audio_hidden_size)

            if frozen_audio_Qformer:
                #  todo frozen  llama_proj
                for name, param in self.audio_Qformer.named_parameters():
                    param.requires_grad = False
                self.audio_query_tokens.requires_grad = False
                for name, param in self.audio_llama_proj.named_parameters():
                    param.requires_grad = False
                for name, param in self.audio_position_embedding.named_parameters():
                    param.requires_grad = False
                logging.info('audio_Qformer and audio-LLAMA proj is frozen')
            else:
                for name, param in self.audio_Qformer.named_parameters():
                    param.requires_grad = True
                self.audio_query_tokens.requires_grad = True
                for name, param in self.audio_llama_proj.named_parameters():
                    param.requires_grad = True
                for name, param in self.audio_position_embedding.named_parameters():
                    param.requires_grad = True
                logging.info('audio_Qformer is not frozen')

        self.num_thin_patch_token = num_thin_patch_token
        self.full_gap = full_gap
    '''
    for name, module in self.named_modules():
        if next(module.parameters()).device == "cpu":
            print(name)
            for name, module in self.named_modules():
                if next(module.parameters()) == "cpu":
                    module.to('cuda:0')
    '''


        #  self.audio_hidden_size
    def vit_to_cpu(self):
        self.ln_vision.to("cpu")
        self.ln_vision.float()
        self.visual_encoder.to("cpu")
        self.visual_encoder.float()
    
    def cal_lightPredictor_loss(self, tokens, tokens_full):
        tokens = F.log_softmax(tokens, dim=-1)
        tokens_full = F.softmax(tokens_full, dim=-1)

        loss = F.kl_div(tokens, tokens_full, reduction='batchmean') / (tokens.shape[0])
        return loss

    def encode_lightPredictor_segment(self, image, timestamps, n_f):
        #先按照每个batch=1来写
        #根据分段，每段生成num_f的视频特征
        # 获取视频长度
        video_len = image.shape[2]
        num_seg = timestamps.size(0)
        # 初始化结果张量
        result = torch.zeros((num_seg, 3, n_f, 224, 224), dtype=torch.float32)

        start = timestamps[0,:,0]
        end = timestamps[0,:,1]

        lengths = torch.tensor([end - start for start, end in timestamps[0]]).to(start.device)
        indexes = torch.linspace(0, 1, n_f).view(1,n_f).to(start.device)
        indexes = (indexes * lengths.view(-1, 1)).long() + start.view(-1, 1)
        indexes = torch.clamp(indexes, min=0, max=video_len - 1)
        # 生成采样索引
        # indexes = torch.arange(0, n_f).float() / (n_f - 1) * (end - start) + start
        # indexes = torch.clamp(indexes.long(), min=0, max=video_len - 1)
        
        # 使用索引获取对应帧
        sampled_frames = image[:, :, indexes]
        sampled_frames = sampled_frames.squeeze(0).transpose(0,1)
        
        return sampled_frames

    def encode_videoQformer_lightPredictor_visual_eval_2(self, image, label_seg = None, long_video = False, video_path="", seg_len = 0):
        device = image.device
        
        # input shape b,c,t,h,w
        batch_size,_,time_length,_,_ = image.size()
        with self.maybe_autocast():
            image_full_path = image[:,:,::seg_len,...]
            image = einops.rearrange(image, 'b c t h w -> (b t) c h w')
            # embed image features with blip2, out: (b t) q h
            # image_embeds_shallow = self.ln_vision_shallow(self.visual_encoder.forward_features_shallow(image, self.light_encoder_lnum)).to(device)
            image_embeds_shallow = self.ln_vision_shallow(self.visual_encoder.forward_features_thin(image, topk=self.num_thin_patch_token, full_layer_num=self.full_layer_num)).to(device)  #32
            #image_embeds_shallow = einops.rearrange(image_embeds_shallow, '(b t) q h -> b t q h', b=batch_size, t=time_length)
            #image_embeds_shallow = einops.rearrange(image_embeds_shallow[:,1:,...], 'b t q h -> (b t) q h')

            # image_embeds = self.ln_vision(self.visual_encoder(image)).to(device)
            
            image_full_path = einops.rearrange(image_full_path, 'b c t h w -> (b t) c h w')
            image_embeds_full = self.ln_vision(self.visual_encoder(image_full_path)).to(device)

            image_atts = torch.ones(image_embeds_full.size()[:-1], dtype=torch.long).to(device)

            query_tokens = self.query_tokens.expand(image_embeds_full.shape[0], -1, -1)
            query_output = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds_full,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )


            # add frame_pos embedding
            position_ids = torch.arange(time_length, dtype=torch.long, device=query_tokens.device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
            frame_position_embeddings = self.video_frame_position_embedding(position_ids)

            frame_position_embeddings = frame_position_embeddings.unsqueeze(-2)

            q_hidden_state = query_output.last_hidden_state
            # frame_hidden_state_ori = einops.rearrange(q_hidden_state, '(b t) q h -> b t q h',b=batch_size,t=time_length)
            frame_hidden_state_ori = einops.rearrange(q_hidden_state, '(b t) q h -> b t q h',b=batch_size)
            
            frame_hidden_state_ls = []
            # frame_hidden_state = frame_hidden_state[:,:1,...] + frame_position_embeddings[:,:1,...]
            frame_hidden_state = frame_hidden_state_ori[:,:1,...] + frame_position_embeddings[:,:1,...]
                
            frame_hidden_state = frame_hidden_state.squeeze(1)
            #自回归预测
            #initialize with query_output, cross attention to image_feats_shallow
            query_tokens_shallow = query_output.last_hidden_state
            query_tokens_shallow = einops.rearrange(query_tokens_shallow, '(b t) q h -> b t q h', b=batch_size)
            query_tokens_shallow = einops.rearrange(query_tokens_shallow[:,:1,...], 'b t q h -> (b t) q h')

            image_embeds_shallow = einops.rearrange(image_embeds_shallow, '(b t) q h -> b t q h', b=batch_size, t=time_length)
            for i in range(1, time_length): 
                if i % seg_len == 0:
                    #取query_output的对应下标结果
                    idx_cur = i // seg_len
                    q_hidden_state_shallow = frame_hidden_state_ori[:,idx_cur:idx_cur+1,...].squeeze(1)
                else:
                    image_embeds_shallow_cur = einops.rearrange(image_embeds_shallow[:,i:(i+1),...], 'b t q h -> (b t) q h')
                    query_output_shallow = self.light_predictor.bert(
                        query_embeds=query_tokens_shallow,
                        encoder_hidden_states=image_embeds_shallow_cur,   #[]这里应该image错位一下
                        encoder_attention_mask=image_atts[:1, :image_embeds_shallow_cur.size(1),...],
                        return_dict=True,
                    )
                    q_hidden_state_shallow = query_output_shallow.last_hidden_state

                #更新query_output_shallow
                # query_tokens_shallow = query_tokens_shallow * i/(i+1) + q_hidden_state_shallow * 1/(i+1)
                query_tokens_shallow = self.merge_memory(query_tokens_shallow, q_hidden_state_shallow)

                frame_hidden_state_shallow = einops.rearrange(q_hidden_state_shallow, '(b t) q h -> b t q h',b=batch_size,t=1)

                frame_hidden_state_shallow = frame_position_embeddings[:,i:(i+1),...] + frame_hidden_state_shallow
                #选出显著变化的token
                #预测头
                weights = self.weight_head(frame_hidden_state_shallow).sigmoid()  #要sigmoid
                added_idx = torch.topk(weights, k=self.delta_token_num, dim=2)[1]
                expand_idx = added_idx.expand(-1,-1, -1, frame_hidden_state_shallow.size(-1))
                added_tokens_cur = frame_hidden_state_shallow.gather(dim=2, index=expand_idx)

                #和第一帧链接起来
                added_tokens =  einops.rearrange(added_tokens_cur, 'b t q h -> b (t q) h', b=batch_size, t=1)
                frame_hidden_state = torch.cat((frame_hidden_state, added_tokens),dim=1)

                
            
            if long_video:
                d_emb = frame_hidden_state.size(-1)

                position_ids = torch.arange(frame_hidden_state.size(0), dtype=torch.long, device=frame_hidden_state.device)
                position_ids = position_ids.unsqueeze(0)
                frame_position_embeddings = self.video_frame_position_embedding(position_ids)
                frame_position_embeddings = frame_position_embeddings.squeeze(0).unsqueeze(1)

                frame_hidden_state = frame_hidden_state + frame_position_embeddings
                frame_hidden_state = frame_hidden_state.view(-1,d_emb).unsqueeze(0)   #片段放到一起，但是应该加位置信息？
            frame_atts = torch.ones(frame_hidden_state.size()[:-1], dtype=torch.long).to(device)
            video_query_tokens = self.video_query_tokens.expand(frame_hidden_state.shape[0], -1, -1)


            video_query_output = self.video_Qformer.bert(
                query_embeds=video_query_tokens,
                encoder_hidden_states=frame_hidden_state,   #[:,:32,:],
                encoder_attention_mask=frame_atts,  #[:,:32],
                return_dict=True,
                )
            video_hidden = video_query_output.last_hidden_state

            inputs_llama = self.llama_proj(video_hidden)
            atts_llama = torch.ones(inputs_llama.size()[:-1], dtype=torch.long).to(image_embeds_shallow.device)
        return inputs_llama, atts_llama

    def encode_videoQformer_lightPredictor_visual_eval_FLOPs(self, image, label_seg = None, long_video = False, video_path="", seg_len = 0):
        device = image.device
        
        # input shape b,c,t,h,w
        batch_size,_,time_length,_,_ = image.size()
        image = einops.rearrange(image, 'b c t h w -> (b t) c h w')
        with self.maybe_autocast():
            # embed image features with blip2, out: (b t) q h
            # image_embeds_shallow = self.ln_vision_shallow(self.visual_encoder.forward_features_shallow(image, self.light_encoder_lnum)).to(device)
            if self.light_type == "MOD":
                image_embeds_shallow = self.ln_vision_shallow(self.visual_encoder.forward_features_MOD(image, topk=self.num_thin_patch_token, full_layer_num=self.full_layer_num)).to(device)  #32
            else:
                image_embeds_shallow = self.ln_vision_shallow(self.visual_encoder.forward_features_thin(image, topk=self.num_thin_patch_token, full_layer_num=self.full_layer_num)).to(device)  #32
            #image_embeds_shallow = einops.rearrange(image_embeds_shallow, '(b t) q h -> b t q h', b=batch_size, t=time_length)
            #image_embeds_shallow = einops.rearrange(image_embeds_shallow[:,1:,...], 'b t q h -> (b t) q h')
            image_embeds = self.ln_vision(self.visual_encoder(image[:1,...])).to(device)
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(device)

            query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
            query_output = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )

            # add frame_pos embedding
            position_ids = torch.arange(time_length, dtype=torch.long, device=query_tokens.device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
            frame_position_embeddings = self.video_frame_position_embedding(position_ids)

            frame_position_embeddings = frame_position_embeddings.unsqueeze(-2)

            q_hidden_state = query_output.last_hidden_state
            frame_hidden_state = einops.rearrange(q_hidden_state, '(b t) q h -> b t q h',b=batch_size,t=1)
            
            frame_hidden_state_ls = []
            frame_hidden_state = frame_hidden_state[:,:1,...] + frame_position_embeddings[:,:1,...]
                
            frame_hidden_state = frame_hidden_state.squeeze(1)
            #自回归预测
            #initialize with query_output, cross attention to image_feats_shallow
            query_tokens_shallow = query_output.last_hidden_state
            query_tokens_shallow = einops.rearrange(query_tokens_shallow, '(b t) q h -> b t q h', b=batch_size, t=1)
            query_tokens_shallow = einops.rearrange(query_tokens_shallow[:,:1,...], 'b t q h -> (b t) q h')

            image_embeds_shallow = einops.rearrange(image_embeds_shallow, '(b t) q h -> b t q h', b=batch_size, t=time_length)
            for i in range(1, time_length): 
                image_embeds_shallow_cur = einops.rearrange(image_embeds_shallow[:,i:(i+1),...], 'b t q h -> (b t) q h')
                query_output_shallow = self.light_predictor.bert(
                    query_embeds=query_tokens_shallow,
                    encoder_hidden_states=image_embeds_shallow_cur,   #[]这里应该image错位一下
                    encoder_attention_mask=image_atts[:1, :image_embeds_shallow_cur.size(1),...],
                    return_dict=True,
                )
                q_hidden_state_shallow = query_output_shallow.last_hidden_state

                #更新query_output_shallow
                # query_tokens_shallow = query_tokens_shallow * i/(i+1) + q_hidden_state_shallow * 1/(i+1)
                query_tokens_shallow = self.merge_memory(query_tokens_shallow, q_hidden_state_shallow)

                frame_hidden_state_shallow = einops.rearrange(q_hidden_state_shallow, '(b t) q h -> b t q h',b=batch_size,t=1)

                frame_hidden_state_shallow = frame_position_embeddings[:,i:(i+1),...] + frame_hidden_state_shallow
                #选出显著变化的token
                if self.pick_delta_token == "":
                    #预测头
                    weights = self.weight_head(frame_hidden_state_shallow).sigmoid()  #要sigmoid
                    added_idx = torch.topk(weights, k=self.delta_token_num, dim=2)[1]
                    expand_idx = added_idx.expand(-1,-1, -1, frame_hidden_state_shallow.size(-1))
                    added_tokens_cur = frame_hidden_state_shallow.gather(dim=2, index=expand_idx)
                elif self.pick_delta_token == "change":
                    frame_hidden_state_before = einops.rearrange(query_tokens_shallow, '(b t) q h -> b t q h',b=batch_size)
                    distance = -1*F.cosine_similarity(frame_hidden_state_shallow, frame_hidden_state_before, dim=-1).unsqueeze(-1)
                    added_idx = torch.topk(distance, k=self.delta_token_num, dim=2)[1]
                    # added_idx, _ = torch.sort(added_idx, dim=2)
                    expand_idx = added_idx.expand(-1,-1, -1, frame_hidden_state_shallow.size(-1))
                    added_tokens_cur = frame_hidden_state_shallow.gather(dim=2, index=expand_idx)
                elif self.pick_delta_token == "all":
                    added_tokens_cur = frame_hidden_state_shallow
                # #预测头
                # weights = self.weight_head(frame_hidden_state_shallow).sigmoid()  #要sigmoid
                # added_idx = torch.topk(weights, k=self.delta_token_num, dim=2)[1]
                # expand_idx = added_idx.expand(-1,-1, -1, frame_hidden_state_shallow.size(-1))
                # added_tokens_cur = frame_hidden_state_shallow.gather(dim=2, index=expand_idx)

                #和第一帧链接起来
                added_tokens =  einops.rearrange(added_tokens_cur, 'b t q h -> b (t q) h', b=batch_size, t=1)
                frame_hidden_state = torch.cat((frame_hidden_state, added_tokens),dim=1)

                
            
            if long_video:
                d_emb = frame_hidden_state.size(-1)

                position_ids = torch.arange(frame_hidden_state.size(0), dtype=torch.long, device=frame_hidden_state.device)
                position_ids = position_ids.unsqueeze(0)
                frame_position_embeddings = self.video_frame_position_embedding(position_ids)
                frame_position_embeddings = frame_position_embeddings.squeeze(0).unsqueeze(1)

                frame_hidden_state = frame_hidden_state + frame_position_embeddings
                frame_hidden_state = frame_hidden_state.view(-1,d_emb).unsqueeze(0)   #片段放到一起，但是应该加位置信息？
            frame_atts = torch.ones(frame_hidden_state.size()[:-1], dtype=torch.long).to(device)
            video_query_tokens = self.video_query_tokens.expand(frame_hidden_state.shape[0], -1, -1)


            video_query_output = self.video_Qformer.bert(
                query_embeds=video_query_tokens,
                encoder_hidden_states=frame_hidden_state,   #[:,:32,:],
                encoder_attention_mask=frame_atts,  #[:,:32],
                return_dict=True,
                )
            video_hidden = video_query_output.last_hidden_state

            inputs_llama = self.llama_proj(video_hidden)
            atts_llama = torch.ones(inputs_llama.size()[:-1], dtype=torch.long).to(image_embeds.device)
        return inputs_llama, atts_llama

    def encode_videoQformer_lightPredictor_visual_visualize_feat(self, image, label_seg = None, long_video = False, video_path="", topk=-1, full_layer_num=-1, seg_len = 0):
        device = image.device
        
        # input shape b,c,t,h,w
        batch_size,_,time_length,_,_ = image.size()
        image = einops.rearrange(image, 'b c t h w -> (b t) c h w')
        with self.maybe_autocast():
            # embed image features with blip2, out: (b t) q h
            # image_embeds_shallow = self.ln_vision_shallow(self.visual_encoder.forward_features_shallow(image, self.light_encoder_lnum)).to(device)
            if topk==-1 or full_layer_num == -1:
                x= self.visual_encoder.forward_features_thin_visualize_feat(image, topk=self.num_thin_patch_token, full_layer_num=self.full_layer_num)
            else:
                x = self.visual_encoder.forward_features_thin_visualize_feat(image, topk=topk, full_layer_num=full_layer_num)
        return x

    def encode_videoQformer_lightPredictor_visual_visualize(self, image, label_seg = None, long_video = False, video_path="", topk=-1, full_layer_num=-1, seg_len = 0):
        device = image.device
        
        # input shape b,c,t,h,w
        batch_size,_,time_length,_,_ = image.size()
        image = einops.rearrange(image, 'b c t h w -> (b t) c h w')
        with self.maybe_autocast():
            # embed image features with blip2, out: (b t) q h
            # image_embeds_shallow = self.ln_vision_shallow(self.visual_encoder.forward_features_shallow(image, self.light_encoder_lnum)).to(device)
            if topk==-1 or full_layer_num == -1:
                delta, mask = self.visual_encoder.forward_features_thin_visualize(image, topk=self.num_thin_patch_token, full_layer_num=self.full_layer_num)
            else:
                delta, mask = self.visual_encoder.forward_features_thin_visualize(image, topk=topk, full_layer_num=full_layer_num)
        return delta, mask[1:,:]

    def encode_videoQformer_lightPredictor_visual_eval(self, image, label_seg = None, long_video = False, video_path="", seg_len = 0, first_num=1):
        device = image.device
        
        # input shape b,c,t,h,w
        batch_size,_,time_length,_,_ = image.size()
        image = einops.rearrange(image, 'b c t h w -> (b t) c h w')
        with self.maybe_autocast():
            # embed image features with blip2, out: (b t) q h
            # image_embeds_shallow = self.ln_vision_shallow(self.visual_encoder.forward_features_shallow(image, self.light_encoder_lnum)).to(device)
            image_embeds_shallow = self.ln_vision_shallow(self.visual_encoder.forward_features_thin(image, topk=self.num_thin_patch_token, full_layer_num=self.full_layer_num)).to(device)  #32
            #image_embeds_shallow = einops.rearrange(image_embeds_shallow, '(b t) q h -> b t q h', b=batch_size, t=time_length)
            #image_embeds_shallow = einops.rearrange(image_embeds_shallow[:,1:,...], 'b t q h -> (b t) q h')
            image_embeds = self.ln_vision(self.visual_encoder(image)).to(device)
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(device)

            query_tokens = self.query_tokens.expand(image_embeds_shallow.shape[0], -1, -1)
            query_output = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )

            # add frame_pos embedding
            position_ids = torch.arange(time_length, dtype=torch.long, device=query_tokens.device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
            frame_position_embeddings = self.video_frame_position_embedding(position_ids)

            frame_position_embeddings = frame_position_embeddings.unsqueeze(-2)

            q_hidden_state = query_output.last_hidden_state
            frame_hidden_state = einops.rearrange(q_hidden_state, '(b t) q h -> b t q h',b=batch_size,t=time_length)
            
            frame_hidden_state_ls = []
            # frame_hidden_state = frame_hidden_state[:,:1,...] + frame_position_embeddings[:,:1,...]
            # frame_hidden_state = frame_hidden_state.squeeze(1)
            frame_hidden_state = frame_hidden_state[:,:first_num,...] + frame_position_embeddings[:,:first_num,...]
            frame_hidden_state =  einops.rearrange(frame_hidden_state, 'b t q h -> b (t q) h')
                
            #自回归预测
            #initialize with query_output, cross attention to image_feats_shallow
            query_tokens_shallow_a = query_output.last_hidden_state
            query_tokens_shallow_a = einops.rearrange(query_tokens_shallow_a, '(b t) q h -> b t q h', b=batch_size, t=time_length)
            # query_tokens_shallow = einops.rearrange(query_tokens_shallow[:,:1,...], 'b t q h -> (b t) q h')
            query_tokens_shallow = einops.rearrange(query_tokens_shallow_a[:,(first_num-1):(first_num),...], 'b t q h -> (b t) q h')

            image_embeds_shallow = einops.rearrange(image_embeds_shallow, '(b t) q h -> b t q h', b=batch_size, t=time_length)
            for i in range(first_num, time_length):
                if i % self.full_gap ==0:
                    query_tokens_shallow = einops.rearrange(query_tokens_shallow_a[:,(i-1):i,...], 'b t q h -> (b t) q h')
                    frame_hidden_state_shallow = einops.rearrange(q_hidden_state_shallow, '(b t) q h -> b t q h',b=batch_size,t=1)

                    frame_hidden_state_shallow = frame_position_embeddings[:,i:(i+1),...] + frame_hidden_state_shallow
                    added_tokens_cur = frame_hidden_state_shallow
                else:
                    image_embeds_shallow_cur = einops.rearrange(image_embeds_shallow[:,i:(i+1),...], 'b t q h -> (b t) q h')
                    query_output_shallow = self.light_predictor.bert(
                        query_embeds=query_tokens_shallow,
                        encoder_hidden_states=image_embeds_shallow_cur,   #[]这里应该image错位一下
                        encoder_attention_mask=image_atts[:1, :image_embeds_shallow_cur.size(1),...],
                        return_dict=True,
                    )
                    q_hidden_state_shallow = query_output_shallow.last_hidden_state

                    #更新query_output_shallow
                    # query_tokens_shallow = query_tokens_shallow * i/(i+1) + q_hidden_state_shallow * 1/(i+1)
                    query_tokens_shallow = self.merge_memory(query_tokens_shallow, q_hidden_state_shallow)

                    frame_hidden_state_shallow = einops.rearrange(q_hidden_state_shallow, '(b t) q h -> b t q h',b=batch_size,t=1)

                    frame_hidden_state_shallow = frame_position_embeddings[:,i:(i+1),...] + frame_hidden_state_shallow
                    #选出显著变化的token
                    if self.pick_delta_token == "":
                        #预测头
                        weights = self.weight_head(frame_hidden_state_shallow).sigmoid()  #要sigmoid
                        added_idx = torch.topk(weights, k=self.delta_token_num, dim=2)[1]
                        expand_idx = added_idx.expand(-1,-1, -1, frame_hidden_state_shallow.size(-1))
                        added_tokens_cur = frame_hidden_state_shallow.gather(dim=2, index=expand_idx)
                    elif self.pick_delta_token == "change":
                        frame_hidden_state_before = einops.rearrange(query_tokens_shallow, '(b t) q h -> b t q h',b=batch_size)
                        distance = -1*F.cosine_similarity(frame_hidden_state_shallow, frame_hidden_state_before, dim=-1).unsqueeze(-1)
                        added_idx = torch.topk(distance, k=self.delta_token_num, dim=2)[1]
                        # added_idx, _ = torch.sort(added_idx, dim=2)
                        expand_idx = added_idx.expand(-1,-1, -1, frame_hidden_state_shallow.size(-1))
                        added_tokens_cur = frame_hidden_state_shallow.gather(dim=2, index=expand_idx)
                    elif self.pick_delta_token == "all":
                        added_tokens_cur = frame_hidden_state_shallow
                # #预测头
                # weights = self.weight_head(frame_hidden_state_shallow).sigmoid()  #要sigmoid
                # added_idx = torch.topk(weights, k=self.delta_token_num, dim=2)[1]
                # expand_idx = added_idx.expand(-1,-1, -1, frame_hidden_state_shallow.size(-1))
                # added_tokens_cur = frame_hidden_state_shallow.gather(dim=2, index=expand_idx)

                #和第一帧链接起来
                added_tokens =  einops.rearrange(added_tokens_cur, 'b t q h -> b (t q) h', b=batch_size, t=1)
                frame_hidden_state = torch.cat((frame_hidden_state, added_tokens),dim=1)

                
            
            if long_video:
                d_emb = frame_hidden_state.size(-1)

                position_ids = torch.arange(frame_hidden_state.size(0), dtype=torch.long, device=frame_hidden_state.device)
                position_ids = position_ids.unsqueeze(0)
                frame_position_embeddings = self.video_frame_position_embedding(position_ids)
                frame_position_embeddings = frame_position_embeddings.squeeze(0).unsqueeze(1)

                frame_hidden_state = frame_hidden_state + frame_position_embeddings
                frame_hidden_state = frame_hidden_state.view(-1,d_emb).unsqueeze(0)   #片段放到一起，但是应该加位置信息？
            frame_atts = torch.ones(frame_hidden_state.size()[:-1], dtype=torch.long).to(device)
            video_query_tokens = self.video_query_tokens.expand(frame_hidden_state.shape[0], -1, -1)


            video_query_output = self.video_Qformer.bert(
                query_embeds=video_query_tokens,
                encoder_hidden_states=frame_hidden_state,   #[:,:32,:],
                encoder_attention_mask=frame_atts,  #[:,:32],
                return_dict=True,
                )
            video_hidden = video_query_output.last_hidden_state

            inputs_llama = self.llama_proj(video_hidden)
            atts_llama = torch.ones(inputs_llama.size()[:-1], dtype=torch.long).to(image_embeds.device)
        return inputs_llama, atts_llama

    def merge_memory(self, memory_token, cur_token):
        if self.memory_merge == "" or self.memory_merge == "mean":
            result = (memory_token + cur_token) / 2
        elif self.memory_merge == "cos_sim":
            #当前和memory token的距离，
            sim = F.cosine_similarity(memory_token, cur_token, dim=-1).unsqueeze(-1)
            result = sim * memory_token + (1-sim) * cur_token
        elif self.memory_merge == "predict_weight":
            #当前和memory token的差，
            m = nn.Sigmoid()
            sim = self.merge_weight_predictor(cur_token - memory_token)
            sim = m(sim)
            result = sim * memory_token + (1-sim) * cur_token
        elif self.memory_merge == "only_cur":
            result = cur_token

        return result

    def encode_videoQformer_lightPredictor_visual(self, image, label_seg = None):
        device = image.device
        
        # input shape b,c,t,h,w
        batch_size,_,time_length,_,_ = image.size()
        #image0 = einops.rearrange(image[:,:,:1,...], 'b c t h w -> (b t) c h w')
        image = einops.rearrange(image, 'b c t h w -> (b t) c h w')
        with self.maybe_autocast():
            # embed image features with blip2, out: (b t) q h
            # image_embeds_shallow = self.ln_vision_shallow(self.visual_encoder.forward_features_shallow(image, self.light_encoder_lnum)).to(device)
            if self.light2full:
                image_embeds_shallow, light_idx = self.visual_encoder.forward_features_thin(image, self.num_thin_patch_token, full_layer_num=self.full_layer_num, return_idx=self.light2full)
                image_embeds_shallow = self.ln_vision_shallow(image_embeds_shallow).to(device)  #32
            else:
                image_embeds_shallow = self.ln_vision_shallow(self.visual_encoder.forward_features_thin(image, self.num_thin_patch_token, full_layer_num=self.full_layer_num, return_idx=self.light2full)).to(device)  #32
            image_embeds_shallow = einops.rearrange(image_embeds_shallow, '(b t) q h -> b t q h', b=batch_size, t=time_length)
            image_embeds_shallow = einops.rearrange(image_embeds_shallow[:,1:,...], 'b t q h -> (b t) q h')
            image_embeds = self.ln_vision(self.visual_encoder(image)).to(device)
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(device)

            query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
            query_output = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )
            #initialize with query_output, cross attention to image_feats_shallow
            query_tokens_shallow = query_output.last_hidden_state
            query_tokens_shallow = einops.rearrange(query_tokens_shallow, '(b t) q h -> b t q h', b=batch_size, t=time_length)
            query_tokens_shallow = self.generate_query_shallow(query_tokens_shallow)
            # query_tokens_shallow = einops.rearrange(query_tokens_shallow[:,:-1,...], 'b t q h -> (b t) q h')

            if self.light2full:
                # image_embeds_shallow_ = image_embeds_shallow.copy()
                image_embeds = einops.rearrange(image_embeds, '(b t) q h -> b t q h', b=batch_size, t=time_length)
                image_embeds = einops.rearrange(image_embeds[:,1:,...], 'b t q h -> (b t) q h')
                light_idx = einops.rearrange(light_idx, '(b t) q h -> b t q h', b=batch_size, t=time_length)
                light_idx = einops.rearrange(light_idx[:,1:,...], 'b t q h -> (b t) q h')
                image_embeds_shallow = image_embeds.scatter(1, light_idx, image_embeds_shallow[:,:-1,:])
                # x = x.scatter(1, added_idx_expanded, topk_values)
                # image_embeds_shallow = 
            query_output_shallow = self.light_predictor.bert(
                query_embeds=query_tokens_shallow,
                encoder_hidden_states=image_embeds_shallow,   #[]这里应该image错位一下
                encoder_attention_mask=image_atts[:-batch_size, :image_embeds_shallow.size(1), ...],
                return_dict=True,
            )

            # add frame_pos embedding
            position_ids = torch.arange(time_length, dtype=torch.long, device=query_tokens.device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
            frame_position_embeddings = self.video_frame_position_embedding(position_ids)

            frame_position_embeddings = frame_position_embeddings.unsqueeze(-2)

            q_hidden_state_shallow = query_output_shallow.last_hidden_state
            frame_hidden_state_shallow = einops.rearrange(q_hidden_state_shallow, '(b t) q h -> b t q h',b=batch_size,t=time_length-1)

            q_hidden_state = query_output.last_hidden_state
            frame_hidden_state = einops.rearrange(q_hidden_state, '(b t) q h -> b t q h',b=batch_size,t=time_length)
            
            loss = self.cal_lightPredictor_loss(frame_hidden_state_shallow, frame_hidden_state[:,1:,...])

            frame_hidden_state_shallow = frame_position_embeddings[:,1:,...] + frame_hidden_state_shallow
            frame1_hidden_state = frame_position_embeddings[:,:1,...] + frame_hidden_state[:,:1,...]
            #  expand(<shape>)
            # frame attention
            #frame_hidden_state = frame_hidden_state[:,3,:,:].expand(8,32,768).unsqueeze(0)
            #判断哪里是分段的位置
            '''
            if label_seg != None:
                frame_hidden_state_full = torch.cat([torch.mean(frame1_hidden_state, dim=2), torch.mean(frame_hidden_state_shallow, dim=2)], dim=1)
                frame_delta = frame_hidden_state_full[:,1:,:] - frame_hidden_state_full[:,:-1,:]
                seg_predicted = self.seg_head(frame_delta).sigmoid().squeeze()  #要sigmoid
                #seg_predicted = self.seg_head(torch.mean(frame_hidden_state_shallow, dim=2)).sigmoid().squeeze()  #要sigmoid
                loss_seg = self.seg_loss(seg_predicted, label_seg[:,1:])
                loss = loss_seg
                return None, None, loss
            '''
                
            #选出显著变化的token
            if self.pick_delta_token == "":
                #预测头
                weights = self.weight_head(frame_hidden_state_shallow).sigmoid()  #要sigmoid
                added_idx = torch.topk(weights, k=self.delta_token_num, dim=2)[1]
                expand_idx = added_idx.expand(-1,-1, -1, frame_hidden_state_shallow.size(-1))
                added_tokens_cur = frame_hidden_state_shallow.gather(dim=2, index=expand_idx)
            elif self.pick_delta_token == "change":
                frame_hidden_state_before = einops.rearrange(query_tokens_shallow, '(b t) q h -> b t q h',b=batch_size)
                distance = -1*F.cosine_similarity(frame_hidden_state_shallow, frame_hidden_state_before, dim=-1).unsqueeze(-1)
                added_idx = torch.topk(distance, k=self.delta_token_num, dim=2)[1]
                # added_idx, _ = torch.sort(added_idx, dim=2)
                expand_idx = added_idx.expand(-1,-1, -1, frame_hidden_state_shallow.size(-1))
                added_tokens_cur = frame_hidden_state_shallow.gather(dim=2, index=expand_idx)
            elif self.pick_delta_token == "all":
                added_tokens_cur = frame_hidden_state_shallow

            #和第一帧链接起来
            added_tokens =  einops.rearrange(added_tokens_cur, 'b t q h -> b (t q) h', b=batch_size, t=time_length-1)
            frame1_hidden_state = frame1_hidden_state.squeeze(1)
            frame_hidden_state = torch.cat((frame1_hidden_state, added_tokens),dim=1)
            
            if self.replace_video_qformer:
                video_hidden = self.video_MLP(frame_hidden_state)
            else:
                frame_atts = torch.ones(frame_hidden_state.size()[:-1], dtype=torch.long).to(device)
                video_query_tokens = self.video_query_tokens.expand(frame_hidden_state.shape[0], -1, -1)

                video_query_output = self.video_Qformer.bert(
                    query_embeds=video_query_tokens,
                    encoder_hidden_states=frame_hidden_state,  
                    encoder_attention_mask=frame_atts,
                    return_dict=True,
                    )
                video_hidden = video_query_output.last_hidden_state

            inputs_llama = self.llama_proj(video_hidden)
            atts_llama = torch.ones(inputs_llama.size()[:-1], dtype=torch.long).to(image_embeds.device)
        return inputs_llama, atts_llama, loss

    def generate_query_shallow(self, query_tokens_shallow):
        #b, t, emb
        #聚合方式：平均
        if self.memory_merge == "" or self.memory_merge == "mean":
            #前k帧进行聚类，作为query的初始化
            result = torch.cumsum(query_tokens_shallow, dim=1)
            norm = torch.arange(1,result.size(1)+1).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).to(query_tokens_shallow.device)
            # norm = torch.range(1,result.size(1)).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).to(query_tokens_shallow.device)
            norm = 1/norm
            result = torch.mul(result, norm)
            result = result[:,1:,...]
            result = einops.rearrange(result, 'b t q h -> (b t) q h')
        elif self.memory_merge == "cos_sim":
            tokens_memory = query_tokens_shallow[:,0,...]
            result_ls = []
            for i in range(1, query_tokens_shallow.size(1)):
                #当前和memory token的距离，
                sim = F.cosine_similarity(query_tokens_shallow[:,i,...], tokens_memory, dim=-1).unsqueeze(-1)
                tokens_memory = sim * tokens_memory + (1-sim) * query_tokens_shallow[:,i,...]
                result_ls.append(tokens_memory)
            result = torch.stack(result_ls, dim=1)
            result = einops.rearrange(result, 'b t q h -> (b t) q h')
        elif self.memory_merge == "predict_weight":
            tokens_memory = query_tokens_shallow[:,0,...]
            result_ls = []
            for i in range(1, query_tokens_shallow.size(1)):
                #当前和memory token的差，
                m = nn.Sigmoid()
                sim = self.merge_weight_predictor(query_tokens_shallow[:,i,...] - tokens_memory)
                sim = m(sim)
                tokens_memory = sim * tokens_memory + (1-sim) * query_tokens_shallow[:,i,...]
                result_ls.append(tokens_memory)
            result = torch.stack(result_ls, dim=1)
            result = einops.rearrange(result, 'b t q h -> (b t) q h')
        if self.memory_merge == "only_cur":
            #前k帧进行聚类，作为query的初始化
            result = query_tokens_shallow
            result = result[:,1:,...]
            result = einops.rearrange(result, 'b t q h -> (b t) q h')



        return result

    def encode_videoQformer_visual_MOD(self, image):
        device = image.device
        
        # input shape b,c,t,h,w
        batch_size,_,time_length,_,_ = image.size()
        image = einops.rearrange(image, 'b c t h w -> (b t) c h w')
        with self.maybe_autocast():
            # embed image features with blip2, out: (b t) q h
            image_embeds = self.ln_vision(self.visual_encoder.forward_features_MOD(image)).to(device)
            # image_embeds = self.ln_vision(self.visual_encoder.forward_features(image)).to(device)
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(device)

            query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
            query_output = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )

            # add frame_pos embedding
            position_ids = torch.arange(time_length, dtype=torch.long, device=query_tokens.device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
            frame_position_embeddings = self.video_frame_position_embedding(position_ids)
            q_hidden_state = query_output.last_hidden_state

            frame_position_embeddings = frame_position_embeddings.unsqueeze(-2)
            frame_hidden_state = einops.rearrange(q_hidden_state, '(b t) q h -> b t q h',b=batch_size,t=time_length)
            frame_hidden_state = frame_position_embeddings + frame_hidden_state
            #  expand(<shape>)
            # frame attention
            #frame_hidden_state = frame_hidden_state[:,3,:,:].expand(8,32,768).unsqueeze(0)
            frame_hidden_state =  einops.rearrange(frame_hidden_state, 'b t q h -> b (t q) h',b=batch_size,t=time_length)
            frame_atts = torch.ones(frame_hidden_state.size()[:-1], dtype=torch.long).to(device)
            video_query_tokens = self.video_query_tokens.expand(frame_hidden_state.shape[0], -1, -1)

            video_query_output = self.video_Qformer.bert(
                query_embeds=video_query_tokens,
                encoder_hidden_states=frame_hidden_state,
                encoder_attention_mask=frame_atts,
                return_dict=True,
                )
            video_hidden = video_query_output.last_hidden_state

            inputs_llama = self.llama_proj(video_hidden)
            atts_llama = torch.ones(inputs_llama.size()[:-1], dtype=torch.long).to(image_embeds.device)
        return inputs_llama, atts_llama


    def encode_videoQformer_visual(self, image):
        device = image.device
        
        # input shape b,c,t,h,w
        batch_size,_,time_length,_,_ = image.size()
        image = einops.rearrange(image, 'b c t h w -> (b t) c h w')
        with self.maybe_autocast():
            # embed image features with blip2, out: (b t) q h
            image_embeds = self.ln_vision(self.visual_encoder(image)).to(device)
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(device)

            query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
            query_output = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )

            # add frame_pos embedding
            position_ids = torch.arange(time_length, dtype=torch.long, device=query_tokens.device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
            frame_position_embeddings = self.video_frame_position_embedding(position_ids)
            q_hidden_state = query_output.last_hidden_state

            frame_position_embeddings = frame_position_embeddings.unsqueeze(-2)
            frame_hidden_state = einops.rearrange(q_hidden_state, '(b t) q h -> b t q h',b=batch_size,t=time_length)
            frame_hidden_state = frame_position_embeddings + frame_hidden_state
            #  expand(<shape>)
            # frame attention
            #frame_hidden_state = frame_hidden_state[:,3,:,:].expand(8,32,768).unsqueeze(0)
            frame_hidden_state =  einops.rearrange(frame_hidden_state, 'b t q h -> b (t q) h',b=batch_size,t=time_length)
            frame_atts = torch.ones(frame_hidden_state.size()[:-1], dtype=torch.long).to(device)
            video_query_tokens = self.video_query_tokens.expand(frame_hidden_state.shape[0], -1, -1)

            video_query_output = self.video_Qformer.bert(
                query_embeds=video_query_tokens,
                encoder_hidden_states=frame_hidden_state,
                encoder_attention_mask=frame_atts,
                return_dict=True,
                )
            video_hidden = video_query_output.last_hidden_state

            inputs_llama = self.llama_proj(video_hidden)
            atts_llama = torch.ones(inputs_llama.size()[:-1], dtype=torch.long).to(image_embeds.device)
        return inputs_llama, atts_llama
    
    def encode_videoQformer_save(self, image):
        device = image.device
        
        # input shape b,c,t,h,w
        batch_size,_,time_length,_,_ = image.size()
        image = einops.rearrange(image, 'b c t h w -> (b t) c h w')
        with self.maybe_autocast():
            # embed image features with blip2, out: (b t) q h
            image_embeds = self.ln_vision(self.visual_encoder(image)).to(device)
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(device)

            query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
            query_output = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )

            v_hidden_ls = []
            # add frame_pos embedding
            position_ids = torch.arange(time_length, dtype=torch.long, device=query_tokens.device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
            frame_position_embeddings = self.video_frame_position_embedding(position_ids)
            q_hidden_state = query_output.last_hidden_state

            frame_position_embeddings = frame_position_embeddings.unsqueeze(-2)
            frame_hidden_state = einops.rearrange(q_hidden_state, '(b t) q h -> b t q h',b=batch_size,t=time_length)
            frame_hidden_state = frame_position_embeddings + frame_hidden_state
            #  expand(<shape>)
            # frame attention
            #frame_hidden_state = frame_hidden_state[:,3,:,:].expand(8,32,768).unsqueeze(0)
            frame_hidden_state0 =  einops.rearrange(frame_hidden_state, 'b t q h -> b (t q) h',b=batch_size,t=time_length)
            frame_atts0 = torch.ones(frame_hidden_state0.size()[:-1], dtype=torch.long).to(device)
            video_query_tokens0 = self.video_query_tokens.expand(frame_hidden_state0.shape[0], -1, -1)

            video_query_output0 = self.video_Qformer.bert(
                query_embeds=video_query_tokens0,
                encoder_hidden_states=frame_hidden_state0,
                encoder_attention_mask=frame_atts0,
                return_dict=True,
                )
            video_hidden0 = video_query_output0.last_hidden_state
            v_hidden_ls.append(video_hidden0)

            for i in range(8):
                frame_hidden_state_ = frame_hidden_state[:,i,:,:].expand(8,32,768).unsqueeze(0)
                frame_hidden_state_ =  einops.rearrange(frame_hidden_state_, 'b t q h -> b (t q) h',b=batch_size,t=time_length)
                frame_atts_ = torch.ones(frame_hidden_state_.size()[:-1], dtype=torch.long).to(device)
                video_query_tokens_ = self.video_query_tokens.expand(frame_hidden_state_.shape[0], -1, -1)

                video_query_output_ = self.video_Qformer.bert(
                    query_embeds=video_query_tokens_,
                    encoder_hidden_states=frame_hidden_state_,
                    encoder_attention_mask=frame_atts_,
                    return_dict=True,
                    )
                video_hidden_ = video_query_output_.last_hidden_state
                v_hidden_ls.append(video_hidden_)

        return v_hidden_ls
    
    def encode_videoQformer_load_feats_eval(self, frame_hidden_state, frame_atts, indexes):
        device = frame_hidden_state.device
        
        frame_atts = torch.ones(frame_hidden_state.size()[:-1], dtype=torch.long).to(device)
        video_query_tokens = self.video_query_tokens.expand(frame_hidden_state.shape[0], -1, -1)

        video_query_output = self.video_Qformer.bert(
            query_embeds=video_query_tokens,
            encoder_hidden_states=frame_hidden_state,
            encoder_attention_mask=frame_atts,
            return_dict=True,
            )
        video_hidden = video_query_output.last_hidden_state

        inputs_llama = self.llama_proj(video_hidden)
        atts_llama = torch.ones(inputs_llama.size()[:-1], dtype=torch.long).to(video_hidden.device)
        return inputs_llama, atts_llama
        
    def prompt_wrap(self, img_embeds, atts_img, prompt):
        if prompt:
            batch_size = img_embeds.shape[0]
            # print(prompt)
            p_before, p_after = prompt.split('<ImageHere>')
            p_before_tokens = self.llama_tokenizer(
                p_before, return_tensors="pt", add_special_tokens=False).to(img_embeds.device)
            p_after_tokens = self.llama_tokenizer(
                p_after, return_tensors="pt", add_special_tokens=False).to(img_embeds.device)
            p_before_embeds = self.llama_model.model.embed_tokens(p_before_tokens.input_ids).expand(batch_size, -1, -1)
            p_after_embeds = self.llama_model.model.embed_tokens(p_after_tokens.input_ids).expand(batch_size, -1, -1)
            wrapped_img_embeds = torch.cat([p_before_embeds, img_embeds, p_after_embeds], dim=1)
            wrapped_atts_img = atts_img[:, :1].expand(-1, wrapped_img_embeds.shape[1])
            
            return wrapped_img_embeds, wrapped_atts_img
        else:
            return img_embeds, atts_img
    #  input audio shape [b t c h w] 
    def encode_audioQformer(self, audio,modality_type=ModalityType.AUDIO):
        device = audio.device
        with self.maybe_autocast():
            audio_feature, audio_imagebind_finalout = self.audio_encoder.get_audio_feature(audio,modality_type=modality_type)
            batch_size,time_length = audio.size()[:2]


            position_ids = torch.arange(time_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)

            audio_position_embeddings = self.audio_position_embedding(position_ids)
            audio_imagebind_finalout = audio_imagebind_finalout + audio_position_embeddings

            audio_query_tokens = self.audio_query_tokens.expand(audio_imagebind_finalout.shape[0], -1, -1)
            frame_atts = torch.ones(audio_imagebind_finalout.size()[:-1], dtype=torch.long).to(device)

            audio_query_output = self.audio_Qformer.bert(
                query_embeds=audio_query_tokens, #[32,768]
                encoder_hidden_states=audio_imagebind_finalout,
                encoder_attention_mask=frame_atts,
                return_dict=True,
                )
            audio_hidden = audio_query_output.last_hidden_state

            inputs_llama = self.audio_llama_proj(audio_hidden)
            atts_llama = torch.ones(inputs_llama.size()[:-1], dtype=torch.long).to(device)
    
        return inputs_llama, atts_llama

    def encode_videoQformer_audiovideo(self, image, audio):
        device = image.device
        
        # input shape b,c,t,h,w
        batch_size,_,time_length,_,_ = image.size()
        image = einops.rearrange(image, 'b c t h w -> (b t) c h w')
        with self.maybe_autocast():
            # embed image features with blip2, out: (b t) q h
            image_embeds = self.ln_vision(self.visual_encoder(image)).to(device)
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(device)

            query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
            query_output = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )
            
            # add frame_pos embedding
            position_ids = torch.arange(time_length, dtype=torch.long, device=query_tokens.device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
            frame_position_embeddings = self.video_frame_position_embedding(position_ids)
            q_hidden_state = query_output.last_hidden_state

            frame_position_embeddings = frame_position_embeddings.unsqueeze(-2)
            frame_hidden_state = einops.rearrange(q_hidden_state, '(b t) q h -> b t q h',b=batch_size,t=time_length)
            frame_hidden_state = frame_position_embeddings + frame_hidden_state

            # encode audio 
            audio_feature, audio_imagebind_finalout = self.audio_encoder.get_audio_feature(audio,modality_type=ModalityType.AUDIO) # [batch,8*1,768]    8*32, 768
            audio_frame_position_embeddings = frame_position_embeddings.squeeze(-2)
            audio_feature = audio_feature + audio_frame_position_embeddings

            # frame attention a
            frame_hidden_state =  einops.rearrange(frame_hidden_state, 'b t q h -> b (t q) h',b=batch_size,t=time_length)
            frame_hidden_state = torch.cat([frame_hidden_state,audio_feature],dim = 1)
            video_query_tokens = self.video_query_tokens.expand(frame_hidden_state.shape[0], -1, -1)
            frame_atts = torch.ones(frame_hidden_state.size()[:-1], dtype=torch.long).to(device)

            video_query_output = self.video_Qformer.bert(
                query_embeds=video_query_tokens, #[32,768]
                encoder_hidden_states=frame_hidden_state,
                encoder_attention_mask=frame_atts,
                return_dict=True,
                )
            video_hidden = video_query_output.last_hidden_state

            inputs_llama = self.llama_proj(video_hidden)
            atts_llama = torch.ones(inputs_llama.size()[:-1], dtype=torch.long).to(image_embeds.device)
    
        return inputs_llama, atts_llama

    def forward(self, samples):
        if 'conv_type' in samples.keys() and samples['conv_type']=='multi':
        # if len(samples['images'].size())==4:
            
            im_patch_token_id = self.IMAGE_PATCH_TOKEN_ID
            image = samples["images"]
            input_ids = samples['input_ids']
            if len(image.size())==4:
                time = 1
                image = einops.repeat(image, 'b c h w -> b c t h w',t = time)

            if self.train_flag == 0:
                num_patch_tokens = self.num_video_query_token
                if image.size()[2]==1:
                    img_embeds, atts_img = self.encode_videoQformer_visual(image)
                else:
                    if self.light_type == "MOD":
                        img_embeds, atts_img = self.encode_videoQformer_visual_MOD(image)
                    else:
                        img_embeds, atts_img, shallow_loss = self.encode_videoQformer_lightPredictor_visual(image)
            elif self.train_flag == 1:
                num_patch_tokens = self.num_audio_query_token
                image = einops.rearrange(image, 'b c t h w -> b t c h w')
                img_embeds, atts_img = self.encode_audioQformer(image, modality_type=ModalityType.VISION)
                
            temp_input_ids = copy.deepcopy(input_ids)
            temp_input_ids[temp_input_ids == im_patch_token_id] = 0
            temp_input_embedding = self.llama_model.model.embed_tokens(temp_input_ids)

            new_input_embeds=[]
            cur_image_idx = 0
            for cur_input_ids, cur_input_embeds in zip(input_ids, temp_input_embedding):
                cur_image_features = img_embeds[cur_image_idx]

                if (cur_input_ids == im_patch_token_id).sum() != num_patch_tokens:
                        raise ValueError("The number of image patch tokens should be the same as the number of image patches.")
                masked_indices = torch.where(cur_input_ids == im_patch_token_id)[0]
                mask_index_start = masked_indices[0]
                if (masked_indices != torch.arange(mask_index_start, mask_index_start+num_patch_tokens, device=masked_indices.device, dtype=masked_indices.dtype)).any():
                    raise ValueError("The image patch tokens should be consecutive.")
                
                cur_new_input_embeds = torch.cat((cur_input_embeds[:mask_index_start], cur_image_features, cur_input_embeds[mask_index_start+num_patch_tokens:]), dim=0)
                new_input_embeds.append(cur_new_input_embeds)
                
                cur_image_idx+=1
            inputs_embeds = torch.stack(new_input_embeds, dim=0)
            targets = samples['labels']
            attention_mask = samples['attention_mask']
            with self.maybe_autocast():
                outputs = self.llama_model(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    return_dict=True,
                    labels=targets,
                )
            loss = outputs.loss
            return {"loss": loss}
        else:
            try:
                image = samples["images"]
            except:
                image = samples["image"]

            if len(image.size()) != 5:
                time = 1
                image = einops.repeat(image, 'b c h w -> b c t h w',t = time)
            
            if self.train_flag == 1:
                image = einops.rearrange(image, 'b c t h w -> b t c h w')
                img_embeds, atts_img = self.encode_audioQformer(image, modality_type=ModalityType.VISION)
            #elif 'label_seg' not in samples.keys():
            #    img_embeds, atts_img, shallow_loss = self.encode_videoQformer_lightPredictor_visual(image)
                #img_embeds, atts_img = self.encode_videoQformer_visual(image)
            elif "segment_idx" in samples.keys():
                img_input = self.encode_lightPredictor_segment(samples['image'], samples['segment_idx'], 4)
                img_embeds, atts_img = self.encode_videoQformer_lightPredictor_visual_eval(img_input, long_video=True)
            elif len(samples["image"].size()) == 4:
                img_embeds, atts_img = self.encode_videoQformer_visual(image)
            else:
                if 'label_seg' in samples:
                    if self.light_type == "MOD":
                        img_embeds, atts_img = self.encode_videoQformer_visual_MOD(image)
                    else:
                        img_embeds, atts_img, shallow_loss = self.encode_videoQformer_lightPredictor_visual(image, label_seg = samples['label_seg'])
                else:
                    if self.light_type == "MOD":
                        img_embeds, atts_img = self.encode_videoQformer_visual_MOD(image)
                    else:
                        img_embeds, atts_img, shallow_loss = self.encode_videoQformer_lightPredictor_visual(image)
                # return {"loss": shallow_loss}

            if self.prompt_list:
                prompt = random.choice(self.prompt_list)
                img_embeds, atts_img = self.prompt_wrap(img_embeds, atts_img, prompt)
                

            self.llama_tokenizer.padding_side = "right"

            text = [t + self.end_sym for t in samples["text_input"]]

            to_regress_tokens = self.llama_tokenizer(
                text,
                return_tensors="pt",
                padding="longest",
                truncation=True,
                max_length=self.max_txt_len,
                add_special_tokens=False
            ).to(image.device)

            targets = to_regress_tokens.input_ids.masked_fill(
                to_regress_tokens.input_ids == self.llama_tokenizer.pad_token_id, -100
            )

            empty_targets = (
                torch.ones([atts_img.shape[0], atts_img.shape[1]+1],
                        dtype=torch.long).to(image.device).fill_(-100)  # plus one for bos
            )
            targets = torch.cat([empty_targets, targets], dim=1)

            batch_size = img_embeds.shape[0]
            bos = torch.ones([batch_size, 1],
                            dtype=to_regress_tokens.input_ids.dtype,
                            device=to_regress_tokens.input_ids.device) * self.llama_tokenizer.bos_token_id
            bos_embeds = self.llama_model.model.embed_tokens(bos)
            atts_bos = atts_img[:, :1]

            to_regress_embeds = self.llama_model.model.embed_tokens(to_regress_tokens.input_ids)
            inputs_embeds = torch.cat([bos_embeds, img_embeds, to_regress_embeds], dim=1)
            attention_mask = torch.cat([atts_bos, atts_img, to_regress_tokens.attention_mask], dim=1)

            with self.maybe_autocast():
                outputs = self.llama_model(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    return_dict=True,
                    labels=targets,
                )
            loss = outputs.loss

        if "segment_idx" in samples.keys() or len(samples["image"].size()) == 4 or self.light_type == "MOD":
            return {"loss": loss}
        else:
            return {"loss": loss+shallow_loss*self.light_loss_weight, "generate_loss": loss, "distill_loss": shallow_loss}

    @classmethod
    def from_config(cls, cfg):
        vit_model = cfg.get("vit_model", "eva_clip_g")
        q_former_model = cfg.get("q_former_model", "https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/blip2_pretrained_flant5xxl.pth")
        img_size = cfg.get("image_size")
        num_query_token = cfg.get("num_query_token")
        llama_model = cfg.get("llama_model")

        drop_path_rate = cfg.get("drop_path_rate", 0)
        use_grad_checkpoint = cfg.get("use_grad_checkpoint", False)
        vit_precision = cfg.get("vit_precision", "fp16")
        freeze_vit = cfg.get("freeze_vit", True)
        freeze_qformer = cfg.get("freeze_qformer", True)
        low_resource = cfg.get("low_resource", False)
        device_8bit = cfg.get("device_8bit", 0)
        
        freeze_light_predictor = cfg.get("freeze_light_predictor", False)

        prompt_path = cfg.get("prompt_path", "")
        prompt_template = cfg.get("prompt_template", "")
        max_txt_len = cfg.get("max_txt_len", 32)
        end_sym = cfg.get("end_sym", '\n')
        
        frozen_llama_proj = cfg.get("frozen_llama_proj", True)
        frozen_video_Qformer = cfg.get("frozen_video_Qformer", True)
        frozen_audio_Qformer = cfg.get("frozen_audio_Qformer", True)



        llama_proj_model = cfg.get("llama_proj_model", '')
        
        fusion_header_type = cfg.get("fusion_header_type", 'seqTransf')
        max_frame_pos = cfg.get("max_frame_pos", 32)
        fusion_head_layers = cfg.get("fusion_head_layers", 2)
        num_video_query_token =  cfg.get("num_video_query_token", 32)

        equip_audio_branch= cfg.get("equip_audio_branch", True)
        num_audio_query_token =  cfg.get("num_audio_query_token", 8)
        imagebind_ckpt_path = cfg.get("imagebind_ckpt_path", '/mnt/workspace/ckpt')

        num_thin_patch_token = cfg.get("num_thin_patch_token", 32)
        memory_merge = cfg.get("memory_merge", "")
        delta_token_num = cfg.get("delta_token_num", 5)
        replace_video_qformer = cfg.get("replace_video_qformer", False)
        pick_delta_token = cfg.get("pick_delta_token", "")

        full_layer_num = cfg.get("full_layer_num", 20)
        light_loss_weight = cfg.get("light_loss_weight", 1.0)
        light_weight = cfg.get("light_weight", "")
        light_type = cfg.get("light_type", "")
        light2full = cfg.get("light2full", False)
        full_gap = cfg.get("full_gap", 8)

        model = cls(
            vit_model=vit_model,
            q_former_model=q_former_model,
            img_size=img_size,
            drop_path_rate=drop_path_rate,
            use_grad_checkpoint=use_grad_checkpoint,
            vit_precision=vit_precision,
            freeze_vit=freeze_vit,
            freeze_qformer=freeze_qformer,
            num_query_token=num_query_token,
            llama_model=llama_model,
            prompt_path=prompt_path,
            prompt_template=prompt_template,
            max_txt_len=max_txt_len,
            end_sym=end_sym,
            low_resource=low_resource,
            device_8bit=device_8bit,
            fusion_header_type=fusion_header_type,
            max_frame_pos=max_frame_pos,
            fusion_head_layers=fusion_head_layers,
            frozen_llama_proj=frozen_llama_proj,
            frozen_video_Qformer=frozen_video_Qformer,
            frozen_audio_Qformer=frozen_audio_Qformer,
            num_video_query_token=num_video_query_token,
            num_audio_query_token = num_audio_query_token,
            imagebind_ckpt_path = imagebind_ckpt_path,
            equip_audio_branch = equip_audio_branch,
            llama_proj_model = llama_proj_model,
            freeze_light_predictor = freeze_light_predictor,
            num_thin_patch_token = num_thin_patch_token,
            memory_merge = memory_merge,
            delta_token_num = delta_token_num,
            replace_video_qformer = replace_video_qformer,
            pick_delta_token = pick_delta_token,
            full_layer_num = full_layer_num,
            light_loss_weight = light_loss_weight,
            light_weight = light_weight,
            light_type = light_type,
            light2full = light2full,
            full_gap = full_gap
        )

        ckpt_path = cfg.get("ckpt", "")  # load weights of MiniGPT-4
        if ckpt_path:
            print("Load first Checkpoint: {}".format(ckpt_path))
            ckpt = torch.load(ckpt_path, map_location="cpu")
            msg = model.load_state_dict(ckpt['model'], strict=False)


        ckpt_path_2 = cfg.get("ckpt_2", "")  


        #加载ckpt_path_2的config
        ckpt_path_1 = os.path.join(os.path.dirname(ckpt_path_2), "codes/train_configs/visionbranch_stage2_finetune_my.yaml")
        if os.path.exists(ckpt_path_1):
            with open(ckpt_path_1, 'r') as file:
                config = yaml.safe_load(file)
                ckpt_path_1 = config.get('model', {}).get('ckpt_2', None)
                if os.path.exists(ckpt_path_1):
                    ckpt_1 = torch.load(ckpt_path_1, map_location="cpu")
                    msg = model.load_state_dict(ckpt_1['model'], strict=False)

        
        if os.path.exists(ckpt_path_2):
            if ckpt_path_2:
                print("Load second Checkpoint: {}".format(ckpt_path_2))
                ckpt = torch.load(ckpt_path_2, map_location="cpu")
                msg = model.load_state_dict(ckpt['model'], strict=False)
        return model
