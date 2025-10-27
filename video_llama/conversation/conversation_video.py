"""
Conversation prompt template of Video-LLaMA.
Adapted from: https://github.com/Vision-CAIR/MiniGPT-4/blob/main/minigpt4/conversation/conversation.py 
"""
import argparse
import time
from PIL import Image
import sys
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer
from transformers import StoppingCriteria, StoppingCriteriaList

import dataclasses
from enum import auto, Enum
from typing import List, Tuple, Any
import os
from video_llama.common.registry import registry
from video_llama.processors.video_processor import ToTHWC,ToUint8,load_video
from video_llama.processors import Blip2ImageEvalProcessor
            
from video_llama.models.ImageBind.data import load_and_transform_audio_data
import numpy as np
from thop import profile
import einops

class SeparatorStyle(Enum):
    """Different separator style."""
    SINGLE = auto()
    TWO = auto()
    LLAMA_2 = auto()


@dataclasses.dataclass
class Conversation:
    """A class that keeps all conversation history."""
    system: str
    roles: List[str]
    messages: List[List[str]]
    offset: int
    # system_img: List[Image.Image] = []
    sep_style: SeparatorStyle = SeparatorStyle.SINGLE
    sep: str = "###"
    sep2: str = None

    skip_next: bool = False
    conv_id: Any = None

    def get_prompt(self):
        if self.sep_style == SeparatorStyle.SINGLE:
            ret = self.system + self.sep
            for role, message in self.messages:
                if message:
                    ret += role + ": " + message + self.sep
                else:
                    ret += role + ":"
            return ret
        elif self.sep_style == SeparatorStyle.TWO:
            seps = [self.sep, self.sep2]
            ret = self.system + seps[0]
            for i, (role, message) in enumerate(self.messages):
                if message:
                    ret += role + ": " + message + seps[i % 2]
                else:
                    ret += role + ":"
            return ret
        elif self.sep_style == SeparatorStyle.LLAMA_2:
            wrap_sys = lambda msg: f"<<SYS>>\n{msg}\n<</SYS>>\n\n"
            wrap_inst = lambda msg: f"[INST] {msg} [/INST]"
            ret = ""

            for i, (role, message) in enumerate(self.messages):
                if i == 0:
                    assert message, "first message should not be none"
                    assert role == self.roles[0], "first message should come from user"
                if message:
                    if type(message) is tuple:
                        message, _, _ = message
                    if i == 0: message = wrap_sys(self.system) + message
                    if i % 2 == 0:
                        message = wrap_inst(message)
                        ret += self.sep + message
                    else:
                        ret += " " + message + " " + self.sep2
                else:
                    ret += ""
            ret = ret.lstrip(self.sep)
            return ret
        else:
            raise ValueError(f"Invalid style: {self.sep_style}")

    def append_message(self, role, message):
        self.messages.append([role, message])

    def to_gradio_chatbot(self):
        ret = []
        for i, (role, msg) in enumerate(self.messages[self.offset:]):
            if i % 2 == 0:
                ret.append([msg, None])
            else:
                ret[-1][-1] = msg
        return ret

    def copy(self):
        return Conversation(
            system=self.system,
            # system_img=self.system_img,
            roles=self.roles,
            messages=[[x, y] for x, y in self.messages],
            offset=self.offset,
            sep_style=self.sep_style,
            sep=self.sep,
            sep2=self.sep2,
            conv_id=self.conv_id)

    def dict(self):
        return {
            "system": self.system,
            # "system_img": self.system_img,
            "roles": self.roles,
            "messages": self.messages,
            "offset": self.offset,
            "sep": self.sep,
            "sep2": self.sep2,
            "conv_id": self.conv_id,
        }


class StoppingCriteriaSub(StoppingCriteria):

    def __init__(self, stops=[], encounters=1):
        super().__init__()
        self.stops = stops

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        for stop in self.stops:
            if torch.all((stop == input_ids[0][-len(stop):])).item():
                return True

        return False


CONV_VISION = Conversation(
    system="Give the following image: <Img>ImageContent</Img>. "
           "You will be able to see the image once I provide it to you. Please answer my questions.",
    roles=("Human", "Assistant"),
    messages=[],
    offset=0,
    sep_style=SeparatorStyle.SINGLE,
    sep="###",
)

default_conversation = Conversation(
    system="",
    roles=("Human", "Assistant"),
    messages=[],
    offset=0,
    sep_style=SeparatorStyle.SINGLE,
    sep="###",
)
conv_llava_llama_2 = Conversation(
    system="You are a helpful language and vision assistant. "
           "You are able to understand the visual content that the user provides, "
           "and assist the user with a variety of tasks using natural language.",
    roles=("USER", "ASSISTANT"),
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.LLAMA_2,
    sep="<s>",
    sep2="</s>",
)

def generate_seg_idx(timestamps, duration, vid_len):
    result = []
    for stamp in timestamps:
        s_idx = int(vid_len * stamp[0] / duration)
        e_idx = int(vid_len * stamp[1] / duration)
        result.append([s_idx, e_idx])
    result = torch.tensor(result)
    return result

class Chat:
    def __init__(self, model, vis_processor, device='cuda:0'):
        self.device = device
        self.model = model
        self.vis_processor = vis_processor
        self.image_vis_processor = Blip2ImageEvalProcessor()
        # stop_words_ids = [torch.tensor([835]).to(self.device),
        #                   torch.tensor([2277, 29937]).to(self.device)]  # '###' can be encoded in two different ways.
        # self.stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])

    def ask(self, text, conv):
        if len(conv.messages) > 0 and conv.messages[-1][0] == conv.roles[0] \
                and ('</Video>' in conv.messages[-1][1] or '</Image>' in conv.messages[-1][1]):  # last message is image.
            conv.messages[-1][1] = ' '.join([conv.messages[-1][1], text])
        else:
            conv.append_message(conv.roles[0], text)

    def answer(self, conv, img_list, max_new_tokens=300, num_beams=1, min_length=1, top_p=0.9,
               repetition_penalty=1.0, length_penalty=1, temperature=1.0, max_length=2000):
        conv.append_message(conv.roles[1], None)
        embs = self.get_context_emb(conv, img_list)

        current_max_len = embs.shape[1] + max_new_tokens
        if current_max_len - max_length > 0:
            print('Warning: The number of tokens in current conversation exceeds the max length. '
                  'The model will not see the contexts outside the range.')
        begin_idx = max(0, current_max_len - max_length)

        embs = embs[:, begin_idx:]
        if conv.sep =="###":
            stop_words_ids = [torch.tensor([835]).to(self.device),
                          torch.tensor([2277, 29937]).to(self.device)]  # '###' can be encoded in two different ways.
            stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])
        else:
            stop_words_ids = [torch.tensor([2]).to(self.device)]
            stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])

        # stopping_criteria
        outputs = self.model.llama_model.generate(
            inputs_embeds=embs,
            max_new_tokens=max_new_tokens,
            stopping_criteria=stopping_criteria,
            num_beams=num_beams,
            do_sample=True,
            min_length=min_length,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            temperature=temperature,
        )
        output_token = outputs[0]
        if output_token[0] == 0:  # the model might output a unknow token <unk> at the beginning. remove it
            output_token = output_token[1:]
        if output_token[0] == 1:  # some users find that there is a start token <s> at the beginning. remove it
            output_token = output_token[1:]
        output_text = self.model.llama_tokenizer.decode(output_token, add_special_tokens=False)
        if conv.sep =="###":
            output_text = output_text.split('###')[0]  # remove the stop sign '###'
            output_text = output_text.split('Assistant:')[-1].strip()
        else:
            output_text = output_text.split(conv.sep2)[0]  # remove the stop sign '###'
            output_text = output_text.split(conv.roles[1]+':')[-1].strip()
        conv.messages[-1][1] = output_text
        return output_text, output_token.cpu().numpy()
    
    def upload_video(self, video_path, conv, img_list, middle=-1):

        msg = ""
        if isinstance(video_path, str):  # is a video path
            ext = os.path.splitext(video_path)[-1].lower()
            print(video_path)
            # image = self.vis_processor(image).unsqueeze(0).to(self.device)
            if middle == -1:
                video, msg = load_video(
                    video_path=video_path,
                    n_frms=8,
                    height=224,
                    width=224,
                    sampling ="uniform", return_msg = True
                )
            else:
                video, msg = load_video(
                    video_path=video_path,
                    n_frms=8,
                    height=224,
                    width=224,
                    sampling ="breakpoint", return_msg = True, middle=middle
                )
            video = self.vis_processor.transform(video)
            video = video.unsqueeze(0).to(self.device)
            # print(image)
        else:
            raise NotImplementedError
        
        try:
            audio_flag = 1
            audio = load_and_transform_audio_data([video_path],"cpu",  clips_per_video=8)
            audio = audio.to(self.device)
        except :
            print('no audio is found')
            audio_flag = 0
        finally:
            if audio_flag == 1:
                # image_emb, _ = self.model.encode_videoQformer_audiovideo(video,audio)
                image_emb, _ = self.model.encode_videoQformer_visual(video)
                audio_emb,_  = self.model.encode_audioQformer(audio)
                img_list.append(audio_emb)
                img_list.append(image_emb)
                conv.system = ""
                # conv.append_message(conv.roles[0], "The audio of this video is <Video><ImageHere></Video> ")
                conv.append_message(conv.roles[0], "Close your eyes, open your ears and you imagine only based on the sound that: <ImageHere>. \
                Close your ears, open your eyes and you see that <Video><ImageHere></Video>.  \
                Now answer my question based on what you have just seen and heard.")

            else:  # only vison no audio
                # conv.system = "You can understand the video that the user provides. Follow the instructions carefully and explain your answers in detail."
                image_emb, _ = self.model.encode_videoQformer_visual(video)
                img_list.append(image_emb)
                conv.append_message(conv.roles[0], "<Video><ImageHere></Video> "+ msg)
            return "Received."

    def upload_video_without_audio_visualize(self, video_path, conv, img_list, n_frms=5, topk=-1, full_layer_num=-1):
        msg = ""
        video_id = video_path.split('/')[-1].split('.')[0]
        if isinstance(video_path, str):  # is a video path
            ext = os.path.splitext(video_path)[-1].lower()
            # print(video_path)
            # image = self.vis_processor(image).unsqueeze(0).to(self.device)
            video, msg = load_video(
                video_path=video_path,
                n_frms=n_frms,
                height=224,
                width=224,
                sampling ="uniform", return_msg = True, video_id = video_id
            )
            for i in range(video.size(1)):
                ts = video[:,i].permute(1,2,0).to(torch.uint8)
                image = Image.fromarray(ts.numpy())
                image.save("figs/"+str(i)+".jpg", 'JPEG')
            video = self.vis_processor.transform(video)
            video = video.unsqueeze(0).to(self.device)
            # print(image)
        else:
            raise NotImplementedError
        
        # video = einops.rearrange(video.squeeze(),'c b h w -> b c h w').unsqueeze(dim=2)
        # delta, mask = self.model.ln_vision_shallow(self.model.visual_encoder.forward_features_thin_visualize(video, topk=self.model.num_thin_patch_token, full_layer_num=self.model.full_layer_num))  #32
        delta, mask = self.model.encode_videoQformer_lightPredictor_visual_visualize(video, topk=topk, full_layer_num=full_layer_num)
        return delta, mask
        # x = self.model.encode_videoQformer_lightPredictor_visual_visualize_feat(video, topk=topk, full_layer_num=full_layer_num)
        # return x


    def upload_video_without_audio(self, video_path, conv, img_list, n_frms=8, middle=-1, first_num=1):
        msg = ""
        video_id = video_path.split('/')[-1].split('.')[0]
        if isinstance(video_path, str):  # is a video path
            ext = os.path.splitext(video_path)[-1].lower()
            # print(video_path)
            # image = self.vis_processor(image).unsqueeze(0).to(self.device)
            if middle == -1:
                video, msg = load_video(
                    video_path=video_path,
                    n_frms=n_frms,
                    height=224,
                    width=224,
                    sampling ="uniform", return_msg = True, video_id = video_id
                )
            else:
                video, msg = load_video(
                    video_path=video_path,
                    n_frms=n_frms,
                    height=224,
                    width=224,
                    sampling ="breakpoint", return_msg = True, video_id = video_id, middle=middle
                )

            video = self.vis_processor.transform(video)
            video = video.unsqueeze(0).to(self.device)
            # print(image)
        else:
            raise NotImplementedError
        
        # conv.system = "You can understand the video that the user provides.  Follow the instructions carefully and explain your answers in detail."

        if hasattr(self.model, "light_type") and self.model.light_type == "MOD":
            image_emb, atts_img = self.model.encode_videoQformer_visual_MOD(video)
        # if self.model.light_type == "MOD":
        #     image_emb, atts_img = self.model.encode_videoQformer_visual_MOD(video)
        elif self.model.__class__.__name__ == "VideoLLAMA":
            image_emb, _ = self.model.encode_videoQformer_visual(video)  #encode_videoQformer_lightPredictor_visual
        elif self.model.__class__.__name__ == "VideoLLAMA_effi":
            # flops, params = profile(self.model.encode_videoQformer_lightPredictor_visual_eval_FLOPs, (video, video_path=video_path, seg_len = 8))
            image_emb, _ = self.model.encode_videoQformer_lightPredictor_visual_eval(video, video_path=video_path, seg_len = 8, first_num=first_num)  #encode_videoQformer_lightPredictor_visual
        # image_emb, _, _ = self.model.encode_videoQformer_lightPredictor_visual(video) 
        img_list.append(image_emb)
        conv.append_message(conv.roles[0], "<Video><ImageHere></Video> "+ msg)
        #conv.append_message(conv.roles[0], "<Video><ImageHere></Video> ")
        return "Received."

    def upload_video_without_audio_load_feats(self, video_path, conv, img_list, indexes=None):
        msg = ""
        if isinstance(video_path, str):  # is a video path
            ext = os.path.splitext(video_path)[-1].lower()
            print(video_path)
            # image = self.vis_processor(image).unsqueeze(0).to(self.device)
        else:
            raise NotImplementedError
        #加载token, atts
        video_id = video_path.split('/')[-1].split('.')[0]
        feat = np.load("video_llama/output/mid_feats/"+video_id+"/feats.npy", allow_pickle=True).item()
        tokens = feat['frame_hidden_state'][:,indexes,:]
        atts = feat['frame_atts']
        # tokens = tokens[]
 
        image_emb, _ = self.model.encode_videoQformer_load_feats_eval(tokens, atts, indexes)  #encode_videoQformer_lightPredictor_visual
        img_list.append(image_emb)
        conv.append_message(conv.roles[0], "<Video><ImageHere></Video> "+ msg)
        #conv.append_message(conv.roles[0], "<Video><ImageHere></Video> ")
        return "Received."

    def upload_long_video_without_audio(self, video_path, conv, img_list, batch=None):
        msg = ""
        if isinstance(video_path, str):  # is a video path
            image = self.vis_processor(video_path).unsqueeze(0).to(self.device)
        else:
            raise NotImplementedError
        seg_idx =  generate_seg_idx(batch['timestamps'], batch['duration'], image.size(1))
        img_input = self.model.encode_lightPredictor_segment(image, seg_idx.unsqueeze(0), 4)
        image_emb, atts_img = self.model.encode_videoQformer_lightPredictor_visual_eval(img_input, long_video=True)
        # conv.system = "You can understand the video that the user provides.  Follow the instructions carefully and explain your answers in detail."
        #image_emb, _ = self.model.encode_videoQformer_lightPredictor_visual_eval(video)  #encode_videoQformer_lightPredictor_visual
        img_list.append(image_emb)
        conv.append_message(conv.roles[0], "<Video><ImageHere></Video> "+ msg)
        #conv.append_message(conv.roles[0], "<Video><ImageHere></Video> ")
        return "Received."

    def upload_image_as_video(self, video_path, conv, img_list, batch=None):
        msg = ""
        result = []
        for image in video_path:
            if isinstance(image, str):  # is a image path
                raw_image = Image.open(image).convert('RGB') # 增加一个时间维度
                image_cur = self.image_vis_processor(raw_image).unsqueeze(0).unsqueeze(2).to(self.device)
            elif isinstance(image, Image.Image):
                raw_image = image
                image_cur = self.image_vis_processor(raw_image).unsqueeze(0).unsqueeze(2).to(self.device)
            elif isinstance(image, torch.Tensor):
                if len(image.shape) == 3:
                    image = image.unsqueeze(0)
                image_cur = image.to(self.device)
            else:
                raise NotImplementedError
            result.append(image_cur)
        video = torch.cat(result, dim=0)
        video = video.squeeze()

        # delta, mask = self.model.ln_vision_shallow(self.model.visual_encoder.forward_features_thin_visualize(video, topk=self.model.num_thin_patch_token, full_layer_num=self.model.full_layer_num))  #32
        # return delta, mask
        image_emb, _ = self.model.encode_videoQformer_lightPredictor_visual_eval(video, video_path=video_path)  #encode_videoQformer_lightPredictor_visual
        # image_emb, _ = self.model.encode_videoQformer_visual(video)
        img_list.append(image_emb)
        conv.append_message(conv.roles[0], "<Video><ImageHere></Video> "+ msg)
        return "Received."


    
    def upload_img(self, image, conv, img_list):

        msg = ""
        if isinstance(image, str):  # is a image path
            raw_image = Image.open(image).convert('RGB') # 增加一个时间维度
            image = self.image_vis_processor(raw_image).unsqueeze(0).unsqueeze(2).to(self.device)
        elif isinstance(image, Image.Image):
            raw_image = image
            image = self.image_vis_processor(raw_image).unsqueeze(0).unsqueeze(2).to(self.device)
        elif isinstance(image, torch.Tensor):
            if len(image.shape) == 3:
                image = image.unsqueeze(0)
            image = image.to(self.device)
        else:
            raise NotImplementedError

        image_emb, _ = self.model.encode_videoQformer_visual(image)
        img_list.append(image_emb)
        # Todo msg=""
        conv.append_message(conv.roles[0], "<Image><ImageHere></Image> "+ msg)

        return "Received."

    def get_context_emb(self, conv, img_list):
        prompt = conv.get_prompt()
        prompt_segs = prompt.split('<ImageHere>')
        assert len(prompt_segs) == len(img_list) + 1, "Unmatched numbers of image placeholders and images."
        seg_tokens = [
            self.model.llama_tokenizer(
                seg, return_tensors="pt", add_special_tokens=i == 0).to(self.device).input_ids
            # only add bos to the first seg
            for i, seg in enumerate(prompt_segs)
        ]
        seg_embs = [self.model.llama_model.model.embed_tokens(seg_t) for seg_t in seg_tokens]
        mixed_embs = [emb for pair in zip(seg_embs[:-1], img_list) for emb in pair] + [seg_embs[-1]]
        mixed_embs = torch.cat(mixed_embs, dim=1)
        return mixed_embs

if __name__ =='__main__':
    video_path = '/mnt/workspace/videoGPT/Video-LLaMA/examples/applausing.mp4'
    # import torch.classes.torchaudio.ffmpeg_StreamReader
    # ffmpeg_StreamReader(video_path)
    load_and_transform_audio_data([video_path],"cpu",  clips_per_video=8)
