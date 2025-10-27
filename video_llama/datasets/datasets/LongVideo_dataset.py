
from collections import defaultdict
from itertools import chain
import json

import os

import scipy
import torch
import numpy as np
import random
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from scipy.interpolate import interp1d

from decord import VideoReader, cpu
from PIL import Image
from transformers import Blip2Processor

from random import choice, randint


class ActivityNetDataset(Dataset):
    def __init__(self, vis_processor, text_processor, ann_root, dataset="webvid", subset="training", is_test = False, sample_num=4):
        ann_path = ann_root
        self.vis_processor = vis_processor
        self.text_processor = text_processor

        self.data = []
        if subset == "training":
            self.dir_path = os.path.join(ann_path, "rawvideos"+"_train/")
            file_path = os.path.join(ann_path, "train.json")  #results_2M.csv
        else:
            self.dir_path = os.path.join(ann_path, "rawvideos"+"_val/")
            file_path = os.path.join(ann_path, "val_1.json")  #results_2M.csv
            
        ann_f = open(file_path, 'r')
        df_data = json.load(ann_f)
        idx = 0
        file_ls = os.listdir(self.dir_path)
        file_ls = [item.split('.')[0] for item in file_ls]
        del_ls = []
        for key in df_data.keys():
            if key not in file_ls:
                del_ls.append(key)
        for key in del_ls:
            del df_data[key]
        self.data = df_data
        self.keys = list(df_data.keys())
        # for data in self.data:
        #     frames = self.save_img(data)

        self.prompt_ls = ["Describe the following video concisely.", "Provide a brief description of the given video clip.", "Offer a succinct explanation of the footage presented.", "Summarize the visual content of the following video.", "Give a short and clear explanation of the subsequent video clip.", "Share a concise interpretation of the video provided.", "Present a compact description of the clip’s key features.", "Relay a brief, clear account of the video shown.", "Render a clear and concise summary of the video below.", "Write a terse but informative summary of the following video clip.", "Create a compact narrative representing the video presented."]

    def __len__(self):
        return len(self.data.keys())

    def __getitem__(self, index):
        #frames
        while(True):
            try:
                video_id = self.keys[index]
                video_path = os.path.join(self.dir_path, video_id + ".mp4")
                video = self.vis_processor(video_path)
                prompt_cur = self.generate_prompt(self.data[video_id]['timestamps'], self.data[video_id]['duration'])
                caption_cur = self.generate_target(self.data[video_id]['sentences'], self.data[video_id]['timestamps'])
                caption = self.text_processor(prompt_cur + caption_cur)
                seg_idx = self.generate_seg_idx(self.data[video_id]['timestamps'], self.data[video_id]['duration'], video.size(1))
                break
            except: #FileNotFoundError:
                index = (index + 1) % len(self.data)
                #frames = self.load_segment(self.da8ta[index+1])
                #print("{} got error".format(self.data[index]['path']))
        return {
            "image": video,
            "text_input": caption,
            "segment_idx": seg_idx,
            "type":'video',
        }
    
    def generate_prompt(self, timestamps, duration):
        prompt = "This video has " + str(duration) + "seconds."
        prompt += " The segments respectively start and end at "
        num = len(timestamps)
        for idx, stamp in enumerate(timestamps):
            s, e = stamp[0], stamp[1]
            prompt += str(s)
            prompt += " and "
            prompt += str(e)
            if idx == num-1:
                prompt += "."
            else:
                prompt += ", "
        return prompt
    
    def generate_target(self, captions, timestamps):
        result = ""
        for idx, caption in enumerate(captions):
            s, e = timestamps[idx][0], timestamps[idx][1]
            cur_caption = " Segment from "+ str(s) + " seconds to " + str(e) + " seconds describes that "
            cur_caption += caption
            result += cur_caption
        return result

    def generate_seg_idx(self, timestamps, duration, vid_len):
        result = []
        for stamp in timestamps:
            s_idx = int(vid_len * stamp[0] / duration)
            e_idx = int(vid_len * stamp[1] / duration)
            result.append([s_idx, e_idx])
        result = torch.tensor(result)
        return result
        

    def get_by_name(self, file):
        for i in range(len(self.data)):
            if self.data[i]['path'] == file:
                return self.__getitem__(i)
        
        return 
    
    def rec_time(self, dur):
        h = int(dur[2:4])
        m = int(dur[5:7])
        s = int(dur[8:10])
        return h*3600 + m*60 + s
        
    def get_index(self, frame_s, frame_e, num_segments):
        num_frames = frame_e - frame_s
        seg_size = float(num_frames - 1) / num_segments
        start = int(seg_size / 2) + frame_s
        offsets = np.array([
            start + int(np.round(seg_size * idx)) for idx in range(num_segments)
        ])
        return offsets

    def save_img(self, video_id):
        path = os.path.join("datasets/activitynet/rawvideos_train/", video_id+".mp4")
        vr = VideoReader(path, ctx=cpu(0))
        num_frames = len(vr)
        frame_s = 0
        frame_e = num_frames
        frame_indices = self.get_index(frame_s, frame_e, 16)

        cur_path = "data/activitynet/" + video_id
        if not os.path.exists(cur_path):
            os.mkdir(cur_path)
        for id,frame_index in enumerate(frame_indices):
            image = Image.fromarray(vr[frame_index].numpy())   #vr[frame_index].asnumpy()
            image.save(os.path.join(cur_path, str(id)+".jpg"))

    def load_segment(self, seg_dict):
        if os.path.isdir(seg_dict['path']):
            #存了八帧图片
            images_group = list()
            for i in range(8):
                f_cur = os.path.join(seg_dict['path'], str(i)+'.jpg')
                img = Image.open(f_cur)
                images_group.append(img)
            return images_group
            
        vr = VideoReader(seg_dict['path'], ctx=cpu(0))
        num_frames = len(vr)
        duration = seg_dict['duration']
        #把起止时间转化为frame下标
        frame_s = 0
        frame_e = num_frames
        frame_indices = self.get_index(frame_s, frame_e, self.num_samples)

        images_group = list()
        for frame_index in frame_indices:
            img = Image.fromarray(vr[frame_index].numpy())   #vr[frame_index].asnumpy()
            images_group.append(img)
        return images_group
        
    def process(self, images_group, file, prompt):
        result = []
        file_name = file.split('.')[0]
        for id,image in enumerate(images_group):
            inputs = self.processor(image, return_tensors="pt")
            result.append(inputs['pixel_values'])
            if self.is_test:
                image.save("save/test_pics/"+file_name+"_"+str(id)+".jpg")
        result = torch.cat(result, dim=0)
        input_id = self.processor(text=prompt, return_tensors='pt')
        input_id['text_input'] = input_id['input_ids'][0]
        #input_id['attention_mask'] = input_id['attention_mask'][0]
        input_id['image']= result
        return input_id


class ActivityNetDataset_test(Dataset):
    def __init__(self, ann_root, dataset="webvid", subset="training", is_test = False, sample_num=4):
        ann_path = ann_root

        self.data = []
        if subset == "training":
            self.dir_path = os.path.join(ann_path, "rawvideos"+"_train/")
            file_path = os.path.join(ann_path, "train.json")  #results_2M.csv
        else:
            self.dir_path = os.path.join(ann_path, "rawvideos"+"_val/")
            file_path = os.path.join(ann_path, "val_1.json")  #results_2M.csv
            
        ann_f = open(file_path, 'r')
        df_data = json.load(ann_f)
        idx = 0
        file_ls = os.listdir(self.dir_path)
        file_ls = [item.split('.')[0] for item in file_ls]
        del_ls = []
        for key in df_data.keys():
            if key not in file_ls:
                del_ls.append(key)
        for key in del_ls:
            del df_data[key]
        self.data = df_data
        self.keys = list(df_data.keys())

        self.prompt_ls = ["Describe the following video concisely.", "Provide a brief description of the given video clip.", "Offer a succinct explanation of the footage presented.", "Summarize the visual content of the following video.", "Give a short and clear explanation of the subsequent video clip.", "Share a concise interpretation of the video provided.", "Present a compact description of the clip’s key features.", "Relay a brief, clear account of the video shown.", "Render a clear and concise summary of the video below.", "Write a terse but informative summary of the following video clip.", "Create a compact narrative representing the video presented."]

    def __len__(self):
        return len(self.data.keys())

    def __getitem__(self, index):
        #frames
        while(True):
            try:
                video_id = self.keys[index]
                video_path = os.path.join(self.dir_path, video_id + ".mp4")

                prompt_cur = self.generate_prompt(self.data[video_id]['timestamps'], self.data[video_id]['duration'])
                caption_cur = self.generate_target(self.data[video_id]['sentences'], self.data[video_id]['timestamps'])

                # seg_idx = self.generate_seg_idx(self.data[video_id]['timestamps'], self.data[video_id]['duration'], video.size(1))
                item_dict = {'path':video_path, 'sentence_prompt':prompt_cur, 'sentence_target': caption_cur, 'timestamps': self.data[video_id]['timestamps'], 'duration':self.data[video_id]['duration']}

                break
            except: #FileNotFoundError:
                index = (index + 1) % len(self.data)
                #frames = self.load_segment(self.da8ta[index+1])
                #print("{} got error".format(self.data[index]['path']))
        return item_dict
    
    def generate_prompt(self, timestamps, duration):
        prompt = "This video has " + str(duration) + "seconds."
        prompt += " The segments respectively start and end at "
        num = len(timestamps)
        for idx, stamp in enumerate(timestamps):
            s, e = stamp[0], stamp[1]
            prompt += str(s)
            prompt += " and "
            prompt += str(e)
            if idx == num-1:
                prompt += "."
            else:
                prompt += ", "
        return prompt
    
    def generate_target(self, captions, timestamps):
        result = ""
        for idx, caption in enumerate(captions):
            s, e = timestamps[idx][0], timestamps[idx][1]
            cur_caption = " Segment from "+ str(s) + " seconds to " + str(e) + " seconds describes that "
            cur_caption += caption
            result += cur_caption
        return result

    def generate_seg_idx(self, timestamps, duration, vid_len):
        result = []
        for stamp in timestamps:
            s_idx = int(vid_len * stamp[0] / duration)
            e_idx = int(vid_len * stamp[1] / duration)
            result.append([s_idx, e_idx])
        result = torch.tensor(result)
        return result
