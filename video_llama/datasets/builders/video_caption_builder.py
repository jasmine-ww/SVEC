import os
import logging
import warnings

from video_llama.common.registry import registry
from video_llama.datasets.builders.base_dataset_builder import BaseDatasetBuilder
#from video_llama.datasets.datasets.webvid_datasets import WebvidDataset
from video_llama.datasets.datasets.webvid_my_datasets import WebvidDataset, WebVidDataset_seg
from video_llama.datasets.datasets.LongVideo_dataset import ActivityNetDataset


@registry.register_builder("webvid")
class WebvidBuilder(BaseDatasetBuilder):
    train_dataset_cls = WebvidDataset
    DATASET_CONFIG_DICT = {"default": "configs/datasets/webvid/defaults.yaml"}
    
    def _download_ann(self):
        pass

    def _download_vis(self):
        pass

    def build(self):
        self.build_processors()
        datasets = dict()
        split = "train"

        build_info = self.config.build_info
        dataset_cls = self.train_dataset_cls
        # datasets[split] = dataset_cls(
        #     "/data1/qzhb/dvcflow_pretrain/dvcflow_pretrain/video_feature/", subset="training",
        #     vis_processor=self.vis_processors[split],
        #     text_processor=self.text_processors[split],
        #     sample_num=8
        # )
        datasets[split] = dataset_cls(
            vis_processor=self.vis_processors[split],
            text_processor=self.text_processors[split],
            vis_root=build_info.videos_dir,
            ann_root=build_info.anno_dir
        )

        return datasets
        
@registry.register_builder("webvid_seg")
class Webvid_segBuilder(BaseDatasetBuilder):
    train_dataset_cls = WebVidDataset_seg
    DATASET_CONFIG_DICT = {"default": "configs/datasets/webvid/defaults_seg.yaml"}
    
    def _download_ann(self):
        pass

    def _download_vis(self):
        pass

    def build(self):
        self.build_processors()
        datasets = dict()
        split = "train"

        build_info = self.config.build_info
        dataset_cls = self.train_dataset_cls
        # datasets[split] = dataset_cls(
        #     "/data1/qzhb/dvcflow_pretrain/dvcflow_pretrain/video_feature/", subset="training",
        #     vis_processor=self.vis_processors[split],
        #     text_processor=self.text_processors[split],
        #     sample_num=8
        # )
        datasets[split] = dataset_cls(
            vis_processor=self.vis_processors[split],
            text_processor=self.text_processors[split],
            vis_root=build_info.videos_dir,
            ann_root=build_info.anno_dir
        )

        return datasets

@registry.register_builder("Activitynet")
class Activitynet_Builder(BaseDatasetBuilder):
    train_dataset_cls = ActivityNetDataset
    DATASET_CONFIG_DICT = {"default": "configs/datasets/long_video/defaults.yaml"}
    
    def _download_ann(self):
        pass

    def _download_vis(self):
        pass

    def build(self):
        self.build_processors()
        datasets = dict()
        split = "train"

        build_info = self.config.build_info
        dataset_cls = self.train_dataset_cls
        # datasets[split] = dataset_cls(
        #     "/data1/qzhb/dvcflow_pretrain/dvcflow_pretrain/video_feature/", subset="training",
        #     vis_processor=self.vis_processors[split],
        #     text_processor=self.text_processors[split],
        #     sample_num=8
        # )
        datasets[split] = dataset_cls(
            vis_processor=self.vis_processors[split],
            text_processor=self.text_processors[split],
            ann_root=build_info.anno_dir
        )

        return datasets