import torch
import torch.utils.data as data
import numpy as np
import json as jsonmod

from basic.util import getVideoId
from vocab import clean_str

import sys

VIDEO_MAX_LEN=64

def collate_frame_gru_fn(data):
    """
    Build mini-batch tensors from a list of (video, caption) tuples.
    """
    # Sort a data list by caption length
    if len(data[0]) == 10:
        if data[0][1] is not None:
            data.sort(key=lambda x: len(x[1]), reverse=True)
        videos, captions, cap_bows, idxs, cap_ids, video_ids, videos_target,  video_ids_target, cap_tensor_target, cap_bow_target= zip(*data)

        # Merge videos (convert tuple of 1D tensor to 4D tensor)
        video_lengths = [min(VIDEO_MAX_LEN,len(frame)) for frame in videos]

        frame_vec_len = len(videos[0][0])
        vidoes = torch.zeros(len(videos), max(video_lengths), frame_vec_len)
        videos_origin = torch.zeros(len(videos), frame_vec_len)
        vidoes_mask = torch.zeros(len(videos), max(video_lengths))
        for i, frames in enumerate(videos):
                end = video_lengths[i]
                vidoes[i, :end, :] = frames[:end,:]
                videos_origin[i,:] = torch.mean(frames,0)
                vidoes_mask[i,:end] = 1.0

        video_lengths_target = [min(VIDEO_MAX_LEN,len(frame)) for frame in videos_target]

        frame_vec_len = len(videos_target[0][0])
        vidoes_target = torch.zeros(len(videos_target), max(video_lengths_target), frame_vec_len)
        videos_origin_target = torch.zeros(len(videos_target), frame_vec_len)
        vidoes_mask_target = torch.zeros(len(videos_target), max(video_lengths_target))
        for i, frames in enumerate(videos_target):
                end = video_lengths_target[i]
                vidoes_target[i, :end, :] = frames[:end,:]
                videos_origin_target[i,:] = torch.mean(frames,0)
                vidoes_mask_target[i,:end] = 1.0
        
        if captions[0] is not None:
            # Merge captions (convert tuple of 1D tensor to 2D tensor)
            lengths = [len(cap) for cap in captions]
            target = torch.zeros(len(captions), max(lengths)).long()
            words_mask = torch.zeros(len(captions), max(lengths))
            for i, cap in enumerate(captions):
                end = lengths[i]
                target[i, :end] = cap[:end]
                words_mask[i, :end] = 1.0
        else:
            target = None
            lengths = None
            words_mask = None

        if cap_tensor_target[0] is not None:
            # Merge captions (convert tuple of 1D tensor to 2D tensor)
            lengths_target = [len(cap) for cap in cap_tensor_target]
            target_target = torch.zeros(len(cap_tensor_target), max(lengths_target)).long()
            words_mask_target = torch.zeros(len(cap_tensor_target), max(lengths_target))
            for i, cap in enumerate(cap_tensor_target):
                end = lengths_target[i]
                target_target[i, :end] = cap[:end]
                words_mask_target[i, :end] = 1.0
        else:
            target_target = None
            lengths_target = None
            words_mask_target = None


        cap_bows = torch.stack(cap_bows, 0) if cap_bows[0] is not None else None

        cap_bow_target = torch.stack(cap_bow_target, 0) if cap_bow_target[0] is not None else None

        video_data = (vidoes, videos_origin, video_lengths, vidoes_mask)
        text_data = (target, cap_bows, lengths, words_mask)
        text_data_target = (target_target, cap_bow_target, lengths_target, words_mask_target)
        video_data_target = (vidoes_target, videos_origin_target, video_lengths_target, vidoes_mask_target)

        return video_data, text_data, idxs, cap_ids, video_ids, video_ids_target, video_data_target, text_data_target

    elif len(data[0]) == 14:
        if data[0][1] is not None:
            data.sort(key=lambda x: len(x[1]), reverse=True)
        videos, captions, cap_bows, idxs, cap_ids, video_ids, videos_target,  video_ids_target, cap_tensor_target, cap_bow_target, videos_source2,  video_ids_source2, cap_tensor_source2, cap_bow_source2= zip(*data)

        # Merge videos (convert tuple of 1D tensor to 4D tensor)
        video_lengths = [min(VIDEO_MAX_LEN,len(frame)) for frame in videos]

        frame_vec_len = len(videos[0][0])
        vidoes = torch.zeros(len(videos), max(video_lengths), frame_vec_len)
        videos_origin = torch.zeros(len(videos), frame_vec_len)
        vidoes_mask = torch.zeros(len(videos), max(video_lengths))
        for i, frames in enumerate(videos):
                end = video_lengths[i]
                vidoes[i, :end, :] = frames[:end,:]
                videos_origin[i,:] = torch.mean(frames,0)
                vidoes_mask[i,:end] = 1.0

        video_lengths_target = [min(VIDEO_MAX_LEN,len(frame)) for frame in videos_target]

        frame_vec_len = len(videos_target[0][0])
        vidoes_target = torch.zeros(len(videos_target), max(video_lengths_target), frame_vec_len)
        videos_origin_target = torch.zeros(len(videos_target), frame_vec_len)
        vidoes_mask_target = torch.zeros(len(videos_target), max(video_lengths_target))
        for i, frames in enumerate(videos_target):
                end = video_lengths_target[i]
                vidoes_target[i, :end, :] = frames[:end,:]
                videos_origin_target[i,:] = torch.mean(frames,0)
                vidoes_mask_target[i,:end] = 1.0

        video_lengths_source2 = [min(VIDEO_MAX_LEN,len(frame)) for frame in videos_source2]

        frame_vec_len = len(videos_source2[0][0])
        vidoes_source2 = torch.zeros(len(videos_source2), max(video_lengths_source2), frame_vec_len)
        videos_origin_source2 = torch.zeros(len(videos_source2), frame_vec_len)
        vidoes_mask_source2 = torch.zeros(len(videos_source2), max(video_lengths_source2))
        for i, frames in enumerate(videos_source2):
                end = video_lengths_source2[i]
                vidoes_source2[i, :end, :] = frames[:end,:]
                videos_origin_source2[i,:] = torch.mean(frames,0)
                vidoes_mask_source2[i,:end] = 1.0

        if captions[0] is not None:
            # Merge captions (convert tuple of 1D tensor to 2D tensor)
            lengths = [len(cap) for cap in captions]
            target = torch.zeros(len(captions), max(lengths)).long()
            words_mask = torch.zeros(len(captions), max(lengths))
            for i, cap in enumerate(captions):
                end = lengths[i]
                target[i, :end] = cap[:end]
                words_mask[i, :end] = 1.0
        else:
            target = None
            lengths = None
            words_mask = None

        if cap_tensor_target[0] is not None:
            # Merge captions (convert tuple of 1D tensor to 2D tensor)
            lengths_target = [len(cap) for cap in cap_tensor_target]
            target_target = torch.zeros(len(cap_tensor_target), max(lengths_target)).long()
            words_mask_target = torch.zeros(len(cap_tensor_target), max(lengths_target))
            for i, cap in enumerate(cap_tensor_target):
                end = lengths_target[i]
                target_target[i, :end] = cap[:end]
                words_mask_target[i, :end] = 1.0
        else:
            target_target = None
            lengths_target = None
            words_mask_target = None

        if cap_tensor_source2[0] is not None:
            # Merge captions (convert tuple of 1D tensor to 2D tensor)
            lengths_source2 = [len(cap) for cap in cap_tensor_source2]
            target_source2 = torch.zeros(len(cap_tensor_source2), max(lengths_source2)).long()
            words_mask_source2 = torch.zeros(len(cap_tensor_source2), max(lengths_source2))
            for i, cap in enumerate(cap_tensor_source2):
                end = lengths_source2[i]
                target_source2[i, :end] = cap[:end]
                words_mask_source2[i, :end] = 1.0
        else:
            target_source2 = None
            lengths_source2 = None
            words_mask_source2 = None

        cap_bows = torch.stack(cap_bows, 0) if cap_bows[0] is not None else None
        cap_bow_target = torch.stack(cap_bow_target, 0) if cap_bow_target[0] is not None else None
        cap_bow_source2 = torch.stack(cap_bow_source2, 0) if cap_bow_source2[0] is not None else None

        video_data = (vidoes, videos_origin, video_lengths, vidoes_mask)
        text_data = (target, cap_bows, lengths, words_mask)
        text_data_target = (target_target, cap_bow_target, lengths_target, words_mask_target)
        video_data_target = (vidoes_target, videos_origin_target, video_lengths_target, vidoes_mask_target)
        text_data_source2 = (target_source2, cap_bow_source2, lengths_source2, words_mask_source2)
        video_data_source2 = (vidoes_source2, videos_origin_source2, video_lengths_source2, vidoes_mask_source2)


        return video_data, text_data, idxs, cap_ids, video_ids, video_ids_target, video_data_target, text_data_target, video_ids_source2, video_data_source2, text_data_source2


    else:
        if data[0][1] is not None:
            data.sort(key=lambda x: len(x[1]), reverse=True)
        videos, captions, cap_bows, idxs, cap_ids, video_ids = zip(*data)

        # Merge videos (convert tuple of 1D tensor to 4D tensor)
        video_lengths = [min(VIDEO_MAX_LEN,len(frame)) for frame in videos]
        frame_vec_len = len(videos[0][0])
        vidoes = torch.zeros(len(videos), max(video_lengths), frame_vec_len)
        videos_origin = torch.zeros(len(videos), frame_vec_len)
        vidoes_mask = torch.zeros(len(videos), max(video_lengths))
        for i, frames in enumerate(videos):
                end = video_lengths[i]
                vidoes[i, :end, :] = frames[:end,:]
                videos_origin[i,:] = torch.mean(frames,0)
                vidoes_mask[i,:end] = 1.0
        
        if captions[0] is not None:
            # Merge captions (convert tuple of 1D tensor to 2D tensor)
            lengths = [len(cap) for cap in captions]
            target = torch.zeros(len(captions), max(lengths)).long()
            words_mask = torch.zeros(len(captions), max(lengths))
            for i, cap in enumerate(captions):
                end = lengths[i]
                target[i, :end] = cap[:end]
                words_mask[i, :end] = 1.0
        else:
            target = None
            lengths = None
            words_mask = None


        cap_bows = torch.stack(cap_bows, 0) if cap_bows[0] is not None else None

        video_data = (vidoes, videos_origin, video_lengths, vidoes_mask)
        text_data = (target, cap_bows, lengths, words_mask)

        return video_data, text_data, idxs, cap_ids, video_ids


def collate_frame(data):

    videos, idxs, video_ids = zip(*data)

    # Merge videos (convert tuple of 1D tensor to 4D tensor)
    video_lengths = [min(VIDEO_MAX_LEN,len(frame)) for frame in videos]
    frame_vec_len = len(videos[0][0])
    vidoes = torch.zeros(len(videos), max(video_lengths), frame_vec_len)
    videos_origin = torch.zeros(len(videos), frame_vec_len)
    vidoes_mask = torch.zeros(len(videos), max(video_lengths))
    for i, frames in enumerate(videos):
            end = video_lengths[i]
            vidoes[i, :end, :] = frames[:end,:]
            videos_origin[i,:] = torch.mean(frames,0)
            vidoes_mask[i,:end] = 1.0

    video_data = (vidoes, videos_origin, video_lengths, vidoes_mask)

    return video_data, idxs, video_ids


def collate_text(data):
    if data[0][0] is not None:
        data.sort(key=lambda x: len(x[0]), reverse=True)
    captions, cap_bows, idxs, cap_ids = zip(*data)

    if captions[0] is not None:
        # Merge captions (convert tuple of 1D tensor to 2D tensor)
        lengths = [len(cap) for cap in captions]
        target = torch.zeros(len(captions), max(lengths)).long()
        words_mask = torch.zeros(len(captions), max(lengths))
        for i, cap in enumerate(captions):
            end = lengths[i]
            target[i, :end] = cap[:end]
            words_mask[i, :end] = 1.0
    else:
        target = None
        lengths = None
        words_mask = None


    cap_bows = torch.stack(cap_bows, 0) if cap_bows[0] is not None else None

    text_data = (target, cap_bows, lengths, words_mask)

    return text_data, idxs, cap_ids


class Dataset4DualEncoding(data.Dataset):
    """
    Load captions and video frame features by pre-trained CNN model.
    """

    def __init__(self, cap_file, visual_feat, bow2vec, vocab, n_caption=None, video2frames=None, visual_feat_target=None, video2frames_target=None, caption_file_target=None, visual_feat_source2=None, video2frames_source2=None, caption_file_source2=None):
        # Captions
        self.captions = {}
        self.captions_target = {}
        self.captions_source2 = {}

        self.cap_ids = []
        self.cap_ids_target = []
        self.cap_ids_source2 = []

        self.visual_feat = visual_feat
        self.video2frames = video2frames
        self.visual_feat_target = visual_feat_target
        self.video2frames_target = video2frames_target
        self.visual_feat_source2 = visual_feat_source2
        self.video2frames_source2 = video2frames_source2


        with open(cap_file, 'r') as cap_reader:
            for line in cap_reader.readlines():
                cap_id, caption = line.strip().split(' ', 1)
                self.captions[cap_id] = caption
                self.cap_ids.append(cap_id)
                

        if visual_feat_target!=None and video2frames_target!=None:
            with open(caption_file_target, 'r') as cap_reader:
                for line in cap_reader.readlines():
                    cap_id, caption = line.strip().split(' ', 1)
                    self.captions_target[cap_id] = caption
                    self.cap_ids_target.append(cap_id)
        
        if visual_feat_source2!=None and video2frames_source2!=None:
            with open(caption_file_source2, 'r') as cap_reader:
                for line in cap_reader.readlines():
                    cap_id, caption = line.strip().split(' ', 1)
                    self.captions_source2[cap_id] = caption
                    self.cap_ids_source2.append(cap_id)


        self.bow2vec = bow2vec
        self.vocab = vocab
        self.length = len(self.cap_ids)
#        self.cap_ids_target = self.cap_ids_target * 2

        print(self.length)
        print(len(self.cap_ids_target))

#        if n_caption is not None:
#            assert len(self.video_ids) * n_caption == self.length, "%d != %d" % (len(self.video_ids) * n_caption, self.length)

    def __getitem__(self, index):
        cap_id = self.cap_ids[index]
        video_id = getVideoId(cap_id)
        frame_list = self.video2frames[video_id]
        frame_vecs = []
        for frame_id in frame_list:
            frame_vecs.append(self.visual_feat.read_one(frame_id))
        frames_tensor = torch.Tensor(frame_vecs)

        caption = self.captions[cap_id]
        if self.bow2vec is not None:
            cap_bow = self.bow2vec.mapping(caption)
            if cap_bow is None:
                cap_bow = torch.zeros(self.bow2vec.ndims)
            else:
                cap_bow = torch.Tensor(cap_bow)
        else:
            cap_bow = None

        if self.vocab is not None:
            tokens = clean_str(caption)
            caption = []
            caption.append(self.vocab('<start>'))
            caption.extend([self.vocab(token) for token in tokens])
            caption.append(self.vocab('<end>'))
            cap_tensor = torch.Tensor(caption)
        else:
            cap_tensor = None

        if self.visual_feat_target != None and self.video2frames_target != None:

            cap_id_target = self.cap_ids_target[index]
            video_id_target = getVideoId(cap_id_target)
                
            frame_list = self.video2frames_target[video_id_target]
            frame_vecs_target = []
            for frame_id in frame_list:
                frame_vecs_target.append(self.visual_feat_target.read_one(frame_id))
            frames_tensor_target = torch.Tensor(frame_vecs_target)

            caption_target = self.captions_target[cap_id_target]
            # caption_target = self.cap_ids_target[index]
            # video_id_target = 'a'
            # frame_list = self.video2frames_target[self.target_video_id_list[index]]
            # frame_vecs_target = []
            # for frame_id in frame_list:
            #     frame_vecs_target.append(self.visual_feat_target.read_one(frame_id))
            # frames_tensor_target = torch.Tensor(frame_vecs_target)

            if self.bow2vec is not None:
                cap_bow_target = self.bow2vec.mapping(caption_target)
                if cap_bow_target is None:
                    cap_bow_target = torch.zeros(self.bow2vec.ndims)
                else:
                    cap_bow_target = torch.Tensor(cap_bow_target)
            else:
                cap_bow_target = None

            if self.vocab is not None:
                tokens = clean_str(caption_target)
                caption_target = []
                caption_target.append(self.vocab('<start>'))
                caption_target.extend([self.vocab(token) for token in tokens])
                caption_target.append(self.vocab('<end>'))
                cap_tensor_target = torch.Tensor(caption_target)
            else:
                cap_tensor_target = None

            if self.visual_feat_source2 != None and self.video2frames_source2 != None: 
                cap_id_source2 = self.cap_ids_source2[index]
                video_id_source2 = getVideoId(cap_id_source2)
    
                frame_list = self.video2frames_source2[video_id_source2]
                frame_vecs_source2 = []
                for frame_id in frame_list:
                    frame_vecs_source2.append(self.visual_feat_source2.read_one(frame_id))
                frames_tensor_source2 = torch.Tensor(frame_vecs_source2)
    
                caption_source2 = self.captions_source2[cap_id_source2]
                # caption_source2 = self.cap_ids_source2[index]
                # video_id_source2 = 'a'
                # frame_list = self.video2frames_source2[self.source2_video_id_list[index]]
                # frame_vecs_source2 = []
                # for frame_id in frame_list:
                #     frame_vecs_source2.append(self.visual_feat_source2.read_one(frame_id))
                # frames_tensor_source2 = torch.Tensor(frame_vecs_source2)
    
                if self.bow2vec is not None:
                    cap_bow_source2 = self.bow2vec.mapping(caption_source2)
                    if cap_bow_source2 is None:
                        cap_bow_source2 = torch.zeros(self.bow2vec.ndims)
                    else:
                        cap_bow_source2 = torch.Tensor(cap_bow_source2)
                else:
                    cap_bow_source2 = None
    
                if self.vocab is not None:
                    tokens = clean_str(caption_source2)
                    caption_source2 = []
                    caption_source2.append(self.vocab('<start>'))
                    caption_source2.extend([self.vocab(token) for token in tokens])
                    caption_source2.append(self.vocab('<end>'))
                    cap_tensor_source2 = torch.Tensor(caption_source2)
                else:
                    cap_tensor_source2 = None

               

                return frames_tensor, cap_tensor, cap_bow, index, cap_id, video_id, frames_tensor_target, video_id_target, cap_tensor_target, cap_bow_target, frames_tensor_source2, video_id_source2, cap_tensor_source2, cap_bow_source2

            else:
                return frames_tensor, cap_tensor, cap_bow, index, cap_id, video_id, frames_tensor_target, video_id_target, cap_tensor_target, cap_bow_target
        else:
            return frames_tensor, cap_tensor, cap_bow, index, cap_id, video_id

    def __len__(self):
        return self.length
     

class VisDataSet4DualEncoding(data.Dataset):
    """
    Load video frame features by pre-trained CNN model.
    """
    def __init__(self, visual_feat, video2frames=None):
        self.visual_feat = visual_feat
        self.video2frames = video2frames

        self.video_ids = video2frames.keys()
        self.length = len(self.video_ids)

    def __getitem__(self, index):
        video_id = self.video_ids[index]

        frame_list = self.video2frames[video_id]
        frame_vecs = []
        for frame_id in frame_list:
            frame_vecs.append(self.visual_feat.read_one(frame_id))
        frames_tensor = torch.Tensor(frame_vecs)

        return frames_tensor, index, video_id

    def __len__(self):
        return self.length


class TxtDataSet4DualEncoding(data.Dataset):
    """
    Load captions
    """
    def __init__(self, cap_file, bow2vec, vocab):
        # Captions
        self.captions = {}
        self.cap_ids = []
        with open(cap_file, 'r') as cap_reader:
            for line in cap_reader.readlines():
                cap_id, caption = line.strip().split(' ', 1)
                self.captions[cap_id] = caption
                self.cap_ids.append(cap_id)
        self.bow2vec = bow2vec
        self.vocab = vocab
        self.length = len(self.cap_ids)

    def __getitem__(self, index):
        cap_id = self.cap_ids[index]

        caption = self.captions[cap_id]
        if self.bow2vec is not None:
            cap_bow = self.bow2vec.mapping(caption)
            if cap_bow is None:
                cap_bow = torch.zeros(self.bow2vec.ndims)
            else:
                cap_bow = torch.Tensor(cap_bow)
        else:
            cap_bow = None

        if self.vocab is not None:
            tokens = clean_str(caption)
            caption = []
            caption.append(self.vocab('<start>'))
            caption.extend([self.vocab(token) for token in tokens])
            caption.append(self.vocab('<end>'))
            cap_tensor = torch.Tensor(caption)
        else:
            cap_tensor = None

        return cap_tensor, cap_bow, index, cap_id

    def __len__(self):
        return self.length

def get_data_loaders(cap_files, visual_feats, vocab, bow2vec, batch_size=100, num_workers=2, n_caption=2, video2frames=None, video2frames_target=None, visual_feats_target=None, caption_file_target=None, multi_flag=0):
    """
    Returns torch.utils.data.DataLoader for train and validation datasets
    Args:
        cap_files: caption files (dict) keys: [train, val]
        visual_feats: image feats (dict) keys: [train, val]
    """
    if video2frames_target!=None and visual_feats_target!=None:
        if multi_flag == 0:
            dset = {'train': Dataset4DualEncoding(cap_files['train'], visual_feats['train'], bow2vec, vocab, video2frames=video2frames['train'], video2frames_target=video2frames_target['train'], visual_feat_target=visual_feats_target['train'], caption_file_target=caption_file_target),
                    'val': Dataset4DualEncoding(cap_files['val'], visual_feats['val'], bow2vec, vocab, n_caption, video2frames=video2frames['val']),
                    'test': Dataset4DualEncoding(cap_files['test'], visual_feats['test'], bow2vec, vocab, n_caption, video2frames=video2frames['test'])}
        else:
            dset = {'train': Dataset4DualEncoding(cap_files['train'], visual_feats['train'], bow2vec, vocab, video2frames=video2frames['train'], video2frames_target=video2frames_target['train'], visual_feat_target=visual_feats_target['train'], caption_file_target=caption_file_target, visual_feat_source2=visual_feats['train2'], video2frames_source2=video2frames['train2'], caption_file_source2=cap_files['train2']),
                    'val': Dataset4DualEncoding(cap_files['val'], visual_feats['val'], bow2vec, vocab, n_caption, video2frames=video2frames['val']),
                    'test': Dataset4DualEncoding(cap_files['test'], visual_feats['test'], bow2vec, vocab, n_caption, video2frames=video2frames['test'])}



    else:
        dset = {'train': Dataset4DualEncoding(cap_files['train'], visual_feats['train'], bow2vec, vocab, video2frames=video2frames['train']),
                'val': Dataset4DualEncoding(cap_files['val'], visual_feats['val'], bow2vec, vocab, n_caption, video2frames=video2frames['val']),
                'test': Dataset4DualEncoding(cap_files['test'], visual_feats['test'], bow2vec, vocab, n_caption, video2frames=video2frames['test'])}

    data_loaders = {x: torch.utils.data.DataLoader(dataset=dset[x],
                                    batch_size=batch_size,
                                    shuffle=(x=='train'),
                                    pin_memory=True,
                                    num_workers=num_workers,
                                    collate_fn=collate_frame_gru_fn)
                        for x in ['train', 'val', 'test']}
    return data_loaders


def get_test_data_loaders(cap_files, visual_feats, vocab, bow2vec, batch_size=100, num_workers=2, n_caption=2, video2frames = None):
    """
    Returns torch.utils.data.DataLoader for test dataset
    Args:
        cap_files: caption files (dict) keys: [test]
        visual_feats: image feats (dict) keys: [test]
    """
    dset = {'test': Dataset4DualEncoding(cap_files['test'], visual_feats['test'], bow2vec, vocab, n_caption, video2frames = video2frames['test'])}


    data_loaders = {x: torch.utils.data.DataLoader(dataset=dset[x],
                                    batch_size=batch_size,
                                    shuffle=False,
                                    pin_memory=True,
                                    num_workers=num_workers,
                                    collate_fn=collate_frame_gru_fn)
                        for x in cap_files }
    return data_loaders


def get_vis_data_loader(vis_feat, batch_size=100, num_workers=2, video2frames=None):
    dset = VisDataSet4DualEncoding(vis_feat, video2frames)

    data_loader = torch.utils.data.DataLoader(dataset=dset,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              pin_memory=True,
                                              num_workers=num_workers,
                                              collate_fn=collate_frame)
    return data_loader


def get_txt_data_loader(cap_file, vocab, bow2vec, batch_size=100, num_workers=2):
    dset = TxtDataSet4DualEncoding(cap_file, bow2vec, vocab)

    data_loader = torch.utils.data.DataLoader(dataset=dset,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              pin_memory=True,
                                              num_workers=num_workers,
                                              collate_fn=collate_text)
    return data_loader



if __name__ == '__main__':
    pass
