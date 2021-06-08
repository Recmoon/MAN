import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import pdb



def cosine_sim(im, s):
    """Cosine similarity between all the image and sentence pairs
    """
    return im.mm(s.t())


def order_sim(im, s):
    """Order embeddings similarity measure $max(0, s-im)$
    """
    YmX = (s.unsqueeze(1).expand(s.size(0), im.size(0), s.size(1))
           - im.unsqueeze(0).expand(s.size(0), im.size(0), s.size(1)))
    score = -YmX.clamp(min=0).pow(2).sum(2).sqrt().t()
    return score


def euclidean_sim(im, s):
    """Order embeddings similarity measure $max(0, s-im)$
    """
    YmX = (s.unsqueeze(1).expand(s.size(0), im.size(0), s.size(1))
           - im.unsqueeze(0).expand(s.size(0), im.size(0), s.size(1)))
    score = -YmX.pow(2).sum(2).t()
    return score    

class improveTripletLoss(nn.Module):
    """
    triplet ranking loss
    """

    def __init__(self, margin=0, measure=False, max_violation=False, cost_style='sum', direction='all'):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.cost_style = cost_style
        self.direction = direction
        if measure == 'order':
            self.sim = order_sim
        elif measure == 'euclidean':
            self.sim = euclidean_sim
        else:
            self.sim = cosine_sim

        self.max_violation = max_violation

    def forward(self, s, im, s_idx, t_idx):
        # compute image-sentence score matrix
        scores = self.sim(im, s)
        seten_sim = self.sim(s, s)
        video_sim = self.sim(im, im)

        diagonal = scores.diag().view(im.size(0), 1)
        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)

        # clear diagonals
        mask = torch.eye(scores.size(0)) > .5
        I = Variable(mask)
        
        # generate_index
        inter_domain_pair_index = []
        intra_domain_pair_index = []
        if len(s_idx) > len(t_idx):
            add_ele = len(s_idx) - len(t_idx)
        else:
            add_ele = len(t_idx) - len(s_idx)

        for ii in range(scores.size(0)):
            if len(s_idx) > len(t_idx):
                if ii in s_idx:
                    inter_domain_pair_index.append(t_idx+[ii]*add_ele)
                    intra_domain_pair_index.append(s_idx)
                else:
                    inter_domain_pair_index.append(s_idx)
                    intra_domain_pair_index.append(t_idx+[ii]*add_ele)
            elif len(s_idx) < len(t_idx):
                if ii in s_idx:
                    inter_domain_pair_index.append(t_idx)
                    intra_domain_pair_index.append(s_idx+[ii]*add_ele)
                else:
                    inter_domain_pair_index.append(s_idx+[ii]*add_ele)
                    intra_domain_pair_index.append(t_idx)
            else:
                if ii in s_idx:
                    inter_domain_pair_index.append(t_idx)
                    intra_domain_pair_index.append(s_idx)
                else:
                    inter_domain_pair_index.append(s_idx)
                    intra_domain_pair_index.append(t_idx)


        inter_domain_pair_index = Variable(torch.LongTensor(inter_domain_pair_index), volatile=True).cuda()
        intra_domain_pair_index = Variable(torch.LongTensor(intra_domain_pair_index), volatile=True).cuda()
        
        # for re_idx in range(scores.size(0)):
        #     if re_idx in s_idx:
        #         for xx in t_idx:
        #             I[re_idx, xx] = True
        #     else:
        #         for xx in s_idx:
        #             I[re_idx, xx] = True
        if torch.cuda.is_available():
            I = I.cuda()

        cost_s = None
        cost_im = None
        # compare every diagonal score to scores in its column
        if self.direction in  ['i2t', 'all']:
            # caption retrieval
            cost_s = (self.margin + scores - d1).clamp(min=0)
            cost_s = cost_s.masked_fill_(I, 0)
        # compare every diagonal score to scores in its row
        if self.direction in ['t2i', 'all']:
            # image retrieval
            cost_im = (self.margin + scores - d2).clamp(min=0)
            cost_im = cost_im.masked_fill_(I, 0)

        # keep the maximum violating negative for each query
        if self.max_violation:
            if cost_s is not None:
                # max_idx = cost_s.max(1)[1].cpu().data.numpy()
                # not_match_idx = []
                # for re_idx in range(len(max_idx)):
                #     if re_idx in s_idx:
                #         if max_idx[re_idx] not in s_idx:
                #             not_match_idx.append(re_idx)
                #     else:
                #         if max_idx[re_idx] not in t_idx:
                #             not_match_idx.append(re_idx)
                inter_pair_sim = torch.gather(cost_s, 1, inter_domain_pair_index)
                intra_pair_sim = torch.gather(cost_s, 1, intra_domain_pair_index)

                intra_mask = intra_pair_sim.max(1)[0] > inter_pair_sim.max(1)[0]
                inter_mask = intra_pair_sim.max(1)[0] < inter_pair_sim.max(1)[0]

                stn_max_sim = torch.gather(torch.gather(seten_sim, 1, inter_domain_pair_index), 1, inter_pair_sim.max(1)[1].unsqueeze(1)).squeeze()
                stn_sim_mask_for_inter = stn_max_sim < -0.5
                # print(inter_mask * stn_sim_mask_for_inter)

                loss_for_intra = torch.masked_select(intra_pair_sim.max(1)[0], intra_mask).sum()
                loss_for_inter = torch.masked_select(inter_pair_sim.max(1)[0], inter_mask * stn_sim_mask_for_inter).sum() + \
                    torch.masked_select(intra_pair_sim.max(1)[0], inter_mask * (stn_sim_mask_for_inter^1)).sum()
                

                cost_s = loss_for_intra + loss_for_inter
            if cost_im is not None:
                # cost_im = cost_im.max(0)[0]
                inter_im_pair_sim = torch.gather(cost_im, 0, inter_domain_pair_index.t())
                intra_im_pair_sim = torch.gather(cost_im, 0, intra_domain_pair_index.t())

                intra_mask_im = intra_im_pair_sim.max(0)[0] > inter_im_pair_sim.max(0)[0]
                inter_mask_im = intra_im_pair_sim.max(0)[0] < inter_im_pair_sim.max(0)[0]

                stn_max_sim_im = torch.gather(torch.gather(video_sim, 0, inter_domain_pair_index.t()), 0, inter_im_pair_sim.max(0)[1].unsqueeze(0)).squeeze()
                stn_sim_mask_for_inter_im = stn_max_sim_im < -0.5

                # print(inter_mask_im * stn_sim_mask_for_inter_im)

                loss_for_intra_im = torch.masked_select(intra_im_pair_sim.max(0)[0], intra_mask_im).sum()
                loss_for_inter_im = torch.masked_select(inter_im_pair_sim.max(0)[0], inter_mask_im * stn_sim_mask_for_inter_im).sum() + \
                    torch.masked_select(intra_im_pair_sim.max(0)[0], inter_mask_im * (stn_sim_mask_for_inter_im^1)).sum()
                

                cost_im = loss_for_intra_im + loss_for_inter_im



        if cost_s is None:
            cost_s = Variable(torch.zeros(1)).cuda()
        if cost_im is None:
            cost_im = Variable(torch.zeros(1)).cuda()

        if self.cost_style == 'sum':
            return cost_s + cost_im
        else:
            return cost_s.mean() + cost_im.mean()

class WeightedTripletLoss(nn.Module):
    """
    triplet ranking loss
    """

    def __init__(self, margin=0, measure=False, max_violation=False, cost_style='sum', direction='all'):
        super(WeightedTripletLoss, self).__init__()
        self.margin = margin
        self.cost_style = cost_style
        self.direction = direction
        if measure == 'order':
            self.sim = order_sim
        elif measure == 'euclidean':
            self.sim = euclidean_sim
        else:
            self.sim = cosine_sim

        self.max_violation = max_violation

    def forward(self, s, im, s_prob_matrix, im_prob_matrix):

        # compute the probabity that the sample belongs to the source domain
        s_softmax = F.softmax(s_prob_matrix, dim=1)   
        s_prob_tensor = s_softmax[:, 1]
        im_softmax = F.softmax(im_prob_matrix, dim=1)
        im_prob_tensor = im_softmax[:, 1]

        print('s', s_prob_tensor)
        print('im', im_prob_tensor)
        sys.exit(0)

        # compute image-sentence score matrix
        scores = self.sim(im, s)
        diagonal = scores.diag().view(im.size(0), 1)
        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)

        # clear diagonals
        mask = torch.eye(scores.size(0)) > .5
        I = Variable(mask)
        if torch.cuda.is_available():
            I = I.cuda()

        cost_s = None
        cost_im = None
        # compare every diagonal score to scores in its column
        if self.direction in  ['i2t', 'all']:
            # caption retrieval
            cost_s = (self.margin + scores - d1).clamp(min=0)
            cost_s = cost_s.masked_fill_(I, 0)
        # compare every diagonal score to scores in its row
        if self.direction in ['t2i', 'all']:
            # image retrieval
            cost_im = (self.margin + scores - d2).clamp(min=0)
            cost_im = cost_im.masked_fill_(I, 0)

        # keep the maximum violating negative for each query
        if self.max_violation:
            if cost_s is not None:
                cost_s = cost_s.max(1)[0]           
                cost_s = cost_s * (0 + s_prob_tensor)
            if cost_im is not None:
                cost_im = cost_im.max(0)[0]
                cost_im = cost_im * (0 + im_prob_tensor)

        if cost_s is None:
            cost_s = Variable(torch.zeros(1)).cuda()
        if cost_im is None:
            cost_im = Variable(torch.zeros(1)).cuda()

        if self.cost_style == 'sum':
            return cost_s.sum() + cost_im.sum()
        else:
            return cost_s.mean() + cost_im.mean()

class TripletLoss(nn.Module):
    """
    triplet ranking loss
    """

    def __init__(self, margin=0, measure=False, max_violation=False, cost_style='sum', direction='all'):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.cost_style = cost_style
        self.direction = direction
        if measure == 'order':
            self.sim = order_sim
        elif measure == 'euclidean':
            self.sim = euclidean_sim
        else:
            self.sim = cosine_sim

        self.max_violation = max_violation

    def forward(self, s, im):
        # compute image-sentence score matrix
        scores = self.sim(im, s)
        diagonal = scores.diag().view(im.size(0), 1)
        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)

        # clear diagonals
        mask = torch.eye(scores.size(0)) > .5
        I = Variable(mask)
        if torch.cuda.is_available():
            I = I.cuda()

        cost_s = None
        cost_im = None
        # compare every diagonal score to scores in its column
        if self.direction in  ['i2t', 'all']:
            # caption retrieval
            cost_s = (self.margin + scores - d1).clamp(min=0)
            cost_s = cost_s.masked_fill_(I, 0)
        # compare every diagonal score to scores in its row
        if self.direction in ['t2i', 'all']:
            # image retrieval
            cost_im = (self.margin + scores - d2).clamp(min=0)
            cost_im = cost_im.masked_fill_(I, 0)

        # keep the maximum violating negative for each query
        if self.max_violation:
            if cost_s is not None:
                cost_s = cost_s.max(1)[0]
            if cost_im is not None:
                cost_im = cost_im.max(0)[0]

        if cost_s is None:
            cost_s = Variable(torch.zeros(1)).cuda()
        if cost_im is None:
            cost_im = Variable(torch.zeros(1)).cuda()

        if self.cost_style == 'sum':
            return cost_s.sum() + cost_im.sum()
        else:
            return cost_s.mean() + cost_im.mean()

class MultidomainTripletLoss(nn.Module):
    """
    triplet ranking loss
    """

    def __init__(self, margin=0, measure=False, max_violation=False, cost_style='sum', direction='all'):
        super(MultidomainTripletLoss, self).__init__()
        self.margin = margin
        self.cost_style = cost_style
        self.direction = direction
        if measure == 'order':
            self.sim = order_sim
        elif measure == 'euclidean':
            self.sim = euclidean_sim
        else:
            self.sim = cosine_sim

        self.max_violation = max_violation

    def forward(self, s_s, t_s, s_im, t_im):
        # compute image-sentence score matrix

        # source N + target N
        s = torch.cat((s_s, t_s), dim=0)
        im = torch.cat((s_im, t_im), dim=0)
        scores = self.sim(im, s)
        diagonal = scores.diag().view(im.size(0), 1)
        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)

        # clear diagonals
        mask_for_i2t = torch.eye(scores.size(0)) 
        mask_for_t2i = torch.eye(scores.size(0)) 
        mask_for_i2t[s_s.size(0):,:] = 1
        mask_for_t2i[:,s_s.size(0):] = 1
        
        mask_for_i2t = mask_for_i2t > .5
        mask_for_t2i = mask_for_t2i > .5

        I_for_i2t = Variable(mask_for_i2t)
        if torch.cuda.is_available():
            I_for_i2t = I_for_i2t.cuda()

        I_for_t2i = Variable(mask_for_t2i)
        if torch.cuda.is_available():
            I_for_t2i = I_for_t2i.cuda()

        cost_s = None
        cost_im = None
        # compare every diagonal score to scores in its column
        if self.direction in  ['i2t', 'all']:
            # caption retrieval
            cost_s = (self.margin + scores - d1).clamp(min=0)
            cost_s = cost_s.masked_fill_(I_for_i2t, 0)
        # compare every diagonal score to scores in its row
        if self.direction in ['t2i', 'all']:
            # image retrieval
            cost_im = (self.margin + scores - d2).clamp(min=0)
            cost_im = cost_im.masked_fill_(I_for_t2i, 0)

        # keep the maximum violating negative for each query
        if self.max_violation:
            if cost_s is not None:
                cost_s_s = cost_s[:s_s.size(0), :s_s.size(0)].max(1)[0]
                msk_s_t = (cost_s[:s_s.size(0), s_s.size(0):]-cost_s_s) < 0
                cost_s_t = (cost_s[:s_s.size(0), s_s.size(0):]*msk_s_t.float()).max(1)[0]
                # print(msk_s_t.float().sum().cpu().data[0])
                if msk_s_t.float().sum().cpu().data[0] == s_s.size(0)*s_s.size(0):
                    lamda_s = 0.001
                else:
                    lamda_s = 0 
                cost_s = cost_s_s + lamda_s*cost_s_t
            if cost_im is not None:
                cost_im_s = cost_im[:s_s.size(0), :s_s.size(0)].max(0)[0]
                msk_im_t = (cost_im[s_s.size(0):, :s_s.size(0)]-cost_im_s) < 0
                cost_im_t = (cost_im[s_s.size(0):, :s_s.size(0)]*msk_im_t.float()).max(0)[0]
                # print(msk_im_t.float().sum().cpu().data[0])
                if msk_im_t.float().sum().cpu().data[0] == s_s.size(0)*s_s.size(0):
                    lamda_im = 0.001
                else:
                    lamda_im = 0
                cost_im = cost_im_s + lamda_im*cost_im_t
        if cost_s is None:
            cost_s = Variable(torch.zeros(1)).cuda()
        if cost_im is None:
            cost_im = Variable(torch.zeros(1)).cuda()

        if self.cost_style == 'sum':
            return cost_s.sum() + cost_im.sum()
        else:
            return cost_s.mean() + cost_im.mean()
