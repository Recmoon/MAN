import torch
import torch.nn as nn
import torch.nn.init
import torchvision.models as models
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.backends.cudnn as cudnn
from torch.nn.utils.clip_grad import clip_grad_norm  # clip_grad_norm_ for 0.4.0, clip_grad_norm for 0.3.1
import numpy as np
from collections import OrderedDict
import torch.nn.functional as F
from loss import WeightedTripletLoss, TripletLoss, MultidomainTripletLoss
from basic.bigfile import BigFile
from torch.autograd import Function

multi_source_flag = 0

class ReverseLayer(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None

class domain_classifier(nn.Module):

    def __init__(self):
        super(domain_classifier, self).__init__()

        self.domain_discriminator = nn.Sequential()
        self.domain_discriminator.add_module('d_fc1', nn.Linear(2048, 2048))
        self.domain_discriminator.add_module('d_relu1', nn.ReLU())
        self.domain_discriminator.add_module('d_fc2', nn.Linear(2048, 2))
    
    def forward(self, sinputs, tinputs, alpha):
        sinputs = ReverseLayer.apply(sinputs, alpha)
        sinputs = sinputs.view(sinputs.size(0), -1)
        tinputs = ReverseLayer.apply(tinputs, alpha)
        tinputs = tinputs.view(tinputs.size(0), -1)
        domain_s = self.domain_discriminator(sinputs)
        domain_t = self.domain_discriminator(tinputs)
        return domain_s, domain_t

def CORAL(source, target):
    d = source.size(1)
    ns, nt = source.size(0), target.size(0)

    # source covariance
    tmp_s = torch.matmul(Variable(torch.ones((1, ns))).cuda(), source)
    cs = (torch.matmul(source.t(), source) - (torch.matmul(tmp_s.t(), tmp_s)) / ns) / (ns - 1)

    # target covariance
    tmp_t = torch.matmul(Variable(torch.ones((1, nt))).cuda(), target)
    ct = (torch.matmul(target.t(), target) - (torch.matmul(tmp_t.t(), tmp_t)) / nt) / (nt - 1)

    # frobenius norm
    loss = (cs - ct).pow(2).sum().sqrt()
    loss = loss / (4 * d * d)

    return loss

def get_we_parameter(vocab, w2v_file):
    w2v_reader = BigFile(w2v_file)
    ndims = w2v_reader.ndims

    we = []
    # we.append([0]*ndims)
    for i in range(len(vocab)):
        try:
            vec = w2v_reader.read_one(vocab.idx2word[i])
        except:
            vec = np.random.uniform(-1, 1, ndims)
        we.append(vec)
    print('getting pre-trained parameter for word embedding initialization', np.shape(we)) 
    return np.array(we)

def l2norm(X):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=1, keepdim=True).sqrt()
    X = torch.div(X, norm)
    return X

def xavier_init_fc(fc):
    """Xavier initialization for the fully connected layer
    """
    r = np.sqrt(6.) / np.sqrt(fc.in_features +
                             fc.out_features)
    fc.weight.data.uniform_(-r, r)
    fc.bias.data.fill_(0)



class MFC(nn.Module):
    """
    Multi Fully Connected Layers
    """
    def __init__(self, fc_layers, dropout, have_dp=True, have_bn=False, have_last_bn=False):
        super(MFC, self).__init__()
        # fc layers
        self.n_fc = len(fc_layers)
        if self.n_fc > 1:
            if self.n_fc > 1:
                self.fc1 = nn.Linear(fc_layers[0], fc_layers[1])

            # dropout
            self.have_dp = have_dp
            if self.have_dp:
                self.dropout = nn.Dropout(p=dropout)

            # batch normalization
            self.have_bn = have_bn
            self.have_last_bn = have_last_bn
            if self.have_bn:
                if self.n_fc == 2 and self.have_last_bn:
                    self.bn_1 = nn.BatchNorm1d(fc_layers[1])

        self.init_weights()

    def init_weights(self):
        """Xavier initialization for the fully connected layer
        """
        if self.n_fc > 1:
            xavier_init_fc(self.fc1)

    def forward(self, inputs):

        if self.n_fc <= 1:
            features = inputs

        elif self.n_fc == 2:
            features = self.fc1(inputs)
            # batch noarmalization
            if self.have_bn and self.have_last_bn:
                features = self.bn_1(features)
            if self.have_dp:
                features = self.dropout(features)

        return features

class Video_multilevel_encoding(nn.Module):
    """
    Section 3.1. Video-side Multi-level Encoding
    """
    def __init__(self, opt):
        super(Video_multilevel_encoding, self).__init__()

        self.rnn_output_size = opt.visual_rnn_size*2
        self.dropout = nn.Dropout(p=opt.dropout)
        self.visual_norm = opt.visual_norm
        self.concate = opt.concate

        # visual bidirectional rnn encoder
        self.rnn = nn.GRU(opt.visual_feat_dim, opt.visual_rnn_size, batch_first=True, bidirectional=True)

        # visual 1-d convolutional network
        self.convs1 = nn.ModuleList([
            nn.Conv2d(1, opt.visual_kernel_num, (window_size, self.rnn_output_size), padding=(window_size - 1, 0)) 
            for window_size in opt.visual_kernel_sizes
            ])

        # visual mapping
        self.visual_mapping = MFC(opt.visual_mapping_layers, opt.dropout, have_bn=True, have_last_bn=True)


    def forward(self, videos):
        """Extract video feature vectors."""

        videos, videos_origin, lengths, vidoes_mask = videos
        
        # Level 1. Global Encoding by Mean Pooling According
        org_out = videos_origin

        # Level 2. Temporal-Aware Encoding by biGRU
        gru_init_out, _ = self.rnn(videos)
        mean_gru = Variable(torch.zeros(gru_init_out.size(0), self.rnn_output_size)).cuda()
        for i, batch in enumerate(gru_init_out):
            mean_gru[i] = torch.mean(batch[:lengths[i]], 0)
        gru_out = mean_gru
        gru_out = self.dropout(gru_out)

        # Level 3. Local-Enhanced Encoding by biGRU-CNN
        vidoes_mask = vidoes_mask.unsqueeze(2).expand(-1,-1,gru_init_out.size(2)) # (N,C,F1)
        gru_init_out = gru_init_out * vidoes_mask
        con_out = gru_init_out.unsqueeze(1)
        con_out = [F.relu(conv(con_out)).squeeze(3) for conv in self.convs1]
        con_out = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in con_out]
        con_out = torch.cat(con_out, 1)
        con_out = self.dropout(con_out)

        # concatenation
        if self.concate == 'full': # level 1+2+3
            features = torch.cat((gru_out,con_out,org_out), 1)
        elif self.concate == 'reduced':  # level 2+3
            features = torch.cat((gru_out,con_out), 1)

        # mapping to common space
        features = self.visual_mapping(features)
        if self.visual_norm:
            features = l2norm(features)

        return features

    def load_state_dict(self, state_dict):
        """Copies parameters. overwritting the default one to
        accept state_dict from Full model
        """
        own_state = self.state_dict()
        new_state = OrderedDict()
        for name, param in state_dict.items():
            if name in own_state:
                new_state[name] = param

        super(Video_multilevel_encoding, self).load_state_dict(new_state)



class Text_multilevel_encoding(nn.Module):
    """
    Section 3.2. Text-side Multi-level Encoding
    """
    def __init__(self, opt):
        super(Text_multilevel_encoding, self).__init__()
        self.text_norm = opt.text_norm
        self.word_dim = opt.word_dim
        self.we_parameter = opt.we_parameter
        self.rnn_output_size = opt.text_rnn_size*2
        self.dropout = nn.Dropout(p=opt.dropout)
        self.concate = opt.concate
        
        # visual bidirectional rnn encoder
        self.embed = nn.Embedding(opt.vocab_size, opt.word_dim)
        self.rnn = nn.GRU(opt.word_dim, opt.text_rnn_size, batch_first=True, bidirectional=True)

        # visual 1-d convolutional network
        self.convs1 = nn.ModuleList([
            nn.Conv2d(1, opt.text_kernel_num, (window_size, self.rnn_output_size), padding=(window_size - 1, 0)) 
            for window_size in opt.text_kernel_sizes
            ])
        
        # multi fc layers
        self.text_mapping = MFC(opt.text_mapping_layers, opt.dropout, have_bn=True, have_last_bn=True)

        self.init_weights()

    def init_weights(self):
        if self.word_dim == 500 and self.we_parameter is not None:
            self.embed.weight.data.copy_(torch.from_numpy(self.we_parameter))
        else:
            self.embed.weight.data.uniform_(-0.1, 0.1)


    def forward(self, text, *args):
        # Embed word ids to vectors
        # cap_wids, cap_w2vs, cap_bows, cap_mask = x
        cap_wids, cap_bows, lengths, cap_mask = text
        

        # Level 1. Global Encoding by Mean Pooling According
        org_out = cap_bows

        # Level 2. Temporal-Aware Encoding by biGRU
        cap_wids = self.embed(cap_wids)
        packed = pack_padded_sequence(cap_wids, lengths, batch_first=True)
        gru_init_out, _ = self.rnn(packed)
        # Reshape *final* output to (batch_size, hidden_size)
        padded = pad_packed_sequence(gru_init_out, batch_first=True)
        gru_init_out = padded[0]
        gru_out = Variable(torch.zeros(padded[0].size(0), self.rnn_output_size)).cuda()
        for i, batch in enumerate(padded[0]):
            gru_out[i] = torch.mean(batch[:lengths[i]], 0)
        gru_out = self.dropout(gru_out)

        # Level 3. Local-Enhanced Encoding by biGRU-CNN
        con_out = gru_init_out.unsqueeze(1)
        con_out = [F.relu(conv(con_out)).squeeze(3) for conv in self.convs1]
        con_out = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in con_out]
        con_out = torch.cat(con_out, 1)
        con_out = self.dropout(con_out)

        # concatenation
        if self.concate == 'full': # level 1+2+3
            features = torch.cat((gru_out,con_out,org_out), 1)
        elif self.concate == 'reduced': # level 2+3
            features = torch.cat((gru_out,con_out), 1)
        
        # mapping to common space
        features = self.text_mapping(features)
        if self.text_norm:
            features = l2norm(features)

        return features




class BaseModel(object):

    def state_dict(self):
        state_dict = [self.vid_encoding.state_dict(), self.text_encoding.state_dict(), self.video_domain_classifier.state_dict(), self.text_domain_classifier.state_dict(), self.video_domain_classifier2.state_dict(), self.text_domain_classifier2.state_dict(), self.s2_vt_domain_classifier.state_dict()]
        return state_dict

    def load_state_dict(self, state_dict):
        self.vid_encoding.load_state_dict(state_dict[0])
        self.text_encoding.load_state_dict(state_dict[1])
        # self.domain_classifier.load_state_dict(state_dict[2])

    def train_start(self):
        """switch to train mode
        """
        self.vid_encoding.train()
        self.text_encoding.train()
        self.video_domain_classifier.train()
        self.text_domain_classifier.train()
        self.s_vt_domain_classifier.train()
        self.t_vt_domain_classifier.train()
        self.video_domain_classifier2.train()
        self.text_domain_classifier2.train()
        self.s2_vt_domain_classifier.train()

    def val_start(self):
        """switch to evaluate mode
        """
        self.vid_encoding.eval()
        self.text_encoding.eval()
        self.video_domain_classifier.eval()
        self.text_domain_classifier.eval()
        self.s_vt_domain_classifier.eval()
        self.t_vt_domain_classifier.eval()
        self.video_domain_classifier2.eval()
        self.text_domain_classifier2.eval()
        self.s2_vt_domain_classifier.eval()

#    def forward_loss_ms(self, cap_emb, vid_emb, vid_emb_target, cap_emb_target, vid_emb_source, cap_emb_source, *agrs, **kwargs):
#        """Compute the loss given pairs of video and caption embeddings
#        """
#        domain_s_v, domain_t_v = self.video_domain_classifier(vid_emb_source, vid_emb_target, 0.5)
#        domain_s_t, domain_t_t = self.text_domain_classifier(cap_emb_source, cap_emb_target, 0.5)
#        cross_domain_s_v, cross_domain_s_t = self.s_vt_domain_classifier(vid_emb_source, cap_emb_source, 0.5)
#        cross_domain_t_v, cross_domain_t_t = self.t_vt_domain_classifier(vid_emb_target, cap_emb_target, 0.5)
#        slabels = Variable(torch.ones(vid_emb_source.size(0)).type(torch.LongTensor), volatile=True).cuda()
#        tlabels = Variable(torch.zeros(vid_emb_target.size(0)).type(torch.LongTensor), volatile=True).cuda()
#
#        domain_loss = F.cross_entropy(domain_s_v, slabels) + F.cross_entropy(domain_t_v, tlabels) + \
#                      F.cross_entropy(domain_s_t, slabels) + F.cross_entropy(domain_t_t, tlabels) + \
#                      F.cross_entropy(cross_domain_s_v, slabels) + F.cross_entropy(cross_domain_s_t, slabels) + \
#                      F.cross_entropy(cross_domain_t_v, tlabels) + F.cross_entropy(cross_domain_t_t, tlabels)
#
#        # domain_loss = F.cross_entropy(domain_s_v, slabels) + F.cross_entropy(domain_t_v, tlabels) + \
#        #               F.cross_entropy(domain_s_t, slabels) + F.cross_entropy(domain_t_t, tlabels)
#
#        # domain_loss = F.cross_entropy(cross_domain_s_v, slabels) + F.cross_entropy(cross_domain_s_t, slabels) + \
#        #               F.cross_entropy(cross_domain_t_v, tlabels) + F.cross_entropy(cross_domain_t_t, tlabels)
#        # print(type(vid_emb_target))
#        retrieval_loss = self.criterion(cap_emb, vid_emb)
#        # retrieval_loss = self.criterion(cap_emb, cap_emb_target, vid_emb, vid_emb_target)
#        if vid_emb_target is not None:
#            # adapt_loss = CORAL(vid_emb, vid_emb_target) + CORAL(cap_emb, cap_emb_target)
#            # print(retrieval_loss, adapt_loss)
#            loss = retrieval_loss + 0.01*domain_loss
#            # print(retrieval_loss, domain_loss)
#        else:
#            loss = retrieval_loss
#        if torch.__version__ == '0.3.1':  # loss.item() for 0.4.0, loss.data[0] for 0.3.1
#            self.logger.update('Le', loss.data[0], vid_emb.size(0)) 
#        else:
#            self.logger.update('Le', loss.item(), vid_emb.size(0)) 
#        return loss

    def forward_loss(self, cap_emb, vid_emb, vid_emb_target=None, cap_emb_target=None, vid_emb_source2=None, cap_emb_source2=None, *agrs, **kwargs):
        """Compute the loss given pairs of video and caption embeddings
        """

        if vid_emb_target is not None:

            domain_s_v, domain_t_v = self.video_domain_classifier(vid_emb, vid_emb_target, 0.5)
            domain_s_t, domain_t_t = self.text_domain_classifier(cap_emb, cap_emb_target, 0.5)
            cross_domain_s_v, cross_domain_s_t = self.s_vt_domain_classifier(vid_emb, cap_emb, 0.5)
            cross_domain_t_v, cross_domain_t_t = self.t_vt_domain_classifier(vid_emb_target, cap_emb_target, 0.5)
            slabels = Variable(torch.ones(vid_emb.size(0)).type(torch.LongTensor), volatile=True).cuda()
            tlabels = Variable(torch.zeros(vid_emb.size(0)).type(torch.LongTensor), volatile=True).cuda()
   
            domain_loss = self.domain_weight * (F.cross_entropy(domain_s_v, slabels) + F.cross_entropy(domain_t_v, tlabels) + \
                          F.cross_entropy(domain_s_t, slabels) + F.cross_entropy(domain_t_t, tlabels)) + \
                          self.modality_weight * (F.cross_entropy(cross_domain_s_v, slabels) + F.cross_entropy(cross_domain_s_t, tlabels) + \
                          F.cross_entropy(cross_domain_t_v, slabels) + F.cross_entropy(cross_domain_t_t, tlabels))

            if vid_emb_source2 is not None:
                domain_s_v, domain_t_v = self.video_domain_classifier2(vid_emb_source2, vid_emb_target, 0.5)
                domain_s_t, domain_t_t = self.text_domain_classifier2(cap_emb_source2, cap_emb_target, 0.5)
                cross_domain_s_v, cross_domain_s_t = self.s2_vt_domain_classifier(vid_emb_source2, cap_emb_source2, 0.5)

                slabels = Variable(torch.ones(vid_emb.size(0)).type(torch.LongTensor), volatile=True).cuda()
                tlabels = Variable(torch.zeros(vid_emb.size(0)).type(torch.LongTensor), volatile=True).cuda()

                domain_loss += self.domain_weight * (F.cross_entropy(domain_s_v, slabels) + F.cross_entropy(domain_t_v, tlabels) + \
                               F.cross_entropy(domain_s_t, slabels) + F.cross_entropy(domain_t_t, tlabels)) + \
                               self.modality_weight * (F.cross_entropy(cross_domain_s_v, slabels) + F.cross_entropy(cross_domain_s_t, tlabels))

        retrieval_loss = self.criterion(cap_emb, vid_emb)
        if vid_emb_source2 is not None:
            retrieval_loss += self.criterion(cap_emb_source2, vid_emb_source2)

        if vid_emb_target is not None:
            loss = retrieval_loss + domain_loss
            # print(retrieval_loss, domain_loss)
        else:
            loss = retrieval_loss
        if torch.__version__ == '0.3.1':  # loss.item() for 0.4.0, loss.data[0] for 0.3.1
            self.logger.update('Le', loss.data[0], vid_emb.size(0)) 
        else:
            self.logger.update('Le', loss.item(), vid_emb.size(0)) 
        return loss

    def train_emb_ada_multi(self, videos, captions, lengths, cap_ids, video_ids, video_ids_target, video_data_target, captions_target, video_ids_source2, video_data_source2, captions_source2, *args):
        """One training step given videos and captions.
        """
        self.Eiters += 1
        self.logger.update('Eit', self.Eiters)
        self.logger.update('lr', self.optimizer.param_groups[0]['lr'])

#        print("multi branch")
        
        # compute the embeddings
        vid_emb, vid_emb_target, vid_emb_source2, cap_emb, cap_emb_target, cap_emb_source2 = self.forward_emb(videos, captions, cap_ids, video_data_target, captions_target, video_data_source2, captions_source2, False)
        # measure accuracy and record loss
        self.optimizer.zero_grad()
        loss = self.forward_loss(cap_emb, vid_emb, vid_emb_target, cap_emb_target, vid_emb_source2, cap_emb_source2)

        if torch.__version__ == '0.3.1':
            loss_value = loss.data[0]
        else:
            loss_value = loss.item()

        # compute gradient and do SGD step
        loss.backward()
        if self.grad_clip > 0:
            clip_grad_norm(self.params, self.grad_clip)
        self.optimizer.step()

        return vid_emb.size(0), loss_value


    def train_emb_ada(self, videos, captions, lengths, cap_ids, video_ids, video_ids_target, video_data_target, captions_target, *args):
        """One training step given videos and captions.
        """
        self.Eiters += 1
        self.logger.update('Eit', self.Eiters)
        self.logger.update('lr', self.optimizer.param_groups[0]['lr'])

#        print("single branch")        

        # compute the embeddings
        vid_emb, vid_emb_target, cap_emb, cap_emb_target = self.forward_emb(videos, captions, cap_ids, video_data_target, captions_target)
        # measure accuracy and record loss
        self.optimizer.zero_grad()
    
        loss = self.forward_loss(cap_emb, vid_emb, vid_emb_target, cap_emb_target)
        
        if torch.__version__ == '0.3.1':
            loss_value = loss.data[0]
        else:
            loss_value = loss.item()

        # compute gradient and do SGD step
        loss.backward()
        if self.grad_clip > 0:
            clip_grad_norm(self.params, self.grad_clip)
        self.optimizer.step()

        return vid_emb.size(0), loss_value

    def train_emb(self, videos, captions, lengths, *args):
        """One training step given videos and captions.
        """
        self.Eiters += 1
        self.logger.update('Eit', self.Eiters)
        self.logger.update('lr', self.optimizer.param_groups[0]['lr'])

        # compute the embeddings
        vid_emb, cap_emb = self.forward_emb(videos, captions, volatile=False)

        # measure accuracy and record loss
        self.optimizer.zero_grad()
        loss = self.forward_loss(cap_emb, vid_emb)
        
        if torch.__version__ == '0.3.1':
            loss_value = loss.data[0]
        else:
            loss_value = loss.item()

        # compute gradient and do SGD step
        loss.backward()
        if self.grad_clip > 0:
            clip_grad_norm(self.params, self.grad_clip)
        self.optimizer.step()

        return vid_emb.size(0), loss_value



class Dual_Encoding(BaseModel):
    """
    dual encoding network
    """
    def __init__(self, opt):
        # Build Models
        self.grad_clip = opt.grad_clip
        self.vid_encoding = Video_multilevel_encoding(opt)
        self.text_encoding = Text_multilevel_encoding(opt)
        self.video_domain_classifier = domain_classifier()
        self.text_domain_classifier = domain_classifier()
        self.video_domain_classifier2 = domain_classifier()
        self.text_domain_classifier2 = domain_classifier()
        self.s_vt_domain_classifier = domain_classifier()
        self.t_vt_domain_classifier = domain_classifier()
        self.s2_vt_domain_classifier = domain_classifier()
        print(self.vid_encoding)
        print(self.text_encoding)
        if torch.cuda.is_available():
            self.vid_encoding.cuda()
            self.text_encoding.cuda()
            self.video_domain_classifier.cuda()
            self.s_vt_domain_classifier.cuda()
            self.text_domain_classifier.cuda()
            self.t_vt_domain_classifier.cuda()
            self.video_domain_classifier2.cuda()
            self.s2_vt_domain_classifier.cuda()
            self.text_domain_classifier2.cuda()
            cudnn.benchmark = True

        # Loss and Optimizer
        if opt.loss_fun == 'sample_weight_mrl':
            self.criterion = WeightedTripletLoss(margin=opt.margin,
                                                 measure=opt.measure,
                                                 max_violation=opt.max_violation,
                                                 cost_style=opt.cost_style,
                                                 direction=opt.direction)

        if opt.loss_fun == 'mrl':
            self.criterion = TripletLoss(margin=opt.margin,
                                            measure=opt.measure,
                                            max_violation=opt.max_violation,
                                            cost_style=opt.cost_style,
                                            direction=opt.direction)

        params = list(self.text_encoding.parameters())
        params += list(self.vid_encoding.parameters())
        params += list(self.video_domain_classifier.parameters())
        params += list(self.text_domain_classifier.parameters())
        params += list(self.s_vt_domain_classifier.parameters())
        params += list(self.t_vt_domain_classifier.parameters())
        params += list(self.video_domain_classifier2.parameters())
        params += list(self.text_domain_classifier2.parameters())
        params += list(self.s2_vt_domain_classifier.parameters())
        self.params = params

        if opt.optimizer == 'adam':
            self.optimizer = torch.optim.Adam(params, lr=opt.learning_rate)
        elif opt.optimizer == 'rmsprop':
            self.optimizer = torch.optim.RMSprop(params, lr=opt.learning_rate)

        self.Eiters = 0
        self.domain_weight = opt.domain_weight
        self.modality_weight = opt.modality_weight

    def forward_emb(self, videos, targets, cap_ids=None, video_target=None, targets_target=None, video_source2=None, targets_source2=None, volatile=False, *args):
        """Compute the video and caption embeddings
        """
        # video data
        videos_data = (Variable(videos[0], volatile=volatile).cuda()
        , Variable(videos[1], volatile=volatile).cuda()
        , videos[2], Variable(videos[3], volatile=volatile).cuda())

        text_data = (Variable(targets[0], volatile=volatile).cuda(), 
        Variable(targets[1], volatile=volatile).cuda(), 
        targets[2], 
        Variable(targets[3], volatile=volatile).cuda())

        if video_target != None:

            reorder_idex = sorted(range(len(targets_target[2])), key=lambda k: targets_target[2][k], reverse=True)
            targets_target[2].sort(reverse=True)
            reorder_idex = torch.LongTensor(reorder_idex)
            text_data_target = (Variable(torch.index_select(targets_target[0],0,reorder_idex), volatile=volatile).cuda(), 
            Variable(torch.index_select(targets_target[1],0,reorder_idex), volatile=volatile).cuda(), 
            targets_target[2], 
            Variable(torch.index_select(targets_target[3],0,reorder_idex), volatile=volatile).cuda())

            video_target[2].sort(reverse=True)
            videos_data_target = (Variable(torch.index_select(video_target[0],0,reorder_idex), volatile=volatile).cuda()
            , Variable(torch.index_select(video_target[1],0,reorder_idex), volatile=volatile).cuda()
            , video_target[2], 
            Variable(torch.index_select(video_target[3],0,reorder_idex), volatile=volatile).cuda())

        if video_source2 != None:
            
            reorder_idex = sorted(range(len(targets_source2[2])), key=lambda k: targets_source2[2][k], reverse=True)
            targets_source2[2].sort(reverse=True)
            reorder_idex = torch.LongTensor(reorder_idex)
            text_data_source2 = (Variable(torch.index_select(targets_source2[0],0,reorder_idex), volatile=volatile).cuda(),     
            Variable(torch.index_select(targets_source2[1],0,reorder_idex), volatile=volatile).cuda(),     
            targets_source2[2],     
            Variable(torch.index_select(targets_source2[3],0,reorder_idex), volatile=volatile).cuda())

            video_source2[2].sort(reverse=True)
            videos_data_source2 = (Variable(torch.index_select(video_source2[0],0,reorder_idex), volatile=volatile).cuda()
            , Variable(torch.index_select(video_source2[1],0,reorder_idex), volatile=volatile).cuda()
            , video_source2[2],     
            Variable(torch.index_select(video_source2[3],0,reorder_idex), volatile=volatile).cuda())

        vid_emb = self.vid_encoding(videos_data)
        cap_emb = self.text_encoding(text_data)

        if video_target != None:
#            if multi_source_flag == 1:
#                print('debug')
#                source_vid_emb = self.vid_encoding(source_videos_data)
#                print('debug1: ok')
#                target_vid_emb = self.vid_encoding(target_videos_data)
#                print('debug2: ok')
#                source_cap_emb = self.text_encoding(source_text_data)
#                target_cap_emb = self.text_encoding(target_text_data)
#                return vid_emb, target_vid_emb, cap_emb, target_cap_emb, source_vid_emb, source_cap_emb
#            else:
            vid_emb_target = self.vid_encoding(videos_data_target)
            cap_emb_target = self.text_encoding(text_data_target)

            if video_source2 != None:
                vid_emb_source2 = self.vid_encoding(videos_data_source2)
                cap_emb_source2 = self.text_encoding(text_data_source2)
                return vid_emb, vid_emb_target, vid_emb_source2, cap_emb, cap_emb_target, cap_emb_source2

#            source_idx_raw=target_idx_raw=1
#            return vid_emb, vid_emb_target, cap_emb, cap_emb_target, source_idx_raw, target_idx_raw
            return vid_emb, vid_emb_target, cap_emb, cap_emb_target


        return vid_emb, cap_emb

    def embed_vis(self, vis_data, volatile=True):
        # video data
        frames, mean_origin, video_lengths, vidoes_mask = vis_data
        frames = Variable(frames, volatile=volatile)
        if torch.cuda.is_available():
            frames = frames.cuda()

        mean_origin = Variable(mean_origin, volatile=volatile)
        if torch.cuda.is_available():
            mean_origin = mean_origin.cuda()

        vidoes_mask = Variable(vidoes_mask, volatile=volatile)
        if torch.cuda.is_available():
            vidoes_mask = vidoes_mask.cuda()
        vis_data = (frames, mean_origin, video_lengths, vidoes_mask)

        return self.vid_encoding(vis_data)


    def embed_txt(self, txt_data, volatile=True):
        # text data
        captions, cap_bows, lengths, cap_masks = txt_data
        if captions is not None:
            captions = Variable(captions, volatile=volatile)
            if torch.cuda.is_available():
                captions = captions.cuda()

        if cap_bows is not None:
            cap_bows = Variable(cap_bows, volatile=volatile)
            if torch.cuda.is_available():
                cap_bows = cap_bows.cuda()

        if cap_masks is not None:
            cap_masks = Variable(cap_masks, volatile=volatile)
            if torch.cuda.is_available():
                cap_masks = cap_masks.cuda()
        txt_data = (captions, cap_bows, lengths, cap_masks)

        return self.text_encoding(txt_data)



NAME_TO_MODELS = {'dual_encoding': Dual_Encoding}

def get_model(name):
    assert name in NAME_TO_MODELS, '%s not supported.'%name
    return NAME_TO_MODELS[name]
