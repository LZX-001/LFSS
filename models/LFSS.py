import copy

import torch
from torch import nn
import torch.nn.functional as F

from models.Accuracy import clustering, best_match, test_torch_times
from models.util import adjust_learning_rate
from models.util import nt_xent, nt_xent_self, get_embedding_for_test
from network import backbone_dict
from utils.grad_scaler import NativeScalerWithGradNormCount
from torch.nn.functional import cosine_similarity

class LFSS(nn.Module):

    def __init__(self,opt):
        nn.Module.__init__(self)
        encoder_type, dim_in = backbone_dict[opt.encoder_name]
        encoder = encoder_type()
        self.m = opt.momentum_base
        self.shuffling_bn = opt.shuffling_bn
        self.num_cluster = opt.num_cluster
        self.temperature = opt.temperature
        self.fea_dim = opt.fea_dim
        self.sigma = opt.sigma
        self.lamb_da = opt.lamb_da

        # create the encoders
        self.encoder_q = encoder
        self.projector_q = nn.Sequential(
            nn.Linear(dim_in, opt.hidden_size),
            nn.BatchNorm1d(opt.hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(opt.hidden_size, opt.fea_dim)
        )
        self.encoder_k = copy.deepcopy(self.encoder_q)
        self.projector_k = copy.deepcopy(self.projector_q)

        self.predictor = nn.Sequential(nn.Linear(opt.fea_dim, opt.hidden_size),
                                       nn.BatchNorm1d(opt.hidden_size),
                                       nn.ReLU(inplace=True),
                                       nn.Linear(opt.hidden_size, opt.fea_dim)
                                       )
        self.q_params = list(self.encoder_q.parameters()) + list(self.projector_q.parameters())
        self.k_params = list(self.encoder_k.parameters()) + list(self.projector_k.parameters())

        for param_q, param_k in zip(self.q_params, self.k_params):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        for m in self.predictor.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                if hasattr(m.bias, 'data'):
                    m.bias.data.fill_(0)
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d,
                                nn.GroupNorm, nn.SyncBatchNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        self.encoder = nn.Sequential(self.encoder_k, self.projector_k)

        self.pre_encoder = copy.deepcopy(self.encoder)
        for param_o in list(self.pre_encoder.parameters()):
            param_o.requires_grad = False
        if opt.syncbn:
            if opt.shuffling_bn:
                self.encoder_q = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.encoder_q)
                self.projector_q = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.projector_q)
                self.predictor = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.predictor)
        self.feature_extractor_copy = copy.deepcopy(self.encoder).cuda()

        self.scaler = NativeScalerWithGradNormCount(amp=opt.amp)

        self.pseudo_labels = None
        self.num_cluster = opt.num_cluster

    def get_old_embedding(self,im_q,im_k):
        q = self.pre_encoder(im_q)
        k = self.pre_encoder(im_k)

        q = F.normalize(q, dim=1)
        k = F.normalize(k, dim=1)
        return q, k
    def forward_noise_loss(self,q, k):

        noise_q = q + torch.randn_like(q) * self.sigma
        contrastive_loss = (2 - 2 * F.cosine_similarity(self.predictor(noise_q), k)).mean()
        # contrastive_loss = (-2 * F.cosine_similarity(self.predictor(noise_q), k)).mean()

        return contrastive_loss
    def forward_nt_loss(self, old, new):
        nt_xent_loss = nt_xent_self(new, 0.5, old)
        return nt_xent_loss
        # return self.forward_noise_loss(new,old)
    def forward_prt_loss(self, q, k, pseudo_labels):
        valid_index = torch.where(pseudo_labels!=-1)[0]
        n_samples = len(pseudo_labels)
        # print(n_samples)
        # print(pseudo_labels)
        weight = torch.zeros(self.num_cluster, n_samples).to('cuda')
        samples_index = torch.arange(n_samples).cuda()
        weight[pseudo_labels[valid_index].to(torch.long), samples_index[valid_index].to(torch.long)] = 1
        non_zero_mask = weight.any(dim=1)
        weight = weight[non_zero_mask]
        weight = F.normalize(weight, p=1, dim=1)  # l1 normalization
        q_centers = torch.mm(weight, q)
        q_centers = F.normalize(q_centers, dim=1)
        k_centers = torch.mm(weight, k)
        k_centers = F.normalize(k_centers, dim=1)
        loss = nt_xent(q_centers,0.5,k_centers)
        return loss
    def get_embedding(self, inputs, mode = 'k'):
        if mode == 'k':
            feature = self.encoder_k(inputs)
            feature = self.projector_k(feature)
        elif mode == 'q':
            feature = self.encoder_q(inputs)
            feature = self.projector_q(feature)
        elif mode == 'p':
            feature = self.encoder_q(inputs)
            feature = self.projector_q(feature)
            feature = self.predictor(feature)
        elif mode == 'q+p':
            feature = self.encoder_q(inputs)
            feature1 = self.projector_q(feature)
            feature2 = self.predictor(feature1)
            return feature1, feature2
        return feature
    def forward_nt(self, inputs1, inputs2, momentum_update=True):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets
        """

        q_1, p_1 = self.get_embedding(inputs1, 'q+p')
        with torch.no_grad():
            k_2 = self.get_embedding(inputs2, 'k')
        loss_noise1 = self.forward_noise_loss(q_1, k_2)
        q_2, p_2 = self.get_embedding(inputs2, 'q+p')
        with torch.no_grad():
            k_1 = self.get_embedding(inputs1, 'k')
        loss_noise2 = self.forward_noise_loss(q_2, k_1)
        loss_noise = (loss_noise2 + loss_noise1) * 0.5
        with torch.no_grad():
            old_1, old_2 = self.get_old_embedding(inputs1, inputs2)
        # 1
        loss_nt1 = self.forward_nt_loss(old_1, q_1)
        loss_nt2 = self.forward_nt_loss(old_2, q_2)
        loss_nt = (loss_nt2 + loss_nt1) * 0.5
        loss = loss_noise + loss_nt * self.lamb_da
        return loss
    def forward_prt(self, inputs1, inputs2, momentum_update = True, idx = None):

        q_1, p_1 = self.get_embedding(inputs1, 'q+p')
        with torch.no_grad():
            k_2 = self.get_embedding(inputs2, 'k')
        loss_noise1 = self.forward_noise_loss(q_1, k_2)
        q_2, p_2 = self.get_embedding(inputs2, 'q+p')
        with torch.no_grad():
            k_1 = self.get_embedding(inputs1, 'k')
        loss_noise2 = self.forward_noise_loss(q_2, k_1)
        loss_noise = (loss_noise2 + loss_noise1) * 0.5
        pseudo_labels = self.pseudo_labels[idx]
        loss_prt1 = self.forward_prt_loss(q_1, k_2, pseudo_labels)
        loss_prt2 = self.forward_prt_loss(q_2, k_1, pseudo_labels)
        loss_prt = (loss_prt2 + loss_prt1) * 0.5
        # print(loss_noise,loss_prt)
        loss = loss_noise + loss_prt * self.lamb_da
        return loss
    def forward_noise_only(self, inputs1, inputs2, momentum_update = True):

        q_1, p_1 = self.get_embedding(inputs1, 'q+p')
        with torch.no_grad():
            k_2 = self.get_embedding(inputs2, 'k')
        loss_noise1 = self.forward_noise_loss(q_1, k_2)
        q_2, p_2 = self.get_embedding(inputs2, 'q+p')
        with torch.no_grad():
            k_1 = self.get_embedding(inputs1, 'k')
        loss_noise2 = self.forward_noise_loss(q_2, k_1)
        loss_noise = (loss_noise2 + loss_noise1) * 0.5
        return loss_noise
    def forward_nt_prt(self,inputs1, inputs2, momentum_update = True, idx = None):

        q_1, p_1 = self.get_embedding(inputs1, 'q+p')
        with torch.no_grad():
            k_2 = self.get_embedding(inputs2, 'k')
        loss_noise1 = self.forward_noise_loss(q_1, k_2)
        q_2, p_2 = self.get_embedding(inputs2, 'q+p')
        with torch.no_grad():
            k_1 = self.get_embedding(inputs1, 'k')
        loss_noise2 = self.forward_noise_loss(q_2, k_1)
        loss_noise = (loss_noise2 + loss_noise1) * 0.5
        with torch.no_grad():
            old_1, old_2 = self.get_old_embedding(inputs1, inputs2)
        loss_nt1 = self.forward_nt_loss(old_1, q_1)
        loss_nt2 = self.forward_nt_loss(old_2, q_2)
        loss_nt = (loss_nt2 + loss_nt1) * 0.5
        pseudo_labels = self.pseudo_labels[idx]
        loss_prt1 = self.forward_prt_loss(q_1, k_2, pseudo_labels)
        loss_prt2 = self.forward_prt_loss(q_2, k_1, pseudo_labels)
        loss_prt = (loss_prt2 + loss_prt1) * 0.5
        # print(loss_noise,loss_prt,loss_nt)
        loss = loss_noise + loss_prt * self.lamb_da + loss_nt * self.lamb_da
        return loss
    def forward(self,inputs1, inputs2, epoch, eta, momentum_update=True, idx = None):
        if epoch <= eta:
            return self.forward_nt(inputs1, inputs2, momentum_update)
        else:
            return self.forward_nt_prt(inputs1, inputs2, momentum_update,idx)

    def get_pseudo_labels(self,data_loader,opt):
        features = torch.zeros([len(data_loader.dataset), opt.fea_dim]).cuda()
        old_features = torch.zeros([len(data_loader.dataset), opt.fea_dim]).cuda()
        for i, (inputs, target, idx) in enumerate(data_loader):
            with torch.no_grad():
                inputs_1, inputs_2, inputs_3 = inputs
                inputs_3 = inputs_3.cuda(non_blocking=True)

                feature = self.encoder_k(inputs_3)
                feature = self.projector_k(feature)
                features[idx] = feature
                old_feature = self.pre_encoder(inputs_3)
                old_features[idx] = old_feature
        y_pred, _ = clustering(features, opt.num_cluster)
        old_y_pred, _ = clustering(old_features, opt.num_cluster)
        y_pred = best_match(old_y_pred.cpu().numpy(), y_pred.cpu().numpy())
        y_pred = torch.from_numpy(y_pred).cuda()
        feature1 = torch.nn.functional.normalize(old_features, dim=-1)
        feature2 = torch.nn.functional.normalize(features, dim=-1)
        s = cosine_similarity(feature1, feature2, dim=1)
        sample_number = len(s)

        threshold, _ = torch.kthvalue(s, int(opt.delta * sample_number))
        change_idx = torch.where(s < threshold)[0]

        y_pred[change_idx] = -1
        print(len(y_pred), len(change_idx), len(change_idx) / len(y_pred))
        self.pseudo_labels = y_pred
        return y_pred

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.q_params, self.k_params):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)


def train_LFSS(opt, model, optimizer, train_loader, epoch, log):
    n_iter = opt.num_batch * (epoch - 1) + 1
    if epoch > opt.eta and (((epoch - 1) % opt.prototype_freq == 0) or model.pseudo_labels == None):
        model.get_pseudo_labels(train_loader, opt)

    for i, (inputs, target, idx) in enumerate(train_loader):
        inputs_1, inputs_2, inputs_3 = inputs
        inputs_1 = inputs_1.cuda(non_blocking=True)
        inputs_2 = inputs_2.cuda(non_blocking=True)
        inputs_3 = inputs_3.cuda(non_blocking=True)

        model.train()
        lr = adjust_learning_rate(opt, model, optimizer, n_iter)
        update_params = (n_iter % opt.acc_grd_step == 0)
        with torch.autocast('cuda', enabled=opt.amp):
            loss = model(inputs_1, inputs_2, epoch, opt.eta, update_params, idx)
        loss = loss / opt.acc_grd_step
        # model.pre_encoder = copy.deepcopy(model.encoder)
        model.scaler(loss, optimizer=optimizer, update_grad=update_params)
        with torch.no_grad():
            model._momentum_update_key_encoder()
        if i == 0:
            log.info('Epoch {} loss: {} lr: {}'.format(epoch, loss, lr))
        n_iter += 1
    model.pre_encoder = copy.deepcopy(model.encoder)

def test_LFSS(model,data_loader,class_num, mode='k'):
    features, labels = get_embedding_for_test(model,data_loader,mode)
    test_torch_times(features, labels, 10, class_num)




