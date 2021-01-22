import sys
import numpy
import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from runs.layers import build_stem, build_classifier
from torch.nn.init import kaiming_normal, kaiming_uniform, xavier_uniform, xavier_normal, constant, uniform
from functools import partial
from runs.data_attribute_random import category_idx, color_idx, brightness_idx, size_idx

# for one object we do not need to have tau and alpha as tensors
# of the original shape

QUESTIONS = [0, 1, 2, 3]
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def _single_tau():
    tau_0 = torch.zeros(1, 2)
    tau_1 = torch.zeros(1, 2)
    tau_0[0][1] = tau_1[0][0] = 1
    return tau_0, tau_1


def _random_tau(num_modules):
    tau_0 = torch.zeros(num_modules, num_modules+1)
    tau_1 = torch.zeros(num_modules, num_modules+1)
    xavier_uniform(tau_0)
    xavier_uniform(tau_1)
    return tau_0, tau_1


def _chain_tau():
    tau_0 = torch.zeros(3, 4)
    tau_1 = torch.zeros(3, 4)
    tau_0[0][1] = tau_1[0][0] = 100  # 1st block - lhs inp img, rhs inp sentinel
    tau_0[1][2] = tau_1[1][0] = 100  # 2nd block - lhs inp 1st block, rhs inp sentinel
    tau_0[2][3] = tau_1[2][0] = 100  # 3rd block - lhs inp 2nd block, rhs inp sentinel
    return tau_0, tau_1


def _chain_with_shortcuts_tau():
    tau_0 = torch.zeros(3, 4)
    tau_1 = torch.zeros(3, 4)
    tau_0[0][1] = tau_1[0][0] = 100  # 1st block - lhs inp img, rhs inp sentinel
    tau_0[1][2] = tau_1[1][1] = 100  # 2nd block - lhs inp 1st block, rhs img
    tau_0[2][3] = tau_1[2][1] = 100  # 3rd block - lhs inp 2nd block, rhs img
    return tau_0, tau_1


def _chain_with_shortcuts_tau_flipped():
    tau_0 = torch.zeros(3, 4)
    tau_1 = torch.zeros(3, 4)
    tau_0[0][1] = tau_1[0][0] = 100  # 1st block - lhs inp img, rhs inp sentinel
    tau_0[1][1] = tau_1[1][2] = 100  # 2nd block - lhs inp img, rhs 1st block
    tau_0[2][1] = tau_1[2][3] = 100  # 3rd block - lhs inp img, rhs 2nd block
    return tau_0, tau_1


def _tree_tau():
    tau_0 = torch.zeros(3, 4)
    tau_1 = torch.zeros(3, 4)
    tau_0[0][1] = tau_1[0][0] = 100  # 1st block - lhs inp img, rhs inp sentinel
    tau_0[1][1] = tau_1[1][0] = 100  # 2st block - lhs inp img, rhs inp sentinel
    tau_0[2][2] = tau_1[2][3] = 100  # 3rd block - lhs inp 1st block, rhs inp 2nd block
    return tau_0, tau_1


def correct_alpha_init_xyr(alpha):
    alpha.zero_()
    alpha[0][0] = 100
    alpha[1][2] = 100
    alpha[2][1] = 100

    return alpha


def correct_alpha_init_rxy(alpha, use_stopwords=True):
    alpha.zero_()
    alpha[0][1] = 100
    alpha[1][0] = 100
    alpha[2][2] = 100

    return alpha


def correct_alpha_init_xry(alpha, use_stopwords=True):
    alpha.zero_()
    alpha[0][0] = 100
    alpha[1][1] = 100
    alpha[2][2] = 100

    return alpha


def single_alpha(alpha):
    alpha.zero_()
    alpha[0][0] = 1
    print(alpha)


def _shnmn_func(question, img, num_modules, alpha, tau_0, tau_1, func):

    sentinel = torch.zeros_like(img)  # B x 1 x C x H x W
    h_prev = torch.cat([sentinel, img], dim=1)  # B x 2 x C x H x W

    if num_modules == 3:
        for i in range(num_modules):
            alpha_curr = F.softmax(alpha[i], dim=0)
            tau_0_curr = F.softmax(tau_0[i, :(i+2)], dim=0)
            tau_1_curr = F.softmax(tau_1[i, :(i+2)], dim=0)

            if type(func) is list:
                if len(func) == 3:
                    question_indexes = [0, 2, 1]
                    question_rep = question[question_indexes[i]]
                    func_ = func[question_indexes[i]]
            else:
                question_rep = torch.sum(alpha_curr.view(1,-1,1)*question, dim=1)  # (B,D)
            # B x C x H x W

            lhs_rep = torch.sum((tau_0_curr.view(1, (i+2), 1, 1, 1))*h_prev, dim=1)
            # B x C x H x W
            rhs_rep = torch.sum((tau_1_curr.view(1, (i+2), 1, 1, 1))*h_prev, dim=1)

            if type(func) is list:
                if len(func) == 3:
                    h_i = func_(question_rep, lhs_rep, rhs_rep)  # B x C x H x W
                else:
                    raise ValueError("Not valid list of modules' functions")
            else:
                h_i = func(question_rep, lhs_rep, rhs_rep)  # B x C x H x W

            # print(h_prev.shape, h_i.shape)
            h_prev = torch.cat([h_prev, h_i.unsqueeze(1)], dim=1)

    else:
        lhs_rep = torch.squeeze(sentinel)  # torch.sum((tau_0.view(1, 3, 1, 1, 1)) * h_prev, dim=1)
        # B x C x H x W
        rhs_rep = torch.squeeze(img)  # torch.sum((tau_1.view(1, 3, 1, 1, 1)) * h_prev, dim=1)
        question_rep = question
        h_prev = func(question_rep, lhs_rep, rhs_rep)  # B x C x H x W  # here it breaks
    return torch.unsqueeze(h_prev, dim=1)


class FindModule(nn.Module):
    def __init__(self, dim, kernel_size):
        super().__init__()
        self.dim = dim
        self.kernel_size = kernel_size
        self.conv_1 = nn.Conv2d(2*dim, dim, kernel_size=1, padding=0)
        self.conv_2 = nn.Conv2d(dim, dim, kernel_size=kernel_size, padding = kernel_size // 2)

    def forward(self, question_rep, lhs_rep, rhs_rep):
        out = F.relu(self.conv_1(torch.cat([lhs_rep, rhs_rep], 1)))  # concat along depth
        question_rep = question_rep.view(-1, self.dim, 1, 1)
        return F.relu(self.conv_2(out*question_rep))


class ResidualFunc:
    def __init__(self, dim, kernel_size):
        self.dim = dim
        self.kernel_size = kernel_size

    def __call__(self, question_rep, lhs_rep, rhs_rep):
        cnn_weight_dim = self.dim * self.dim * self.kernel_size * self.kernel_size
        cnn_bias_dim = self.dim
        proj_cnn_weight_dim = 2 * self.dim * self.dim
        proj_cnn_bias_dim = self.dim
        if (question_rep.size(1) !=
              proj_cnn_weight_dim + proj_cnn_bias_dim
              + 2 * (cnn_weight_dim + cnn_bias_dim)):
            raise ValueError

        # pick out CNN and projection CNN weights/biases
        cnn1_weight = question_rep[:,:cnn_weight_dim]
        cnn2_weight = question_rep[:,cnn_weight_dim:2 * cnn_weight_dim]
        cnn1_bias = question_rep[:,2 * cnn_weight_dim:(2 * cnn_weight_dim) + cnn_bias_dim]
        cnn2_bias = question_rep[:,(2 * cnn_weight_dim) + cnn_bias_dim:2 * (cnn_weight_dim + cnn_bias_dim)]
        proj_weight = question_rep[:, 2 * (cnn_weight_dim + cnn_bias_dim) :
                                  2 * (cnn_weight_dim + cnn_bias_dim) + proj_cnn_weight_dim]
        proj_bias   = question_rep[:, 2*(cnn_weight_dim + cnn_bias_dim) + proj_cnn_weight_dim:]

        cnn_out_total = []
        bs = question_rep.size(0)

        for i in range(bs):
            cnn1_weight_curr = cnn1_weight[i].view(self.dim, self.dim, self.kernel_size, self.kernel_size)
            cnn1_bias_curr   = cnn1_bias[i]
            cnn2_weight_curr = cnn2_weight[i].view(self.dim, self.dim, self.kernel_size, self.kernel_size)
            cnn2_bias_curr   = cnn2_bias[i]

            proj_weight_curr = proj_weight[i].view(self.dim, 2*self.dim, 1, 1)
            proj_bias_curr = proj_bias[i]

            cnn_inp = F.relu(F.conv2d(torch.cat([lhs_rep[[i]], rhs_rep[[i]]], 1),
                               proj_weight_curr,
                               bias=proj_bias_curr, padding=0))

            cnn1_out = F.relu(F.conv2d(cnn_inp, cnn1_weight_curr, bias=cnn1_bias_curr, padding=self.kernel_size // 2))
            cnn2_out = F.conv2d(cnn1_out, cnn2_weight_curr, bias=cnn2_bias_curr,padding=self.kernel_size // 2)

            cnn_out_total.append(F.relu(cnn_inp + cnn2_out) )

        return torch.cat(cnn_out_total)


class ConvFunc:
    def __init__(self, dim, kernel_size):
        self.dim = dim
        self.kernel_size = kernel_size

    def __call__(self, question_rep, lhs_rep, rhs_rep):
        cnn_weight_dim = self.dim*self.dim*self.kernel_size*self.kernel_size
        cnn_bias_dim = self.dim
        proj_cnn_weight_dim = 2*self.dim*self.dim
        proj_cnn_bias_dim = self.dim
        if (question_rep.size(1) !=
              proj_cnn_weight_dim + proj_cnn_bias_dim
              + cnn_weight_dim + cnn_bias_dim):
            raise ValueError

        # pick out CNN and projection CNN weights/biases
        cnn_weight = question_rep[:, : cnn_weight_dim]
        cnn_bias = question_rep[:, cnn_weight_dim : cnn_weight_dim + cnn_bias_dim]
        proj_weight = question_rep[:, cnn_weight_dim+cnn_bias_dim :
                                  cnn_weight_dim+cnn_bias_dim+proj_cnn_weight_dim]
        proj_bias   = question_rep[:, cnn_weight_dim+cnn_bias_dim+proj_cnn_weight_dim:]

        cnn_out_total = []
        bs = question_rep.size(0)

        for i in range(bs):
            cnn_weight_curr = cnn_weight[i].view(self.dim, self.dim, self.kernel_size, self.kernel_size)
            cnn_bias_curr   = cnn_bias[i]
            proj_weight_curr = proj_weight[i].view(self.dim, 2*self.dim, 1, 1)
            proj_bias_curr = proj_bias[i]

            cnn_inp = F.conv2d(torch.cat([lhs_rep[[i]], rhs_rep[[i]]], 1),
                               proj_weight_curr,
                               bias=proj_bias_curr, padding=0)
            cnn_out_total.append(F.relu(F.conv2d(
                cnn_inp, cnn_weight_curr, bias=cnn_bias_curr, padding=self.kernel_size // 2)))

        return torch.cat(cnn_out_total)


INITS = {'xavier_uniform': xavier_uniform,
         'constant': constant,
         'uniform': uniform,
         'correct': correct_alpha_init_xyr,
         'correct_xry': correct_alpha_init_xry,
         'correct_rxy': correct_alpha_init_rxy,
         'single': single_alpha}


class SHNMN(nn.Module):
    def __init__(self,
        vocab,
        feature_dim,
        module_dim,
        module_kernel_size,
        stem_dim,
        stem_num_layers,
        stem_subsample_layers,
        stem_kernel_size,
        stem_padding,
        stem_batchnorm,
        classifier_fc_layers,
        classifier_proj_dim,
        classifier_downsample,classifier_batchnorm,
        num_modules,
        hard_code_alpha=False,
        hard_code_tau=False,
        tau_init='random',
        alpha_init='xavier_uniform',
        model_type ='soft',
        model_bernoulli=0.5,
        use_module='conv',
        separated_stem=False,
        use_stopwords = True,
        **kwargs):

        super().__init__()
        self.question_token_to_idx = vocab["question_token_to_idx"]
        self.num_modules = num_modules
        # alphas and taus from Overleaf Doc.
        self.hard_code_alpha = hard_code_alpha
        self.hard_code_tau = hard_code_tau
        self.use_module = use_module
        self.separated_stem = separated_stem

        answers_category = [v_ for v_ in self.question_token_to_idx.values()
                            if v_ in category_idx]
        answers_color = [v_ for v_ in self.question_token_to_idx.values()
                         if v_ in color_idx]
        answers_brightness = [v_ for v_ in self.question_token_to_idx.values()
                              if v_ in brightness_idx]
        answers_size = [v_ for v_ in self.question_token_to_idx.values()
                        if v_ in size_idx]

        self.n_classes_per_attr = []
        len_embedding = 0
        for ans_ in [answers_category, answers_color,
                     answers_brightness, answers_size]:
            if len(ans_) > 0:
                len_embedding += 1
                self.n_classes_per_attr.append(ans_)
        self.len_embedding = len_embedding
        num_question_tokens = 3

        if alpha_init.startswith('correct'):
            print('using correct initialization')
            alpha = INITS[alpha_init](torch.Tensor(num_modules, num_question_tokens))
        elif alpha_init == 'constant':
            alpha = INITS[alpha_init](torch.Tensor(num_modules, num_question_tokens), 1)
        elif alpha_init == 'single':
            alpha = INITS[alpha_init](torch.Tensor(num_modules, 1))
        else:
            alpha = INITS[alpha_init](torch.Tensor(num_modules, num_question_tokens),1)

        if hard_code_alpha:
            if num_modules == 3:
                assert(alpha_init.startswith('correct'))

            self.alpha = Variable(alpha)
            self.alpha = self.alpha.to(device)
        else:
            self.alpha = nn.Parameter(alpha)

        # create taus
        if tau_init == 'tree':
            tau_0, tau_1 = _tree_tau()
            print("initializing with tree.")
        elif tau_init == 'chain':
            tau_0, tau_1 = _chain_tau()
            print("initializing with chain")
        elif tau_init == 'chain_with_shortcuts':
            tau_0, tau_1 = _chain_with_shortcuts_tau() 
            print("initializing with chain and shortcuts")
        elif tau_init == 'chain_with_shortcuts_flipped':
            tau_0, tau_1 = _chain_with_shortcuts_tau_flipped()
            print("initializing with chain and shortcuts")
        elif tau_init == "single":
            tau_0, tau_1 = _single_tau()
            print("initializing with single module")
        else:
            tau_0, tau_1 = _random_tau(num_modules)

        if hard_code_tau:
            if num_modules == 3:
                assert(tau_init in ['chain', 'tree', 'chain_with_shortcuts', 'chain_with_shortcuts_flipped'])
            self.tau_0 = Variable(tau_0)
            self.tau_1 = Variable(tau_1)
            self.tau_0 = self.tau_0.to(device)
            self.tau_1 = self.tau_1.to(device)
        else:
            self.tau_0 = nn.Parameter(tau_0)
            self.tau_1 = nn.Parameter(tau_1)

        if use_module == 'conv':
            embedding_dim_1 = module_dim + (module_dim*module_dim*module_kernel_size*module_kernel_size)
            embedding_dim_2 = module_dim + (2*module_dim*module_dim)

            question_embeddings_1 = nn.Embedding(len_embedding, embedding_dim_1)
            question_embeddings_2 = nn.Embedding(len_embedding, embedding_dim_2)

            stdv_1 = 1. / math.sqrt(module_dim*module_kernel_size*module_kernel_size)
            stdv_2 = 1. / math.sqrt(2*module_dim)

            question_embeddings_1.weight.data.uniform_(-stdv_1, stdv_1)
            question_embeddings_2.weight.data.uniform_(-stdv_2, stdv_2)
            # TODO: adapt to query type of questions
            self.question_embeddings = nn.Embedding(len_embedding, embedding_dim_1+embedding_dim_2)
            self.question_embeddings.weight.data = torch.cat([question_embeddings_1.weight.data,
                                                              question_embeddings_2.weight.data],dim=-1)

            self.func = ConvFunc(module_dim, module_kernel_size)

        elif use_module == 'residual':
            embedding_dim_1 = module_dim + (module_dim*module_dim*module_kernel_size*module_kernel_size)
            embedding_dim_2 = module_dim + (2*module_dim*module_dim)

            question_embeddings_a = nn.Embedding(len_embedding, embedding_dim_1)
            question_embeddings_b = nn.Embedding(len_embedding, embedding_dim_1)
            question_embeddings_2 = nn.Embedding(len_embedding, embedding_dim_2)

            stdv_1 = 1. / math.sqrt(module_dim*module_kernel_size*module_kernel_size)
            stdv_2 = 1. / math.sqrt(2*module_dim)

            question_embeddings_a.weight.data.uniform_(-stdv_1, stdv_1)
            question_embeddings_b.weight.data.uniform_(-stdv_1, stdv_1)
            question_embeddings_2.weight.data.uniform_(-stdv_2, stdv_2)
            self.question_embeddings = nn.Embedding(len_embedding,
                                                    2*embedding_dim_1+embedding_dim_2)
            self.question_embeddings.weight.data = torch.cat([question_embeddings_a.weight.data,
                                                              question_embeddings_b.weight.data,
                                                              question_embeddings_2.weight.data],
                                                             dim=-1)
            self.func = ResidualFunc(module_dim, module_kernel_size)

        elif use_module == 'find':
            self.question_embeddings = nn.Embedding(len_embedding, module_dim)
            self.func = FindModule(module_dim, module_kernel_size)

        # stem for processing the image into a 3D tensor
        # print(feature_dim, stem_dim, module_dim, stem_num_layers)
        # TODO: adapt to separated_stem
        if self.separated_stem:
            self.stem = nn.ModuleList([build_stem(feature_dim[0], stem_dim, module_dim,
                                                  num_layers=stem_num_layers,
                                                  subsample_layers=stem_subsample_layers,
                                                  kernel_size=stem_kernel_size,
                                                  padding=stem_padding,
                                                  with_batchnorm=stem_batchnorm).to(device)
                                      for k__ in range(len_embedding)])
            tmp = self.stem[0](Variable(torch.zeros([1,
                                                     feature_dim[0],
                                                     feature_dim[1],
                                                     feature_dim[2]])).to(device)).to(device)

        else:
            self.stem = build_stem(feature_dim[0], stem_dim, module_dim,
                                   num_layers=stem_num_layers,
                                   subsample_layers=stem_subsample_layers,
                                   kernel_size=stem_kernel_size,
                                   padding=stem_padding,
                                   with_batchnorm=stem_batchnorm)
            tmp = self.stem(Variable(torch.zeros([1,
                                                  feature_dim[0],
                                                  feature_dim[1],
                                                  feature_dim[2]])))

        module_H = tmp.size(2)
        module_W = tmp.size(3)

        classifiers_list = []
        # 0: category
        # if num_answers_category > 0:
        # classifier0 = build_classifier(module_dim, module_H, module_W,
        #                                num_answers_category,
        #                                classifier_fc_layers,
        #                                classifier_proj_dim,
        #                                classifier_downsample,
        #                                with_batchnorm=classifier_batchnorm)
        # self.classifiers_list.append(classifier0.to(device))
        for ans_ in self.n_classes_per_attr:
            if len(ans_) > 0:
                classifiers_list.append(build_classifier(module_dim, module_H, module_W,
                                                         len(ans_),
                                                         classifier_fc_layers,
                                                         classifier_proj_dim,
                                                         classifier_downsample,
                                                         with_batchnorm=classifier_batchnorm).to(device))
        self.classifiers_list = nn.ModuleList(classifiers_list)
        self.model_type = model_type
        self.use_module = use_module
        p = model_bernoulli
        tree_odds = -numpy.log((1 - p) / p)
        self.tree_odds = nn.Parameter(torch.Tensor([tree_odds]))

        self.h_list = [None for k__ in range(len_embedding)]
        # self.h_0 = None
        # self.h_1 = None
        # self.h_2 = None
        # self.h_3 = None

    def forward_hard(self, image, question):
        # TODO: adapt to query
        # NOT ADAPTED TO THE QUERY TYPE OF QUESTION
        question = self.question_embeddings(question)
        stemmed_img = self.stem(image).unsqueeze(1)  # B x 1 x C x H x W
        chain_tau_0, chain_tau_1 = _chain_tau()
        chain_tau_0 = chain_tau_0.to(device)
        chain_tau_1 = chain_tau_1.to(device)
        h_chain = _shnmn_func(question, stemmed_img,
                              self.num_modules, self.alpha,
                              Variable(chain_tau_0), Variable(chain_tau_1), self.func)
        h_final_chain = h_chain[:, -1, :, :, :]
        tree_tau_0, tree_tau_1 = _tree_tau()
        tree_tau_0 = tree_tau_0.to(device)
        tree_tau_1 = tree_tau_1.to(device)
        h_tree = _shnmn_func(question, stemmed_img,
                             self.num_modules, self.alpha,
                             Variable(tree_tau_0), Variable(tree_tau_1), self.func)
        h_final_tree = h_tree[:, -1, :, :, :]

        p_tree = torch.sigmoid(self.tree_odds[0])
        self.tree_scores = self.classifier(h_final_tree)
        self.chain_scores = self.classifier(h_final_chain)
        output_probs_tree = F.softmax(self.tree_scores, dim=1)
        output_probs_chain = F.softmax(self.chain_scores, dim=1)
        probs_mixture = p_tree * output_probs_tree + (1.0 - p_tree) * output_probs_chain
        eps = 1e-6
        probs_mixture = (1 - eps) * probs_mixture + eps
        return torch.log(probs_mixture)

    def forward_soft(self, image, question):
        # here it is simpler
        if self.separated_stem:
            stemmed_lst = []
            for j_ in range(self.len_embedding):
                stemmed_lst.append(self.stem[j_](image).unsqueeze(1))
        else:
            stemmed_img = self.stem(image).unsqueeze(1)  # B x 1 x C x H x W

        embedded_questions = self.question_embeddings(question)
        for j_ in range(self.len_embedding):
            if self.separated_stem:
                stemmed_img = stemmed_lst[j_]
            self.h_list[j_] = _shnmn_func(embedded_questions[:, j_],
                                          stemmed_img,
                                          self.num_modules,
                                          self.alpha,
                                          self.tau_0,
                                          self.tau_1,
                                          self.func)

        h_final = [h_[:, -1, :, :, :].to(device) for h_ in self.h_list]
        return [cl_(h_) for cl_, h_ in zip(self.classifiers_list, h_final)]

    def forward(self, image, question):
        if self.model_type == 'hard':
            return self.forward_hard(image, question)
        else:
            return self.forward_soft(image, question)


# this code is useful for the shnmn and residual
"""if self.separated_stem:
    stemmed_img = []
    for j in range(self.len_embedding):
        stemmed_img.append(self.stem[j](image).unsqueeze(1))
    count = 0
    B, Q = question.shape
    tmp_ = np.zeros((B, Q))
    question_np = question.cpu().detach().numpy()
    for ans_ in enumerate(self.n_classes_per_attr):
        if len(ans_) > 0:
            q_attribute = question_np[:, count]
            tmp_[:, count] = np.array([q_ in ans_ for q_ in q_attribute]).squeeze()
            count += 1
    indexes_select_stem = np.array([np.where(tmp__)[0][0] for tmp__ in tmp_])

    for id_img, ind_ in enumerate(indexes_select_stem):
        tmp_img = self.stem[ind_](image[id_img].unsqueeze(0)).unsqueeze(1)

        if id_img == 0:
            stemmed_img = tmp_img
        else:
            stemmed_img = torch.cat([stemmed_img, tmp_img])"""