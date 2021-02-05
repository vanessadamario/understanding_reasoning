import sys
import numpy
import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from runs.layers import build_stem, build_classifier
# from vr.models.layers import init_modules, ResidualBlock, SimpleVisualBlock, GlobalAveragePool, Flatten
# from vr.models.layers import build_classifier, build_stem, ConcatBlock
# import vr.programs
from torch.nn.init import kaiming_normal, kaiming_uniform, xavier_uniform, xavier_normal, constant, uniform
# from vr.models.filmed_net import FiLM, FiLMedResBlock, ConcatFiLMedResBlock, coord_map
# from vr.models.film_gen import FiLMGen
from functools import partial
# for one object we do not need to have tau and alpha as tensors
# of the original shape

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


def _tree_sep_input_tau():
    # it works if we do not need the soft-max
    tau_0 = torch.zeros(3, 5)
    tau_1 = torch.zeros(3, 5)
    tau_0[0][1] = tau_1[0][0] = 1
    tau_0[1][2] = tau_1[1][0] = 1
    tau_0[2][3] = tau_1[2][4] = 1
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
    # print(isinstance(img, list)) separated_residual/ separated find
    flag_separated_stem = isinstance(img, list)
    if flag_separated_stem:
        sentinel = torch.zeros_like(img[0])  # B x 1 x C x H x W
        h_prev = torch.cat([sentinel, img[0], img[1]], dim=1)  # B x 3 x C x H x W
        look_back = 3
    else:
        sentinel = torch.zeros_like(img)  # B x 1 x C x H x W
        h_prev = torch.cat([sentinel, img], dim=1)  # B x 2 x C x H x W
        look_back = 2

    if num_modules == 3:
        for i in range(num_modules):
            if not flag_separated_stem:
                alpha_curr = F.softmax(alpha[i], dim=0)
                tau_0_curr = F.softmax(tau_0[i, :(i+look_back)], dim=0)
                tau_1_curr = F.softmax(tau_1[i, :(i+look_back)], dim=0)
            else:
                alpha_curr = F.softmax(alpha[i], dim=0)
                tau_0_curr = tau_0[i, :(i + look_back)]
                tau_1_curr = tau_1[i, :(i + look_back)]

            if type(func) is list and len(func) == 3:
                # asymmetric residual, mixed find, mixed architectures

                question_indexes = [0, 2, 1]
                question_rep = question[question_indexes[i]]  # check
                func_ = func[question_indexes[i]]
                # else:
                #     question_indexes = [0, 2, 1]
                #     question_rep = question[:, question_indexes[i]]
            else:
                question_rep = torch.sum(alpha_curr.view(1, -1, 1)*question, dim=1)  # (B,D)
            # B x C x H x W

            lhs_rep = torch.sum((tau_0_curr.view(1, (i+look_back), 1, 1, 1))*h_prev, dim=1)
            # B x C x H x W
            rhs_rep = torch.sum((tau_1_curr.view(1, (i+look_back), 1, 1, 1))*h_prev, dim=1)

            if type(func) is list:
                batch_size = len(func)
                if len(func) == 3:
                    h_i = func_(question_rep, lhs_rep, rhs_rep)  # B x C x H x W
                else:  # a func_ for each sample
                    h_i = torch.cat([func[id_][i](question_rep[id_].unsqueeze(0),
                                                  lhs_rep[id_].unsqueeze(0),
                                                  rhs_rep[id_].unsqueeze(0))
                                     for id_ in range(batch_size)])
            else:
                h_i = func(question_rep, lhs_rep, rhs_rep)  # B x C x H x W

            h_prev = torch.cat([h_prev, h_i.unsqueeze(1)], dim=1)
        return h_prev

    else:
        lhs_rep = torch.squeeze(sentinel)  # torch.sum((tau_0.view(1, 3, 1, 1, 1)) * h_prev, dim=1)
        # B x C x H x W
        rhs_rep = torch.squeeze(img)  # torch.sum((tau_1.view(1, 3, 1, 1, 1)) * h_prev, dim=1)
        question_rep = question
        # TODO: here you need to change depending if the function is a
        # list or single object
        if type(func) == list:
            h_prev = torch.stack([f_(q_.unsqueeze(0), l_.unsqueeze(0), r_.unsqueeze(0))
                                  for f_, q_, l_, r_ in zip(func, question_rep, lhs_rep, rhs_rep)])
        else:
            h_prev = func(question_rep, lhs_rep, rhs_rep)
        return torch.unsqueeze(h_prev, dim=1)


class FindModule(nn.Module):
    def __init__(self, dim, kernel_size):
        super().__init__()
        self.dim = dim
        self.kernel_size = kernel_size
        self.conv_1 = nn.Conv2d(2*dim, dim, kernel_size=1, padding=0)
        self.conv_2 = nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=kernel_size // 2)

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
        use_module = 'conv',
        use_stopwords = True,
        separated_stem=False,
        separated_module=False,
        separated_classifier=False,
        **kwargs):

        super().__init__()
        # alphas and taus from Overleaf Doc.
        self.vocab = vocab
        self.num_modules = num_modules
        self.hard_code_alpha = hard_code_alpha
        self.hard_code_tau = hard_code_tau
        self.use_module = use_module
        self.separated_stem = separated_stem
        self.separated_module = separated_module
        self.separated_classifier = separated_classifier

        self.image_pair = False
        if 'image_pair' in self.vocab.keys():
            self.image_pair = self.vocab['image_pair']
        print("image pair: ", self.image_pair)
        print(type(self.image_pair))

        if self.separated_stem:
            # if the specialization is at the beginning
            # we will have it in all the architecture
            self.separated_module = True
            self.separated_classifier = True

        from runs.data_attribute_random import vocab as vocab_sqoop
        from runs.data_comparison_relations import vocab as vocab_attribute_sqoop
        if self.vocab['question_token_to_idx'] == vocab_sqoop:
            from runs.data_attribute_random import map_question_idx_to_attribute_category as map_
        elif self.vocab['question_token_to_idx'] == vocab_attribute_sqoop['question_token_to_idx']:
            from runs.data_comparison_relations import map_question_idx_to_group as map_
        else:
            raise ValueError('vocab variable does not match')

        # TODO: old version for the shnmn with multiple attributes
        if self.use_module == "find":
            self.func_ = map_
            tmp = np.array([self.func_(k_) for k_ in self.vocab['question_token_to_idx'].values()])
            self.subgroups = np.unique(tmp).size
        else:
            self.func_ = lambda x: x
            self.subgroups = None
        num_question_tokens = 3
        # len_embedding = len(vocab["question_token_to_idx"])

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
            if self.num_modules == 3 and self.separated_stem:
                tau_0, tau_1 = _tree_sep_input_tau()
            elif self.image_pair:
                tau_0, tau_1 = _tree_sep_input_tau()
            else:
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

            question_embeddings_1 = nn.Embedding(len(vocab['question_idx_to_token']),embedding_dim_1)
            question_embeddings_2 = nn.Embedding(len(vocab['question_idx_to_token']),embedding_dim_2)

            stdv_1 = 1. / math.sqrt(module_dim*module_kernel_size*module_kernel_size)
            stdv_2 = 1. / math.sqrt(2*module_dim)

            question_embeddings_1.weight.data.uniform_(-stdv_1, stdv_1)
            question_embeddings_2.weight.data.uniform_(-stdv_2, stdv_2)
            self.question_embeddings = nn.Embedding(len(vocab['question_idx_to_token']), embedding_dim_1+embedding_dim_2)
            self.question_embeddings.weight.data = torch.cat([question_embeddings_1.weight.data,
                                                              question_embeddings_2.weight.data],dim=-1)

            self.func = ConvFunc(module_dim, module_kernel_size)

        elif use_module == 'residual':
            embedding_dim_1 = module_dim + (module_dim*module_dim*module_kernel_size*module_kernel_size)
            embedding_dim_2 = module_dim + (2*module_dim*module_dim)

            question_embeddings_a = nn.Embedding(len(vocab['question_idx_to_token']),embedding_dim_1)
            question_embeddings_b = nn.Embedding(len(vocab['question_idx_to_token']),embedding_dim_1)
            question_embeddings_2 = nn.Embedding(len(vocab['question_idx_to_token']),embedding_dim_2)

            stdv_1 = 1. / math.sqrt(module_dim*module_kernel_size*module_kernel_size)
            stdv_2 = 1. / math.sqrt(2*module_dim)

            question_embeddings_a.weight.data.uniform_(-stdv_1, stdv_1)
            question_embeddings_b.weight.data.uniform_(-stdv_1, stdv_1)
            question_embeddings_2.weight.data.uniform_(-stdv_2, stdv_2)
            self.question_embeddings = nn.Embedding(len(vocab['question_idx_to_token']), 2*embedding_dim_1+embedding_dim_2)
            self.question_embeddings.weight.data = torch.cat([question_embeddings_a.weight.data, question_embeddings_b.weight.data,
                                                              question_embeddings_2.weight.data],dim=-1)
            self.func = ResidualFunc(module_dim, module_kernel_size)

        # TODO new code
        # TODO elif use_module == "find" and self.separated_module:
            # TODO self.func = nn.ModuleDict({str(k_): FindModule(module_dim, module_kernel_size) for k_ in range(4)})
            # TODO self.question_embedding = nn.Embedding(len(vocab["question_idx_to_token"]), module_dim)

        # TODO delete this
        # elif use_module == "find" and self.separated_stem: old version
        elif use_module == "find" and self.separated_module:
            #     self.question_embeddings = nn.ModuleDict # something
            # TODO : self.subgroups  # independent from type of question
            self.func = nn.ModuleDict({str(k_): FindModule(module_dim, module_kernel_size)
                                       for k_ in range(self.subgroups)})
            self.question_embeddings = nn.Embedding(len(vocab['question_idx_to_token']), module_dim)
            # hard-coded four attributes

        elif use_module == 'find':
            # comment if you want find with GRU
            self.question_embeddings = nn.Embedding(len(vocab['question_idx_to_token']), module_dim)

            # decomment if you want to see find with GRU
            """gru = FiLMGen(encoder_vocab_size=len(vocab['question_idx_to_token']),
                          wordvec_dim=200,
                          num_modules=3,
                          module_dim=module_dim//2,  # gamma and beta in FiLM
                          taking_context=False,
                          use_attention=False,
                          parameter_efficient=True,
                          gamma_baseline=0)
            self.question_embeddings = gru  # here we need the gru"""
            # until here
            self.func = FindModule(module_dim, module_kernel_size)

        elif use_module == 'mixed':
            embedding_dim_1 = module_dim + (module_dim * module_dim * module_kernel_size * module_kernel_size)
            embedding_dim_2 = module_dim + (2 * module_dim * module_dim)
            question_embeddings_a = nn.Embedding(len(vocab['question_idx_to_token']), embedding_dim_1)
            question_embeddings_b = nn.Embedding(len(vocab['question_idx_to_token']), embedding_dim_1)
            question_embeddings_2 = nn.Embedding(len(vocab['question_idx_to_token']), embedding_dim_2)

            stdv_1 = 1. / math.sqrt(module_dim * module_kernel_size * module_kernel_size)
            stdv_2 = 1. / math.sqrt(2 * module_dim)

            question_embeddings_a.weight.data.uniform_(-stdv_1, stdv_1)
            question_embeddings_b.weight.data.uniform_(-stdv_1, stdv_1)
            question_embeddings_2.weight.data.uniform_(-stdv_2, stdv_2)
            self.question_embeddings_res = nn.Embedding(len(vocab['question_idx_to_token']),
                                                        2 * embedding_dim_1 + embedding_dim_2)
            self.question_embeddings_res.weight.data = torch.cat(
                [question_embeddings_a.weight.data, question_embeddings_b.weight.data,
                 question_embeddings_2.weight.data], dim=-1)
            self.func_objects = ResidualFunc(module_dim, module_kernel_size)

            self.question_embeddings_find = nn.Embedding(len(vocab['question_idx_to_token']), module_dim)
            self.func_relation = FindModule(module_dim, module_kernel_size)
            self.func = [self.func_objects, self.func_relation, self.func_objects]

        elif use_module == 'mixed_find':
            self.question_embeddings_find = nn.Embedding(len(vocab['question_idx_to_token']), module_dim)
            self.func_objects = FindModule(module_dim, module_kernel_size)
            self.func_relation = FindModule(module_dim, module_kernel_size)
            self.func = [self.func_objects, self.func_relation, self.func_objects]  # find w/o check is

        elif use_module == 'asymmetric_residual':
            embedding_dim_1 = module_dim + (module_dim * module_dim * module_kernel_size * module_kernel_size)
            embedding_dim_2 = module_dim + (2 * module_dim * module_dim)

            question_embeddings_a_X = nn.Embedding(len(vocab['question_idx_to_token']), embedding_dim_1)
            question_embeddings_b_X = nn.Embedding(len(vocab['question_idx_to_token']), embedding_dim_1)
            question_embeddings_2_X = nn.Embedding(len(vocab['question_idx_to_token']), embedding_dim_2)

            stdv_1 = 1. / math.sqrt(module_dim * module_kernel_size * module_kernel_size)
            stdv_2 = 1. / math.sqrt(2 * module_dim)

            question_embeddings_a_X.weight.data.uniform_(-stdv_1, stdv_1)
            question_embeddings_b_X.weight.data.uniform_(-stdv_1, stdv_1)
            question_embeddings_2_X.weight.data.uniform_(-stdv_2, stdv_2)
            self.question_embeddings_X = nn.Embedding(len(vocab['question_idx_to_token']),
                                                    2 * embedding_dim_1 + embedding_dim_2)
            self.question_embeddings_X.weight.data = torch.cat(
                [question_embeddings_a_X.weight.data, question_embeddings_b_X.weight.data,
                 question_embeddings_2_X.weight.data], dim=-1)
            self.func_X = ResidualFunc(module_dim, module_kernel_size)

            question_embeddings_a_Y = nn.Embedding(len(vocab['question_idx_to_token']), embedding_dim_1)
            question_embeddings_b_Y = nn.Embedding(len(vocab['question_idx_to_token']), embedding_dim_1)
            question_embeddings_2_Y = nn.Embedding(len(vocab['question_idx_to_token']), embedding_dim_2)

            stdv_1 = 1. / math.sqrt(module_dim * module_kernel_size * module_kernel_size)
            stdv_2 = 1. / math.sqrt(2 * module_dim)

            question_embeddings_a_Y.weight.data.uniform_(-stdv_1, stdv_1)
            question_embeddings_b_Y.weight.data.uniform_(-stdv_1, stdv_1)
            question_embeddings_2_Y.weight.data.uniform_(-stdv_2, stdv_2)
            self.question_embeddings_Y = nn.Embedding(len(vocab['question_idx_to_token']),
                                                      2 * embedding_dim_1 + embedding_dim_2)
            self.question_embeddings_Y.weight.data = torch.cat(
                [question_embeddings_a_Y.weight.data, question_embeddings_b_Y.weight.data,
                 question_embeddings_2_Y.weight.data], dim=-1)
            self.func_Y_R = ResidualFunc(module_dim, module_kernel_size)
            self.func = [self.func_X, self.func_Y_R, self.func_Y_R]

        # stem for processing the image into a 3D tensor
        # print(feature_dim, stem_dim, module_dim, stem_num_layers)
        if self.separated_stem:
            self.stem = nn.ModuleDict({str(self.func_(qv_)): build_stem(feature_dim[0],
                                                                        stem_dim,
                                                                        module_dim,
                                                                        num_layers=stem_num_layers,
                                                                        subsample_layers=stem_subsample_layers,
                                                                        kernel_size=stem_kernel_size,
                                                                        padding=stem_padding,
                                                                        with_batchnorm=stem_batchnorm)
                                                             for qv_ in vocab["question_token_to_idx"].values()})
            rnd_key = str(self.func_(list(vocab["question_token_to_idx"].values())[0]))
            tmp = (self.stem[rnd_key])(Variable(torch.zeros([1, feature_dim[0],
                                                             feature_dim[1],
                                                             feature_dim[2]])))
            # #self.subgroups different find modules when separated_stem
            # we need to see how many elements we have per attribute family and generate

        else:
            self.stem = build_stem(feature_dim[0], stem_dim, module_dim,
                                   num_layers=stem_num_layers,
                                   subsample_layers=stem_subsample_layers,
                                   kernel_size=stem_kernel_size,
                                   padding=stem_padding,
                                   with_batchnorm=stem_batchnorm)
            tmp = self.stem(Variable(torch.zeros([1, feature_dim[0],
                                                  feature_dim[1],
                                                  feature_dim[2]])))
        print("feature dims")
        sys.stdout.flush()
        print(feature_dim[0], feature_dim[1], feature_dim[2])
        sys.stdout.flush()
        print("tmp dims")
        sys.stdout.flush()
        print(tmp.shape)
        sys.stdout.flush()
        module_H = tmp.size(2)
        module_W = tmp.size(3)
        print(module_H, module_W)
        sys.stdout.flush()

        num_answers = len(vocab['answer_idx_to_token'])

        # old version: if self.separated_stem:
        if self.separated_classifier:
            self.classifier = nn.ModuleDict({str(self.func_(qv_)): build_classifier(module_dim, module_H, module_W, num_answers,
                                                                                    classifier_fc_layers,
                                                                                    classifier_proj_dim,
                                                                                    classifier_downsample,
                                                                                    with_batchnorm=classifier_batchnorm).to(device)
                                                                   for qv_ in vocab["question_token_to_idx"].values()})

        else:
            print("classifier dimensions")
            sys.stdout.flush()
            print(module_dim, module_H, module_W, num_answers, classifier_fc_layers,
                  classifier_proj_dim, classifier_downsample, classifier_batchnorm)
            sys.stdout.flush()
            self.classifier = build_classifier(module_dim, module_H, module_W, num_answers,
                                               classifier_fc_layers,
                                               classifier_proj_dim,
                                               classifier_downsample,
                                               with_batchnorm=classifier_batchnorm)
        self.model_type = model_type
        self.use_module = use_module
        p = model_bernoulli
        tree_odds = -numpy.log((1 - p) / p)
        self.tree_odds = nn.Parameter(torch.Tensor([tree_odds]))
        self.h = None

    def forward_hard(self, image, question):
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
        # the question must drive the stem to use
        question_copy = torch.clone(question)

        if (image.ndimension() == 5) != self.image_pair:
            raise ValueError('Incongruent')

        # STEM GENERATION
        if self.image_pair:
            if self.separated_stem:
                if self.num_modules == 3:
                    stemmed_img_lhs = torch.cat([self.stem[str(self.func_(int(qv_)))](img.unsqueeze(0)).unsqueeze(1)
                                                 for img, qv_ in zip(image[:, 0], question[:, 0])])
                    stemmed_img_rhs = torch.cat([self.stem[str(self.func_(int(qv_)))](img.unsqueeze(0)).unsqueeze(1)
                                                 for img, qv_ in zip(image[:, 1], question[:, 2])])
                    stemmed_img = [stemmed_img_lhs, stemmed_img_rhs]
                else:
                    raise ValueError("Pair of images with n_modules != 3 not implemented")

            else:
                if self.num_modules == 3:
                    stemmed_img_lhs = self.stem(image[:, 0]).unsqueeze(1)
                    stemmed_img_rhs = self.stem(image[:, 1]).unsqueeze(1)
                    print(stemmed_img_lhs.shape)
                    stemmed_img = [stemmed_img_lhs, stemmed_img_rhs]
                else:
                    raise ValueError("Pair of images with n_modules != 3 not implemented")

        else:
            if self.separated_stem:
                if self.num_modules == 1:
                    stemmed_img = torch.cat([self.stem[str(self.func_(int(qv_)))](img.unsqueeze(0)).unsqueeze(1)
                                             for img, qv_ in zip(image, question)])
                elif self.num_modules == 3:
                    stemmed_img_lhs = torch.cat([self.stem[str(self.func_(int(qv_)))](img.unsqueeze(0)).unsqueeze(1)
                                                 for img, qv_ in zip(image, question[:, 0])])
                    stemmed_img_rhs = torch.cat([self.stem[str(self.func_(int(qv_)))](img.unsqueeze(0)).unsqueeze(1)
                                                 for img, qv_ in zip(image, question[:, 2])])
                    stemmed_img = [stemmed_img_lhs, stemmed_img_rhs]
                else:
                    raise ValueError("This case is not implemented")
            else:
                stemmed_img = self.stem(image).unsqueeze(1)  # B x 1 x C x H x W

        # MODULES OTHER THAN FIND AND RESIDUAL IN TREE CONFIG
        if self.use_module == 'find' and self.separated_module:
            if self.num_modules == 1:
                func = [self.func[str(self.func_(int(q_)))] for q_ in question_copy]
            elif self.num_modules == 3:
                func = [[self.func[str(self.func_(int(q_[0])))],  # x
                         self.func[str(self.func_(int(q_[2])))],  # y and relation
                         self.func[str(self.func_(int(q_[1])))]] for q_ in question_copy]

        if self.use_module == 'mixed':
            question = [self.question_embeddings_res(question[:, 0]),
                        self.question_embeddings_find(question[:, 1]),
                        self.question_embeddings_res(question[:, 2])]

        elif self.use_module == 'mixed_find':
            question = [self.question_embeddings_find(question[:, i_]) for i_ in range(3)]

        elif self.use_module == 'asymmetric_residual':
            question = [self.question_embeddings_X(question[:, 0]),
                        self.question_embeddings_Y(question[:, 1]),
                        self.question_embeddings_Y(question[:, 2])]
        else:
            question = self.question_embeddings(question)
        # print('question shape', question.shape)

        # GENERATE MODULE AND HIDDEN REPRESENTATION
        # here depending on the question, the self.func we call is different
        # old if self.separated_stem and self.use_module == "find":
        if self.separated_module and self.use_module == "find":
            # with self.num_modules == 3, the input
            # to stemmed_img can be a list with lhs and rhs representations
            self.h = _shnmn_func(question, stemmed_img, self.num_modules,
                                 self.alpha, self.tau_0, self.tau_1, func)

        else:  # residual, or find wo separated_module
            self.h = _shnmn_func(question, stemmed_img, self.num_modules,
                                 self.alpha, self.tau_0, self.tau_1, self.func)
        h_final = self.h[:, -1, :, :, :]

        # GENERATE CLASSIFIER
        if self.separated_classifier:
            if self.num_modules == 1:
                if self.use_module == 'find':
                    classifier_output = torch.cat([self.classifier[str(self.func_(int(qv_)))](h_final[i_])
                                                   for i_, qv_ in enumerate(question_copy)])
                elif self.use_module == 'residual':
                    classifier_output = torch.cat([self.classifier[str(self.func_(int(qv_)))](h_final[i_].unsqueeze(0))
                                                   for i_, qv_ in enumerate(question_copy)])
                else:
                    raise ValueError('Separated classifier not implemented')

            elif self.num_modules == 3: 

                if self.use_module == 'find':
                    classifier_output = torch.cat([self.classifier[str(self.func_(int(qv_)))](h_final[i_].unsqueeze(0))
                                                   for i_, qv_ in enumerate(question_copy[:, 1])])
                elif self.use_module == 'residual':
                    classifier_output = torch.cat([self.classifier[str(self.func_(int(qv_)))](h_final[i_].unsqueeze(0))
                                                   for i_, qv_ in enumerate(question_copy[:, 1])])
            else:
                raise ValueError('Separated classifier not implemented')
            return classifier_output

        else:
            return self.classifier(h_final)

    def forward(self, image, question):
        if self.model_type == 'hard':
            return self.forward_hard(image, question)
        else:
            return self.forward_soft(image, question)
