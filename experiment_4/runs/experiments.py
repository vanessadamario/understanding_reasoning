import json
import os
from os.path import join


# TODO: remember to change the parameters for the dataset : n_training
experiment_case_list = [1]  # [1] for VQA and binary answers
lr_array = [1e-2, 5e-3, 1e-3, 1e-5, 1e-4]  # [1e-4, 1e-5]  #
method_type_list = ["SHNMN"]
batch_list = [64]
dataset_dict = {"dataset_name": ["dataset_0",
                                 "dataset_1",
                                 "dataset_2",
                                 "dataset_3",
                                 "dataset_4",
                                 "dataset_5",
                                 ]
                }

dict_method_type = {"use_module": "residual"
                                  "",
                    "model_type": 'soft',
                    "tau_init": "tree",
                    "alpha_init": "correct",
                    "model_bernoulli": 0.5,
                    "hard_code_alpha": True,
                    "hard_code_tau": True,
                    "feature_dim": [3, 28, 28],  # TODO: input dimensions
                    "module_dim": 64,
                    "module_kernel_size": 3,
                    "stem_dim": 64,
                    "stem_num_layers": 6,  # TODO: stem_num_layer changed 2
                    "stem_subsample_layers": [1, 3],  # TODO: changed from []
                    "stem_kernel_size": [3],
                    "stem_padding": None,
                    "stem_batchnorm": 0,  # TODO: stem_batchnorm changed 0
                    "classifier_fc_layers": [1024],
                    "classifier_proj_dim": 512,
                    "classifier_batchnorm": 0,  # TODO: classifier_batchnorm changed 0
                    "classifier_downsample": "maxpoolfull",
                    "num_modules": 3,
                    "separated_stem": 1,
                    "separated_module": 1,
                    "separated_classifier": 1
                    }


class OptimizationHyperParameters(object):
    """ Add hyper-parameters in init so when you read a json,
    it will get updated as your latest code. """
    def __init__(self,
                 learning_rate=5e-2,
                 architecture=None,
                 # epochs=500,  # it is num_iterations  # we can call the code multiple times
                 batch_size=64,
                 optimizer='Adam',
                 lr_at_plateau=True,  # TODO: not implemented in sysgen
                 reduction_factor=None,  # TODO: not implemented in sysgen
                 validation_check=True,  # TODO: not implemented in sysgen
                 num_iterations=200000,
                 sensitive_learning_rate=1e-3,
                 reward_decay=0.9,
                 weight_decay=0.,
                 allow_resume=True,
                 randomize_checkpoint_path=False,
                 avoid_checkpoint_override=False,
                 record_loss_every=10,
                 checkpoint_every=1000,
                 time=0,
                 num_val_samples=1000,
                 early_stopping=False,
                 previous_epochs=2,
                 min_epochs=3,
                 max_epochs=500,
                 n_checkpoint_every_epoch=1000,
                 n_record_loss_every_epoch=1000):
        """
        :param learning_rate: float, the initial value for the learning rate.
        :param architecture: str, the architecture types.
        :param batch_size: int, the dimension of the batch size.
        :param optimizer: str, the optimizer type, choices
            ['Adadelta', 'Adagrad', 'Adam', 'Adamax', 'ASGD', 'RMSprop', 'SGD']
        :param lr_at_plateau: bool, protocol to decrease the learning rate.
        :param reduction_factor, int, the factor which we use to reduce the learning rate.
        :param validation_check: bool, if we want to keep track of validation loss as a stopping criterion.
        :param num_iterations: max number of iterations
        :param sensitive_learning_rate:
        :param reward_decay:
        :param weight_decay:
        :param allow_resume:
        :param randomize_checkpoint_path: bool, path of output file
        :param avoid_checkpoint_override: bool, default False
        :param record_loss_every: number of iterations at which we record the loss
        :param checkpoint_every: save the model every checkpoint_every iterations
        :param time: default 0
        :param num_val_samples: int, max number of examples in evaluation
        """
        self.learning_rate = learning_rate
        self.architecture = architecture
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.lr_at_plateau = lr_at_plateau
        self.reduction_factor = reduction_factor
        self.validation_check = validation_check
        self.num_iterations = num_iterations
        self.sensitive_learning_rate = sensitive_learning_rate
        self.reward_decay = reward_decay
        self.weight_decay = weight_decay
        self.allow_resume = allow_resume
        self.randomize_checkpoint_path = randomize_checkpoint_path
        self.avoid_checkpoint_override = avoid_checkpoint_override
        self.record_loss_every = record_loss_every
        self.checkpoint_every = checkpoint_every
        self.time = time
        self.num_val_samples = num_val_samples
        self.early_stopping = early_stopping
        self.previous_epochs = previous_epochs
        self.min_epochs = min_epochs
        self.max_epochs = max_epochs
        self.n_checkpoint_every_epoch = n_checkpoint_every_epoch
        self.n_record_loss_every_epoch = n_record_loss_every_epoch


class ArchitectureHyperParameters(object):
    """ """
    def __init__(self,
                 model_type='PG',
                 train_program_generator=True,
                 train_execution_engine=True,
                 baseline_train_only_rnn=False,
                 program_generator_start_from=None,  # starts from existing checkpoint
                 execution_engine_start_from=None,
                 baseline_start_from=None,
                 rnn_wordvec_dim=300,  # RNN options for PG
                 rnn_hidden_dim=256,
                 rnn_num_layers=2,
                 rnn_dropout=0,
                 rnn_attention=True,
                 module_stem_num_layers=2,
                 module_stem_subsample_layers=[],  # it must be a list of ints
                 module_stem_batchnorm=0,
                 module_dim=128,
                 stem_dim=64,
                 module_residual=1,
                 module_batchnorm=0,
                 module_intermediate_batchnorm=0,
                 use_color=0,
                 nmn_type='chain1',
                 classifier_proj_dim=512,
                 classifier_downsample='maxpool2',
                 classifier_fc_dims=[1024],
                 classifier_batchnorm=0,
                 classifier_dropout=0.,
                 ):

        """
        :param model_type: choices=['RTfilm', 'Tfilm', 'FiLM', 'PG', 'EE',
            'PG+EE', 'LSTM', 'CNN+LSTM', 'CNN+LSTM+SA', 'Hetero',
            'MAC', 'TMAC', 'SimpleNMN', 'RelNet', 'SHNMN', 'ConvLSTM']
        :param train_program_generator:
        :param train_execution_engine:
        :param baseline_train_only_rnn:
        :param program_generator_start_from:
        :param execution_engine_start_from:
        :param baseline_start_from:
        :param rnn_wordvec_dim:
        :param rnn_hidden_dim:
        :param rnn_num_layers:
        :param rnn_dropout:
        :param rnn_attention:
        :param module_stem_num_layers:
        :param module_stem_subsample_layers: it must be a list of ints
        :param module_stem_batchnorm:
        :param module_dim:
        :param stem_dim:
        :param module_residual:
        :param module_batchnorm:
        :param module_intermediate_batchnorm:
        :param use_color:
        :param nmn_type:
        :param classifier_proj_dim: int,
        :param classifier_downsample: str, choice in ['maxpool2', 'maxpool3', 'maxpool4',
            'maxpool5', 'maxpool7', 'maxpoolfull', 'none', 'avgpool2', 'avgpool3',
            'avgpool4', 'avgpool5', 'avgpool7', 'avgpoolfull', 'aggressive']
        :param classifier_fc_dims: list of int,
        :param classifier_batchnorm: int,
        :param classifier_dropout: int or list of float(?)
        """
        self.model_type = model_type
        self.train_program_generator = train_program_generator
        self.train_execution_engine = train_execution_engine
        self.baseline_train_only_rnn = baseline_train_only_rnn
        self.program_generator_start_from = program_generator_start_from  # starts from existing checkpoint
        self.execution_engine_start_from = execution_engine_start_from
        self.baseline_start_from = baseline_start_from
        self.rnn_wordvec_dim = rnn_wordvec_dim  # RNN options for PG
        self.rnn_hidden_dim = rnn_hidden_dim
        self.rnn_num_layers = rnn_num_layers
        self.rnn_dropout = rnn_dropout
        self.rnn_attention = rnn_attention
        self.module_stem_num_layers = module_stem_num_layers
        self.module_stem_subsample_layers = module_stem_subsample_layers
        self.module_stem_batchnorm = module_stem_batchnorm
        self.module_dim = module_dim
        self.stem_dim = stem_dim
        self.module_residual = module_residual
        self.module_batchnorm = module_batchnorm
        self.module_intermediate_batchnorm = module_intermediate_batchnorm
        self.use_color = use_color
        self.nmn_type = nmn_type
        self.classifier_proj_dim = classifier_proj_dim
        self.classifier_downsample = classifier_downsample
        self.classifier_fc_dims = classifier_fc_dims
        self.classifier_batchnorm = classifier_batchnorm
        self.classifier_dropout = classifier_dropout


class CNNHyperParameters(object):
    def __init__(self,
                 cnn_res_block_dim=128,
                 cnn_num_res_blocks=0,
                 cnn_proj_dim=512,
                 cnn_pooling='maxpool2'):
        """
        :param cnn_res_block_dim: int,
        :param cnn_num_res_blocks: int,
        :param cnn_proj_dim: int,
        :param cnn_pooling: str, choice in ['none', 'maxpool2']
        """
        self.cnn_res_block_dim = cnn_res_block_dim
        self.cnn_num_res_blocks = cnn_num_res_blocks
        self.cnn_proj_dim = cnn_proj_dim
        self.cnn_pooling = cnn_pooling


class StackedAttentionHyperParameters(object):
    def __init__(self,
                 stacked_attn_dim=512,
                 num_stacked_attn=2,
                 ):
        """
        :param stacked_attn_dim: int,
        :param num_stacked_attn: int,
        """
        self.stacked_attn_dim = stacked_attn_dim
        self.num_stacked_attn = num_stacked_attn


class FiLMHyperParameters(object):
    def __init__(self,
                 set_execution_engine_eval=0,
                 program_generator_parameter_efficient=1,
                 rnn_output_batchnorm=0,
                 bidirectional=0,
                 encoder_type='gru',
                 decoder_type='linear',
                 gamma_option='linear',
                 gamma_baseline=1,
                 num_modules=4,
                 module_stem_kernel_size=[3],
                 module_stem_stride=[1],
                 module_stem_padding=None,
                 module_num_layers=1,
                 module_batchnorm_affine=0,
                 module_dropout=5e-2,
                 module_input_proj=1,
                 module_kernel_size=3,
                 condition_method='bn-film',
                 condition_pattern=[],
                 use_gamma=1,
                 use_beta=1,
                 use_coords=1,
                 grad_clip=0,
                 debug_every=float('inf'),
                 print_verbose_every=float('inf'),
                 film_use_attention=0):
        """
        Initialization parameters for the FiLM architecture
        :param encoder_type: str, choices in ['linear', 'gru', 'lstm', 'null']
        :param decoder_type: str choices in ['linear', 'gru', 'lstm']
        :param gamma_option: str choices in ['linear', 'sigmoid', 'tanh', 'exp']
        :param condition_method: str, choices in['nothing', 'block-input-film', 'block-output-film',
            'bn-film', 'concat', 'conv-film', 'relu-film']
        :param condition_pattern: # type=parse_int_list)  # List of 0/1's (len = # FiLMs)
        :param use_coords:  0: none, 1: low usage, 2: high usage
        :param grad_clip:  <= 0 for no grad clipping
        :param debug_every: inf for no pdb
        :param module_num_layers: only mnl=1 currently implemented
        :param module_input_proj: Inp conv kernel size (0 for None)
        :param print_verbose_every: inf for min print
        :param module_batchnorm_affine: int, 1 overrides other factors
        """
        self.set_execution_engine_eval = set_execution_engine_eval
        self.program_generator_parameter_efficient = program_generator_parameter_efficient
        self.rnn_output_batchnorm = rnn_output_batchnorm
        self.bidirectional = bidirectional
        self.encoder_type = encoder_type
        self.decoder_type = decoder_type
        self.gamma_option = gamma_option
        self.gamma_baseline = gamma_baseline
        self.num_modules = num_modules
        self.module_stem_kernel_size = module_stem_kernel_size
        self.module_stem_stride = module_stem_stride
        self.module_stem_padding = module_stem_padding
        self.module_num_layers = module_num_layers
        self.module_batchnorm_affine = module_batchnorm_affine
        self.module_dropout = module_dropout
        self.module_input_proj = module_input_proj
        self.module_kernel_size = module_kernel_size
        self.condition_method = condition_method
        self.condition_pattern = condition_pattern
        self.use_gamma = use_gamma
        self.use_beta = use_beta
        self.use_coords = use_coords
        self.grad_clip = grad_clip
        self.debug_every = debug_every
        self.print_verbose_every = print_verbose_every
        self.film_use_attention = film_use_attention


class MACHyperParameters(object):
    def __init__(self,
                 mac_write_unit='original',
                 mac_read_connect='last',
                 mac_vib_start=0,
                 mac_vib_coof=0.,
                 mac_use_self_attention=1,
                 mac_use_memory_gate=1,
                 mac_nonlinearity='ELU',
                 mac_question2output=1,
                 mac_question_embedding_dropout=0.08,
                 mac_stem_dropout=0.18,
                 mac_memory_dropout=0.15,
                 mac_read_dropout=0.15,
                 mac_use_prior_control_in_control_unit=0,
                 variational_embedding_dropout=0.15,
                 mac_embedding_uniform_boundary=1.,
                 hard_code_control=True,
                 exponential_moving_average_weight=1.):
        """Initialization parameters
            :param mac_write_unit: str, default 'original'
            :param mac_read_connect: str, default 'last'
            :param mac_vib_start: float, default 0,
            :param mac_vib_coof: float, default 0,
            :param mac_use_self_attention: int, default 1
            :param mac_use_memory_gate: int, default 1
            :param mac_nonlinearity: str, default 'ELU'
            :param mac_question2output: int, default 1
            :param mac_question_embedding_dropout: float, default 0.08
            :param mac_stem_dropout: float, default 0.18
            :param mac_memory_dropout: float, default 0.15
            :param mac_read_dropout: float, default 0.15
            :param mac_use_prior_control_in_control_unit: int 0
            :param variational_embedding_dropout: float, default 0.15
            :param mac_embedding_uniform_boundary: float, default 1.
            :param hard_code_control: bool, default True
            :param exponential_moving_average_weight: float, default 1.
        """
        self.mac_write_unit = mac_write_unit
        self.mac_read_connect = mac_read_connect
        self.mac_vib_start = mac_vib_start
        self.mac_vib_coof = mac_vib_coof
        self.mac_use_self_attention = mac_use_self_attention
        self.mac_use_memory_gate = mac_use_memory_gate
        self.mac_nonlinearity = mac_nonlinearity
        self.mac_question2output = mac_question2output
        self.mac_question_embedding_dropout = mac_question_embedding_dropout
        self.mac_stem_dropout = mac_stem_dropout
        self.mac_memory_dropout = mac_memory_dropout
        self.mac_read_dropout = mac_read_dropout
        self.mac_use_prior_control_in_control_unit = mac_use_prior_control_in_control_unit
        self.variational_embedding_dropout = variational_embedding_dropout
        self.mac_embedding_uniform_boundary = mac_embedding_uniform_boundary
        self.hard_code_control = hard_code_control
        self.exponential_moving_average_weight = exponential_moving_average_weight


class NMNFiLM2HyperParameters(object):  # NMNFilm2 options
    def __init__(self,
                 nmnfilm2_sharing_params_patterns=[0, 0],
                 nmn_use_film=0,
                 nmn_use_simple_block=0):
        """
        :param nmnfilm2_sharing_params_patterns: list of int
        :param nmn_use_film: int
        :param nmn_use_simple_block: int
        """
        self.nmnfilm2_sharing_params_patterns = nmnfilm2_sharing_params_patterns
        self.nmn_use_film = nmn_use_film
        self.nmn_use_simple_block = nmn_use_simple_block


class SHNMNHyperParameters(object):
    def __init__(self,
                 model_type='soft',
                 use_module='residual',
                 tau_init='random',
                 model_bernoulli=0.5,
                 alpha_init='uniform',
                 hard_code_alpha=True,
                 hard_code_tau=True,
                 feature_dim=64,
                 module_dim=64,
                 module_kernel_size=3,
                 stem_dim=64,
                 stem_num_layers=2,
                 stem_subsample_layers=[],
                 stem_kernel_size=[3],
                 stem_padding=None,
                 stem_batchnorm=0,
                 classifier_fc_layers=[1024],
                 classifier_proj_dim=512,
                 classifier_downsample="maxpoolfull",
                 classifier_batchnorm=0,
                 num_modules=3,
                 separated_stem=False,
                 separated_module=True,
                 separated_classifier=True
                 ):
        """
        :param shnmn_type: str, choice in ['hard', 'soft']
        :param use_module: str, choice in ['conv', 'find', 'residual', 'mixed',
            'asymmetric_residual', 'mixed_find']
        :param tau_init: str, choice in  ['random', 'tree', 'chain',
            'chain_with_shortcuts', 'chain_with_shortcuts_flipped']
        :param model_bernoulli: float,
        :param alpha_init: str, choice in ['xavier_uniform', 'constant',
            'uniform', 'correct', 'correct_xry', 'correct_rxy']
        :param hard_code_alpha:
        :param hard_code_tau:
        :param separated_stem: we want stem to be specialized
        :param separated_module: this works only for find, if we want to separate some of the representation
        :param separated_classifier: if true, we will have a number of classifier equal to number of modules
        """
        self.model_type = model_type
        self.use_module = use_module
        self.tau_init = tau_init
        self.model_bernoulli = model_bernoulli
        self.alpha_init = alpha_init
        self.hard_code_alpha = hard_code_alpha
        self.hard_code_tau = hard_code_tau
        self.feature_dim = feature_dim
        self.module_dim = module_dim
        self.module_kernel_size = module_kernel_size
        self.stem_dim = stem_dim
        self.stem_num_layers = stem_num_layers
        self.stem_subsample_layers = stem_subsample_layers
        self.stem_kernel_size = stem_kernel_size
        self.stem_padding = stem_padding
        self.stem_batchnorm = stem_batchnorm
        self.classifier_fc_layers = classifier_fc_layers
        self.classifier_proj_dim = classifier_proj_dim
        self.classifier_downsample = classifier_downsample
        self.classifier_batchnorm = classifier_batchnorm
        self.num_modules = num_modules
        self.separated_stem = separated_stem
        self.separated_module = separated_module
        self.separated_classifier = separated_classifier


architecture_dict  = {"SHNMN": SHNMNHyperParameters,
                      "CNN": CNNHyperParameters,
                      "StackedAttention": StackedAttentionHyperParameters,
                      "FiLM": FiLMHyperParameters,
                      "MAC": MACHyperParameters,
                      "NMNFiLM2": NMNFiLM2HyperParameters}


# we are supposed to return
# (question, image, feats, answer, program_seq, program_json)
class Dataset(object):
    """ Here we save the dataset specific related to each experiment.
    The name of the dataset. Its hyper-parameters can be recovered using
    a pd.DataFrame, for the training or test case.
    We specify the dimensions of the output.
    """
    def __init__(self,
                 dataset_id="0",
                 dataset_split="train",
                 dataset_id_path="",
                 experiment_case=1,
                 image_size=28,
                 n_training=210000,
                 policy=None):
        """
        :param dataset_id: str, identifier of the dataset used
        :param dataset_split: str, "train", "validation", or "test"
        :param dataset_id_path: str, path to the id
        :param experiment_case: int, the learning paradigm
        :param image_size: str, size of the image
        :param n_training: int, number of training examples
        :param policy: int, if None, we use the entire training set (divided by train and val)
        """
        self.dataset_id = dataset_id
        self.dataset_split = dataset_split
        self.dataset_id_path = dataset_id_path
        self.experiment_case = experiment_case
        self.image_size = image_size
        self.n_training = n_training
        self.policy = policy


class Experiment(object):
    def __init__(self,
                 id,
                 output_path,
                 train_completed,
                 method_type,
                 hyper_opt,
                 hyper_arch,
                 hyper_method,
                 dataset):
        self.id = id
        self.output_path = output_path
        self.train_completed = train_completed
        self.method_type = method_type
        self.hyper_opt = hyper_opt
        self.hyper_arch = hyper_arch
        self.hyper_method = hyper_method
        self.dataset = dataset


def generate_dataset_json(df_tr,
                          df_ts,
                          output_data="./"):
    """ This function is called to make your train.json file or append to it.
    You should change the loops for your own usage.
    The info variable is a dictionary that first reads the json file if there exists any,
    appends your new experiments to it, and dumps it into the json file again
    We need to pass at training the new type of df_tr and df_ts here
    :param df_tr: DataFrame for the objects at training
    :param df_ts: DataFrame for the objects at test
    :param output_data: str, data folder path
    """
    # TODO: include the dataset path, generate everything from there
    # WARNING: if the *.json is empty it complains
    info = {}
    df_tr = df_tr.to_dict()
    df_ts = df_ts.to_dict()
    info_path = output_data + 'data_list.json'
    print(info_path)
    dirname = os.path.dirname(info_path)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
        idx_base = 0
    elif os.path.isfile(info_path):
        with open(info_path) as infile:
            info = json.load(infile)
            if info:  # it is not empty
                idx_base = int(list(info.keys())[-1]) + 1  # concatenate
            else:
                idx_base = 0
    else:  # we have the folder but the file is empty
        info = dict()
        idx_base = 0

    already_exists_flag = False

    for k_dataset in info.keys():
        for k_attribute in info[k_dataset]["dataset_train"].keys():  #  category, size
            compare_keys = True
            for key_json, key_new in zip(info[k_dataset]["dataset_train"][k_attribute].keys(),
                                         df_tr[k_attribute].keys()):
                compare_keys *= key_json == str(key_new)
                # we verified that these two are equivalent
        if not compare_keys:  #  not even the keys are comparable
            with open(info_path, 'w') as f:
                json.dump(info, f)
            raise ValueError(
                "The keys are not comparable \n between dataset %i and the given training keys" % k_dataset)

        # after verifying that the keys are identical
        # we verify that each item is identical

        tmp_flag = True  # the dataset is identical to k_dataset in the json file
        for key_json, key_new in zip(info[k_dataset]["dataset_train"].keys(),
                                     df_tr.keys()):
            for val_json, val_new in zip(info[k_dataset]["dataset_train"][key_json].values(),
                                         df_tr[key_new].values()):
                tmp_flag *= val_json == val_new  # the train is different for the k_dataset

        if tmp_flag:
            # for category, size, etc.
            for key_json, key_new in zip(info[k_dataset]["dataset_test"].keys(),
                                         df_ts.keys()):
                for val_json, val_new in zip(info[k_dataset]["dataset_test"][key_json].values(),
                                             df_ts[key_new].values()):
                    tmp_flag *= val_json == val_new

        if tmp_flag:  #  if it is true for both the dataset
            already_exists_flag = True
            # the dataset with this characteristics already exists
            # close the json
            with open(info_path, 'w') as f:
                json.dump(info, f)
            return

    if not already_exists_flag:
        info[idx_base] = {"dataset_name": "dataset_%i" % idx_base,
                          "dataset_path": output_data + "dataset_%i" % idx_base,
                          "dataset_train": df_tr,
                          "dataset_test": df_ts}

        print(info)

    with open(info_path, 'w') as f:
        json.dump(info, f)

    if not already_exists_flag:
        return info[idx_base]
    else:
        return None


def exp_exists(exp, info):
    """ This function checks if the experiment exists in your json file to avoid duplicate experiments.
    """
    # TODO: is this function called in other parts, except from generate_experiments?
    # do we want to put also the flag train_completed here, correct?
    dict_new = json.loads(json.dumps(exp, default=lambda o: o.__dict__))
    dict_new_wo_id = {i: dict_new[i]
                      for i in dict_new if (i != 'id' and i != 'output_path' and i != 'train_completed')}
    for idx in info:
        dict_old = info[idx]
        dict_old_wo_id = {i: dict_old[i]
                          for i in dict_old if (i != 'id' and i != 'output_path' and i != 'train_completed')}
        if dict_old_wo_id == dict_new_wo_id:
            return idx
    return False


def generate_experiments(output_path,
                         data_path):
    """ This function is called to make your train.json file or append to it.
    You should change the loops for your own usage.
    The info variable is a dictionary that first reads the json file if there exists any,
    appends your new experiments to it, and dumps it into the json file again
    """
    # TODO: include the dataset path, generate everything from there
    # WARNING: if the *.json is empty it complains
    info = {}

    info_path = output_path + 'train.json'
    dirname = os.path.dirname(info_path)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
        idx_base = 0
    elif os.path.isfile(info_path):
        with open(info_path) as infile:
            info = json.load(infile)
            if info:  # it is not empty
                idx_base = int(list(info.keys())[-1]) + 1  # concatenate
            else:
                idx_base = 0
    else:
        idx_base = 0

    # These loops indicate your experiments. Change them accordingly.
    for experiment_case_ in experiment_case_list:  # for each experiment type
        for method_type_ in method_type_list:
            for dataset_name_ in dataset_dict["dataset_name"]:
                for lr_ in lr_array:
                    for bs_ in batch_list:
                        dataset_ = Dataset(dataset_id=dataset_name_,
                                           dataset_id_path=join(data_path, dataset_name_),
                                           experiment_case=experiment_case_)

                        hyper_opt_ = OptimizationHyperParameters(learning_rate=lr_,
                                                                 batch_size=bs_)
                        hyper_arch_ = ArchitectureHyperParameters(model_type=method_type_)

                        hyper_method_ = architecture_dict[method_type_](**dict_method_type)
                        exp = Experiment(id=idx_base,
                                         output_path=output_path+'train_'+str(idx_base),
                                         train_completed=False,
                                         method_type=method_type_,
                                         hyper_opt=hyper_opt_,
                                         hyper_arch=hyper_arch_,
                                         hyper_method=hyper_method_,
                                         dataset=dataset_)
                        idx = exp_exists(exp, info)
                        if idx is not False:
                            print("The experiment already exists with id:", idx)
                            continue
                        s = json.loads(json.dumps(exp, default=lambda o: o.__dict__))
                        print(s)
                        info[str(idx_base)] = s
                        idx_base += 1

        # for each experiment type we have a correspondent json
        with open(info_path, 'w') as outfile:
            json.dump(info, outfile, indent=4)


def decode_exp(dct):
    """ When reading a json file, it is originally a dictionary
    which is hard to work with in other parts of the code.
    IF YOU ADD ANOTHER CLASS TO EXPERIMENT, MAKE SURE TO INCLUDE IT HERE.
    This function goes through the dictionary and turns it into an instance of Experiment class.
        :parameter dct: dictionary of parameters as saved in the *.json file.
        :returns exp: instance of the Experiment class.
    """
    dct_copy = dct.copy()
    hyper_dict = {'hyper_opt': OptimizationHyperParameters(),
                  'hyper_arch': ArchitectureHyperParameters(),
                  'hyper_method': architecture_dict[dct['method_type']](),
                  'dataset': Dataset()
                  }

    for hyper_name_, hyper_obj_ in hyper_dict.items():  # for all the elements in the dictionary
        for key in hyper_obj_.__dict__.keys():  # for everything that the defines the hyper_obj_ object
            if key in dct_copy[hyper_name_].keys():
                hyper_obj_.__setattr__(key, dct_copy[hyper_name_][key])
    dct_copy.update(hyper_dict)

    exp = Experiment(**dct_copy)

    return exp


def get_experiment(output_path, id):
    """
    This function is called when you want to get the details of your experiment
    given the index (id) and the path to train.json
    """
    info_path = join(output_path, 'train.json')
    with open(info_path) as infile:
        trained = json.load(infile)
    opt = trained[str(id)]  # access to the experiment details through the ID
    exp = decode_exp(opt)   # return an Experiment object

    print('Retrieved experiment:')
    for key, val in exp.__dict__.items():
        if key.startswith('hyper') or key is 'dataset':  # hyper-parameters details
            print('%s: ' % key, val.__dict__)
        else:
            print('%s: ' % key, exp.__getattribute__(key))

    return exp

