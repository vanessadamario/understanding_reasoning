import sys
import json
import torch
from tqdm import tqdm
import os
import logging
import time
import h5py
import numpy as np
from torch.autograd import Variable
from os.path import join
import torch.nn.functional as F
from runs.utils import load_vocab, load_cpu, get_updated_args
from runs.data_loader import DataTorchLoader
from runs.train_loop import set_mode, check_accuracy

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def load_execution_engine(path,
                          model_type="SHNMN",
                          verbose=True,
                          query=False):
    checkpoint = load_cpu(path)
    # TODO: remember to change this
    # TODO depending on query or residual
    kwargs = checkpoint['execution_engine_kwargs']
    state = checkpoint['execution_engine_state']
    kwargs['verbose'] = verbose
    if model_type == 'FiLM':
        kwargs = get_updated_args(kwargs, FiLMedNet)
        model = FiLMedNet(**kwargs)
    elif model_type == 'EE':
        model = ModuleNet(**kwargs)
    elif model_type == 'MAC':
        kwargs.setdefault('write_unit', 'original')
        kwargs.setdefault('read_connect', 'last')
        kwargs.setdefault('noisy_controls', False)
        kwargs.pop('sharing_params_patterns', None)
        model = MAC(**kwargs)
    elif model_type == 'RelNet':
        model = RelationNet(**kwargs)
    elif model_type == 'SHNMN':
        if query:
            from runs.shnmn_query import SHNMN
        else:
            from runs.shnmn import SHNMN
        model = SHNMN(**kwargs)
    else:
        raise ValueError()
    cur_state = model.state_dict()
    # TODO: modified
    model.load_state_dict(state)
    return model, kwargs


def get_execution_engine(query=False, **kwargs):
    """ Load the model """
    # print("\nQUERY VALUE: ", query)
    if query:
        from runs.shnmn_query import SHNMN
    else:
        from runs.shnmn import SHNMN
    if kwargs["execution_engine_start_from"] is not None:
        ee, kwargs = load_execution_engine(
            kwargs["execution_engine_start_from"],
            kwargs["method_type"],
            query=query)
    else:
        ee = SHNMN(**kwargs)
    ee.to(device)
    return ee


def check_accuracy_test(opt, flag_out_of_sample, test_loader, dtype, ee, pg=None):
    if opt.dataset.experiment_case == 0:
        from runs.shnmn_query import SHNMN
        query_flag = True

    else:
        from runs.shnmn import SHNMN
        query_flag = False

    # if pg is None:
    #     pg = ee
    print("dtype", dtype)
    ee.to(device)
    ee.type(dtype)
    ee.eval()

    all_scores = []
    all_correct = []
    all_probs = []
    all_preds = []
    all_film_scores = []
    all_read_scores = []
    all_control_scores = []
    all_connections = []
    all_vib_costs = []
    num_correct, num_samples = 0, 0
    q_types = []

    start = time.time()
    for i__, batch in enumerate(tqdm(test_loader)):
        assert(not ee.training)
        feats, questions, answers = batch
        if query_flag:
            # we need as many num_correct as the size of the question
            if i__ == 0:
                num_correct = [0 for k_ in range(questions.shape[1])]
                all_correct = [0 for k_ in range(questions.shape[1])]
        if isinstance(questions, list):
            questions_var = questions[0].type(dtype).long()
            q_types += [questions[1].cpu().numpy()]
        else:
            questions_var = questions.type(dtype).long()
        feats_var = feats.type(dtype)

        questions_var = Variable(questions_var.to(device))
        feats_var = Variable(feats_var.to(device))

        kwargs = {}
        pos_args = [feats_var]
        if isinstance(ee, SHNMN):
            pos_args.append(questions_var)
        else:
            pos_args.append(programs_pred)

        scores = ee(*pos_args, **kwargs)
        if query_flag:
            preds_list = []
            for id_s, s_ in enumerate(scores):
                _, preds = s_.data.cpu().max(1)
                preds_list.append(preds)  # .cpu().detach().numpy())
                if i__ == 0:
                    all_correct[id_s] = preds == answers[:, id_s]
                else:
                    all_correct[id_s] = torch.cat((all_correct[id_s],
                                                   preds == answers[:, id_s]))

        else:
            probs = F.softmax(scores, dim=1)
            _, preds = scores.data.cpu().max(1)

            all_scores.append(scores.data.cpu().clone())
            all_probs.append(probs.data.cpu().clone())
            all_preds.append(preds.cpu().clone())
            all_correct.append(preds == answers)
        # if isinstance(pg, FiLMGen) and pg.scores is not None:
        #     all_film_scores.append(pg.scores.data.cpu().clone())
        # if isinstance(ee, MAC):
        #    all_control_scores.append(ee.control_scores.data.cpu().clone())
        #     all_read_scores.append(ee.read_scores.data.cpu().clone())
        # if hasattr(ee, 'vib_costs'):
        #    all_vib_costs.append(ee.vib_costs.data.cpu().clone())
        # if hasattr(ee, 'connections') and ee.connections:
        #     all_connections.append(torch.cat([conn.unsqueeze(1) for conn in ee.connections], 1).data.cpu().clone())

        if query_flag:
            if answers[0] is not None:
                for j__, p_ in enumerate(preds_list):
                    num_correct[j__] += np.array(p_ == answers[:, j__]).sum()
            num_samples += preds_list[0].shape[0]  # one of the attributes
            acc = np.array([float(num_correct_attr)/num_samples for num_correct_attr in num_correct])
            print(acc)
        else:
            if answers[0] is not None:
                # print(answers, preds)
                num_correct += np.array(preds == answers).sum()
            num_samples += preds.size(0)
            acc = float(num_correct) / num_samples

            print('Got %d / %d = %.2f correct' % (num_correct, num_samples, 100 * acc))
            print('%.2fs to evaluate' % (time.time() - start))

    if all_control_scores:
        max_len = max(cs.size(2) for cs in all_control_scores)
        for i in range(len(all_control_scores)):
            tmp = torch.zeros(
                (all_control_scores[i].size(0), all_control_scores[i].size(1), max_len))
            tmp[:, :, :all_control_scores[i].size(2)] = all_control_scores[i]
            all_control_scores[i] = tmp

    filename = "oos_output.h5" if flag_out_of_sample else "output.h5"
    output_path = join(opt.output_path, filename)
    print(len(all_correct))
    print(all_correct[0].shape)

    print('Writing output to "%s"' % output_path)
    with h5py.File(output_path, 'w') as fout:
        if not query_flag:
            fout.create_dataset('scores', data=torch.cat(all_scores, 0).numpy())
            fout.create_dataset('probs', data=torch.cat(all_probs, 0).numpy())
            fout.create_dataset('correct', data=torch.cat(all_correct, 0).numpy())
        else:
            fout.create_dataset('correct', data=torch.stack(all_correct, 0).numpy().T)
        if all_film_scores:
            fout.create_dataset('film_scores', data=torch.cat(all_film_scores, 1).numpy())
        if all_vib_costs:
            fout.create_dataset('vib_costs', data=torch.cat(all_vib_costs, 0).numpy())
        if all_read_scores:
            fout.create_dataset('read_scores', data=torch.cat(all_read_scores, 0).numpy())
        if all_control_scores:
            fout.create_dataset('control_scores', data=torch.cat(all_control_scores, 0).numpy())
        if all_connections:
            fout.create_dataset('connections', data=torch.cat(all_connections, 0).numpy())

    # Save FiLM param stats
    # if args.output_program_stats_dir:
    #    if not os.path.isdir(args.output_program_stats_dir):
    #         os.mkdir(args.output_program_stats_dir)
    #     gammas = all_programs[:,:,:pg.module_dim]
    #    betas = all_programs[:,:,pg.module_dim:2*pg.module_dim]
    #     gamma_means = gammas.mean(0)
    #     torch.save(gamma_means, os.path.join(args.output_program_stats_dir, 'gamma_means'))
    #     beta_means = betas.mean(0)
    #     torch.save(beta_means, os.path.join(args.output_program_stats_dir, 'beta_means'))
    #     gamma_medians = gammas.median(0)[0]
    #     torch.save(gamma_medians, os.path.join(args.output_program_stats_dir, 'gamma_medians'))
    #     beta_medians = betas.median(0)[0]
    #     torch.save(beta_medians, os.path.join(args.output_program_stats_dir, 'beta_medians'))

    #     Note: Takes O(10GB) space
    #     torch.save(gammas, os.path.join(args.output_program_stats_dir, 'gammas'))
    #     torch.save(betas, os.path.join(args.output_program_stats_dir, 'betas'))

    # if args.output_preds is not None:
    #     vocab = load_vocab(args)
    #     all_preds_strings = []
    #     for i in range(len(all_preds)):
    #         all_preds_strings.append(vocab['answer_idx_to_token'][all_preds[i]])
    #     save_to_file(all_preds_strings, args.output_preds)

    # if args.debug_every <= 1:
    #     pdb.set_trace()
    return acc


def check_and_test(opt, flag_out_of_sample, use_gpu=True):
    # this must happen in the main.py

    # TODO remove comment, this is for testing the load function
    # if not opt.train_completed:
    #     raise ValueError("Experiment %i did not train." % opt.id)

    split_name = "oos_test" if flag_out_of_sample else "test"
    test_loader = DataTorchLoader(opt, split=split_name)

    vocab = load_vocab(join(opt.dataset.dataset_id_path, "vocab.json"))
    kkwargs_exec_engine_ = opt.hyper_method.__dict__.copy()
    kkwargs_exec_engine_["execution_engine_start_from"] = join(opt.output_path,
                                                               "model.best")
    kkwargs_exec_engine_["vocab"] = vocab
    kkwargs_exec_engine_["method_type"] = opt.method_type
    sys.stdout.flush()
    if opt.dataset.experiment_case == 0:
        query = True
    else:
        query = False

    ee = get_execution_engine(query=query, **kkwargs_exec_engine_)
    if use_gpu:
        dtype = torch.cuda.FloatTensor
    else:
        dtype = torch.FloatTensor

    test_acc = check_accuracy_test(opt, flag_out_of_sample, test_loader, dtype, ee)
    print("\nTest accuracy: ",  test_acc)
    return test_acc