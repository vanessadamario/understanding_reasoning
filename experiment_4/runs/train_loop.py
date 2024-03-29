import os
import sys
import json
import torch
import logging
import time
import numpy as np
from torch.autograd import Variable
from os.path import join
import torch.nn.functional as F
from runs.shnmn import SHNMN
from runs.utils import load_vocab, load_execution_engine


# here we need to the the same of the train_loop function

logger = logging.getLogger(__name__)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def get_execution_engine(**kwargs):
    """ Load the model """
    # consider the case where you want to load
    # print(kwargs)
    #     if args.execution_engine_start_from is not None:
    #         ee, kwargs = vr.utils.load_execution_engine(
    #             args.execution_engine_start_from, model_type=args.model_type)
    print(kwargs)
    load_last = True
    print('shaping flags')
    print(kwargs['shaping'])
    print(kwargs['path_shaping'])
    print(kwargs['module_per_subtask'])
    if kwargs['load_model']:
        # if we load we do not care about path shaping anymore
        print("Loading model")
        files = os.listdir(kwargs['output_path'])
        load_last = 'model' in files  # TODO:
        if load_last:  # standard brute force
            ee, kwargs = load_execution_engine(join(kwargs['output_path'], 'model'),
                                               model_type="SHNMN")
        else:  # early stopping
            ee, kwargs = load_execution_engine(join(kwargs['output_path'], 'model.best'),
                                               model_type="SHNMN")
            # save the kwargs shaping in the model and check that it exists
        print(kwargs)
    else:
        # ee = torch.jit.script(SHNMN(**kwargs))
        ee = SHNMN(**kwargs)
    ee.to(device)
    ee.train()

    if kwargs['shaping']:
        # TODO: evaluation mode after check_accuracy
        for id_st_, st_ in enumerate(ee.stem):
            if isinstance(st_, torch.nn.BatchNorm2d):
                print('evaluation mode for', id_st_)
                ee.stem[id_st_].eval()

    return ee, load_last


def set_mode(mode, models):
    assert mode in ['train', 'eval']
    for m in models:
        if m is None:
            continue
        if mode == 'train':
            m.train()
            if m.__dict__['shaping']:
                for id_st_, st_ in enumerate(m.stem):
                    if isinstance(st_, torch.nn.BatchNorm2d):
                        print('evaluation mode for', id_st_)
                        m.stem[id_st_].eval()
        if mode == 'eval':
            m.eval()


def get_state(m):
    if m is None:
        return None
    state = {}
    for k, v in m.state_dict().items():
        state[k] = v.clone()
    return state


def check_accuracy(opt, execution_engine, loader, test=False):
    max_n_samples = opt.hyper_opt.num_val_samples if not test else None
    set_mode('eval', [execution_engine])
    num_correct, num_samples = 0, 0
    # print("\n\nEvaluation")
    for batch in loader:
        (feats, questions, answers) = batch
        # print(questions)
        # print(feats.shape)
        if isinstance(questions, list):
            questions = questions[0]
        questions_var = questions.to(device)
        feats_var = feats.to(device)
        scores = None  # Use this for everything but PG
        if opt.method_type in ['SimpleNMN', 'SHNMN']:
            scores = execution_engine(feats_var, questions_var)
        else:
            raise NotImplementedError('model ', opt.method_type, ' check_accuracy not implemented')
        if scores is not None:
            _, preds = scores.data.cpu().max(1)
            num_correct += (preds == answers).sum()
            num_samples += preds.size(0)

        if max_n_samples is not None and num_samples >= max_n_samples:
            # print("We are in the condition to exit")
            # sys.stdout.flush()
            # print(num_samples)
            # sys.stdout.flush()
            break
    set_mode('train', [execution_engine])
    acc = float(num_correct) / num_samples
    print('accuracy: ', acc)
    print("num check samples", num_samples)

    return acc


def train_loop(opt, train_loader, val_loader, load=False, 
               shaping=False, path_shaping=None,
               module_per_subtask=False):
    print("We load the model: ", load)
    init_training = time.time()
    vocab = load_vocab(join(opt.dataset.dataset_id_path, "vocab.json"))
    execution_engine, ee_kwargs, ee_optimizer = None, None, None
    # vocabulary and execution engine

    stats = {
        'train_losses': [], 'train_rewards': [], 'train_losses_ts': [],
        'train_accs': [], 'val_accs': [], 'val_accs_ts': [], 'alphas': [], 'grads': [],
        'best_val_acc': -1, 'model_t': 0, 'model_epoch': 0,
        'p_tree': [], 'tree_loss': [], 'chain_loss': [], 'best_model_t': 0, 'best_model_epoch': 0
    }
    kkwargs_exec_engine_ = opt.hyper_method.__dict__
    kkwargs_exec_engine_["vocab"] = vocab
    kkwargs_exec_engine_["load_model"] = load
    kkwargs_exec_engine_["shaping"] = shaping
    kkwargs_exec_engine_["path_shaping"] = path_shaping
    kkwargs_exec_engine_['module_per_subtask'] = module_per_subtask
    if load:
        kkwargs_exec_engine_["output_path"] = opt.output_path

    if opt.method_type in ['SHNMN']:
        # TODO load the model if it exists already
        execution_engine, load_last = get_execution_engine(**kkwargs_exec_engine_)  # TODO: new
    logger.info('Here is the conditioned network:')
    logger.info(execution_engine)

    # print(execution_engine)
    # print(execution_engine.__dict__.keys())
    # return

    optim_method = getattr(torch.optim, opt.hyper_opt.optimizer)

    if execution_engine:
        # separate learning rate for p(model) for the stochastic tree NMN
        base_parameters = []
        sensitive_parameters = []
        logger.info("PARAMETERS:")
        for name, param in execution_engine.named_parameters():
            print(name)
            if name == 'question_embeddings.weight':
                print('here')
                # continue
            if not param.requires_grad:
                print('no gradient', name)
                print('\n')
                continue
            logger.info(name)
            if name.startswith('tree_odds') or name.startswith('alpha'):
                sensitive_parameters.append(param)
            else:
                base_parameters.append(param)
        logger.info("SENSITIVE PARAMS ARE: {}".format(sensitive_parameters))
        ee_optimizer = optim_method([{'params': sensitive_parameters,
                                      'lr': opt.hyper_opt.sensitive_learning_rate,
                                      'weight_decay': 0.0},
                                     {'params': base_parameters}],
                                    lr=opt.hyper_opt.learning_rate,
                                    weight_decay=opt.hyper_opt.weight_decay)

    compute_start_time = time.time()
    loss_fn = torch.nn.CrossEntropyLoss().to(device)

    if load:
        with open(join(opt.output_path, 'model.json'), 'r') as f:
            checkpoint = json.load(f)
        for key in list(stats.keys()):
            if key in checkpoint:
                stats[key] = checkpoint[key]
        # stats['model_epoch'] -= 1

    t, epoch, reward_moving_average = stats['model_t'], stats['model_epoch'], 0
    best_model_t, best_model_epoch = stats['best_model_t'], stats['best_model_epoch']

    if not load_last:
        t = best_model_t
        epoch = best_model_epoch
        tr_ls_times = np.array(stats['train_losses_ts'])
        vl_ac_times = np.array(stats['val_accs_ts'])
        id_tr_ls_ts = np.sum(tr_ls_times <= t)
        id_vl_ac_ts = np.sum(vl_ac_times <= t)

        stats['train_losses'] = stats['train_losses'][:id_tr_ls_ts]
        stats['train_losses_ts'] = stats['train_losses_ts'][:id_tr_ls_ts]
        stats['val_accs_ts'] = stats['val_accs_ts'][:id_vl_ac_ts]
        stats['val_accs'] = stats['val_accs'][:id_vl_ac_ts]
        stats['train_accs'] = stats['train_accs'][:id_vl_ac_ts]

    num_checkpoints = 0
    num_recordloss = 0
    reward = None  # TODO figure out what this means
    running_loss = 0.0
    epoch_start_time = 0.0
    epoch_total_time = 0.0
    train_pass_total_time = 0.0
    val_pass_total_time = 0.0

    temp_old = time.time()
    satisfied_early_stop = False


    if opt.hyper_opt.early_stopping:
        n_iters_per_epoch = opt.dataset.n_training // opt.hyper_opt.batch_size
        max_iterations = opt.hyper_opt.max_epochs * n_iters_per_epoch
        checkpoint_every = n_iters_per_epoch // opt.hyper_opt.n_checkpoint_every_epoch  # 1/5 epoch
        record_loss = n_iters_per_epoch // opt.hyper_opt.n_record_loss_every_epoch
        print("\nCheckpoint accs: ", checkpoint_every)
        print("\nCheckpoint loss: ", record_loss)

    else:
        max_iterations = opt.hyper_opt.num_iterations
        checkpoint_every = opt.hyper_opt.checkpoint_every
        record_loss = opt.hyper_opt.record_loss_every

    print("Saving steps")
    print(max_iterations, checkpoint_every, record_loss)

    if not load_last:
        print("\nStarting point")
        print(t, epoch)
        print(len(stats['train_losses']), len(stats['train_losses_ts']), stats['train_losses_ts'][-1])
        print(len(stats['val_accs']), len(stats['val_accs_ts']), stats['val_accs_ts'][-1])
    else:
        print(stats)
    print("Start training")
    sys.stdout.flush()

    while t < max_iterations:  # this is the number of steps in tf
        epoch += 1
        print("Iteration: %i " % t)
        if (epoch > 0) and (opt.hyper_opt.time == 1):
            epoch_time = time.time() - epoch_start_time
            epoch_total_time += epoch_time
            logger.info('EPOCH PASS AVG TIME: ' + str(epoch_total_time / epoch), 'white')
            logger.info('Epoch Pass Time      : ' + str(epoch_time), 'white')
        epoch_start_time = time.time()

        logger.info('Starting epoch %d' % epoch)

        batch_start_time = time.time()

        for batch in train_loader:
            # print("Starting a new batch")
            sys.stdout.flush()
            t += 1
            (feats, questions, answers) = batch
            if t == 2:
                np.save('/om2/user/vanessad/understanding_reasoning/experiment_4/feats_check.npy', feats.detach().cpu().numpy())
                np.save('/om2/user/vanessad/understanding_reasoning/experiment_4/questions_check.npy', questions.detach().cpu().numpy())
                np.save('/om2/user/vanessad/understanding_reasoning/experiment_4/answers_check.npy', answers.detach().cpu().numpy())

            if isinstance(questions, list):
                questions = questions[0]
            questions_var = Variable(questions.to(device))
            feats_var = Variable(feats.to(device))
            answers_var = Variable(answers.to(device))
            time1 = time.time()
            if opt.method_type in ['SimpleNMN', 'SHNMN']:
                # Train execution engine with ground-truth programs
                ee_optimizer.zero_grad()
                # print("zero grad", time.time() - time_i__)
                # sys.stdout.flush()
            scores = execution_engine(feats_var, questions_var)
            # print("scores", time.time() - time_i__)
            # sys.stdout.flush()
            if opt.hyper_method.model_type == 'hard':
                tree_loss = loss_fn(execution_engine.tree_scores, answers_var)
                chain_loss = loss_fn(execution_engine.chain_scores, answers_var)
            loss = loss_fn(scores, answers_var)
            # print("compute loss", time.time() - time_i__)
            loss.backward()
            # print("loss back", time.time() - time_i__)
            # sys.stdout.flush()
            # record alphas and gradients and p(model) here : DEBUGGING
            if opt.method_type == 'SHNMN' and opt.hyper_method.model_type == 'hard':
                p_tree = F.sigmoid(execution_engine.tree_odds).item()
                if t % record_loss == 0:
                    print('p_tree:', p_tree)
                    stats['p_tree'].append(p_tree)
                    stats['tree_loss'].append(tree_loss.item())
                    stats['chain_loss'].append(chain_loss.item())

            if opt.hyper_method.model_type == 'SHNMN' and not opt.hyper_method.hard_code_alpha:  # TODO check alpha here
                alphas = [execution_engine.alpha[i]
                          for i in range(3)]
                alphas = [t.data.cpu().numpy() for t in alphas]
                alphas_grad = execution_engine.alpha.grad.data.cpu().numpy()
                if t % 10 == 0:
                    for i, (alphas_i, alphas_i_grad) in enumerate(zip(alphas, alphas_grad)):
                        print('data_%d: %s' % (i, " ".join(['{:.3f}'.format(float(x)) for x in alphas_i])))
                        print('grad_%d: %s' % (i, " ".join(['{:.3f}'.format(float(x)) for x in alphas_i_grad])))
                        stats['alphas_{}'.format(i)].append(alphas_i.tolist())
                        stats['alphas_{}_grad'.format(i)].append(alphas_i_grad.tolist())
                # print("not hard", time.time() - time_i__)
                # sys.stdout.flush()
            # print("time scores by ee", time.time() - time_i__)
            ee_optimizer.step()
            # time_e__ = time.time()
            # print("TIME SINGLE BATCH: ")
            # sys.stdout.flush()
            # print(time_e__ - time_i__)
            # sys.stdout.flush()

            if t % record_loss == 0:
                running_loss += loss.item()
                num_recordloss += 1
                # print(t)
                # print('Time for 10 iterations')
                # print(time.time() - batch_start_time)
                batch_start_time = time.time()
                # print("\n\nNth CHECKPOINT LOSS: %i" % num_recordloss)
                avg_loss = running_loss / opt.hyper_opt.record_loss_every
                logger.info("{} {:.5f} {:.5f} {:.5f}".format(t,
                                                             time.time() - batch_start_time, time.time() - compute_start_time,
                                                             loss.item()))

                # for id_st_, st_ in enumerate(execution_engine._modules['stem']):
                #     print(st_)
                #     if isinstance(st_, torch.nn.BatchNorm2d):
                #         torch.save(st_, join(opt.output_path, 'BN_iter_%i_%i' % (t, id_st_)))

                stats['train_losses'].append(avg_loss)
                stats['train_losses_ts'].append(t)
                if reward is not None:
                    stats['train_rewards'].append(reward.item())
                running_loss = 0.0
            else:
                running_loss += loss.item()

            if t % checkpoint_every == 0:
                num_checkpoints += 1
                # print("\n\nNth CHECKPOINT ACCURACY: %i" % num_checkpoints)

                temp_time = time.time()

                # print("\nTIME: %f \n" %(temp_time - temp_old))
                sys.stdout.flush()

                temp_old = temp_time
                logger.info('Checking training accuracy ... ')
                start = time.time()
                print('Checking train accuracy ... ')
                train_acc = check_accuracy(opt, execution_engine, train_loader)
                print('Done')
                train_pass_time = (time.time() - start)
                train_pass_total_time += train_pass_time
                logger.info('TRAIN PASS AVG TIME:' + str(train_pass_total_time /
                                                         num_checkpoints))
                logger.info('Train Pass Time : ' + str(train_pass_time))
                logger.info('train accuracy is {}'.format(train_acc))
                logger.info('Checking validation accuracy ...')
                start = time.time()

                print('Checking validation accuracy ... ')
                val_acc = check_accuracy(opt, execution_engine, val_loader)
                print('Done')
                val_pass_time = (time.time() - start)
                val_pass_total_time += val_pass_time
                logger.info('VAL PASS AVG TIME: ' + str(val_pass_total_time /
                                                        num_checkpoints))
                logger.info('Val Pass Time: ' + str(val_pass_time))
                logger.info('val accuracy is {}'.format(val_acc))
                stats['train_accs'].append(train_acc)
                stats['val_accs'].append(val_acc)
                stats['val_accs_ts'].append(t)

                stats['model_t'] = t
                stats['model_epoch'] = epoch

                if not opt.hyper_opt.early_stopping:
                    ee_state = get_state(execution_engine)
                    checkpoint = {
                        'optimization_kwargs': opt.hyper_opt.__dict__,  # in case we change the lr
                        'execution_engine_kwargs': opt.hyper_method.__dict__,
                        'execution_engine_state': ee_state,
                        'vocab': vocab
                    }
                    for k, v in stats.items():
                        checkpoint[k] = v

                    # Save current model
                    logger.info('Saving checkpoint to %s' % opt.output_path)
                    torch.save(checkpoint, join(opt.output_path, 'model'))
                    print("SAVED!")
                    # Save the best model separately
                    if val_acc > stats['best_val_acc']:
                        logger.info('Saving best so far checkpoint to %s' % (join(opt.output_path, 'model.best')))
                        stats['best_val_acc'] = val_acc
                        checkpoint['execution_engine_state'] = ee_state
                        torch.save(checkpoint, join(opt.output_path, 'model.best'))

                    # Save training status in a human-readable format
                    del checkpoint['execution_engine_state']
                    with open(join(opt.output_path, 'model.json'), 'w') as f:
                        json.dump(checkpoint, f, indent=2, sort_keys=True)

                else:
                    checkpoint = {
                        'optimization_kwargs': opt.hyper_opt.__dict__,  # in case we change the lr
                        'execution_engine_kwargs': opt.hyper_method.__dict__,
                        'vocab': vocab
                    }
                    for k, v in stats.items():
                        checkpoint[k] = v
                    with open(join(opt.output_path, 'model.json'), 'w') as f:
                        json.dump(checkpoint, f, indent=2, sort_keys=True)

                    if val_acc > stats['best_val_acc']:
                        stats['best_val_acc'] = val_acc
                        checkpoint['execution_engine_state'] = get_state(execution_engine)
                        # Save current model and exit
                        logger.info('Saving checkpoint to %s' % opt.output_path)
                        stats['best_model_t'] = t
                        stats['best_model_epoch'] = epoch
                        torch.save(checkpoint, join(opt.output_path, 'model.best'))

                    if epoch > opt.hyper_opt.min_epochs:
                        # no improvement in the last two epochs
                        val_acc_idxs = opt.hyper_opt.previous_epochs * opt.hyper_opt.n_checkpoint_every_epoch

                        exit_condition = np.all(np.array(stats['val_accs'])[-val_acc_idxs:] - stats['best_val_acc'] < 0)

                        if exit_condition:
                            checkpoint['execution_engine_state'] = get_state(execution_engine)
                            # Save current model and exit
                            logger.info('Saving checkpoint to %s' % opt.output_path)
                            torch.save(checkpoint, join(opt.output_path, 'model'))
                            satisfied_early_stop = True

                        # this attempt has been done to save the best model -- validation accuracy comparable to model
                        # else:
                        #     checkpoint['execution_engine_state'] = get_state(execution_engine)
                        #     torch.save(checkpoint, join(opt.output_path, 'model.best'))

            if t == max_iterations or satisfied_early_stop:
                # Save the best model separately
                print("\nTraining time")
                print(time.time() - init_training)
                return

            # if t == 50000:
            #     print('50k iterations!')
            #     return
            # if t == 100000:
            #     print('100k iterations!')
            #     return

            batch_start_time = time.time()

