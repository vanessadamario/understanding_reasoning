import sys
import json
import torch
import logging
import time
from torch.autograd import Variable
from os.path import join
import torch.nn.functional as F
from runs.shnmn_query import SHNMN
from runs.utils import load_vocab, load_cpu
from runs.data_attribute_random import category_idx, color_idx, brightness_idx, size_idx


# here we need to the the same of the train_loop function
logger = logging.getLogger(__name__)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def load_execution_engine(path,
                          model_type="SHNMN",
                          verbose=True):
    checkpoint = load_cpu(path)
    from runs.shnmn_quey import SHNMN
    kwargs = checkpoint['execution_engine_kwargs']
    state = checkpoint['execution_engine_state']
    kwargs['verbose'] = verbose
    if model_type == 'SHNMN':
        model = SHNMN(**kwargs)
    else:
        raise ValueError()
    cur_state = model.state_dict()
    # TODO: modified
    model.load_state_dict(state)
    return model, kwargs


def get_execution_engine(**kwargs):
    """ Load the model """
    # TODO load_execution_engine
    # consider the case where you want to load
    print("Loading model")
    print(kwargs)
    if kwargs['load_model']:
        ee, kwargs = load_execution_engine(join(kwargs['output_path'], 'model'),
                                           model_type="SHNMN")
        print(kwargs)
    else:
        ee = SHNMN(**kwargs)
    ee.to(device)
    ee.train()
    return ee


def set_mode(mode, models):
    assert mode in ['train', 'eval']
    for m in models:
        if m is None:
            continue
        if mode == 'train':
            m.train()
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
    separated_scores = [0, 0, 0, 0]
    for batch in loader:
        (feats, questions, answers) = batch
        # if isinstance(questions, list):
        #     questions = questions[0]
        questions_var = questions.to(device)
        feats_var = feats.to(device)
        scores = None  # Use this for everything but PG
        if opt.method_type in ['SimpleNMN', 'SHNMN']:
            scores = execution_engine(feats_var, questions_var)
        else:
            raise NotImplementedError('model ', opt.method_type, ' check_accuracy not implemented')
        if scores is not None:
            for id_attribute, scores_attribute in enumerate(scores):
                _, preds = scores_attribute.data.cpu().max(1)
                num_correct += (preds == answers[:, id_attribute]).sum()
                num_samples += preds.size(0)
                separated_scores[id_attribute] += (preds == answers[:, id_attribute]).sum() / preds.size(0)

        if max_n_samples is not None and num_samples >= max_n_samples:
            print("We are in the condition to exit")
            sys.stdout.flush()
            print(num_samples)
            sys.stdout.flush()
            break

    set_mode('train', [execution_engine])
    acc = float(num_correct) / num_samples
    # print("accuracy overall", acc)
    # print("separate scores", separated_scores)
    return acc


def train_query_loop(opt, train_loader, val_loader):

    vocab = load_vocab(join(opt.dataset.dataset_id_path, "vocab.json"))
    num_answers_category = len([v_ for v_ in vocab["question_token_to_idx"].values()
                                if v_ in category_idx])
    num_answers_color = len([v_ for v_ in vocab["question_token_to_idx"].values()
                             if v_ in color_idx])
    num_answers_brightness = len([v_ for v_ in vocab["question_token_to_idx"].values()
                                  if v_ in size_idx])
    num_answers_size = len([v_ for v_ in vocab["question_token_to_idx"].values()
                            if v_ in size_idx])

    execution_engine, ee_kwargs, ee_optimizer = None, None, None
    # vocabulary and execution engine

    stats = {
        'train_losses': [], 'train_rewards': [], 'train_losses_ts': [],
        'train_accs': [], 'val_accs': [], 'val_accs_ts': [], 'alphas': [], 'grads': [],
        'best_val_acc': -1, 'model_t': 0, 'model_epoch': 0,
        'p_tree': [], 'tree_loss': [], 'chain_loss': []
    }
    kkwargs_exec_engine_ = opt.hyper_method.__dict__
    kkwargs_exec_engine_["vocab"] = vocab

    if opt.method_type in ['SHNMN']:
        execution_engine = get_execution_engine(**kkwargs_exec_engine_)
    logger.info('Here is the conditioned network:')
    logger.info(execution_engine)

    # by default Adam
    optim_method = getattr(torch.optim, opt.hyper_opt.optimizer)

    if execution_engine:
        # separate learning rate for p(model) for the stochastic tree NMN
        base_parameters = []
        sensitive_parameters = []
        logger.info("PARAMETERS:")
        for name, param in execution_engine.named_parameters():
            if not param.requires_grad:
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
    loss_fn_list = []
    for n_ in [num_answers_category, num_answers_color,
               num_answers_brightness, num_answers_size]:
        if n_ > 0:
            loss_fn_list.append(torch.nn.CrossEntropyLoss().to(device))

    t, epoch, reward_moving_average = stats['model_t'], stats['model_epoch'], 0

    num_checkpoints = 0
    reward = None  # TODO figure out what this means
    running_loss = 0.0
    epoch_start_time = 0.0
    epoch_total_time = 0.0
    train_pass_total_time = 0.0
    val_pass_total_time = 0.0

    temp_old = time.time()
    while t < opt.hyper_opt.num_iterations:  # this is the number of steps in tf
        epoch += 1

        if (epoch > 0) and (opt.hyper_opt.time == 1):
            epoch_time = time.time() - epoch_start_time
            epoch_total_time += epoch_time
            logger.info('EPOCH PASS AVG TIME: ' + str(epoch_total_time / epoch), 'white')
            logger.info('Epoch Pass Time      : ' + str(epoch_time), 'white')
        epoch_start_time = time.time()

        logger.info('Starting epoch %d' % epoch)

        batch_start_time = time.time()

        for batch in train_loader:
            t += 1

            (feats, questions, answers) = batch
            if isinstance(questions, list):
                questions = questions[0]
            questions_var = Variable(questions.to(device))
            feats_var = Variable(feats.to(device))
            answers_var = Variable(answers.to(device))

            if opt.method_type in ['SimpleNMN', 'SHNMN']:
                # Train execution engine with ground-truth programs
                ee_optimizer.zero_grad()
            # TODO: questions_var
            scores = execution_engine(feats_var, questions_var)
            # if opt.hyper_method.model_type == 'hard':
            #    tree_loss = loss_fn(execution_engine.tree_scores, answers_var)
            #     chain_loss = loss_fn(execution_engine.chain_scores, answers_var)
            for id_j, loss_ in enumerate(loss_fn_list):
                if id_j == 0:
                    total_loss = loss_(scores[id_j], answers_var[:, id_j])
                else:
                    total_loss += loss_(scores[id_j], answers_var[:, id_j])
            total_loss.backward()
            # record alphas and gradients and p(model) here : DEBUGGING
            # if opt.method_type == 'SHNMN' and opt.hyper_method.model_type == 'hard':
            #     p_tree = F.sigmoid(execution_engine.tree_odds).item()
            #     if t % opt.hyper_opt.record_loss_every == 0:
            #         print('p_tree:', p_tree)
            #         stats['p_tree'].append(p_tree)
            #         stats['tree_loss'].append(tree_loss.item())
            #         stats['chain_loss'].append(chain_loss.item())

            if opt.hyper_method.model_type == 'SHNMN' and not opt.hyper_method.hard_code_alpha:
                # TODO: we can avoid this code
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

            ee_optimizer.step()

            if t % opt.hyper_opt.record_loss_every == 0:
                running_loss += total_loss.item()
                avg_loss = running_loss / opt.hyper_opt.record_loss_every
                logger.info("{} {:.5f} {:.5f} {:.5f}".format(t,
                                                             time.time() - batch_start_time, time.time() - compute_start_time,
                                                             total_loss.item()))
                stats['train_losses'].append(avg_loss)
                stats['train_losses_ts'].append(t)
                if reward is not None:
                    stats['train_rewards'].append(reward.item())
                running_loss = 0.0
            else:
                running_loss += total_loss.item()

            if t % opt.hyper_opt.checkpoint_every == 0:
                num_checkpoints += 1
                temp_time = time.time()

                temp_old = temp_time
                logger.info('Checking training accuracy ... ')
                start = time.time()
                train_acc = check_accuracy(opt, execution_engine, train_loader)
                train_pass_time = (time.time() - start)
                train_pass_total_time += train_pass_time
                logger.info('TRAIN PASS AVG TIME:' + str(train_pass_total_time /
                                                         num_checkpoints))
                logger.info('Train Pass Time : ' + str(train_pass_time))
                logger.info('train accuracy is {}'.format(train_acc))
                logger.info('Checking validation accuracy ...')
                start = time.time()

                val_acc = check_accuracy(opt, execution_engine, val_loader)
                val_pass_time = (time.time() - start)
                val_pass_total_time += val_pass_time
                logger.info('VAL PASS AVG TIME: ' + str(val_pass_total_time /
                                                        num_checkpoints))
                logger.info('Val Pass Time: ' + str(val_pass_time))
                logger.info('val accuracy is {}'.format(val_acc))
                stats['train_accs'].append(train_acc)
                stats['val_accs'].append(val_acc)
                stats['val_accs_ts'].append(t)

                ee_state = get_state(execution_engine)

                stats['model_t'] = t
                stats['model_epoch'] = epoch

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

            if t == opt.hyper_opt.num_iterations:
                # Save the best model separately
                break

            batch_start_time = time.time()
