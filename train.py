"""Train script for Visual Dialog task
   Used in: Factpr Graph Attention
   https://arxiv.org/abs/1904.05880
   Author: Idan Schwartz (https://scholar.google.com/citations?user=5V-yJT4AAAAJ&hl=en)
   Note:  This code is an adaption of an earlier version by Unnat Jain
"""


from __future__ import print_function
import json
from shutil import copy2
import numpy as np
import os
import torch
import torch.nn.functional as F
import torch.optim.lr_scheduler
import torch.utils.data
from progress.bar import Bar
import utils
from utils import DenseAnnotationsReader, Statistics, dir_path, initialize_model_weights, NDCG
from functools import partial
from fga_model import FGA
from sklearn import preprocessing
import sys
from loader import VisDialDataset
from args import get_parser
import h5py

from more_itertools import unique_everseen

def train(train_ds, batch_size, epoch=0):
    """
    Trains a single epoch
    :param train_ds: a visual dialog dataset (VisDialDataset object)
    :param batch_size: the train batch size
    :param epoch: the current epoch for printing
    :return: loss value
    """
    print("Train epoch %d" % epoch)
    model.train()
    train_dl = torch.utils.data.DataLoader(train_ds, shuffle=True,
                                           batch_size=batch_size, drop_last=False)
    total_samples = len(train_ds)
    num_samples = 0.0
    total_loss = 0.0
    correct = 0.0

    for ib, b in enumerate(train_dl):

        ques, opt_list, hist_ques, hist_ans, cap, ques_len, opt_len, cap_len, img, target = [x.cuda() for x in b]
        optimizer.zero_grad()
        scores = model(ques, opt_list, hist_ques, hist_ans, cap, ques_len, opt_len, cap_len, img)
        pred = scores.data.max(1, keepdim=True)[1]
        loss = F.cross_entropy(scores, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.data.item()*ques.size(0) # Since size_average default value is True
        correct += pred.eq(target.data.view_as(pred)).sum()
        num_samples += ques.size(0)

        if ib % args.log_interval == 0 and ib != 0:
            out_str = 'Progress (epoch: {}) {} / {}   ({:.0f}%)\nTrain Loss: {:.6f}\nTrain Accuracy: {}/{} ({:.2f}%)'.format(
                epoch,
                num_samples,
                total_samples,
                100. * (num_samples / total_samples),
                loss.data.item(),
                correct,
                num_samples,
                100. * (correct / num_samples))
            output_file.write(out_str)
            print(out_str)
    return total_loss/total_samples


def test_eval(test_ds, model, batch_size, mydir, epoch=0, ndcg=None, stats=None, submission_text=""):
    """
    Evaluate the model or generate predictions
    :param test_ds: a visual dialog test dataset (VisDialDataset object)
    :param test_ds: a visual dialog val dataset (VisDialDataset object)
    :param model: the model to eval
    :param batch_size: the test batch size
    :param epoch: the current epoch for printing
    :param ndcg: NDCG object, the offical method to calculate NDCG metric(from VisDial team)
    :param stats: Statistics object, monitor, report and save model
    :param submission_text: append submission post text
    :return:
    """
    print("Eval epoch %d" % epoch)
    model.eval()

    test_or_val = submission_text if submission_text == 'val' else 'test'
    test_dl = torch.utils.data.DataLoader(test_ds, shuffle=False, batch_size=batch_size, drop_last=False)
    total_samples = len(test_ds)
    test_loss = 0
    correct = 0


    bar = Bar('Processing', max=len(test_dl))
    all_output = np.array([]).reshape(0, 100)

    ann_reader = DenseAnnotationsReader("data/visdial_1.0_val_dense_annotations.json")

    for bi, b in enumerate(test_dl):
        bar.next()
        if test_or_val == "val":
            ques, opt_list, hist_ques, hist_ans, cap,\
            ques_len, opt_len, cap_len, img, target = [x.cuda() for x in b]
        else:
            ques, opt_list, hist_ques, hist_ans, cap,\
            ques_len, opt_len, cap_len, img = [x.cuda() for x in b]

        with torch.no_grad():
            output = model(ques, opt_list, hist_ques, hist_ans, cap, ques_len,
                           opt_len, cap_len, img)

        all_output = np.vstack([all_output, output.cpu().data])

        if test_or_val == "val":
            pred = output.data.max(1, keepdim=True)[1]
            ordered_options_list = output.data.sort(dim=1, descending=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
            test_loss += F.cross_entropy(output, target, reduction='sum').item()

            assert len(ordered_options_list) == len(target.data)
            groundtruth_rank = (ordered_options_list == target.data.view(-1,1)).max(dim=1)[1] + 1

            assert len(groundtruth_rank.shape) == 1 # 1d tensor
            stats.update_metrics(groundtruth_rank, len(groundtruth_rank))


    bar.finish()


    best_mrr, best_ndcg = False, False
    if test_or_val == "val":


        b = int(all_output.shape[0]/10)
        p_acc = np.zeros(shape=(b,100))
        r_acc = np.zeros(shape=(b,100))
        all_output = np.reshape(all_output,(b, 10, 100))

        for d in range(all_output.shape[0]):
            image_id = int(visdial_meta["unique_img_val"][d][-16:-4])
            dense_annotations = ann_reader[image_id]
            output_true, dense_round = dense_annotations["gt_relevance"], dense_annotations["round_id"]
            dense_round -= 1
            p_acc[d] = all_output[d][dense_round]
            r_acc[d] = output_true
        p_acc = torch.Tensor(p_acc)
        r_acc = torch.Tensor(r_acc)
        ndcg.observe(p_acc, r_acc)
        stats.update_ndcg(ndcg.retrieve(reset=True)["ndcg"])

        stats.report(test_loss, correct, total_samples, output_file, epoch)
        if not args.only_val:
            best_mrr, best_ndcg = stats.save_best_model(model, optimizer, args, epoch, mydir, output_file, all_output)
        stats.reset()
        test_loss /= total_samples

        if best_mrr or args.only_val:
            # Saving ranks
            print("saving val ranks")
            h5 = h5py.File("data/visdial_data.h5", 'r')
            round_test = h5["num_rounds_val"][:]
            all_output = np.reshape(all_output, (-1, 100))
            ranks = utils.scores_to_ranks_submission(torch.FloatTensor(all_output).data)
            ranks = ranks.view(-1, 10, 100)
            bar = Bar('Processing val ranks', max=ranks.size(0))
            ranks_json = []
            for d in range(ranks.size(0)):
                bar.next()
                for r in range(int(round_test[d])):
                # cast into types explicitly to ensure no errors in schema
                    ranks_json.append({
                        'image_id': int(visdial_meta["unique_img_val"][d][-16:-4]),
                        'round_id': r + 1,
                        'ranks': ranks[d][r].tolist()
                    })
            json.dump(ranks_json, open(os.path.join('.' if args.only_val else mydir, "submission_val.json"), 'w'))


    if test_or_val == 'test':
        # Saving scores
        temp_save_path = os.path.join('.', "output_scores_%s" % submission_text)
        print("Saving output score %s" % temp_save_path)
        np.save(temp_save_path, all_output)

        # Saving ranks
        h5 = h5py.File("data/visdial_data.h5", 'r')
        round_test = h5["num_rounds_test"][:]
        ranks = utils.scores_to_ranks_submission(torch.FloatTensor(all_output).data)
        ranks = ranks.view(-1, 10, 100)
        bar = Bar('Processing ranks', max=ranks.size(0))
        ranks_json = []
        for d in range(ranks.size(0)):
            bar.next()
            round = round_test[d] - 1
            # cast into types explicitly to ensure no errors in schema
            ranks_json.append({
                'image_id': int(visdial_meta["unique_img_test"][d][-16:-4]),
                'round_id': int(round + 1),
                'ranks': ranks[d][round].tolist()
            })

        print("\nWriting submission file to {}".format("submission_%s_%s.json" % (test_or_val, submission_text), 'w'))
        json.dump(ranks_json, open("result.json", 'w'))

    return best_mrr, best_ndcg


if __name__ == '__main__':

    '''
    Init Part
    '''
    # Parser from visdial utils
    parser = get_parser()
    args = parser.parse_args()
    args.model_pathname = args.model_pathname
    print(args)
    mydir = dir_path(args)
    output_file = None

    # Finding vocabulary size
    visdial_meta = json.load(open('data/visdial_params.json', 'r'))

    if "word2ind" in visdial_meta.keys():
        word2ind = visdial_meta["word2ind"]
        vocab_size = len(word2ind.values()) + 1  # Zero is not in word2ind but appears in question/answers.
    else:
        sys.exit("Check the json file. It doesn't have 'word2ind' dictionary.")

    args.stop_id = len(word2ind.values()) + 1
    args.empty_id = len(word2ind.values()) + 2



    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    train_ds = VisDialDataset(args, 'train')
    val_ds = VisDialDataset(args, 'val')
    test_ds = VisDialDataset(args, 'test')

    img_features_dim = train_ds.images.shape[2]

    model = FGA(vocab_size=vocab_size,
                word_embed_dim=args.word_embed_dim,
                hidden_ques_dim=args.hidden_ques_dim,
                hidden_ans_dim=args.hidden_ans_dim,
                hidden_hist_dim=args.hidden_hist_dim,
                hidden_cap_dim=args.hidden_cap_dim,
                hidden_img_dim=img_features_dim)


    # Multiple GPUs batch parallel
    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)
        model.cuda()
    else:
        sys.exit(
            "Only GPU version is currently available.")
    #for n, p in model.named_parameters():
    #    print(n, p.numel())
    print("Total params:", sum(p.numel() for p in model.parameters()))
    stats = Statistics(args)
    ndcg = NDCG()

    if args.submission:
        if args.model_pathname:
            args.mrr_pathname = os.path.join(args.model_pathname, 'best_model_mrr.pth.tar')
            args.ndcg_pathname = os.path.join(args.model_pathname, 'best_model_ndcg.pth.tar')
        else:
            args.mrr_pathname = os.path.join(dir_path(args), 'best_model_mrr.pth.tar')
            args.ndcg_pathname = os.path.join(dir_path(args), 'best_model_ndcg.pth.tar')
        print('Creating submissions')
        print("loading best MRR: {}".format(args.mrr_pathname))
        checkpoint = torch.load(args.mrr_pathname)
        model.load_state_dict(checkpoint["model"])
        optimizer = checkpoint["optimizer"]
        loaded_best_epoch = checkpoint["epoch"]
        test_eval(test_ds, model, mydir=mydir, batch_size=args.test_batch_size, submission_text="mrr")
        if not os.path.exists(args.ndcg_pathname):
            print('no ndcg model')
        else:
            print("loading best NDCG: {}".format(args.ndcg_pathname))
            checkpoint = torch.load(args.ndcg_pathname)
            model.load_state_dict(checkpoint["model"])
            optimizer = checkpoint["optimizer"]
            loaded_best_epoch = checkpoint["epoch"]
            test_eval(test_ds, model, mydir=mydir,  batch_size=args.test_batch_size, submission_text="ndcg")
        exit(1)

    if args.model_pathname:
        if not os.path.exists(args.model_pathname):
            print("No model was loaded, model doesn't not exists")
            exit(1)
        else:
            args.mrr_pathname = os.path.join(args.model_pathname, 'best_model_mrr.pth.tar')
            checkpoint = torch.load(args.mrr_pathname)
            model.load_state_dict(checkpoint["model"])
            optimizer = checkpoint["optimizer"]
            loaded_best_epoch = checkpoint["epoch"]
    else:
        # Initialize weights as per args.initialization and lstm weights as per args.lstm_initialization.
        # Kaiming He initialization worked best for both the weights.
        initialize_model_weights(model, args.initialization, args.lstm_initialization)
        if args.opt == 0:
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
        elif args.opt == 1:
            optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
            if args.epochs_to_half != 0:
                print("scheduler kicks in")
                lambda_1 = lambda epoch: 0.5 ** (1.0 * epoch / args.epochs_to_half)
                scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_1)
        else:
            sys.exit("Only Adam/SGD enabled in code. Feel free to tweak the code to enable other optimizers.")

    if args.only_val:
        test_eval(test_ds=val_ds, model=model,
                    batch_size=args.test_batch_size,
                    epoch=loaded_best_epoch, mydir=mydir, stats=stats, ndcg=ndcg,
                    submission_text="val")
        exit(1)





    if not args.submission and not args.fast:
        '''
        Copies model files, to re-eval
        '''
        copy2("./atten.py", mydir + "/")
        copy2("./train.py", mydir + "/")
        copy2("./fga_model.py", mydir + "/")
        copy2("./utils.py", mydir + "/")
        copy2("./run_test", mydir + "/")

    '''
     Creating output files
    '''

    if args.output_file == "":
        output_file = open(os.path.join(mydir, "output.txt"), 'a+', buffering=1)
    else:
        output_file = open(args.output_file, 'a+', buffering=1)

    output_file.write(" ".join(sys.argv) + "\n")
    for key, value in args.__dict__.items():
        output_file.write(str(key) + "\t\t: " + str(value) + "\n")
    output_file.write(str(model))
    print("output file created")




    for epoch in range(args.model_epoch + 1, args.epochs + 1):
        if args.epochs_to_half != 0:
            scheduler.step()
        train_loss = train(train_ds=train_ds, batch_size=args.batch_size,
                              epoch=epoch)
        if epoch % args.test_after_every == 0:
            output_file.write('\nVal Metrics for current model | Epoch: {}\n'.format(epoch))
            print('\nVal Metrics for current model | Epoch: {}\n'.format(epoch))
            best_mrr, best_ndcg = test_eval(test_ds=val_ds, model=model,
                                            batch_size=args.test_batch_size,
                                            epoch=epoch, mydir=mydir,  stats=stats, ndcg=ndcg,
                                            submission_text="val")

            if best_mrr:
                print('Best MRR model, Creating test outputs')
                test_eval(test_ds, model, mydir=mydir, batch_size=args.test_batch_size, submission_text="mrr")

            if best_ndcg:
                print('Best NDCG model, Creating test outputs')
                test_eval(test_ds, model, mydir=mydir, batch_size=args.test_batch_size, submission_text="ndcg")


