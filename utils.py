
import torch
from collections import defaultdict
import socket
import os
from typing import Dict, List, Union
import json
import numpy as np
import smtplib
from email.message import EmailMessage

def initialize_model_weights(model, initialization, lstm_initialization):
    if initialization == "he":
        print("kaiming normal initialization.")
    elif initialization == "xavier":
        print("xavier normal initialization.")
    else:
        print("default initialization, no changes made.")
    if(initialization):
        for name, param in model.named_parameters():
            # Bias params
            if("bias" in name.split(".")[-1]):
                param.data.zero_()

            # Batchnorm weight params
            elif("weight" in name.split(".")[-1] and len(param.size())==1):
                continue
            # LSTM weight params
            elif("weight" in name.split(".")[-1] and "lstm" in name):
                if "xavier" in lstm_initialization:
                    torch.nn.init.xavier_normal_(param)
                elif "he" in lstm_initialization:
                    torch.nn.init.kaiming_normal_(param)
            # Other weight params
            elif("weight" in name.split(".")[-1] and "lstm" not in name):
                if "xavier" in initialization:
                    torch.nn.init.xavier_normal_(param)
                elif "he" in initialization:
                    torch.nn.init.kaiming_normal_(param)


def dir_path(args):
    """
    Creates a  path from hyperparams
    :param args: see args.py
    :return: path
    """
    MODEL_FOLDER = "models"
    folder_name = str(socket.gethostname()) + \
                  "_" + str(args.folder_prefix) + "opt_" + str(args.opt) + "_" +\
                  "lr_" + str(args.lr) + "_" + "bs_" + str(args.batch_size) +\
                  "_" + "epochs_" + str(args.epochs) + "seed_" + str(args.seed) +\
                  "_word_embed_" + str(args.word_embed_dim) \
                  + "_aembed_" + str(args.hidden_ans_dim) \
                  + "_qembed_" + str(args.hidden_ques_dim) \
                  + "_hembed_" + str(args.hidden_hist_dim) \
                  + "_cembed_" + str(args.hidden_cap_dim) \
                  + "_iembed_" + str(args.hidden_img_dim)
    mydir = os.path.join(os.getcwd(), MODEL_FOLDER, folder_name)

    if not os.path.exists(mydir) and not (args.only_test or args.submission):
        os.makedirs(mydir)

    return mydir

class DenseAnnotationsReader(object):
    """
    A reader for dense annotations for val split.
    The json file must have the same structure as mentioned
    on ``https://visualdialog.org/data``.

    Parameters
    ----------
    dense_annotations_jsonpath : str
        Path to a json file containing VisDial v1.0
    """

    def __init__(self, dense_annotations_jsonpath: str):
        with open(dense_annotations_jsonpath, "r") as visdial_file:
            self._visdial_data = json.load(visdial_file)
            self._image_ids = [entry["image_id"] for entry in self._visdial_data]

    def __len__(self):
        return len(self._image_ids)

    def __getitem__(self, image_id: int) -> Dict[str, Union[int, List]]:
        index = self._image_ids.index(image_id)
        # keys: {"image_id", "round_id", "gt_relevance"}
        return self._visdial_data[index]

    @property
    def split(self):
        # always
        return "val"


def scores_to_ranks(scores: torch.Tensor):
    """Convert model output scores into ranks."""
    batch_size, num_rounds, num_options = scores.size()
    scores = scores.view(-1, num_options)

    # sort in descending order - largest score gets highest rank
    sorted_ranks, ranked_idx = scores.sort(1, descending=True)

    # i-th position in ranked_idx specifies which score shall take this
    # position but we want i-th position to have rank of score at that
    # position, do this conversion
    ranks = ranked_idx.clone().fill_(0)
    for i in range(ranked_idx.size(0)):
        for j in range(num_options):
            ranks[i][ranked_idx[i][j]] = j
    # convert from 0-99 ranks to 1-100 ranks
    ranks += 1
    ranks = ranks.view(batch_size, num_rounds, num_options)
    return ranks


class NDCG(object):
    def __init__(self):
        self._ndcg_numerator = 0.0
        self._ndcg_denominator = 0.0

    def observe(
        self, predicted_scores: torch.Tensor, target_relevance: torch.Tensor
    ):
        """
        Observe model output scores and target ground truth relevance and
        accumulate NDCG metric.

        Parameters
        ----------
        predicted_scores: torch.Tensor
            A tensor of shape (batch_size, num_options), because dense
            annotations are available for 1 randomly picked round out of 10.
        target_relevance: torch.Tensor
            A tensor of shape same as predicted scores, indicating ground truth
            relevance of each answer option for a particular round.
        """
        predicted_scores = predicted_scores.detach()
        # shape: (batch_size, 1, num_options)
        predicted_scores = predicted_scores.unsqueeze(1)
        predicted_ranks = scores_to_ranks(predicted_scores)

        # shape: (batch_size, num_options)
        predicted_ranks = predicted_ranks.squeeze(1)
        batch_size, num_options = predicted_ranks.size()
        k = torch.sum(target_relevance != 0, dim=-1)

        # shape: (batch_size, num_options)
        _, rankings = torch.sort(predicted_ranks, dim=-1)
        # Sort relevance in descending order so highest relevance gets top rnk.
        _, best_rankings = torch.sort(
            target_relevance, dim=-1, descending=True
        )

        # shape: (batch_size, )
        batch_ndcg = []
        for batch_index in range(batch_size):
            num_relevant = k[batch_index]
            dcg = self._dcg(
                rankings[batch_index][:num_relevant],
                target_relevance[batch_index],
            )
            best_dcg = self._dcg(
                best_rankings[batch_index][:num_relevant],
                target_relevance[batch_index],
            )
            batch_ndcg.append(dcg / best_dcg)

        self._ndcg_denominator += batch_size
        self._ndcg_numerator += sum(batch_ndcg)

    def _dcg(self, rankings: torch.Tensor, relevance: torch.Tensor):
        sorted_relevance = relevance[rankings].cpu().float()
        discounts = torch.log2(torch.arange(len(rankings)).float() + 2)
        return torch.sum(sorted_relevance / discounts, dim=-1)

    def retrieve(self, reset: bool = True):
        if self._ndcg_denominator > 0:
            metrics = {
                "ndcg": float(self._ndcg_numerator / self._ndcg_denominator)
            }
        else:
            metrics = {}

        if reset:
            self.reset()
        return metrics

    def reset(self):
        self._ndcg_numerator = 0.0
        self._ndcg_denominator = 0.0


class Statistics():
    def __init__(self, args):
        self.best_mrr = 0 # Max mean reverse rank possible
        self.best_ndcg = 0 # Max mean reverse rank possible

        # best_val_loss = 1e5
        self.best_metrics = None
        self.best_epoch = 0
        self.metrics = defaultdict(int)
        self.metrics["num_samples"] = 0
        self.mail_message = ""
        self.step = 0
        self.mod_flag = 3
        self.num_ndcg = 0
        self.mydir = dir_path(args)
        self.mail_server = args.mail_server
        self.mail_to = args.mail_to
        self.mail_from = args.mail_from

    def reset(self):
        self.metrics = defaultdict(int)
        self.metrics["num_samples"] = 0

    def update_ndcg(self, ndcg_score):
        self.metrics["ndcg"] = ndcg_score

    def update_metrics(self, groundtruth_rank, num_batch_samples, ndcg=0.0):
        new_total = num_batch_samples + self.metrics["num_samples"]
        self.metrics["r1"] = (float( (groundtruth_rank == 1).sum() ) + self.metrics["r1"] * self.metrics["num_samples"]) / (new_total)
        self.metrics["r5"] = (float( (groundtruth_rank <= 5).sum() ) + self.metrics["r5"] * self.metrics["num_samples"]) / (new_total)
        self.metrics["r10"] = (float( (groundtruth_rank <= 10).sum() ) + self.metrics["r10"] * self.metrics["num_samples"]) / (new_total)
        self.metrics["mrr"] = ( groundtruth_rank.float().reciprocal().sum() + self.metrics["mrr"] * self.metrics["num_samples"]) / (new_total)
        self.metrics["mrank"] = (float(groundtruth_rank.sum()) + self.metrics["mrank"] * self.metrics["num_samples"]) / (new_total)
        self.metrics["num_samples"] = new_total

    def report(self,  test_loss, correct, total_samples, output_file=None, epoch=-1):
        if output_file is not None:
            output_file.write('Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
              test_loss, correct, total_samples,
              100. * correct / total_samples))
            output_file.write('R1: {:.4f}, R5: {:.4f}, R10: {:.4f}, MRR: {:.4f}, MRank: {:.4f}, NDCG: {:.4f}\n'.format(
                self.metrics["r1"], self.metrics["r5"], self.metrics["r10"],
                self.metrics["mrr"], self.metrics["mrank"], self.metrics["ndcg"]))
        print('Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
          test_loss, correct, total_samples,
          100. * correct / total_samples))
        print('R1: {:.4f}, R5: {:.4f}, R10: {:.4f}, MRR: {:.4f}, MRank: {:.4f}, NDCG: {:.4f}\n'.format(
          self.metrics["r1"], self.metrics["r5"], self.metrics["r10"],
            self.metrics["mrr"], self.metrics["mrank"], self.metrics["ndcg"]))

        self.mail_message += 'R1: {:.4f}, R5: {:.4f}, R10: {:.4f}, MRR: {:.4f}, MRank: {:.4f}, NDCG: {:.4f}\n'.format(
          self.metrics["r1"], self.metrics["r5"], self.metrics["r10"],
            self.metrics["mrr"], self.metrics["mrank"], self.metrics["ndcg"])
        if self.step % self.mod_flag == 0:
            self.send_mail(epoch)

        self.step += 1

    def send_mail(self, epoch):
        '''
        This function sends an email with  experiments details
        :param epoch:
        :return:
        '''
        if self.mail_server == "":
            return
        # Send mail
        # -------------

        # Create the container email message.
        msg = EmailMessage()
        msg['Subject'] = self.mydir
        # me == the sender's email address
        # family = the list of all recipients' email addresses
        family = self.mail_to.split(',')
        msg['From'] = self.mail_from
        msg['To'] = ', '.join(family)
        msg.preamble = 'VisDial'
        msg.set_content(
            "This is automatic mail, step %d (part of epoch %d), experiement details:\n %s \n %s" % (self.step,
                                                                                                     epoch,
                                                                                                     self.mydir,
                                                                                                     self.mail_message))

        # Open the files in binary mode.  Use imghdr to figure out the
        # MIME subtype for each specific image.

        # with open(os.path.join(mydir, "plot.png"), 'rb') as fp:
        #    img_data = fp.read()
        #    msg.add_attachment(img_data, maintype='image',
        #                       subtype=imghdr.what(None, img_data))

        # Send the email via our own SMTP server.
        with smtplib.SMTP(self.mail_server) as s:
            s.send_message(msg)





    def save_best_model(self, model, optimizer, args, epoch, mydir, output_file, scores):

        best_mrr, best_ndcg = False, False

        if args.fast:
            return best_mrr, best_ndcg
        if self.metrics["ndcg"] > self.best_ndcg:
            self.best_ndcg = self.metrics["ndcg"]
            self.best_epoch = epoch
            best_ndcg = True

            #torch.save({'model': model, 'optimizer': optimizer, 'args': args, 'epoch': epoch},
            #          os.path.join(mydir, "%03d" % epoch + '.pth.tar'))
            torch.save(
                {'model': model.state_dict(), 'optimizer': optimizer, 'metrics': self.metrics, 'args': args, 'epoch': epoch},
                os.path.join(mydir, "best_model_ndcg" + '.pth.tar'))
            temp_save_path = os.path.join(mydir, 'ndcg_output_scores')
            print("Saving NDCG output score %s" % temp_save_path)
            np.save(temp_save_path, scores)

        if self.metrics["mrr"] > self.best_mrr:
            best_mrr = True

            self.best_mrr = self.metrics["mrr"]
            self.best_epoch = epoch
            #torch.save({'model': model, 'optimizer': optimizer, 'args': args, 'epoch': epoch},
            #          os.path.join(mydir, "%03d" % epoch + '.pth.tar'))
            torch.save(
                {'model': model.state_dict(), 'optimizer': optimizer, 'metrics': self.metrics, 'args': args, 'epoch': epoch},
                os.path.join(mydir, "best_model_mrr" + '.pth.tar'))
            temp_save_path = os.path.join(mydir, 'mrr_output_scores')
            print("Saving MRR output score %s" % temp_save_path)
            np.save(temp_save_path, scores)

        print("Best epoch till now: {}/{}, with MRR {}".format(self.best_epoch, epoch, self.best_mrr))
        output_file.write("Best epoch till now: {}/{}, with MRR {}\n".format(self.best_epoch, epoch, self.best_mrr))
        return best_mrr, best_ndcg




def scores_to_ranks_submission(scores):
    # sort in descending order - largest score gets highest rank
    sorted_ranks, ranked_idx = scores.sort(1, descending=True)

    # convert from ranked_idx to ranks
    ranks = ranked_idx.clone().fill_(0)
    for i in range(ranked_idx.size(0)):
        for j in range(100):
            #print(i,j)
            ranks[i][ranked_idx[i][j]] = j
    ranks += 1
    return ranks