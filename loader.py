import torch
import h5py
import numpy as np
from sklearn import preprocessing


class VisDialDataset(torch.utils.data.Dataset):
    def __init__(self, args, split):
        self.h5 = h5py.File(args.visdial_data_path, 'r')
        print("load %s split from data file %s: " % (split, str(args.visdial_data_path)))
        limit_ims = None
        self.astop = args.astop
        self.qstop = args.qstop
        if args.submission:
            if split == 'train':
                limit_ims = 1
        if args.fast:  # Fast for testing
            print("Note: Running fast version for debug")
            limit_ims = 100
        self.split = split
        self.empty_id = args.empty_id
        self.stop_id = args.stop_id
        self.ques = self.h5["ques_%s" % split][:limit_ims]
        # Number of QA pairs in one dialog, this is 10 for visdial and 9 for visdial-q task
        self.n_qa_per_dial = self.ques.shape[1]
        # Slice questions into train-val-test.
        # All questions have 20 length, but zero padded after the last word.
        # Length of questions stored in corresponding length variable
        self.ques = self.ques.reshape(-1, args.trunc_length)
        self.ques_length = self.h5["ques_length_%s" % split][:limit_ims].reshape(-1)

        # Slice answers into train-test
        self.ans = self.h5["ans_%s" % split][:limit_ims].reshape(-1, args.trunc_length)
        self.ans_length = self.h5["ans_length_%s" % split][:limit_ims].reshape(-1)

        # Slice options into train-val-test
        self.opt = self.h5["opt_%s" % split][:limit_ims].reshape(-1, 100)

        # Zero indexing, so shifting options from 1/100 to 0/99:
        self.opt -= 1

        # Slice correct option index (1 to 100) into train-val-test
        if split!="test":
            self.ans_index = self.h5["ans_index_%s" % split][:limit_ims].reshape(-1)
            self.ans_index -= 1

        # Set of all unique answers options (20 length truncated)
        self.opt_list = self.h5["opt_list_%s" % split][:]
        # Corresponding lengths of the above.

        self.opt_length_list = self.h5["opt_length_%s" % split][:]

        # Incorporating image cue.
        print('load %s img file from: %s' % (split, args.image_data))
        img_h5 = h5py.File(args.image_data, 'r')
        self.images = img_h5["%s_features" % split][:limit_ims]
        images_shape = self.images.shape
        print(split, images_shape)
        self.images = self.images.reshape(images_shape[0], -1)
        self.images = preprocessing.normalize(self.images, norm="l2", axis=1, copy=False)
        self.images = self.images.reshape(images_shape[0], images_shape[1], images_shape[2])
        '''
        #in case we have detection features
        if args.add_box_location:
            self.location = img_h5["%s_location" % split][:limit_ims]
            self.location_ori = img_h5["%s_location_ori" % split][:limit_ims]
            self.images = np.concatenate((self.images, self.location, self.location_ori),
                                               axis=2)

        self.cls_prob = img_h5["%s_cls_prob" % split][:limit_ims] if args.cls_modal else None
        '''
        self.cap = self.h5["cap_%s" % split][:limit_ims]
        self.cap_length = self.h5["cap_length_%s" % split][:limit_ims]


        self.total_samples = len(self.ques)
        assert self.total_samples == len(self.ans) == len(self.opt) ==\
               len(self.ques_length) == len(self.ans_length)

    def __len__(self):
        'Denotes the total number of samples'
        return self.total_samples

    def __getitem__(self, index):

        if self.astop:
            #add empty cell
            opt_list = np.insert(self.opt_list[self.opt[index]].astype(np.int64),
                                 20, 0, axis=1)
            opt_len = self.opt_length_list[self.opt[index]].astype(np.int64) + 1
            opt_list[np.arange(0,opt_list.shape[0]), opt_len-1] = self.stop_id
        else:
            opt_list = self.opt_list[self.opt[index]].astype(np.int64)
            opt_len = self.opt_length_list[index].astype(np.int64)

        if self.qstop:
            ques = np.insert(self.ques[index].astype(np.int64),
                    20, 0, axis=0)
            ques_len = self.ques_length[index].astype(np.int64) + 1
            ques[ques_len - 1] = self.stop_id
        else:
            ques = self.ques[index].astype(np.int64)
            ques_len = self.ques_length[index]


        history_qa_max_length = 20 + 1  # 20 q/a + <stop>
        hist_ques = np.zeros((self.n_qa_per_dial - 1, history_qa_max_length), dtype=np.int64)
        hist_ans = np.zeros((self.n_qa_per_dial - 1, history_qa_max_length), dtype=np.int64)
        hist_ques[:, 0] = self.empty_id
        hist_ans[:, 0] = self.empty_id
        hist_ques[:, 1] = self.stop_id
        hist_ans[:, 1] = self.stop_id

        qhist = self.ques[np.arange((index // self.n_qa_per_dial) * self.n_qa_per_dial, index), :]
        ahist = self.ans[np.arange((index // self.n_qa_per_dial) * self.n_qa_per_dial, index), :]
        qhist = np.insert(qhist,20, 0, axis=1)
        ahist = np.insert(ahist,20, 0, axis=1)

        hist_index = index % self.n_qa_per_dial

        qhist_len = self.ques_length[np.arange((index // self.n_qa_per_dial) * self.n_qa_per_dial, index)]
        ahist_len = self.ans_length[np.arange((index // self.n_qa_per_dial) * self.n_qa_per_dial, index)]
        qhist[np.arange(0,hist_index), qhist_len] = self.stop_id
        ahist[np.arange(0,hist_index), ahist_len] = self.stop_id

        hist_ques[0:qhist.shape[0], 0:qhist.shape[1]] = qhist
        hist_ans[0:ahist.shape[0], 0:ahist.shape[1]] = ahist

        assert qhist.shape[1] <= (history_qa_max_length)
        assert ahist.shape[1] <= (history_qa_max_length)

        img_index = index // self.n_qa_per_dial
        img = self.images[img_index, :]
        cap = self.cap[img_index].astype(np.int64)
        if self.qstop:
            cap = np.insert(cap, 40, 0)
            cap_len = self.cap_length[img_index].astype(np.int64) + 1
            cap[cap_len - 1] = self.stop_id
        else:
            cap_len = self.cap_length[img_index].astype(np.int64)


        if self.split != 'test':

            target = self.ans_index[index].astype(np.int64)

            return ques, opt_list, hist_ques, hist_ans, cap, ques_len, opt_len, cap_len, img, target

        else:
            return ques, opt_list, hist_ques, hist_ans, cap, ques_len, opt_len, cap_len, img
