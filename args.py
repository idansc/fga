import argparse
def get_parser():
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='input batch size for training (default: 64)')
    parser.add_argument('--val-batch-size', type=int, default=64,
                        help='input batch size for testing or validating (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=64,
                        help='input batch size for testing or validating (default: 64)')
    parser.add_argument('--epochs', type=int, default=10,
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                        help='learning rate (default: 0.1)')
    parser.add_argument('--trunc-length', type=float, default=20,
                        help='trunc questions param')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log_interval', type=int, default=1000,
                        help='how many batches to wait before logging training status')
    parser.add_argument('--opt', type=int, default=0,
                        help='0: adam; 1:sgd')
    parser.add_argument('--output-file', type=str, default="",
                        help='Pathname to the txt file to save stdout.')
    parser.add_argument('--initialization', type=str, default="",
                        help='Default is uniform. Options: "he".')
    parser.add_argument('--mode', type=str, default="", help='No default', required=False)
    parser.add_argument('--test-after-every', type=int, default=1,
                        help='run over test set after how many epochs')
    parser.add_argument('--model-pathname', type=str, default="",
                        help='Pathname to the model to load.')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--folder-prefix', type=str, default="",
                        help='Prefix to folder name.')
    parser.add_argument('--model-epoch', type=int, default=0,
                        help='epoch number for the model provided (default 0)')
    parser.add_argument('--epochs_to_half', type=int, default=0,
                        help='half the lr after these many epochs')
    parser.add_argument('--clip', type=float, default=None,
                        help='norm for clipping gradients')
    parser.add_argument('--lstm-initialization', type=str, default="",
                        help='Default is "he". Options: "he"/"xavier".')
    parser.add_argument('--word-embed-dim', type=int, default=None,
                        help='Dims for word embedding matrix')
    parser.add_argument('--image_data', type=str,
                        default="data/features_frcnn.h5",
                        help='Source to image features')
    parser.add_argument('--log-interval', type=int, default=1000,
                        help='how many batches to wait before logging training status')
    parser.add_argument('--add-box-location', type=bool, default=False,
                        help='concat location features to object detected.')
    parser.add_argument('--cls-modal', type=bool, default=False,
                        help='use predicted box classes(Not working)')
    parser.add_argument('--visdial_data_path', type=str, default="data/visdial_data.h5",
                        help='Path to visdial_data (default: data/visdial_data.h)')
    parser.add_argument('--hidden-cap-dim', type=int, default=None,
                        help='Dims for hidden state of caption')
    parser.add_argument('--hidden-img-dim', type=int, default=None,
                        help='Dims for hidden state of image')
    parser.add_argument('--hidden-hist-dim', type=int, default=None,
                        help='Dims for hidden state of history')
    parser.add_argument('--hidden-ques-dim', type=int, default=None,
                        help='Dims for question-history word embedding matrix')
    parser.add_argument('--hidden-ans-dim', type=int, default=None,
                        help='Dims for answer-history word embedding matrix')
    parser.add_argument('--load_model', type=bool, default=False,
                        help='model to load.')
    parser.add_argument('--load_epoch', type=int, default=-1,
                        help='epoch to load.')
    parser.add_argument('--only-test', type=bool, default=False,
                        help='disables training, simply test the model passed by --model-pathname')
    parser.add_argument('--astop', type=bool, default=True,
                        help='add stop symbol to answer')
    parser.add_argument('--qstop', type=bool, default=True,
                        help='add stop symbol to question')
    parser.add_argument('--fast', action='store_true', default=False,
                        help='fast training for debugging')
    parser.add_argument('--submission', type=bool, default=False, help='Creates challenge output')
    parser.add_argument('--only_val', type=bool, default=False, help='Creates submission val output')
    parser.add_argument('--mail_server',  type=str, default= "", help='mail server to send results')
    parser.add_argument('--mail_to',  type=str, help='mail addresses to send mails to(separated by ,)')
    parser.add_argument('--mail_from',  type=str, help='sender mail')

    return parser