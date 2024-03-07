import argparse

import torch

#torch.backends.cudnn.benchmark = True
import torch.nn as nn
import torch.optim as optim
import numpy as np
import datasetfactory as dfs
import config
import sed_eval
from dcase_evaluate import get_SED_results, process_event, process_event_my
import model
from utils import preprocess_data, preprocess_data_1
from dataset_batch import BatchData


def parse_option():
    parser = argparse.ArgumentParser('arguements for dataset and training')

    parser.add_argument('--learning-rate', type=float, default=0.0001, help='learning rate for training')
    parser.add_argument('--epoch', type=int, default=100, help='total number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='mini bath-size')
    parser.add_argument('--num-workers', type=int, default=0, help='numbers parallel workers to use')
    parser.add_argument('--check-point', type=int, default=50, help='check point')
    args = parser.parse_args()
    return args


if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# Class labels of DCASE SED 2017 task - Events from street scene
__class_labels_dict = {
    'brakes squeaking': 0,
    'car': 1,
    'children': 2,
    'large vehicle': 3,
    'people speaking': 4,
    'people walking': 5
}

# Development and evaluation sets paths
development_folder = '../dataset/SED_2017_street/TUT-sound-events-2017-development/'
evaluation_folder = '../dataset/SED_2017_street/TUT-sound-events-2017-evaluation/'
#evaluation_folder = '../dataset/MyEvaluationSet/'

check_time_stamp_folder = "../dataset/CheckTimeStamps/"

#usage_folder = 'dataset/MyEvaluationSet/'

time_stamp_predict_file = "time_stamp_predict.txt"
time_stamp_label_file = "time_stamp_label.txt"

def train(model, train_loader, epoch, check_point):
    step = 0
    model.to(device)
    criteria = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epoch, eta_min=0.001)

    for epoch_idx in range(1, epoch + 1):
        model.train()
        sum_loss = 0
        for batch_idx, (mel, target) in enumerate(train_loader):
            optimizer.zero_grad()
            mel, target = mel.to(device), target.to(device).float()
            logits = model(mel)
            loss = criteria(logits, target)
            sum_loss += loss.item()
            loss.backward()
            optimizer.step()
            step += 1

            if (batch_idx + 1) % check_point == 0 or (batch_idx + 1) == len(train_loader):
                print('==>>> epoch: {}, batch index: {}, step: {}, train loss: {:.3f}'.
                      format(epoch_idx, batch_idx + 1, step, sum_loss / (batch_idx + 1)))

        scheduler.step()


def predict_time_stamps(model, usage_loader, check_time_stamp_folder):
    model.to(device)
    model.eval()

    preds_list = []
    target_list = []

    for batch_idx, (mel, target) in enumerate(usage_loader):
        mel, target = mel.to(device), target.to(device).float()
        preds = torch.sigmoid(model(mel))

        preds_list.extend(preds.view(-1, preds.size(2)).cpu().detach().numpy())
        target_list.extend(target.view(-1, preds.size(2)).cpu().detach().numpy())



    hop_length_seconds = config.hop_len/config.sr
    threshold = 0.5

    # this is needed to generate  time stamps of predicted events
    process_event_my(list(__class_labels_dict.keys()),
                  np.array(preds_list).T,
                  threshold, hop_length_seconds, time_stamp_predict_file)


    process_event_my(list(__class_labels_dict.keys()),
                     np.array(target_list).T,
                     threshold, hop_length_seconds, time_stamp_label_file)


#    predicted = process_event( list(__class_labels_dict.keys()),
#                              np.array(preds_list).T,
#                              threshold, hop_length_seconds )

def evaluate(model, test_loader):
    model.to(device)
    model.eval()

    preds_list = []
    target_list = []

    for batch_idx, (mel, target) in enumerate(test_loader):

        mel, target = mel.to(device), target.to(device).float()
        preds = torch.sigmoid(model(mel))

        preds_list.extend(preds.view(-1, preds.size(2)).cpu().detach().numpy())
        target_list.extend(target.view(-1, target.size(2)).cpu().detach().numpy())

    segment_based_metrics = sed_eval.sound_event.SegmentBasedMetrics(
        event_label_list=list(__class_labels_dict.keys()),
        time_resolution=1.0
    )

    output, test_ER, test_F1, class_wise_metrics = (
        get_SED_results(np.array(target_list), np.array(preds_list),
                        list(__class_labels_dict.keys()), segment_based_metrics,
                        threshold=0.5,
                        hop_size=config.hop_len, sample_rate=config.sr))

    #nonzeroInd = np.where (np.any(np.array(target_list) != 0, axis =1 ))[0]
#    for i in nonzeroInd :
#        print(np.array(preds_list)[i], np.array(target_list)[i] )
#        input()

    print(output)

    print('F1: {:.3f}, ER: {:.3f}'.format(test_F1, test_ER))


if __name__ == '__main__':
    args = parse_option()

    np.random.seed(1900)
    model = model.CRNN(classes_num=6).to(device)

    development_data = dfs.MelData(development_folder, __class_labels_dict, sample_rate=config.sr,
                                   n_mels=config.nb_mel_bands,
                                   n_fft=config.nfft, hop_length=config.hop_len)

    evaluation_data = dfs.MelData(evaluation_folder, __class_labels_dict, sample_rate=config.sr,
                                  n_mels=config.nb_mel_bands,
                                  n_fft=config.nfft, hop_length=config.hop_len)

    usage_data = dfs.MelData(check_time_stamp_folder, __class_labels_dict, sample_rate=config.sr,
                                  n_mels=config.nb_mel_bands,
                                  n_fft=config.nfft, hop_length=config.hop_len)

    X_dev, Y_dev = development_data.mel_tensor, development_data.label_tensor

    X_eval, Y_eval = evaluation_data.mel_tensor, evaluation_data.label_tensor

    X_usage, Y_usage = usage_data.mel_tensor, usage_data.label_tensor

    X_dev, Y_dev, X_eval, Y_eval = preprocess_data(X_dev, Y_dev, X_eval, Y_eval, config.seq_length)

    X_usage, Y_usage = preprocess_data_1(X_usage, Y_usage, config.seq_length)
    # X_dev, X_eval = torch.from_numpy(X_dev).float(), torch.from_numpy(X_eval).float()

    Answer = input("Train model ? : (Y/N)").lower()

    if (Answer == "y"):

        train_loader = torch.utils.data.DataLoader(BatchData(X_dev, Y_dev), batch_size=args.batch_size, shuffle=True,
                                               num_workers=args.num_workers)

        train(model, train_loader, epoch=args.epoch, check_point=args.check_point)

        torch.save(model, 'best_model_hrtf.pt')

    try:

        model = torch.load('best_model_hrtf.pt')

    except FileNotFoundError:

        raise FileNotFoundError("No best_model_hrtf.pt file found")

    test_loader = torch.utils.data.DataLoader(BatchData(X_eval, Y_eval), batch_size=args.batch_size, shuffle=False,
                                              num_workers=args.num_workers)

    usage_loader = torch.utils.data.DataLoader(BatchData(X_usage, Y_usage), batch_size=args.batch_size, shuffle=False,
                                              num_workers=args.num_workers)

    predict_time_stamps(model, usage_loader, check_time_stamp_folder)
    evaluate(model, test_loader)
