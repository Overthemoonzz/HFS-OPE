import os.path
import torch
import time
from copy import deepcopy
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
import numpy as np

def mtlTrain(model, train_loader, val_loader, test_loader, args):
    device = args.device
    epoch = args.epochs
    early_stop = 5
    path = os.path.join(args.save_path, '{}_{}_seed{}_best_model_{}.pth'.format(args.task_name, args.model_name, args.seed, args.mtl_task_num))
    loss_function = nn.BCELoss(reduction='mean') if args.model_name == 'hfs_ope' else nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    model.to(device)
    # 多少步内验证集的loss没有变小就提前停止
    patience, eval_loss = 0, 0
    # train
    print(f'start Training for {args.model_name}')
    model.train()
    for i in range(epoch):
        y_train_click_true = []
        y_train_click_predict = []
        y_train_like_true = []
        y_train_like_predict = []
        total_loss, count = 0, 0
        for idx, (x, y1, y2) in enumerate(train_loader):
            x, y1, y2 = x.to(device), y1.to(device), y2.to(device)
            predict = model(x)
            y_train_click_true += list(y1.squeeze().cpu().numpy())
            y_train_like_true += list(y2.squeeze().cpu().numpy())
            y_train_click_predict += list(predict[0].squeeze().cpu().detach().numpy())
            y_train_like_predict += list(predict[1].squeeze().cpu().detach().numpy())
            loss_1 = loss_function(predict[0], y1.unsqueeze(1).float())
            loss_2 = loss_function(predict[1], y2.unsqueeze(1).float())
            loss = loss_1 + loss_2
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach())
            count += 1
        click_auc = roc_auc_score(y_train_click_true, y_train_click_predict)
        like_auc = roc_auc_score(y_train_like_true, y_train_like_predict)
        print("Epoch %d train loss is %.3f, task1 auc is %.3f and task2 auc is %.3f" % (i + 1, total_loss / count,
                                                                                            click_auc, like_auc))
        # 验证
        total_eval_loss = 0
        model.eval()
        count_eval = 0
        y_val_click_true = []
        y_val_like_true = []
        y_val_click_predict = []
        y_val_like_predict = []
        for idx, (x, y1, y2) in enumerate(val_loader):
            x, y1, y2 = x.to(device), y1.to(device), y2.to(device)
            predict = model(x)
            y_val_click_true += list(y1.squeeze().cpu().numpy())
            y_val_like_true += list(y2.squeeze().cpu().numpy())
            y_val_click_predict += list(predict[0].squeeze().cpu().detach().numpy())
            y_val_like_predict += list(predict[1].squeeze().cpu().detach().numpy())
            loss_1 = loss_function(predict[0], y1.unsqueeze(1).float())
            loss_2 = loss_function(predict[1], y2.unsqueeze(1).float())
            loss = loss_1 + loss_2
            total_eval_loss += loss.item()
            count_eval += 1
        click_auc = roc_auc_score(y_val_click_true, y_val_click_predict)
        like_auc = roc_auc_score(y_val_like_true, y_val_like_predict)
        print("Epoch %d val loss is %.3f, task1 auc is %.3f and task2 auc is %.3f" % (i + 1,
                                                                                    total_eval_loss / count_eval,
                                                                                    click_auc, like_auc))

        # earl stopping
        if i == 0:
            eval_loss = total_eval_loss / count_eval
        else:
            if total_eval_loss / count_eval < eval_loss:
                eval_loss = total_eval_loss / count_eval
                state = model.state_dict()
                torch.save(state, path)
            else:
                if patience < early_stop:
                    patience += 1
                else:
                    print("val loss is not decrease in %d epoch and break training" % patience)
                    break
    #test
    state = torch.load(path)
    model.load_state_dict(state)
    total_test_loss = 0
    model.eval()
    count_eval = 0
    y_test_click_true = []
    y_test_like_true = []
    y_test_click_predict = []
    y_test_like_predict = []
    for idx, (x, y1, y2) in enumerate(test_loader):
        x, y1, y2 = x.to(device), y1.to(device), y2.to(device)
        predict = model(x)
        y_test_click_true += list(y1.squeeze().cpu().numpy())
        y_test_like_true += list(y2.squeeze().cpu().numpy())
        y_test_click_predict += list(predict[0].squeeze().cpu().detach().numpy())
        y_test_like_predict += list(predict[1].squeeze().cpu().detach().numpy())
        loss_1 = loss_function(predict[0], y1.unsqueeze(1).float())
        loss_2 = loss_function(predict[1], y2.unsqueeze(1).float())
        loss = loss_1 + loss_2
        total_test_loss += float(loss)
        count_eval += 1
    click_auc = roc_auc_score(y_test_click_true, y_test_click_predict)
    like_auc = roc_auc_score(y_test_like_true, y_test_like_predict)
    print("Epoch %d test loss is %.3f, task1 auc is %.3f and task2 auc is %.3f" % (i + 1,
                                                                                    total_test_loss / count_eval,
                                                                                    click_auc, like_auc))
