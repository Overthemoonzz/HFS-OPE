import os.path
import torch
import torch.nn as nn
from loss.focalloss import FocalLoss
from sklearn.metrics import roc_auc_score

def mtlTrain(model, train_loader, val_loader, test_loader, args):
    device = args.device
    epoch = args.epochs
    early_stop = 5
    num_task=args.mtl_task_num
    path = os.path.join(args.save_path, '{}_{}_seed{}_best_model_{}.pth'.format(args.task_name, args.model_name, args.seed, args.mtl_task_num))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    model.to(device)

    # 'is_click', 'is_like', 'is_comment', 'is_follow', 'is_forward','long_view'
    pos_weight = torch.tensor(args.pos_weight, device=device)
    task_loss_weight = torch.tensor(args.task_loss_weight, device=device)

    
#     pos_weight = torch.tensor(
#     [1.5, 65.0, 100.0, 250.0, 250.0, 2.5],   # [num_task]
#     device=device
# )
    
#     task_loss_weight = torch.tensor(
#     [1.0, 0.8, 0.7, 0.6, 0.6, 1.0],         # [num_task]
#     device=device
# )

    # weighted-BCE
    if args.loss_fn == 'weighted_bce':
        criterion_list = [
            nn.BCEWithLogitsLoss(pos_weight=pos_weight[k], reduction="mean")
            for k in range(num_task)
        ]

    # Focal-loss
    elif args.loss_fn == 'focal':
        use_focal = [True, True, True, True, True, True]
        criterion_list = []
        for k, use_f in enumerate(use_focal):
            pw = pos_weight[k]
            if use_f:
                # 稀疏任务：Focal + pos_weight
                criterion_list.append(FocalLoss(gamma=2.0, pos_weight=pw, reduction="mean"))
            else:
                # 主任务：普通 BCE + pos_weight
                criterion_list.append(nn.BCEWithLogitsLoss(pos_weight=pw))

    # mixed
    elif args.loss_fn == 'mixed':
        use_focal = [False, True, True, True, True, False]
        criterion_list = []
        for k, use_f in enumerate(use_focal):
            pw = pos_weight[k]
            if use_f:
                # 稀疏任务：Focal + pos_weight
                criterion_list.append(FocalLoss(gamma=2.0, pos_weight=pw, reduction="mean"))
            else:
                # 主任务：普通 BCE + pos_weight
                criterion_list.append(nn.BCEWithLogitsLoss(pos_weight=pw))

    # 多少步内验证集的loss没有变小就提前停止
    patience, eval_loss = 0, 0
    # train
    print(f'start Training for {args.model_name}')
    for i in range(epoch):
        model.train()
        train_loss_sum = 0.0
        train_batch_cnt = 0

        for step, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)

            logits_list = model(x)
            task_losses = []
            for k in range(num_task):
                # logits_k: [B]
                logits_k = logits_list[k].squeeze(1)
                targets_k = y[:, k]

                # 带 pos_weight 的 Loss
                loss_k = criterion_list[k](logits_k, targets_k)

                # 乘以任务权重
                weighted_loss_k = task_loss_weight[k] * loss_k
                task_losses.append(weighted_loss_k)

            batch_loss = sum(task_losses) / task_loss_weight.sum()

            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

            train_loss_sum += batch_loss.item()
            train_batch_cnt += 1

        epoch_train_loss = train_loss_sum / train_batch_cnt
        print(f'Epoch {i} Train Loss {epoch_train_loss:.4f}')

        # 验证
        val_loss_sum = 0.0
        val_batch_cnt = 0
        model.eval() 
        with torch.no_grad():
            for step, (x, y, sid) in enumerate(val_loader):
                x, y = x.to(device), y.to(device)
                logits_list_val = model(x)

                task_losses_val = []
                for k in range(num_task):
                    # logits_k: [B]
                    logits_k = logits_list_val[k].squeeze(1)
                    targets_k = y[:, k]

                    # 带 pos_weight 的 BCEWithLogitsLoss
                    loss_k = criterion_list[k](logits_k, targets_k)

                    # 乘以任务权重
                    weighted_loss_k = task_loss_weight[k] * loss_k
                    task_losses_val.append(weighted_loss_k)
                
                batch_loss = sum(task_losses_val) / task_loss_weight.sum()

                val_loss_sum += batch_loss.item()
                val_batch_cnt += 1
        epoch_val_loss = val_loss_sum / val_batch_cnt
        print(f'Epoch {i} Val Loss {epoch_val_loss:.4f}')

        # earl stopping
        if i == 0:
            eval_loss = epoch_val_loss
            torch.save(model.state_dict(), path)
        if epoch_val_loss < eval_loss:
            eval_loss = epoch_val_loss
            torch.save(model.state_dict(), path)
            patience = 0 
        else:
            if patience < early_stop:
                patience += 1
            else:
                print("val loss is not decrease in %d epoch and break training" % patience)
                break

    # 测试
    print(f'start Testing for {args.model_name}') 

    state = torch.load(path)
    model.load_state_dict(state)
    model.eval()

    all_labels = [[] for _ in range(num_task)]   # list of num_task lists
    all_preds  = [[] for _ in range(num_task)]   # list of num_task lists


    with torch.no_grad():
        for x, y, session_id in test_loader:
            x = x.to(device)
            y = y.to(device)

            logits_list = model(x)

            # 1. 得到每个任务的概率 p_k = sigmoid(logit_k)
            probs = torch.stack(
                [torch.sigmoid(logit).squeeze(1) for logit in logits_list], dim=1
            )  # shape [B, num_task]

            # 2. 存储预测结果用于 AUC 计算
            # y shape: [B, num_task]
            for k in range(num_task):
                all_labels[k].extend(y[:, k].cpu().tolist())
                all_preds[k].extend(probs[:, k].cpu().tolist())

    # -----------------------------
    # 计算每个任务的 AUC
    # -----------------------------
    task_aucs = []
    tasks = ['is_click', 'is_like', 'is_comment', 'is_follow', 'is_forward','long_view']
    for k, t in enumerate(tasks):
        labels_k = all_labels[k]
        preds_k  = all_preds[k]

        # 防止单类数据报错
        if len(set(labels_k)) < 2:
            print(f"Warning: Task {k} has only one class in test set, AUC is undefined.")
            auc = float('nan')
        else:
            auc = roc_auc_score(labels_k, preds_k)

        task_aucs.append(auc)
        print(f"Task_{t} AUC: {auc:.4f}")





