import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from evaluation import *
from sklearn.linear_model import LogisticRegression
from torch.utils.data import TensorDataset, DataLoader
from model.mlp import MLP
from model.aWELv import AWELV
from utils import *
from loss.listloss import Listloss

def split_dataset(df):
    cols = ['pclick','plike','pcomment','pfollow','pforward','plong_view','is_click','is_like','is_comment','is_follow','is_forward','long_view']
    agg_dict = {col: list for col in cols}
    agg_dict['uid'] = 'first'
    df = df.groupby('sid').agg(agg_dict).reset_index()
    train_len = int(len(df) * 0.8)
    val_len = int(len(df) * 0.1)
    train_df = df[:train_len]
    tmp_df = df[train_len:]
    val_df = tmp_df[:val_len]
    test_df = tmp_df[val_len:]
    train_df = train_df.explode(cols, ignore_index=True)
    val_df = val_df.explode(cols, ignore_index=True) 
    test_df = test_df.explode(cols, ignore_index=True) 
    return train_df, val_df, test_df

def single_sort(df):
    df['score'] = df['pclick'] #is_click
    return df

def train_test_lr(train_df, val_df, test_df, args):
    # 1) 特征列：6 个 pxtr
    pxtr_cols = ["pclick", "plike", "pcomment", "pfollow", "pforward", "plong_view"]

    # 2) 训练集
    X_train = train_df[pxtr_cols].values          # shape: (N_train, 6)
    y_train = train_df["is_click"].values.astype(int)   # 0/1

    # 3) 验证集
    X_val = val_df[pxtr_cols].values
    y_val = val_df["is_click"].values.astype(int)
    
    logit = LogisticRegression(
    random_state=args.seed,
    penalty="l2",
    C=1.0,
    solver="lbfgs",
    max_iter=1000,
    class_weight="balanced"   # 可去掉，看数据是否极度不平衡
)
    logit.fit(X_train, y_train)

    val_pred = logit.predict_proba(X_val)[:, 1]
    val_auc = roc_auc_score(y_val, val_pred)
    print("Val AUC (click):", val_auc)

    print("融合权重：")
    for name, w in zip(pxtr_cols, logit.coef_[0]):
        print(f"{name}: {w:.4f}")
    print("bias:", logit.intercept_[0])
    
    # test
    X_test = test_df[pxtr_cols].values
    test_df["score"] = logit.predict_proba(X_test)[:, 1]
    return test_df

def train_test_mlp(train_df, val_df, test_df, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(getattr(args, "seed", 42))
    # 1) 特征列：6 个 pxtr
    pxtr_cols = ["pclick", "plike", "pcomment", "pfollow", "pforward", "plong_view"]

    # ======== 构造 numpy 数据 ========
    X_train = train_df[pxtr_cols].values.astype("float32")
    y_train = train_df["is_click"].values.astype("float32")

    X_val = val_df[pxtr_cols].values.astype("float32")
    y_val = val_df["is_click"].values.astype("float32")

    X_test = test_df[pxtr_cols].values.astype("float32")

    # ======== 转成 Tensor & DataLoader ========
    train_ds = TensorDataset(
        torch.from_numpy(X_train),
        torch.from_numpy(y_train),
    )
    val_ds = TensorDataset(
        torch.from_numpy(X_val),
        torch.from_numpy(y_val),
    )

    batch_size = getattr(args, "train_batch_size", 1024)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False)

 # ======== 构建模型 ========
    input_dim = len(pxtr_cols)
    hidden_dims = [64, 32]
    dropout = 0.1
    lr = 1e-4
    num_epochs = 50

    model = MLP(input_dim=input_dim, hidden_dims=hidden_dims, dropout=dropout).to(device)

    # 计算 pos_weight，缓解正负样本不平衡（可选）
    pos_num = y_train.sum()
    neg_num = len(y_train) - pos_num
    if pos_num > 0:
        pos_weight = torch.tensor([neg_num / pos_num], dtype=torch.float32, device=device)
    else:
        pos_weight = torch.tensor([1.0], dtype=torch.float32, device=device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        # ======== 训练循环 ========
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            logits = model(xb)                # [batch]
            loss = criterion(logits, yb)      # yb: [batch]
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * xb.size(0)

        avg_train_loss = total_loss / len(train_ds)

        # ======== 验证 AUC ========
        model.eval()
        val_logits_all = []
        val_labels_all = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                logits = model(xb)
                probs = torch.sigmoid(logits)         # P(click=1)
                val_logits_all.append(probs.cpu().numpy())
                val_labels_all.append(yb.numpy())

        val_pred = np.concatenate(val_logits_all, axis=0)
        val_labels = np.concatenate(val_labels_all, axis=0)

        # 防止全 0/1 导致 AUC 报错
        if len(np.unique(val_labels)) > 1:
            val_auc = roc_auc_score(val_labels, val_pred)
        else:
            val_auc = float("nan")

        print(f"[Epoch {epoch+1}/{num_epochs}] Train Loss: {avg_train_loss:.6f}, Val AUC(click): {val_auc:.6f}")

    # ======== 在 test_df 上打分 ========
    model.eval()
    with torch.no_grad():
        X_test_tensor = torch.from_numpy(X_test).to(device)
        test_logits = model(X_test_tensor)
        test_probs = torch.sigmoid(test_logits).cpu().numpy()

    # 写入一列融合分数
    test_df["score"] = test_probs  
    return test_df

def train_test_aWELv(train_df, val_df, test_df, args):
    score_cols = ["pclick","plike","pcomment","pfollow","pforward","plong_view"]
    label_cols = ["is_click","is_like","is_comment","is_follow","is_forward","long_view"]
    model_num = len(score_cols)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(getattr(args, "seed", 42))

    def collate_fn(batch):
        max_len = max(b["len"] for b in batch)
        B = len(batch)
        scores = torch.zeros(B, max_len, model_num)
        labels = torch.zeros(B, max_len)
        mask = torch.zeros(B, max_len)
        uids = []
        lengths = []
        for i, b in enumerate(batch):
            L = b["len"]
            lengths.append(L)
            scores[i, :L] = torch.from_numpy(b["scores"])
            labels[i, :L] = torch.from_numpy(b["labels"])
            mask[i, :L] = 1.0
            uids.append(b["uid"])

        # 基于标签分数生成排名：分数越大排名越靠前，未触发位置为 0
        ranking = torch.zeros_like(labels, dtype=torch.long)
        for i in range(B):
            valid_mask = mask[i] > 0
            if valid_mask.any():
                vals = labels[i, valid_mask]
                order = torch.argsort(vals, descending=True)
                ranks = torch.zeros_like(vals, dtype=torch.long)
                ranks[order] = torch.arange(1, order.numel() + 1)
                ranking_i = torch.zeros_like(labels[i], dtype=torch.long)
                ranking_i[valid_mask] = ranks
                ranking[i] = ranking_i

        return {
            "scores": scores, # [B,L,M]
            "labels": labels, #[B,L]
            "mask": mask, # [B,L]
            "uids": torch.tensor(uids, dtype=torch.long), # [B]
            "session_len": torch.tensor(lengths, dtype=torch.long),
            "ranking": ranking,
            "batch_size": B,
        }

    # 构造数据
    train_ds = SessionDataset(train_df, score_cols=score_cols, label_cols=label_cols)
    val_ds   = SessionDataset(val_df, score_cols=score_cols, label_cols=label_cols)
    user_num = max(train_df["uid"].max(), val_df["uid"].max(), test_df["uid"].max()) + 1

    train_loader = DataLoader(train_ds, batch_size=32,
                              shuffle=True, collate_fn=collate_fn)
    val_loader   = DataLoader(val_ds, batch_size=32,
                              shuffle=False, collate_fn=collate_fn)

    # 模型、优化器、损失
    model = AWELV(user_num=user_num, model_num=model_num, hidden_size=getattr(args, "hidden_size", 32)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=getattr(args, "lr", 1e-4),
                           weight_decay=getattr(args, "l2", 0.0))
    list_criterion = Listloss(args)

    def run_epoch(loader, train_mode=False):
        model.train(train_mode)
        total_loss, total_batches = 0.0, 0
        with torch.set_grad_enabled(train_mode):
            for batch in loader:
                scores = batch["scores"].to(device) # [B,L,M]
                labels = batch["labels"].to(device) # [B,L]
                mask = batch["mask"].to(device) #  [B,L]
                uids = batch["uids"].to(device) # [B]
                session_len = batch["session_len"].to(device)
                ranking = batch["ranking"].to(device)

                pred, w = model(scores, uids, mask) # [B,L]
                out_dict = {"ens_score": pred, "weights": w}
                in_batch = {
                    "scores": scores,
                    "ranking": ranking,
                    "session_len": session_len,
                    "batch_size": scores.size(0),
                }
                loss = list_criterion.forward(out_dict, in_batch)
                if train_mode:
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                total_loss += loss.item()
                total_batches += 1
        return total_loss / max(total_batches, 1)

    # 训练若干 epoch（可用 args.epochs 控制）
    best_state, best_val = None, float("inf")
    epochs = 10
    for _ in range(epochs):
        run_epoch(train_loader, train_mode=True)
        val_loss = run_epoch(val_loader, train_mode=False)
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
    if best_state:
        model.load_state_dict(best_state)

    # 给 test_df 每行打分
    model.eval()
    with torch.no_grad():
        # 预先计算每个 uid 的权重向量
        unique_uids = test_df["uid"].unique()
        uid_tensor = torch.tensor(unique_uids, dtype=torch.long, device=device)
        zeros_scores = torch.zeros(len(unique_uids), 1, model_num, device=device)  # dummy [B,1,M]
        w_cache = {}
        _, wtmp = model(zeros_scores, uid_tensor, mask=None)  # [B,1,M]
        for i, uid in enumerate(unique_uids):
            w_cache[uid] = wtmp[i, 0].cpu()  # [M]

        scores_np = test_df[score_cols].to_numpy(dtype=np.float32) #
        uids_np = test_df["uid"].to_numpy()
        out_scores = []
        for row_scores, uid in zip(scores_np, uids_np):
            w = w_cache[int(uid)]
            out_scores.append(float((torch.from_numpy(row_scores) * w).sum()))
    test_df = test_df.copy()
    test_df["score"] = out_scores
    return test_df



def rank_evaluate(df):
    df = df.sort_values(['sid', 'score'], ascending=[True, False])
    pred = df.groupby('sid')['score'].apply(list)
    cols = ['is_click', 'is_like', 'is_comment', 'is_follow', 'is_forward', 'long_view']
    k = 3
    for col in cols:
        label = df.groupby('sid')[col].apply(list)
        ndcg_score_list = []
        hr_score_list = []
        cnt = 0
        for y_true, y_pred in zip(label.tolist(), pred.tolist()):
            if len(set(y_true)) < 2:
                continue 
            cnt += 1
            ndcg_score_list.append(ndcg_score(y_true, y_pred, k=k))
            hr_score_list.append(hit_rate_at_k(y_true, y_pred, k=k))
        print(f'valid test samples:{cnt}, {col}_ndcg@{k}: {np.mean(ndcg_score_list):.4f}')
        print(f'valid test samples:{cnt}, {col}_hr@{k}: {np.mean(hr_score_list):.4f}')

