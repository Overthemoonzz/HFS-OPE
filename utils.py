import torch.utils.data as data_utils
import torch
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import numpy as np
from sklearn.metrics import roc_auc_score
from model.ple import PLE
import ast

tqdm.pandas()


# kuairand_1k
def kuairand_1k_data(path, args):
    if not path:
        return
    
    is_debug = args.debug
    label_columns = ['is_click', 'is_like', 'is_comment', 'is_follow', 'is_forward','long_view']

    categorical_columns = [ 

        # session
        'hour_sin', 'hour_cos', 'dow', 'dow_sin', 'dow_cos',

        # user
        'user_id', 'user_active_degree','is_live_streamer', 'is_video_author', 'follow_user_num_range',
        'fans_user_num_range', 'friend_user_num_range', 'register_days_range',
        'onehot_feat0', 'onehot_feat1', 'onehot_feat2', 'onehot_feat3',
        'onehot_feat4', 'onehot_feat5', 'onehot_feat6', 'onehot_feat7',
        'onehot_feat8', 'onehot_feat9', 'onehot_feat10', 'onehot_feat11',
        'onehot_feat12', 'onehot_feat13', 'onehot_feat14', 'onehot_feat15',
        'onehot_feat16', 'onehot_feat17', 
        
        # item
        'video_id', 'author_id', 'video_type','upload_type', 'visible_status', 'music_id', 'music_type', 'tag',
        'upload_year', 'upload_month', 'upload_day', 'upload_dow', 'upload_dow_sin', 'upload_dow_cos', 
        'resolution_level', 'aspect_bucket','duration_bucket'
        ]

    user_columns = [
        'user_id', 'user_active_degree','is_live_streamer', 'is_video_author', 'follow_user_num_range',
        'fans_user_num_range', 'friend_user_num_range', 'register_days_range',
        'onehot_feat0', 'onehot_feat1', 'onehot_feat2', 'onehot_feat3',
        'onehot_feat4', 'onehot_feat5', 'onehot_feat6', 'onehot_feat7',
        'onehot_feat8', 'onehot_feat9', 'onehot_feat10', 'onehot_feat11',
        'onehot_feat12', 'onehot_feat13', 'onehot_feat14', 'onehot_feat15',
        'onehot_feat16', 'onehot_feat17', 
        ]

    item_columns = [
        'video_id', 'author_id', 'video_type','upload_type', 'visible_status', 'music_id', 'music_type', 'tag',
        'upload_year', 'upload_month', 'upload_day', 'upload_dow', 'upload_dow_sin', 'upload_dow_cos', 
        'resolution_level', 'aspect_bucket','duration_bucket'
    ]

    context_columns = ['hour_sin', 'hour_cos', 'dow', 'dow_sin', 'dow_cos']
    usecols = ['session_id', 'session_start'] + categorical_columns + label_columns 

    if is_debug:
        df = pd.read_csv(path, usecols=usecols, nrows=1000)
    else:
        df = pd.read_csv(path, usecols=usecols)

    def safe_list(s):
        cleaned = s.replace('nan', 'None')
        parsed = ast.literal_eval(cleaned)
        return [np.nan if v is None else v for v in parsed]

    for col in tqdm(item_columns+label_columns):
        df[col] = df[col].apply(safe_list)

    
    df = df.explode(item_columns+label_columns, ignore_index=True)  
    for col in tqdm(categorical_columns):
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

    new_columns = categorical_columns + label_columns + ['session_id','session_start']
    df = df.reindex(columns=new_columns)

    user_feature_dict, item_feature_dict, context_feature_dict = {}, {}, {}
    for idx, col in tqdm(enumerate(df.columns)):
        if col not in label_columns:
            if col in user_columns:
                user_feature_dict[col] = (len(df[col].unique()), idx)
            elif col in item_columns:
                item_feature_dict[col] = (len(df[col].unique()), idx)
            elif col in context_columns:
                context_feature_dict[col] = (len(df[col].unique()), idx)
    # {feats:(nunique, idx)}

    agg_dict = {col: list for col in item_columns + label_columns}
    for col in user_columns+context_columns+['session_start']:
        agg_dict[col] = 'first'
    df = df.groupby('session_id').agg(agg_dict).reset_index()

    df = df.sort_values('session_start')
    df = df.reindex(columns=new_columns)

    train_len = int(len(df) * 0.8)
    train_df = df[:train_len]
    tmp_df = df[train_len:]
    val_df = tmp_df[:int(len(tmp_df)/2)]
    test_df = tmp_df[int(len(tmp_df)/2):]
    if not args.is_ensemble_rank:
        train_df = train_df.drop(['session_start','session_id'], axis=1).explode(item_columns+label_columns, ignore_index=True)
        val_df = val_df.drop(['session_start'], axis=1).explode(item_columns+label_columns, ignore_index=True) 
        test_df = test_df.drop(['session_start'], axis=1).explode(item_columns+label_columns, ignore_index=True) 
        return train_df, val_df, test_df, user_feature_dict, item_feature_dict, context_feature_dict
    else:
        df = df.drop(['session_start'], axis=1).explode(item_columns+label_columns, ignore_index=True) 
        return None, None, df, user_feature_dict, item_feature_dict, context_feature_dict

class MTLDataset(data_utils.Dataset):
    def __init__(self, df, is_downsampling, args):
        self.task_cols = args.task_cols
        self.is_downsampling = is_downsampling

        click_neg_ratio = getattr(args, 'click_neg_ratio', 1.0)
        rng = np.random.default_rng(getattr(args, 'seed', 42))

        if self.is_downsampling and click_neg_ratio < 1.0:
            pos_mask = df['is_click'] == 1
            pos_df = df[pos_mask]
            neg_df = df[~pos_mask]
            k_keep = int(len(neg_df) * click_neg_ratio)
            keep_neg_idx = rng.choice(neg_df.index.to_numpy(), size=k_keep, replace=False)
            neg_keep_df = neg_df.loc[keep_neg_idx]
            self.df = pd.concat([pos_df, neg_keep_df], axis=0)
            self.df = self.df.sample(frac=1.0, random_state=getattr(args, 'seed', 42)).reset_index(drop=True)
        else:
            self.df = df.reset_index(drop=True)

        task_len = len(self.task_cols)
        if task_len == 0:
            raise ValueError('task_cols must be provided to MTLDataset')

        if 'session_id' in self.df.columns:
            feature_df = self.df.iloc[:, :-task_len-1].astype(np.float32)
            label_df = self.df.iloc[:, -task_len-1:-1].astype(np.float32)
            self.session_ids = self.df.iloc[:, -1].tolist()
        else:
            feature_df = self.df.iloc[:, :-task_len].astype(np.float32)
            label_df = self.df.iloc[:, -task_len:].astype(np.float32)
            self.session_ids = None

        self.features = torch.from_numpy(feature_df.to_numpy())
        self.labels = torch.from_numpy(label_df.to_numpy())

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if self.session_ids is None:
            return self.features[idx], self.labels[idx]
        return self.features[idx], self.labels[idx], self.session_ids[idx]

class SessionDataset(data_utils.Dataset):
    def __init__(self, df, score_cols, label_cols):
        self.data = []
        # 定义各行为的重要性权重（数值越大表示越重要）
        importance = {
            "is_forward": 5.0,
            "is_follow": 5.0,
            "is_comment": 4.0,
            "is_like": 3.0,
            "long_view": 2.0,
            "is_click": 1.0,
        }
        label_weights = np.array([importance.get(col, 0.0) for col in label_cols], dtype=np.float32)
        for sid, g in df.groupby("sid"):
            scores = g[score_cols].values.astype(np.float32)       # [L, M]
            # 将多行为按重要性映射为单一排序标签：取当前曝光对应行为的最高重要性
            labels = (g[label_cols].values * label_weights).sum(axis=1).astype(np.float32)  # [L]
            uid = int(g["uid"].iloc[0])
            self.data.append((uid, sid, scores, labels))

    def __len__(self): return len(self.data)

    def __getitem__(self, idx):
        uid, sid, scores, labels = self.data[idx]
        return {"uid": uid, "sid": sid, "scores": scores, "labels": labels, "len": len(labels)}
    
def _build_loader(dataset, batch_size, shuffle, args):
    loader_kwargs = {
        "batch_size": batch_size,
        "pin_memory": True,
        "num_workers": max(getattr(args, "num_workers", 0), 0),
        "persistent_workers": getattr(args, "num_workers", 0) > 0,
    }
    loader_kwargs["shuffle"] = shuffle
    return DataLoader(dataset, **loader_kwargs)

def get_train_loader(dataset, args):
    return _build_loader(dataset, args.train_batch_size, shuffle=True, args=args)

def get_val_loader(dataset, args):
    return _build_loader(dataset, args.val_batch_size, shuffle=False, args=args)

def get_test_loader(dataset, args):
    return _build_loader(dataset, args.test_batch_size, shuffle=False, args=args)

def get_caterscols(task_name):
    if task_name == 'census_income':
        categorical_columns = [
        'class_worker', 'det_ind_code', 'det_occ_code', 'education',
        'hs_college', 'major_ind_code', 'major_occ_code', 'race',
        'hisp_origin', 'sex', 'union_member', 'unemp_reason',
        'full_or_part_emp', 'tax_filer_stat', 'region_prev_res',
        'state_prev_res', 'det_hh_fam_stat', 'det_hh_summ',
        'mig_chg_msa', 'mig_chg_reg', 'mig_move_reg', 'mig_same',
        'mig_prev_sunbelt', 'fam_under_18', 'country_father',
        'country_mother', 'country_self', 'citizenship',
        'vet_question'
        ]
        return categorical_columns
    elif task_name == 'tenrec':
        categorical_columns = ["user_id", "item_id", "video_category", "gender", "age", "hist_1", "hist_2",
                       "hist_3", "hist_4", "hist_5", "hist_6", "hist_7", "hist_8", "hist_9", "hist_10"]
    elif task_name == 'kuairand_pure':
        categorical_columns = ['user_id', 'user_active_degree', 'is_live_streamer', 'is_video_author', 'follow_user_num_range', 'fans_user_num_range', 'friend_user_num_range', 'register_days_range',
                        'video_id', 'author_id', 'video_type', 'upload_dt', 'upload_type', 'server_width','server_height','music_id','music_type', 'tag'
                        ]
    elif task_name == 'kuairand_1k':
        
        categorical_columns = [ 

            # session
            'hour_sin', 'hour_cos', 'dow', 'dow_sin', 'dow_cos',

            # user
            'user_id', 'user_active_degree','is_live_streamer', 'is_video_author', 'follow_user_num_range',
            'fans_user_num_range', 'friend_user_num_range', 'register_days_range',
            'onehot_feat0', 'onehot_feat1', 'onehot_feat2', 'onehot_feat3',
            'onehot_feat4', 'onehot_feat5', 'onehot_feat6', 'onehot_feat7',
            'onehot_feat8', 'onehot_feat9', 'onehot_feat10', 'onehot_feat11',
            'onehot_feat12', 'onehot_feat13', 'onehot_feat14', 'onehot_feat15',
            'onehot_feat16', 'onehot_feat17', 
            
            # item
            'video_id', 'author_id', 'video_type','upload_type', 'visible_status', 'music_id', 'music_type', 'tag',
            'upload_year', 'upload_month', 'upload_day', 'upload_dow', 'upload_dow_sin', 'upload_dow_cos', 
            'resolution_level', 'aspect_bucket','duration_bucket'
            ]
        
        return categorical_columns

def generate_masked_validation_data(validation_data, user_feature_dict, item_feature_dict, context_feature_dict, categorical_columns):
    masked_validation_data_list = []
    all_feats_dict = user_feature_dict.copy() #(feats:(nunique,col_idx))
    all_feats_dict.update(item_feature_dict)
    all_feats_dict.update(context_feature_dict)
    # print(all_feats_dict)
    for feat, (_, idx) in all_feats_dict.items():
        masked_validation = validation_data.copy()
        col_name = validation_data.columns[idx]
        if feat not in categorical_columns:
            masked_validation[col_name] = masked_validation[col_name].mean()
        else:
            masked_validation[col_name] = 0
        masked_validation_data_list.append((feat, masked_validation))
    
    return masked_validation_data_list #[[feat1, masked_validation1],...]


def test(data_loader, model, user_feature_dict, args):
    device = args.device
    num_tasks = args.mtl_task_num
    y_labels = [[] for _ in range(num_tasks)]
    y_preds = [[] for _ in range(num_tasks)]
    sid_list = []   # 保存 sid
    uid_list = []
    model.to(device)
    model.eval()

    for batch in data_loader:
        if len(batch) == 3:
            x, y, sid = batch
        else:
            x, y = batch
            sid = None

        x, y = x.to(device), y.to(device)
        
        with torch.no_grad():
            logits = model(x)   
            # probs: [B, num_tasks]
        probs = torch.stack(
            [torch.sigmoid(l).view(-1) for l in logits],
            dim=1
        )

        # -------- 收集 label / pred --------
        for i in range(num_tasks):
            y_labels[i].extend(
                y[:, i].detach().cpu().view(-1).tolist()
            )
            y_preds[i].extend(
                probs[:, i].detach().cpu().view(-1).tolist()
            )

        # -------- 保存 sid and uid--------
        if sid is not None:
            sid_list.extend(sid)

        user_idx = user_feature_dict['user_id'][1]
        uid_batch = x[:, user_idx].detach().cpu().to(torch.int64).view(-1).tolist()
        uid_list.extend(uid_batch)
    # ================= ranking 模式：保存 pxtr + label + sid =================
    if args.is_ensemble_rank:

        preds_arr  = np.stack([np.array(lst) for lst in y_preds],  axis=1)  # [N, K]
        labels_arr = np.stack([np.array(lst) for lst in y_labels], axis=1)  # [N, K]

        pred_cols  = ['pclick', 'plike', 'pcomment', 'pfollow', 'pforward', 'plong_view']
        label_cols = ['is_click', 'is_like', 'is_comment', 'is_follow', 'is_forward', 'long_view']

        df_dict = {}

        # 任务 pxtr（已经是 sigmoid 后的概率）
        for i, name in enumerate(pred_cols):
            df_dict[name] = preds_arr[:, i]

        # 任务真实标签
        for i, name in enumerate(label_cols):
            df_dict[name] = labels_arr[:, i]

        # 加入 sid and uid
        df_dict["sid"] = sid_list
        df_dict["uid"] = uid_list
        df = pd.DataFrame(df_dict)
        return df

    # ================= 非 ranking：计算 AUC =================
    auc = [0 for _ in range(num_tasks)]
    for i in range(num_tasks):
        auc[i] = roc_auc_score(y_labels[i], y_preds[i])
    return auc


def compute_feature_importance(user_feature_dict, item_feature_dict, context_feature_dict,
                               task_name, model_path, val_loader,
                               masked_validation_data_list, device, args=None):
    """
    使用训练好的 PLE 模型，基于“特征掩蔽 → AUC 降幅”来计算特征重要性。
    """
    # 1）初始化 PLE 模型
    model = PLE(
        user_feature_dict,
        item_feature_dict,
        context_feature_dict,
        task_name,
        emb_dim=args.embedding_size,
        num_task=args.mtl_task_num,
        num_specific_experts=args.ple_num_specific_experts,
        num_shared_experts=args.ple_num_shared_experts,
        num_levels=args.ple_num_levels,
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    num_tasks = args.mtl_task_num  # 或 args.tasks，看你前面怎么定义的

    # 2）原始验证集 AUC（基线）
    baseline_auc = test(val_loader, model, args)  # 长度为 num_tasks 的 list

    # 3）逐特征掩蔽并计算 AUC drop
    importance_results = []
    for feature, masked_validation in tqdm(masked_validation_data_list):
        mask_dataset = MTLDataset(masked_validation, is_downsampling=False, args=args)
        masked_loader = get_val_loader(mask_dataset, args)

        auc_masked = test(masked_loader, model, args)  # 同样长度为 num_tasks 的 list

        # 为该 feature 构造一个结果字典
        result = {"Feature": feature}
        for i in range(num_tasks):
            # Importance1, Importance2, ..., ImportanceK
            result[f"Importance{i+1}"] = baseline_auc[i] - auc_masked[i]

        importance_results.append(result)

    return importance_results


def get_top_features(importance_results, top_n=5):
    """
    对所有任务分别取出前 top_n 个重要特征。
    返回格式：
    {
        "Task1": [...],
        "Task2": [...],
        ...,
        "TaskK": [...]
    }
    """
    importance_df = pd.DataFrame(importance_results)

    # 找所有 Importance{i} 列
    importance_cols = [col for col in importance_df.columns if col.startswith("Importance")]

    top_features = {}

    for i, col in enumerate(importance_cols):
        top_task = importance_df.sort_values(by=col, ascending=False).head(top_n)
        top_features[f"Task_{i+1}"] = top_task["Feature"].tolist()

    return top_features



