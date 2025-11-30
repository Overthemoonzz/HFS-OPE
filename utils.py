import torch.utils.data as data_utils
import torch
import pandas as pd
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from sklearn.metrics import roc_auc_score
from model.ple import PLE

tqdm.pandas()

# Tenrec
def tenrec_data(path=None, args=None):
    if not path:
        return
    df = pd.read_csv(path, usecols=["user_id", "item_id", "click", "like", "video_category", "gender", "age", "hist_1", "hist_2",
                       "hist_3", "hist_4", "hist_5", "hist_6", "hist_7", "hist_8", "hist_9", "hist_10"])
    # df = df[:100000]
    df['video_category'] = df['video_category'].astype(str)
    df = sample_data(df)
    if args.mtl_task_num == 2:
        label_columns = ['click', 'like']
        categorical_columns = ["user_id", "item_id", "video_category", "gender", "age", "hist_1", "hist_2",
                       "hist_3", "hist_4", "hist_5", "hist_6", "hist_7", "hist_8", "hist_9", "hist_10"]
    elif args.mtl_task_num == 1:
        label_columns = ['click']
        categorical_columns = ["user_id", "item_id", "video_category", "gender", "age", "hist_1", "hist_2",
                               "hist_3", "hist_4", "hist_5", "hist_6", "hist_7", "hist_8", "hist_9", "hist_10"]
    else:
        label_columns = ['like']
        categorical_columns = ["user_id", "item_id", "video_category", "gender", "age", "hist_1", "hist_2",
                               "hist_3", "hist_4", "hist_5", "hist_6", "hist_7", "hist_8", "hist_9", "hist_10"]
    user_columns = ["user_id", "gender", "age"]
    for col in tqdm(categorical_columns):
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

    new_columns = categorical_columns + label_columns
    df = df.reindex(columns=new_columns)

    user_feature_dict, item_feature_dict = {}, {}
    for idx, col in tqdm(enumerate(df.columns)):
        if col not in label_columns:
            if col in user_columns:
                user_feature_dict[col] = (len(df[col].unique()), idx)
            else:
                item_feature_dict[col] = (len(df[col].unique()), idx)

    df = df.sample(frac=1)
    train_len = int(len(df) * 0.8)
    train_df = df[:train_len]
    tmp_df = df[train_len:]
    val_df = tmp_df[:int(len(tmp_df)/2)]
    test_df = tmp_df[int(len(tmp_df)/2):]
    return train_df, val_df, test_df, user_feature_dict, item_feature_dict


def sample_data(df):
    p_df = df[df.click.isin([1])]
    n_df = df[df.click.isin([0])]
    del df
    n_df = n_df.sample(n=len(p_df)*2)
    df = pd.concat([p_df, n_df], ignore_index=True)
    del p_df, n_df
    df = df.sample(frac=1)
    return df

# census-income
def censusincome_data(args=None):
    """
    读取 census-income 数据，做特征处理和数据集划分。
    """

    seed = args.seed
    # 原始数据的列名
    column_names = ['age', 'class_worker', 'det_ind_code', 'det_occ_code', 'education', 'wage_per_hour', 'hs_college',
                    'marital_stat', 'major_ind_code', 'major_occ_code', 'race', 'hisp_origin', 'sex', 'union_member',
                    'unemp_reason', 'full_or_part_emp', 'capital_gains', 'capital_losses', 'stock_dividends',
                    'tax_filer_stat', 'region_prev_res', 'state_prev_res', 'det_hh_fam_stat', 'det_hh_summ',
                    'instance_weight', 'mig_chg_msa', 'mig_chg_reg', 'mig_move_reg', 'mig_same', 'mig_prev_sunbelt',
                    'num_emp', 'fam_under_18', 'country_father', 'country_mother', 'country_self', 'citizenship',
                    'own_or_self', 'vet_question', 'vet_benefits', 'weeks_worked', 'year', 'income_50k']

    # 读取训练集 / 测试集（原作者将“test”再划一半做 valid）
    train_df = pd.read_csv('./data/census_income/census-income.data.gz', delimiter=',', header=None,
                           index_col=None, names=column_names)
    test_df = pd.read_csv('./data/census_income/census-income.test.gz', delimiter=',', header=None,
                          index_col=None, names=column_names)
    df = pd.concat([train_df, test_df], axis=0)
    # 多任务标签列：收入是否 >50K & 婚姻状态
    label_columns = ['income_50k', 'marital_stat']

    # 需要做 labelencode 的类别特征列
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
    for col in tqdm(categorical_columns):
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
    df['marital_stat'] = (df['marital_stat'] == ' Never married').astype(int)
    df['income_50k']   = (df['income_50k'] == ' 50000+.').astype(int)
    cols = [c for c in df.columns if c not in ['marital_stat', 'income_50k']]
    df = df[cols + ['marital_stat', 'income_50k']]

    df = df.sample(frac=1)
    train_len = int(len(df) * 0.8)
    train_df = df[:train_len]
    tmp_df = df[train_len:]
    val_df = tmp_df[:int(len(tmp_df)/2)]
    test_df = tmp_df[int(len(tmp_df)/2):]

    user_feature_dict, item_feature_dict = {}, {}
    for idx, col in tqdm(enumerate(df.columns)):
        if col in ['marital_stat', 'income_50k']:
            continue
        if col in categorical_columns:
            user_feature_dict[col] = (len(df[col].unique()), idx)
        else:
            user_feature_dict[col] = (1, idx)

    return train_df, val_df, test_df, user_feature_dict, item_feature_dict


class mtlDataSet(data_utils.Dataset):
    def __init__(self, data, args):
        self.feature = data[0]
        self.args = args
        if args.mtl_task_num == 2:
            self.label1 = data[1]
            self.label2 = data[2]
        else:
            self.label = data[1]

    def __getitem__(self, index):
        feature = self.feature[index]
        if self.args.mtl_task_num == 2:
            label1 = self.label1[index]
            label2 = self.label2[index]
            return feature, label1, label2
        else:
            label = self.label[index]
            return feature, label

    def __len__(self):
        return len(self.feature)


def get_train_loader(dataset, args):
    if args.is_parallel:
        dataloader = data_utils.DataLoader(dataset, batch_size=args.train_batch_size, sampler=DistributedSampler(dataset))
    else:
        dataloader = data_utils.DataLoader(dataset, batch_size=args.train_batch_size, shuffle=True, pin_memory=True)
    return dataloader

def get_val_loader(dataset, args):
    if args.is_parallel:
        dataloader = data_utils.DataLoader(dataset, batch_size=args.val_batch_size, sampler=DistributedSampler(dataset))
    else:
        dataloader = data_utils.DataLoader(dataset, batch_size=args.val_batch_size, shuffle=False, pin_memory=True)
    return dataloader

def get_test_loader(dataset, args):
    if args.is_parallel:
        dataloader = data_utils.DataLoader(dataset, batch_size=args.test_batch_size, sampler=DistributedSampler(dataset))
    else:
        dataloader = data_utils.DataLoader(dataset, batch_size=args.test_batch_size, shuffle=False, pin_memory=True)
    return dataloader

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
    elif task_name == 'Tenrec':
        categorical_columns = ["user_id", "item_id", "video_category", "gender", "age", "hist_1", "hist_2",
                       "hist_3", "hist_4", "hist_5", "hist_6", "hist_7", "hist_8", "hist_9", "hist_10"]
        return categorical_columns

def generate_masked_validation_data(validation_data, user_feature_dict, item_feature_dict, categorical_columns):
    masked_validation_data_list = []
    all_feats_dict = user_feature_dict.copy() #(feats:(nunique,col_idx))
    all_feats_dict.update(item_feature_dict)
    # print(user_feature_dict)
    # print(item_feature_dict)
    # print(all_feats_dict)
    for feat, (_, idx) in all_feats_dict.items():
        masked_validation = validation_data.copy()
        col_name = validation_data.columns[idx]
        if feat not in categorical_columns:
            masked_validation[col_name] = masked_validation[col_name].mean()
        else:
            masked_validation[col_name] = 0
        masked_validation_data_list.append((feat, masked_validation))
    
    return masked_validation_data_list

def test(data_loader, model, args):
    device = args.device
    y_test_click_true = []
    y_test_like_true = []
    y_test_click_predict = []
    y_test_like_predict = []
    for idx, (x, y1, y2) in enumerate(data_loader):
        x, y1, y2 = x.to(device), y1.to(device), y2.to(device)
        predict = model(x)
        y_test_click_true += list(y1.squeeze().cpu().numpy())
        y_test_like_true += list(y2.squeeze().cpu().numpy())
        y_test_click_predict += list(predict[0].squeeze().cpu().detach().numpy())
        y_test_like_predict += list(predict[1].squeeze().cpu().detach().numpy())
    click_auc = roc_auc_score(y_test_click_true, y_test_click_predict)
    like_auc = roc_auc_score(y_test_like_true, y_test_like_predict)
    return click_auc, like_auc

def compute_feature_importance(user_feature_dict, item_feature_dict, task_name, model_path, val_loader, masked_validation_data_list,
                                device, args=None):
    """
    使用训练好的 PLE 模型，基于“特征掩蔽 → AUC 降幅”来计算特征重要性。
    """
    # 1）初始化 PLE 模型
    model = PLE(user_feature_dict, item_feature_dict, task_name, emb_dim=args.embedding_size, num_task=2)
    model = model.to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # 2）原始验证集 AUC（基线）
    baseline_auc1, baseline_auc2 = test(val_loader, model, args)

    # 3）逐特征掩蔽并计算 AUC drop
    importance_results = []
    for feature, masked_validation in masked_validation_data_list:

        mask_dataset = (masked_validation.iloc[:, :-2].values, masked_validation.iloc[:, -2].values, masked_validation.iloc[:, -1].values)
        mask_dataset = mtlDataSet(mask_dataset, args)
        masked_loader = get_val_loader(mask_dataset, args)

        auc1_masked, auc2_masked = test(masked_loader, model, args)
        importance_results.append({
            "Feature": feature,
            "Importance1": baseline_auc1 - auc1_masked,
            "Importance2": baseline_auc2 - auc2_masked
        })
    
    return importance_results

def get_top_features(importance_results, top_n=5):
    """
    分别取出两个任务的前 top_n 个重要特征。
    """
    importance_df = pd.DataFrame(importance_results)

    top_task1 = importance_df.sort_values(by="Importance1", ascending=False).head(top_n)
    top_task2 = importance_df.sort_values(by="Importance2", ascending=False).head(top_n)
    
    top_features_task1 = top_task1["Feature"].tolist()
    top_features_task2 = top_task2["Feature"].tolist()

    return top_features_task1, top_features_task2


