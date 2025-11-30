import torch
import random
import argparse
from utils import *
from trainer import *
from model.esmm import ESMM
from model.mmoe import MMOE
from model.ple import PLE
from model.hfs_ope import HFSOPE
import os
import numpy as np
from torch.utils.data import DataLoader

def get_data(args):
    name = args.task_name
    path = args.dataset_path
    rng = random.Random(args.seed)
    if name == 'tenrec':
        train_data, val_data, test_data, user_feature_dict, item_feature_dict = tenrec_data(path, args)
    elif name == 'census_income':
        train_data, val_data, test_data, user_feature_dict, item_feature_dict = censusincome_data(args)
    #dataloader
    if args.mtl_task_num == 2:
        train_dataset = (train_data.iloc[:, :-2].values, train_data.iloc[:, -2].values, train_data.iloc[:, -1].values)
        val_dataset = (val_data.iloc[:, :-2].values, val_data.iloc[:, -2].values, val_data.iloc[:, -1].values)
        test_dataset = (test_data.iloc[:, :-2].values, test_data.iloc[:, -2].values, test_data.iloc[:, -1].values)
    else:
        train_dataset = (train_data.iloc[:, :-1].values, train_data.iloc[:, -1].values)
        val_dataset = (val_data.iloc[:, :-1].values, val_data.iloc[:, -1].values)
        test_dataset = (test_data.iloc[:, :-1].values, test_data.iloc[:, -1].values)
    train_dataset = mtlDataSet(train_dataset, args)
    val_dataset = mtlDataSet(val_dataset, args)
    test_dataset = mtlDataSet(test_dataset, args)
    # dataloader
    train_dataloader = get_train_loader(train_dataset, args)
    val_dataloader = get_val_loader(val_dataset, args)
    test_dataloader = get_test_loader(test_dataset, args)
    return train_data, val_data, test_data, train_dataloader, val_dataloader, test_dataloader, user_feature_dict, item_feature_dict



def set_seed(seed, re=True):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    if re:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    else:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--task_name', default='')
    parser.add_argument('--dataset_path', type=str, default='')
    parser.add_argument('--train_batch_size', type=int, default=1024)
    parser.add_argument('--val_batch_size', type=int, default=1024)
    parser.add_argument('--test_batch_size', type=int, default=1024)
    parser.add_argument('--save_path', type=str, default='./experiments/')

    parser.add_argument('--model_name', default='')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--model_path', default='', help='using it when training hfsope')

    #training
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--device', default='cpu')  # cuda:0
    parser.add_argument('--is_parallel', type=bool, default=False)

    # mtl model param
    parser.add_argument('--hidden_size', type=int, default=128, help='Size of hidden vectors (model)')
    parser.add_argument('--dropout', type=float, default=0.3,
                        help='Dropout probability to use throughout the model')
    parser.add_argument('--embedding_size', type=int, default=128, help='embedding_size for model')
    parser.add_argument('--mtl_task_num', type=int, default=1, help='0:like, 1:click, 2:two tasks')
    parser.add_argument('--featuregate', type=bool, default=False)

    # main
    args = parser.parse_args()
    if args.is_parallel:
        torch.distributed.init_process_group(backend="nccl")
        torch.cuda.set_device(args.local_rank)
    device = torch.device(args.device)

    set_seed(args.seed)
    print(args)
    train_data, val_data, test_data, train_dataloader, val_dataloader, test_dataloader, user_feature_dict, item_feature_dict = get_data(args)
    num_task = args.mtl_task_num
    task_name = args.task_name
    if args.model_name == 'esmm':
        model = ESMM(user_feature_dict, item_feature_dict, task_name, emb_dim=args.embedding_size, num_task=num_task)
    elif args.model_name == 'mmoe':
        model = MMOE(user_feature_dict, item_feature_dict, task_name, emb_dim=args.embedding_size, num_task=num_task)
    elif args.model_name == 'ple':
        model = PLE(user_feature_dict, item_feature_dict, task_name, emb_dim=args.embedding_size, num_task=num_task)
    elif args.model_name == 'hfs_ope':
        model_path = args.model_path
        categorical_columns = get_caterscols(args.task_name)
        masked_validation_data_list = generate_masked_validation_data(
        val_data, user_feature_dict, item_feature_dict, categorical_columns
        )

        importance_results = compute_feature_importance(
                        user_feature_dict, 
                        item_feature_dict, 
                        task_name=args.task_name, 
                        model_path=model_path, 
                        val_loader=val_dataloader, 
                        masked_validation_data_list=masked_validation_data_list,
                        device=device, 
                        args=args)
        
        top_features_task1, top_features_task2 = get_top_features(
        importance_results, top_n=5
    )
        print(f'top_features_task1:{top_features_task1}')
        print(f'top_features_task2:{top_features_task2}')
        train_data, val_data, test_data, train_dataloader, val_dataloader, test_dataloader, user_feature_dict, item_feature_dict = get_data(args)


        model = HFSOPE( 
                 num_tasks = 2,
                 user_feature_dict = user_feature_dict,
                 item_feature_dict = item_feature_dict,
                 task1_feats = top_features_task1, #list
                 task2_feats = top_features_task2,
                 emb_dim = 128, #假设每一个头的embedding size 相同以简化代码
                 num_CGC_layers = 2,
                 num_specific_experts = 1,
                 num_shared_experts = 1,
                 experts_out = 128,
                 experts_hidden = 128,
                 towers_hidden = 128,
                 featuregate = args.featuregate,
                 gate_hidden_ratio = 1.0)
        
    mtlTrain(model, train_dataloader, val_dataloader, test_dataloader, args)






