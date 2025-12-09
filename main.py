import torch
import random
import argparse
from utils import *
from trainer import *
from ensemble import *
from model.mmoe import MMOE
from model.ple import PLE
from model.ope import OPE
import os
import numpy as np

def get_data(args):
    name = args.task_name
    path = args.dataset_path
    rng = random.Random(args.seed)
    if name == 'kuairand_1k':
        train_data, val_data, test_data, user_feature_dict, item_feature_dict, context_feature_dict = kuairand_1k_data(path, args)

    if not args.is_ensemble_rank:
        train_dataset = MTLDataset(train_data, is_downsampling=True, args=args)
        val_dataset = MTLDataset(val_data, is_downsampling=False, args=args)
        test_dataset = MTLDataset(test_data, is_downsampling=False, args=args)
        # dataloader
        train_dataloader = get_train_loader(train_dataset, args)
        val_dataloader = get_val_loader(val_dataset, args)
        test_dataloader = get_test_loader(test_dataset, args)
        return train_data, val_data, test_data, train_dataloader, val_dataloader, test_dataloader, user_feature_dict, item_feature_dict, context_feature_dict
    else:
        test_dataset = MTLDataset(test_data, is_downsampling=False, args=args)
        test_dataloader = get_test_loader(test_dataset, args)
        return None, None, test_data, None, None, test_dataloader, user_feature_dict, item_feature_dict, context_feature_dict


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
    parser.add_argument('--save_path', type=str, default='./experiments/')
    parser.add_argument('--model_name', default='')
    parser.add_argument('--is_ensemble_rank', action='store_true', help='enable ensemble ranking mode')
    parser.add_argument('--debug', action='store_true', help='debug mode')

    #MTL training
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--loss_fn', default='weighted_bce')
    parser.add_argument('--device', default='cpu')  # cuda:0
    parser.add_argument('--train_batch_size', type=int, default=1024)
    parser.add_argument('--val_batch_size', type=int, default=1024)
    parser.add_argument('--test_batch_size', type=int, default=1024)
    parser.add_argument('--num_workers', type=int, default=4, help='Number of DataLoader workers')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--click_neg_ratio', type=float, default=1.0)
    parser.add_argument("--pos_weight", nargs="+", type=float, default=[1.0]*6)
    parser.add_argument("--task_loss_weight", nargs="+", type=float, default=[1.0]*6)
    parser.add_argument('--task_cols', nargs='+', type=str, default='is_click is_like is_comment is_follow is_forward long_view')

    # share MTL model param
    parser.add_argument('--embedding_size', type=int, default=128, help='input embedding_size for mtl model')
    parser.add_argument('--mtl_task_num', type=int, default=6)

    # ope model param
    parser.add_argument('--ope_num_shared_experts', type=int, default=2)
    parser.add_argument('--ope_num_specific_experts', type=int, default=2)
    parser.add_argument('--ope_num_levels', type=int, default=2)
    parser.add_argument('--ope_model_load_path', type=str, default='')
    parser.add_argument('--top_n_feature_save_path', type=str, default='./experiments')
    parser.add_argument('--top_n_feature_load_path', type=str, default='')
    parser.add_argument('--top_n_feature_num', type=int, default=5)
    # mmoe model param
    parser.add_argument('--mmoe_num_experts', type=int, default=4)

    # ple model param
    parser.add_argument('--ple_model_path', default='', help='only using it when training ope')
    parser.add_argument('--ple_num_shared_experts', type=int, default=2)
    parser.add_argument('--ple_num_specific_experts', type=int, default=2)
    parser.add_argument('--ple_num_levels', type=int, default=2)

    # ensemble/rank
    parser.add_argument('--ensemble', type=str, default='single_sort', help='choose ensemble method')
    parser.add_argument('--pxtr_load_path', type=str, default='')
    parser.add_argument('--pxtr_save_path', type=str, default='./experiments')
    parser.add_argument('--cal_diversity', action='store_true', help='List-wise loss add diversity norm')
    parser.add_argument('--diversity_alpha', type=float, default=1e-6)



    # main
    args = parser.parse_args()
    device = torch.device(args.device)

    set_seed(args.seed)
    print(args)
    train_data, val_data, test_data, train_dataloader, val_dataloader, test_dataloader, user_feature_dict, item_feature_dict, context_feature_dict = get_data(args)
    task_name = args.task_name
    if args.model_name == 'mmoe':
        model = MMOE(
                    user_feature_dict, item_feature_dict, context_feature_dict, task_name, 
                    emb_dim=args.embedding_size, num_task=args.mtl_task_num, n_expert=args.mmoe_num_experts, 
                    )
    elif args.model_name == 'ple':
        model = PLE(user_feature_dict, item_feature_dict, context_feature_dict, task_name, 
                    emb_dim=args.embedding_size, num_task=args.mtl_task_num,
                    num_specific_experts=args.ple_num_specific_experts, num_shared_experts=args.ple_num_shared_experts, num_levels=args.ple_num_levels,
                    )
    elif args.model_name == 'ope':
        if args.top_n_feature_load_path:
            df = pd.read_csv(args.top_n_feature_load_path)
            top_features_tasks = df.to_dict(orient="list")
        else:
            ple_model_path = args.ple_model_path
            categorical_columns = get_caterscols(args.task_name)
            masked_validation_data_list = generate_masked_validation_data(
            val_data, user_feature_dict, item_feature_dict, context_feature_dict, categorical_columns
            )

            print('start computing feature importance...')
            if args.debug: #随机生成优选特征
                rng = random.Random(args.seed)  
                k = min(args.top_n_feature_num, len(categorical_columns))  
                top_features_tasks = {
                    f"Task_{i+1}": rng.sample(categorical_columns, k=k)
                    for i in range(args.mtl_task_num)
                }
            else:
                importance_results = compute_feature_importance(
                                user_feature_dict, 
                                item_feature_dict, 
                                context_feature_dict,
                                task_name=args.task_name, 
                                model_path=ple_model_path, 
                                val_loader=val_dataloader, 
                                masked_validation_data_list=masked_validation_data_list,
                                device=device, 
                                args=args)
            
                top_features_tasks = get_top_features(
                importance_results, top_n=args.top_n_feature_num
            )
            print(f'top n features of tasks:{top_features_tasks}')
            df = pd.DataFrame(top_features_tasks)
            df.to_csv(args.top_n_feature_save_path+'/top_n_features.csv', index=False)
            print(f'saving to {args.top_n_feature_save_path}')

        # transform dict to list
        task_feats = [[None] for _ in range(args.mtl_task_num)]
        for k, v in top_features_tasks.items():
            idx = int(k[-1])
            task_feats[idx-1]=v
        
        model = OPE( 
                    num_tasks = args.mtl_task_num,
                    user_feature_dict = user_feature_dict, 
                    item_feature_dict = item_feature_dict, 
                    context_feature_dict = context_feature_dict, 
                    task_feats = task_feats, #list
                    emb_dim = args.embedding_size, 
                    num_CGC_layers = args.ope_num_levels,
                    num_specific_experts = args.ope_num_specific_experts,
                    num_shared_experts = args.ope_num_shared_experts,)
    

    # stage1: MTL train 
    if not args.is_ensemble_rank:
        print(f"####################  Satge1: Training MTL models  ####################")
        mtlTrain(model, train_dataloader, val_dataloader, test_dataloader, args)
    # stage2: ensemble rank 
    else:
    # rank
        print(f"####################  Satge2: Ensemble Rank ####################")
        print(f"####################  Getting Pxtrs results ####################")
        if args.pxtr_load_path:
            df = pd.read_csv(args.pxtr_load_path)
        else:
            print(f"####################  Generating Pxtrs results ####################")
            model.to(device)
            model.load_state_dict(torch.load(args.ope_model_load_path, map_location=device))
            model.eval()
            df = test(test_dataloader, model, user_feature_dict, args)
            df.to_csv(args.pxtr_save_path+'/pxtrs.csv', index=False)
            print(f'saving pxtrs results to {args.pxtr_save_path}')
        
        train_df, val_df, test_df = split_dataset(df)
        print(f"#################### Scoring and Test ####################")
        ensemble = args.ensemble
        if ensemble == 'single_sort':
            test_df = single_sort(test_df)
        elif ensemble == 'lr':
            # training lr model
            test_df = train_test_lr(train_df, val_df, test_df, args)
        elif ensemble == 'mlp':
            test_df = train_test_mlp(train_df, val_df, test_df, args)
        elif ensemble == 'awelv':
            test_df = train_test_aWELv(train_df, val_df, test_df, args)
        rank_evaluate(test_df)




