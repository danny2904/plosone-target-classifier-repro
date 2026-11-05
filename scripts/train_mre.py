import argparse
import os
import sys

# Keep old entrypoint, delegate to unified trainer.
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../'))
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)


def parse_args():
    parser = argparse.ArgumentParser(description='VN wrapper -> unified multitask trainer')
    parser.add_argument('--data_path', type=str, default='data/vihos_prepared.jsonl')
    parser.add_argument('--model_name', type=str, default='xlm-roberta-base')
    parser.add_argument('--output_dir', type=str, default='output/mre')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--learning_rate', type=float, default=2e-5)
    parser.add_argument('--max_len', type=int, default=256)
    parser.add_argument('--alpha', type=float, default=5.0)
    parser.add_argument('--use_fusion', type=int, default=1)
    parser.add_argument('--use_consistency', type=int, default=1)
    parser.add_argument('--lambda_const', type=float, default=0.1)
    parser.add_argument('--use_class_weights', type=int, default=1)
    parser.add_argument('--best_metric', type=str, default='val_f1', choices=['val_f1', 'val_acc', 'val_loss'])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--init_checkpoint', type=str, default=None)
    parser.add_argument('--early_stop_patience', type=int, default=0)
    parser.add_argument('--vn_short_unsafe_boost', type=float, default=1.0)
    parser.add_argument('--vn_short_max_words', type=int, default=6)
    parser.add_argument('--use_token_class_weights', type=int, default=0)
    parser.add_argument('--token_focal_gamma', type=float, default=0.0)
    parser.add_argument('--token_dice_lambda', type=float, default=0.0)
    parser.add_argument('--hard_ids_path', type=str, default=None)
    parser.add_argument('--hard_id_boost', type=float, default=1.0)
    parser.add_argument('--hard_id_apply_to_cls', type=int, default=0)
    parser.add_argument('--token_weight_from_cls_reweight', type=int, default=0)
    args = parser.parse_args()
    args.lang = 'vn'
    args.use_fusion = bool(args.use_fusion)
    args.use_consistency = bool(args.use_consistency)
    args.use_class_weights = bool(args.use_class_weights)
    return args


if __name__ == '__main__':
    args = parse_args()
    from train_multitask_unified import run_training, seed_everything

    seed_everything(args.seed)
    run_training(args)
