import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import time
import warnings
import torch
import torch.distributed as dist
from contextlib import nullcontext
from torch import optim, nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from model.model import MiniMindConfig
from dataset.lm_dataset import PretrainDataset
from trainer.trainer_utils import (
    get_lr,
    Logger,
    is_main_process,
    lm_checkpoint,
    init_distributed_mode,
    setup_seed,
    init_model,
    SkipBatchSampler,
)

warnings.filterwarnings('ignore')


def train_epoch(epoch, loader, iters, start_step=0, wandb=None):
    """
    epoch : 第几个epoch
    loader :dataloader
    iters : 一个epoch迭代多少次
    start_step : 用于resume
    """
    # 因为代码中传入了 loss_mask。通常是因为输入数据包含 Padding（填充符），
    # 我们需要保留所有 Loss，稍后手动乘以 Mask（把 Padding 的 Loss 变成 0），然后再求平均
    loss_fct = nn.CrossEntropyLoss(reduction='none')
    start_time = time.time()
    #
    for step, (X, Y, loss_mask) in enumerate(loader, start=start_step + 1):
        X = X.to(args.device)
        Y = Y.to(args.device)
        loss_mask = loss_mask.to(args.device)
        lr = get_lr(epoch * iters + step, iters * args.epochs, args.learning_rate)

        # 可以修改每一个optim的lr, 这里直接统一
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
            # 混合精度:
            # 矩阵乘法和卷积使用f16
            # softmax和求和使用f32
            with autocast_ctx:
                # 前向传播
                res = model(X)
                # loss计算,拍扁所有token,计算每个生成token的交叉熵损失
                loss = loss_fct(res.logits.view(-1, lm_config.vocab_size), Y.view(-1))
                # 3. 应用 Mask (过滤掉 Padding 的 loss)
                # loss_mask.view(-1) 把 mask 也拉平，形状 [8]
                # loss 是刚才算出来的每个 token 的 loss，形状也是 [8]
                loss = (loss * loss_mask.view(-1)).sum() / loss_mask.sum()
                loss = loss / args.accumulation_steps

            # 4. 反向传播
            scaler.scale(loss).backward()  # 如果用了混合精度，通常配合 GradScaler
            if (step + 1) % args.accumulation_steps == 0:

                scaler.unscale_(optimizer)  # 先还原梯度
                # 防止梯度爆炸,进行裁剪
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

                scaler.step(optimizer)
                scaler.update()
                # 梯度设置为none可以节省内存
                optimizer.zero_grad(set_to_none=True)

            # log记录
            if step % args.log_interval == 0 or step == iters - 1:
                spend_time = time.time() - start_time
                current_loss = loss.item() * args.accumulation_steps  # 恢复真实损失值
                current_lr = optimizer.param_groups[-1]["lr"]  # 当前学习率

                eta_min = spend_time / (step + 1) * iters // 60 - spend_time // 60

                Logger(
                    f"Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}) loss:{current_loss:.6f} lr:{current_lr:.12f} epoch_Time:{eta_min}min:"
                )

                # 记录到实验跟踪系统
                if wandb:
                    wandb.log(
                        {"loss": current_loss, "lr": current_lr, "epoch_Time": eta_min}
                    )

            if (
                step % args.save_interval == 0 or step == iters - 1
            ) and is_main_process():
                model.eval()  # 切换到评估模式

                # 构建保存路径
                moe_suffix = (
                    "_moe"
                    if hasattr(lm_config, "use_moe") and lm_config.use_moe
                    else ""
                )
                ckp = f"{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}{moe_suffix}.pth"

                # 📚 分布式模型保存
                # DDP模型需要通过.module访问真正的模型
                if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                    state_dict = model.module.state_dict()
                else:
                    state_dict = model.state_dict()

                # 📚 半精度保存
                # 将float32参数转为float16，减少存储空间
                state_dict = {k: v.half() for k, v in state_dict.items()}
                torch.save(state_dict, ckp)

                # 保存完整训练状态 , 在utils里面有
                lm_checkpoint(
                    lm_config,
                    weight=args.save_weight,
                    model=model,
                    optimizer=optimizer,
                    scaler=scaler,
                    epoch=epoch,
                    step=step,
                    wandb=wandb,
                    save_dir="checkpoints",
                )

                model.train()  # 恢复训练模式


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind Pretraining")
    parser.add_argument("--save_dir", type=str, default="../out", help="模型保存目录")
    parser.add_argument(
        '--save_weight', default='pretrain', type=str, help="保存权重的前缀名"
    )
    parser.add_argument(
        "--epochs", type=int, default=1, help="训练轮数（建议1轮zero或2-6轮充分训练）"
    )
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=5e-4, help="初始学习率")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="训练设备",
    )
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型")
    parser.add_argument("--num_workers", type=int, default=8, help="数据加载线程数")
    parser.add_argument(
        "--accumulation_steps", type=int, default=8, help="梯度累积步数"
    )
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=100, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=1000, help="模型保存间隔")
    parser.add_argument('--hidden_size', default=512, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量")
    parser.add_argument(
        '--max_seq_len',
        default=340,
        type=int,
        help="训练的最大截断长度（中文1token≈1.5~1.7字符）",
    )
    parser.add_argument(
        '--use_moe',
        default=0,
        type=int,
        choices=[0, 1],
        help="是否使用MoE架构（0=否，1=是）",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="../dataset/pretrain_hq.jsonl",
        help="预训练数据路径",
    )
    parser.add_argument(
        '--from_weight',
        default='none',
        type=str,
        help="基于哪个权重训练，为none则从头开始",
    )
    parser.add_argument(
        '--from_resume',
        default=0,
        type=int,
        choices=[0, 1],
        help="是否自动检测&续训（0=否，1=是）",
    )
    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb")
    parser.add_argument(
        "--wandb_project", type=str, default="MiniMind-Pretrain", help="wandb项目名"
    )
    args = parser.parse_args()

    # ========== 1. 初始化环境和随机种子 ==========
    local_rank = init_distributed_mode()
    if dist.is_initialized():
        args.device = f"cuda:{local_rank}"
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))

    # ========== 2. 配置目录、模型参数、检查ckp ==========
    os.makedirs(args.save_dir, exist_ok=True)
    lm_config = MiniMindConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        use_moe=bool(args.use_moe),
    )
    ckp_data = (
        lm_checkpoint(lm_config, weight=args.save_weight, save_dir='../checkpoints')
        if args.from_resume == 1
        else None
    )

    # ========== 3. 设置混合精度 ==========
    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    autocast_ctx = (
        nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)
    )

    # ========== 4. 配wandb ==========
    wandb = None
    if args.use_wandb and is_main_process():
        import swanlab as wandb

        wandb_id = ckp_data.get('wandb_id') if ckp_data else None
        resume = 'must' if wandb_id else None
        wandb_run_name = f"MiniMind-Pretrain-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LearningRate-{args.learning_rate}"
        wandb.init(
            project=args.wandb_project, name=wandb_run_name, id=wandb_id, resume=resume
        )

    # ========== 5. 定义模型、数据、优化器 ==========
    model, tokenizer = init_model(lm_config, args.from_weight, device=args.device)
    train_ds = PretrainDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == 'float16'))
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    # ========== 6. 从ckp恢复状态 ==========
    start_epoch, start_step = 0, 0
    if ckp_data:
        model.load_state_dict(ckp_data['model'])
        optimizer.load_state_dict(ckp_data['optimizer'])
        scaler.load_state_dict(ckp_data['scaler'])
        start_epoch = ckp_data['epoch']
        start_step = ckp_data.get('step', 0)

    # ========== 7. DDP包模型 ==========
    if dist.is_initialized():
        model._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"}
        model = DistributedDataParallel(model, device_ids=[local_rank])

    # ========== 8. 开始训练 ==========
    for epoch in range(start_epoch, args.epochs):
        train_sampler and train_sampler.set_epoch(epoch)
        if epoch == start_epoch and start_step > 0:  # 第一个epoch且存在检查点
            batch_sampler = SkipBatchSampler(
                train_sampler or range(len(train_ds)), args.batch_size, start_step + 1
            )
            loader = DataLoader(
                train_ds,
                batch_sampler=batch_sampler,
                num_workers=args.num_workers,
                pin_memory=True,
            )
            Logger(
                f'Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}个step，从step {start_step + 1}开始'
            )
            train_epoch(epoch, loader, len(loader) + start_step + 1, start_step, wandb)
        else:  # 默认从头开始
            loader = DataLoader(
                train_ds,
                batch_size=args.batch_size,
                shuffle=(train_sampler is None),
                sampler=train_sampler,
                num_workers=args.num_workers,
                pin_memory=True,
            )
            train_epoch(epoch, loader, len(loader), 0, wandb)

    # ========== 9. 清理分布进程 ==========
    if dist.is_initialized():
        dist.destroy_process_group()
