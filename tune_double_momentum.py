import optuna
import subprocess
import os
import re

def objective(trial):
    # 定义要搜索的参数范围
    momentum_0 = trial.suggest_float("momentum_0", 0.8, 0.92)
    momentum_1 = trial.suggest_float("momentum_1", 0.92, 0.99)

    print(f"Running trial {trial.number} with momentum: [{momentum_0:.4f}, {momentum_1:.4f}]")

    # 执行训练脚本并获取结果
    cmd = [
        "torchrun", "--standalone", "--nproc_per_node=4",
        "train_gpt.py",
        "--muon_type", "double_momentum",
        "--muon_momentum_0", str(momentum_0),
        "--muon_momentum_1", str(momentum_1),
    ]

    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace"
    )

    # 将日志写入文件方便调试
    log_file = f"logs/optuna_trial_{trial.number}.log"
    with open(log_file, "w") as f:
        f.write(result.stdout)

    # 从输出中提取最后的 val_loss
    val_loss_match = re.search(r"val_loss:(\d+\.\d+)", result.stdout[::-1])
    if val_loss_match:
        val_loss = float(val_loss_match.group(1)[::-1])
        print(f"Trial {trial.number} finished with val_loss={val_loss:.4f}")
        return val_loss
    else:
        print("Validation loss not found in output. Returning high value.")
        return float("inf")


# ✅ 新增的 callback 函数
def log_best_callback(study: optuna.Study, trial: optuna.Trial):
    print("\n" + "=" * 50)
    print(f"Trial {trial.number} completed.")
    print(f"Current best params: {study.best_trial.params}")
    print(f"Current best val_loss: {study.best_value:.4f}")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=50, callbacks=[log_best_callback])  # 添加 callback

    print("Final Best trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")