
import os
import tqdm
from utils import TrainingModel, create_dataloader, EarlyStopping
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
try:
    from tensorboardX import SummaryWriter
except:
    from torch.utils.tensorboard import SummaryWriter
import argparse
import torch
from utils.training import add_training_arguments
from utils.dataset import add_dataloader_arguments


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str)
    parser = add_training_arguments(parser)
    parser = add_dataloader_arguments(parser)
    parser.add_argument(
        "--num_epoches", type=int, default=1000, help="# of epoches at starting learning rate"
    )
    parser.add_argument(
        "--earlystop_epoch",
        type=int,
        default=5,
        help="Number of epochs without loss reduction before lowering the learning rate",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of steps to accumulate gradients before updating model weights",
    )

    opt = parser.parse_args()
    torch.manual_seed(opt.seed)

    train_data_loader = create_dataloader(opt, subdir="train", is_train=True)
    valid_data_loader = create_dataloader(opt, subdir="valid", is_train=False)

    print()
    print("# training batches = %d" % len(train_data_loader))
    print("# validation batches = %d" % len(valid_data_loader))
    print("# gradient accumulation steps = %d" % opt.gradient_accumulation_steps)
    print("# effective_batch_size = %d" % (opt.gradient_accumulation_steps * opt.batch_size))

    model = TrainingModel(opt, subdir=opt.name)
    writer = SummaryWriter(os.path.join(model.save_dir, "logs"))
    writer_loss_steps = len(train_data_loader) // 32
    early_stopping = None
    start_epoch = model.total_steps // len(train_data_loader)

    for epoch in range(start_epoch, opt.num_epoches + 1):
        if epoch > start_epoch:
            pbar = tqdm.tqdm(train_data_loader)
            ep_corr, ep_seen = 0, 0

            model.optimizer.zero_grad()

            for batch_idx, batch in enumerate(pbar):
                accumulation_step = batch_idx % opt.gradient_accumulation_steps       
                loss, corr, tot = model.train_on_batch(batch, accumulation_step)

                ep_corr += corr
                ep_seen += tot
                pbar.set_description(f"loss {loss:.4f} | acc {ep_corr / ep_seen:.3f}")

                if model.total_steps % writer_loss_steps == 0:
                    writer.add_scalar("train/loss", loss, model.total_steps)

            writer.add_scalar("train/accuracy", ep_corr / ep_seen, model.total_steps)
            
            if epoch % 10 == 0:
                model.save_networks(epoch)

        # Validation
        print("Validation ...", flush=True)
        y_true, y_pred, y_path = model.predict(valid_data_loader)

        acc = balanced_accuracy_score(y_true, y_pred > 0.0)
        lr = model.get_learning_rate()
        writer.add_scalar("lr", lr, model.total_steps)
        
        writer.add_scalar("valid/accuracy", acc, model.total_steps)
        
        
        print("After {} epoches: val acc = {}".format(epoch, acc), flush=True)
        

        # Early Stopping
        if early_stopping is None:
            early_stopping = EarlyStopping(
                init_score=acc, patience=opt.earlystop_epoch,
                delta=0.001, verbose=True,
            )
        else:
            if early_stopping(acc):
                print('Save best model', flush=True)
                model.save_networks('best')
            if early_stopping.early_stop:
                cont_train = model.adjust_learning_rate()
                if cont_train:
                    print("Learning rate dropped by 10, continue training ...", flush=True)
                    early_stopping.reset_counter()
                else:
                    print("Early stopping.", flush=True)
                    break
