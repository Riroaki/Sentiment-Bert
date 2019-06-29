import numpy as np
import torch
from torch import nn
from torch.utils.data import random_split, DataLoader, Dataset, ConcatDataset
from pytorch_pretrained_bert import BertModel
from config import CONFIG, logger
from dataset import Tokenizer, SentimentDataset
from bert_spc import BERTSPC


def split_k_fold(data: Dataset, k: int) -> tuple:
    # Split into train and valid sets
    total_len = len(data)
    valset_len = total_len // k
    splitedsets = random_split(data, tuple(
        [valset_len] * (CONFIG.cross_val_fold - 1) + [
            total_len - valset_len * (CONFIG.cross_val_fold - 1)]))

    for fid in range(CONFIG.cross_val_fold):
        logger.info('Fold : {}'.format(fid))
        logger.info('>' * 100)
        trainset = ConcatDataset(
            [x for i, x in enumerate(splitedsets) if i != fid])
        validset = splitedsets[fid]
        yield trainset, validset


def train(model: nn.Module, criterion: nn.Module, optimizer: nn.Module,
          train_data_loader: DataLoader,
          valid_data_loader: DataLoader) -> str:
    max_valid_accuracy = 0.
    global_step = 0
    path = None
    for epoch in range(CONFIG.num_epoch):
        logger.info('>' * 100)
        logger.info('Epoch: {}'.format(epoch))
        n_correct, n_total, loss_total = 0, 0, 0
        # switch model to training mode
        model.train()
        # Train model using mini-batch
        for i_batch, sample_batched in enumerate(train_data_loader):
            global_step += 1
            # clear gradient accumulators
            optimizer.zero_grad()
            inputs = [sample_batched[col].to(CONFIG.device) for col in
                      ['text_bert_indices', 'bert_segments_ids']]
            outputs = model(inputs)
            targets = sample_batched['polarity'].to(CONFIG.device)

            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            n_correct += (torch.argmax(outputs, -1) == targets).sum().item()
            n_total += len(outputs)
            loss_total += loss.item() * len(outputs)
            if global_step % CONFIG.log_step == 0:
                accuracy = n_correct / n_total
                loss = loss_total / n_total
                logger.info('loss: {:.4f}, accuracy: {:.4f}'.format(loss,
                                                                    accuracy))

        # Check accuracy using validation set
        valid_accuracy = evaluate(model, valid_data_loader)
        logger.info('> [Valid] accuracy: {:.4f}'.format(valid_accuracy))
        # Save best model parameters: the one with highest accuracy
        if valid_accuracy > max_valid_accuracy:
            max_valid_accuracy = valid_accuracy
            path = 'state_dict/{}'.format(CONFIG.dataset)
            torch.save(model.state_dict(), path)
            logger.info('>> saved: {}'.format(path))
    return path


def evaluate(model: nn.Module, data_loader: DataLoader) -> float:
    correct, total = 0, 0
    targets_all, outputs_all = None, None
    # switch model to evaluation mode
    model.eval()
    with torch.no_grad():
        for batch, sample_batched in enumerate(data_loader):
            inputs = [sample_batched[col].to(CONFIG.device) for col in
                      ['text_bert_indices', 'bert_segments_ids']]
            targets = sample_batched['polarity'].to(CONFIG.device)
            outputs = model(inputs)
            correct += (torch.argmax(outputs, -1) == targets).sum().item()
            total += len(outputs)

            if targets_all is None:
                targets_all = targets
                outputs_all = outputs
            else:
                targets_all = torch.cat((targets_all, targets), dim=0)
                outputs_all = torch.cat((outputs_all, outputs), dim=0)

    accuracy = correct / total
    return accuracy


def main():
    # Build tokenizer and model
    tokenizer = Tokenizer(CONFIG.max_seq_len, CONFIG.bert_vocab_path)
    bert = BertModel.from_pretrained(CONFIG.bert_model_path)
    model = BERTSPC(bert, CONFIG).to(CONFIG.device)

    # Tokenize data
    train_data = SentimentDataset(CONFIG.train_file, tokenizer)
    test_data = SentimentDataset(CONFIG.test_file, tokenizer)

    # Log cuda memory information
    if CONFIG.device.type == 'cuda':
        logger.info('Cuda mem allocated: {}'.format(
            torch.cuda.memory_allocated(device=CONFIG.device.index)))

    # Start training, use k-fold
    accuracy_list = []
    criterion = nn.CrossEntropyLoss()
    params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = CONFIG.optimizer(params, lr=CONFIG.learning_rate,
                                 weight_decay=CONFIG.ridge_reg)

    # Train k-fold
    test_data_loader = DataLoader(dataset=test_data,
                                  batch_size=CONFIG.batch_size,
                                  shuffle=False)
    for train_set, valid_set in split_k_fold(train_data,
                                             CONFIG.cross_val_fold):
        train_data_loader = DataLoader(dataset=train_set,
                                       batch_size=CONFIG.batch_size,
                                       shuffle=True)
        valid_data_loader = DataLoader(dataset=valid_set,
                                       batch_size=CONFIG.batch_size,
                                       shuffle=False)
        best_model_path = train(model, criterion, optimizer, train_data_loader,
                                valid_data_loader)
        # Evaluate test accuracy
        model.load_state_dict(torch.load(best_model_path))
        # Change model into evaluation mode
        model.eval()
        accuracy = evaluate(model, test_data_loader)
        logger.info('>> [Test] accuracy: {:.4f}'.format(accuracy))
        accuracy_list.append(accuracy)

    # Calculate mean accuracy
    logger.info('>' * 100)
    mean_accuracy = np.mean(accuracy_list)
    logger.info('>>> [Test] mean accuracy: {:.4f}'.format(mean_accuracy))


if __name__ == '__main__':
    main()
