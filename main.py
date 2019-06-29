import torch
from torch import nn
from sklearn import metrics
from torch.utils.data import random_split, DataLoader
from pytorch_pretrained_bert import BertModel
from config import CONFIG, logger
from dataset import Tokenizer, SentimentDataset
from model import BERTSPC


def train(model: nn.Module, criterion: nn.Module, optimizer: nn.Module,
          train_data_loader: DataLoader,
          valid_data_loader: DataLoader) -> str:
    max_valid_accuracy = 0.
    max_valid_f1 = 0.
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
        valid_accuracy, valid_f1 = evaluate(model, valid_data_loader)
        logger.info('> [Valid] accuracy: {:.4f}, f1: {:.4f}'
                    .format(valid_accuracy, valid_f1))
        # Save best model parameters: the one with highest accuracy
        if valid_accuracy > max_valid_accuracy:
            max_valid_accuracy = valid_accuracy
            path = 'state_dict/{}_valid_accuracy{}'.format(
                CONFIG.dataset, round(valid_accuracy, 4))
            torch.save(model.state_dict(), path)
            logger.info('>> saved: {}'.format(path))
        if valid_f1 > max_valid_f1:
            max_valid_f1 = valid_f1
    return path


def evaluate(model: nn.Module, data_loader: DataLoader) -> tuple:
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
    f1 = metrics.f1_score(targets_all.cpu(),
                          torch.argmax(outputs_all, -1).cpu(),
                          labels=[0, 1, 2], average='macro')
    return accuracy, f1


def main():
    # Build tokenizer and model
    tokenizer = Tokenizer(CONFIG.max_seq_len, CONFIG.bert_vocab_path)
    bert = BertModel.from_pretrained(CONFIG.bert_model_path)
    model = BERTSPC(bert, CONFIG).to(CONFIG.device)

    # Tokenize data
    train_data = SentimentDataset(CONFIG.train_file, tokenizer)
    test_data = SentimentDataset(CONFIG.test_file, tokenizer)
    # Split train data
    total_len = len(train_data)
    valid_len = int(CONFIG.valid_ratio * total_len)
    train_len = total_len - valid_len
    train_set, valid_set = random_split(train_data, (train_len, valid_len))
    # Load data
    train_data_loader = DataLoader(dataset=train_set,
                                   batch_size=CONFIG.batch_size,
                                   shuffle=True)
    test_data_loader = DataLoader(dataset=test_data,
                                  batch_size=CONFIG.batch_size,
                                  shuffle=False)
    valid_data_loader = DataLoader(dataset=valid_set,
                                   batch_size=CONFIG.batch_size,
                                   shuffle=False)

    if CONFIG.device.type == 'cuda':
        logger.info('Cuda mem allocated: {}'.format(
            torch.cuda.memory_allocated(device=CONFIG.device.index)))

    # Start training...
    criterion = nn.CrossEntropyLoss()
    params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = CONFIG.optimizer(params, lr=CONFIG.learning_rate,
                                 weight_decay=CONFIG.ridge_reg)
    best_model_path = train(model, criterion, optimizer, train_data_loader,
                            valid_data_loader)

    # Evaluate test accuracy and f1
    model.load_state_dict(torch.load(best_model_path))
    model.eval()
    accuracy, f1 = evaluate(model, test_data_loader)
    logger.info('>> [Test] accuracy: {:.4f}, f1: {:.4f}'.format(accuracy, f1))


if __name__ == '__main__':
    main()
