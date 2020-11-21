from argparse import ArgumentParser
from tqdm.auto import tqdm
import numpy as np
import torch
from torchtext.data import Field, TabularDataset, BucketIterator, Iterator
from entity_nlm import EntitiyNLM, EntityOverflow
torch.autograd.set_detect_anomaly(True)


device = 0
MAX_ENT_LENGTH = 50
MAX_NUM_ENTITIES = 64
HIDDEN_SIZE = 256
BATCH_SIZE = 8
LR = 0.0005
N_LAYERS = 2
DATA_DIR = '../data/'
loss_fn = torch.nn.CrossEntropyLoss()

np.random.seed(1)
torch.manual_seed(1)


def load_data():
    TEXT, LABELS, LENGTH = Field(sequential=True, use_vocab=True), \
                           Field(sequential=True, use_vocab=False, preprocessing=lambda x: list(map(int, x)),
                                 pad_token=-1), \
                           Field(sequential=False, use_vocab=False)
    train_set = TabularDataset(path=DATA_DIR + 'train.tok.tsv', format='TSV', fields=[('text', TEXT),
                                                                                      ('labels', LABELS),
                                                                                      ('length', LENGTH)],
                               skip_header=True)
    val_set = TabularDataset(path=DATA_DIR + 'dev.tok.tsv', format='TSV', fields=[('text', TEXT),
                                                                                  ('labels', LABELS),
                                                                                  ('length', LENGTH)],
                             skip_header=True)
    train_loader = Iterator(
        dataset=train_set, batch_size=BATCH_SIZE,
        sort_key=lambda x: len(x.text),
        sort_within_batch=True,
        shuffle=True,
    )
    val_loader = BucketIterator(
        dataset=val_set, batch_size=BATCH_SIZE,
        sort_key=lambda x: len(x.text),
        sort_within_batch=False,
        shuffle=False,
    )
    TEXT.build_vocab(train_set)
    # LABELS.build_vocab(train_set)

    return (TEXT, LABELS, LENGTH), (train_set, val_set), (train_loader, val_loader)


def sample(load_model=None):
    (TEXT, LABELS, LENGTH), (train_set, val_set), (train_loader, val_loader) = load_data()

    null_ent = -1
    br_tok_idx = TEXT.vocab.stoi['<br>']
    vocab_len = len(TEXT.vocab)

    model = EntitiyNLM(vocab_len, hidden_dim=HIDDEN_SIZE, entity_dim=HIDDEN_SIZE, max_ent_length=MAX_ENT_LENGTH,
                       max_entities=MAX_NUM_ENTITIES, break_tok_idx=br_tok_idx).cuda(device)
    if load_model:
        model.load_state_dict(torch.load(load_model))

    total_e = 0
    total_e_correct = 0
    total_e_idx_correct = 0
    for (text, labels, length), _ in tqdm(val_loader):
        curr_batch_size = len(length)

        # build entity annotations
        e_present = labels != null_ent
        e_t = e_present.long()
        e_idx = labels
        e_len = torch.zeros_like(labels)
        prev_e = torch.zeros_like(labels[0]).fill_(null_ent)
        prev_len = torch.zeros_like(labels[0])
        for i in range(labels.shape[0]):
            incr_mask = np.logical_and(prev_e == labels[-i - 1], prev_e != null_ent).type(torch.bool)
            prev_len[incr_mask] = prev_len[incr_mask].clone() + 1
            prev_len[~incr_mask] = 1
            e_len[-i - 1] = prev_len.clone()
            prev_e = labels[-i - 1]

        if e_len.max() >= MAX_ENT_LENGTH or e_idx.max() >= MAX_NUM_ENTITIES:
            continue

        # set initial states
        states = torch.zeros(curr_batch_size, HIDDEN_SIZE).to(device), torch.zeros(curr_batch_size, HIDDEN_SIZE).to(device)

        text = text.to(device)
        pred_et, pred_eidx, pred_elen, out_xs = model.predict_entity(text, states)

        for i, (e_t_pred, e_t_tgt, e_idx_pred, e_idx_tgt) in enumerate(zip(pred_et, e_t.to(device), pred_eidx,
                                                                           e_idx.to(device))):
            total_e += e_t_tgt[length >= i].sum().item()
            mask = np.logical_and(e_t_tgt.cpu() > 0, length >= i).bool()
            total_e_correct += (e_t_pred == e_t_tgt)[mask].sum().item()
            total_e_idx_correct += (e_idx_pred == e_idx_tgt)[mask].sum().item()

    e_t_recall = float(total_e_correct) / total_e
    e_idx_recall = float(total_e_idx_correct) / total_e

    print('e_t recall: %.4f, e_idx_recall: %.4f' % (e_t_recall, e_idx_recall))


def main():
    (TEXT, LABELS, LENGTH), (train_set, val_set), (train_loader, val_loader) = load_data()

    null_ent = -1
    br_tok_idx = TEXT.vocab.stoi['<br>']
    pad_idx = TEXT.vocab.stoi['<pad>']
    vocab_len = len(TEXT.vocab)

    model = EntitiyNLM(vocab_len, hidden_dim=HIDDEN_SIZE, entity_dim=HIDDEN_SIZE, max_ent_length=MAX_ENT_LENGTH,
                       max_entities=MAX_NUM_ENTITIES, break_tok_idx=br_tok_idx, n_layers=N_LAYERS).cuda(device)

    optim = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9)

    for (text, labels, length), _ in tqdm(train_loader):

        # debug
        #if length.type(torch.float).mean().item() < 100:
        #    continue

        #print(length.type(torch.float).mean())

        curr_batch_size = len(length)

        # build entity annotations
        e_present = labels != null_ent
        e_t = (e_present).long()
        e_idx = labels
        e_len = torch.zeros_like(labels)
        prev_e = torch.zeros_like(labels[0]).fill_(null_ent)
        prev_len = torch.zeros_like(labels[0])
        for i in range(labels.shape[0]):
            incr_mask = np.logical_and(prev_e == labels[-i-1], prev_e != null_ent).type(torch.bool)
            prev_len[incr_mask] = prev_len[incr_mask].clone() + 1
            prev_len[~incr_mask] = 1
            e_len[-i-1] = prev_len.clone()
            prev_e = labels[-i-1]

        # TODO set lower value of max_ent_length and handle overflows
        if e_len.max() >= MAX_ENT_LENGTH or e_idx.max() >= MAX_NUM_ENTITIES:
            continue

        # set initial states
        states = torch.zeros(N_LAYERS, curr_batch_size, HIDDEN_SIZE).to(device), torch.zeros(N_LAYERS, curr_batch_size, HIDDEN_SIZE).to(device)

        pad_mask = text[1:] != pad_idx

        # only evalutate entity losses after entity has changed (or no entity is being iterated)
        e_t_loss_mask = e_len[:-1] == 1
        e_t_loss_mask = np.logical_and(e_t_loss_mask, pad_mask).type(torch.bool)

        # additionally only evaluate entity selection (idx or len) losses when the next token is part of a mention
        e_sel_loss_mask = e_present[1:]
        e_sel_loss_mask = np.logical_and(e_sel_loss_mask, e_t_loss_mask).type(torch.bool)

        text = text.to(device)
        e_t, e_idx, e_len = e_t.to(device), e_idx.to(device), e_len.to(device)

        try:
            h, (pe_t, pe_idx, pe_len, ptext) = model(text, states, (e_t, e_idx, e_len))
        except EntityOverflow:
            continue

        # compute e_t loss
        e_t_loss = loss_fn(pe_t[e_t_loss_mask], e_t[1:][e_t_loss_mask])

        new_e_mask = pe_idx == float('-inf')  # new entities will have -inf log likelihood fixed by the model
        new_e_mask = new_e_mask[e_sel_loss_mask]
        e_idx_filtered = e_idx[1:][e_sel_loss_mask]
        inval_preds = new_e_mask[(np.arange(len(e_idx_filtered)), e_idx_filtered)]
        e_idx_filtered[inval_preds] = 0  # set labels for new entities to zero
        e_idx_loss = loss_fn(pe_idx[e_sel_loss_mask], e_idx_filtered)

        # compute e_len loss
        e_len_loss = loss_fn(pe_len[e_sel_loss_mask], e_len[1:][e_sel_loss_mask])

        # compute word loss

        w_loss = loss_fn(ptext[pad_mask], text[1:][pad_mask])

        loss = e_t_loss + e_idx_loss + e_len_loss + w_loss
        loss.backward()
        print(loss.item(), w_loss.item(), e_t_loss.item(), e_idx_loss.item(), e_len_loss.item())
        optim.step()
        optim.zero_grad()

    return


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--sample', action='store_true')
    parser.add_argument('--load-model', type=str, default=None)
    args = parser.parse_args()
    if args.sample:
        sample(load_model=args.load_model)
    else:
        main()
