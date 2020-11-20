# adapted from https://github.com/jiyfeng/entitynlm/blob/master/entitynlm.h
# isaac rehg, November 2020, irehg6@gatech.edu
import numpy as np
import torch
import torch.nn as nn


class EntityOverflow(Exception):
    pass


# TODO add dropout
class EntityContext(nn.Module):

    def __init__(self, hidden_dim=256, entity_dim=256, max_ent_length=25, max_entities=64):
        super(EntityContext, self).__init__()
        assert hidden_dim == entity_dim, 'different values of entity and hidden dim not currently supported'

        self.max_entities = max_entities
        self.hidden_dim = hidden_dim
        self.entity_dim = entity_dim
        self.n_entities = n_entities = 1
        self.max_ent_length = max_ent_length

        # Binary entity/no-entity prediction
        self.R_op = nn.Linear(hidden_dim, 2, bias=False)  # entity type prediction

        # Selection over current entities
        self.E_context_op = nn.Linear(hidden_dim, entity_dim, bias=False)  # entity cluster prediction via cosine sim
        self.lambda_dist = nn.Parameter(torch.FloatTensor([1e-6]))  # scales exponential decrease in log-likelihood of entity selection as distance increases

        # Entity length prediction
        self.L_op = nn.Linear(hidden_dim + entity_dim, max_ent_length, bias=True)  # entity length prediction

        # Entity embeddings
        self.entity_init_mean = nn.Parameter(torch.rand(entity_dim))  # mean for distribution of initialized entity vectors
        self.entity_init_var = 1e-4  # variance of sampled entity embeddings

        self.default_context = nn.Parameter(torch.rand(entity_dim))  # default entity context

        # Entity embedding update
        self.E_forget_op = nn.Linear(hidden_dim, entity_dim, bias=False)  # computes retention of prior entity embedding (by sim with output)
        self.E_input_op = nn.Linear(hidden_dim, entity_dim, bias=False)  # computes new info to add to context embedding

        # Word prediction conditioning
        self.X_op = nn.Linear(entity_dim, hidden_dim, bias=False)  # add transformed entity embedding to hidden state to condition word prediction
        self.X_null_op = nn.Linear(entity_dim, hidden_dim, bias=False)  # entity embed transform for default entity context

        # Losses
        self.bce_loss = torch.nn.BCEWithLogitsLoss()
        self.ce_loss = torch.nn.CrossEntropyLoss()

    def _initialize_E(self, batch_size, eps=1e-20, device=0):
        # allocate extra dimension at index 0 for prediction of new entities
        E = torch.zeros(batch_size, self.max_entities + 1, self.entity_dim).to(device)
        E[:, 0] = (self.entity_init_mean / (torch.norm(self.entity_init_mean) + eps))[None].to(device)
        return E

    def _initialize_n_entities(self, batch_size):
        return torch.ones(batch_size).long()

    def _initialize_e_dists(self, batch_size, device=0):
        # filler at index 0 for new entity prediction, remains 0
        return torch.zeros(batch_size, self.max_entities + 1).to(device)

    def _initialize_e_idx_lookup(self, batch_size, device=0):
        # allocate one extra dimension at last index for indexing of non-entity labels (which are set to -1)
        lookup = torch.zeros(batch_size, self.max_entities + 2).to(device).bool()
        #lookup[:, -1] = -1  # set the last entity lookup to -1 to catch non-entities and set them back to -1
        lookup[:, 0] = True
        return lookup

    def _normalize_E(self, E, sample_idxs, entity_idxs, eps=1e-20):
        # gradient hack
        norm = torch.ones(*E.shape[:2]).to(E.device)
        norm[(sample_idxs, entity_idxs)] = torch.norm(E[(sample_idxs, entity_idxs)], dim=1)

        ret = E / (norm[:,:,None] + eps)
        return ret

    def _sample_embeddings(self, n):
        z = self.entity_init_mean[None] + torch.randn(n, self.entity_dim).to(self.entity_init_mean.device) * self.entity_init_var
        norm = torch.sqrt((z ** 2).sum(dim=1))  # [N x 1]
        return z / norm[:,None]

    def _add_embeddings(self, E, sample_idxs, entity_idxs):
        #E[(sample_idxs, entity_idxs)] = self._sample_embeddings(len(sample_idxs))

        # hack to make gradients propagate
        E_update = torch.zeros_like(E).to(E.device)
        #assert E[(sample_idxs, entity_idxs)].sum().item() == 0, 'Space for new entitiy embeddings should be initialized to 0'
        E_update[(sample_idxs, entity_idxs)] = self._sample_embeddings(len(sample_idxs))
        E = E + E_update

        return E

    def _update_embeddings(self, E, h, sample_idxs, entity_idxs):
        proj_e = self.E_forget_op(h)  # [n x entity_dim]
        f = (E[(sample_idxs, entity_idxs)] * proj_e).sum(dim=1)[:, None]  # [n x 1]
        f = torch.sigmoid(f)
        i = self.E_input_op(h)  # [n x entity_dim]

        #E[(sample_idxs, entity_idxs)] = (1 - f) * E[(sample_idxs, entity_idxs)] + f * i

        # hack to make gradients propagate
        E_update1 = torch.ones_like(E).to(E.device)
        E_update1[(sample_idxs, entity_idxs)] = -(f - 1)
        E_update2 = torch.zeros_like(E).to(E.device)
        E_update2[(sample_idxs, entity_idxs)] = f * i
        E = E * E_update1 + E_update2
        E = self._normalize_E(E, sample_idxs, entity_idxs)

        #assert not np.any(np.isnan(E.data.cpu().numpy()))
        return E

    def _update_null_context(self, null_context, curr_e, h, e_mask, br_mask):
        #null_context[e_mask] = curr_e[e_mask]
        #null_context[br_mask] = h[br_mask]

        # hack to make gradients propagate
        ctxt_add = torch.zeros_like(curr_e).to(curr_e.device)
        ctxt_add[e_mask] = curr_e[e_mask]
        ctxt_subtract = torch.zeros_like(curr_e).to(curr_e.device)
        ctxt_subtract[e_mask] = null_context[e_mask]
        null_context = null_context - ctxt_subtract + ctxt_add

        ctxt_add = torch.zeros_like(curr_e).to(curr_e.device)
        ctxt_add[br_mask] = h[br_mask]
        ctxt_subtract = torch.zeros_like(curr_e).to(curr_e.device)
        ctxt_subtract[br_mask] = null_context[br_mask]
        null_context = null_context - ctxt_subtract + ctxt_add

        return null_context

    def initialize_e_cache(self, batch_size, default_context=None, device=0):
        if default_context is None:
            default_context = self.default_context[None].repeat(batch_size, 1).to(device)
        return (
            self._initialize_E(batch_size, device=device),
            self._initialize_n_entities(batch_size),
            self._initialize_e_dists(batch_size, device=device),
            default_context,
            self._initialize_e_idx_lookup(batch_size, device=device)
        )

    # TODO replace all inplace operations
    def cell_forward(self, h, entity_cache, current_annotations, final_tok=None, debug_var=None):
        if final_tok is None:
            final_tok = np.zeros(h.shape[0]).astype(np.bool_)

        e_t, e_idx, e_len = current_annotations  # e_t: entity label {0, 1}, e_idx: entity id, e_len: mention length
        E, n_entities, e_dists, null_context, e_idx_lookup = entity_cache  # E: entity_dim], n_entities: [N x 1]
                                                             # e_dists: [N x 1], null_context: [N x entity_dim]

        """
        # set entity indices for new entities to 0
        e_idx_true = e_idx.clone()
        e_idx = torch.where(e_idx_lookup[(np.arange(len(e_idx)), e_idx)], e_idx, torch.zeros_like(e_idx).to(e_idx.device))
        #e_idx = e_idx_lookup[(np.arange(len(e_idx)), e_idx)]
        e_idx[e_t == 0] = -1

        # update the entity index lookup with new entities
        e_idx_lookup[(np.arange(len(e_idx_true)), e_idx_true)] = True #torch.where(e_t > 0, e_idx_true, selected_idxs)

        # set entity indices to those sorted by order in which they were seen
        e_idx = e_idx_lookup[(np.arange(len(e_idx)), e_idx)].to(e_idx.device)
        assert not np.any(e_idx.cpu() == 0), '0 index is reserved for new entity predictions'

        assert not np.any(e_idx.cpu() > n_entities), 'we ad one entitity at a time and increment n_entities each time so should never be greater'
        new_e_mask = e_idx.cpu() == n_entities  # get samples whose current entity is new
        """
        e_idx = e_idx + 1  # add one to avoid referencing 0th index of E (reserved for new entities)
        e_mask = e_t.cpu() == 1  # get samples with entities present for this time step

        new_e_mask = np.logical_and(~e_idx_lookup[(np.arange(len(e_idx)), e_idx)].clone().cpu(), e_mask)
        new_e_idxs, = np.where(new_e_mask)

        # update e_idx_lookup
        e_idx_lookup[(np.arange(len(e_idx)), e_idx)] = True

        # update n_entities
        n_entities = n_entities.clone()
        n_entities[new_e_idxs] = n_entities[new_e_idxs] + 1

        max_e_reached = n_entities > self.max_entities  # get samples where the entity max has been reached
        if max_e_reached.sum() > 0:
            print("WARNING: Maximum entity threshold reached. Skipping batch...")
            raise EntityOverflow

        # add new embeddings for new entities
        if len(new_e_idxs) > 0:
            E = self._add_embeddings(E, new_e_idxs, e_idx[new_e_idxs])

            #assert not np.any(np.isinf(E.data.cpu().numpy()))

        # update embeddings
        update_e_idxs, = np.where(e_mask)
        if len(update_e_idxs) > 0:
            #assert not np.any(np.isinf(E.data.cpu().numpy()))
            E = self._update_embeddings(E, h[update_e_idxs], update_e_idxs, e_idx[update_e_idxs])

            #assert not np.any(np.isinf(E.data.cpu().numpy()))

        # update e_dists for samples that are starting a new sequence
        e_dists = e_dists.clone()
        e_dists[final_tok] += 1.0
        #e_dists[(new_e_idxs, e_idx[new_e_idxs])] = 0.0
        e_dists[(update_e_idxs, e_idx[update_e_idxs])] = 0.0
        e_dists[:, 0] = 0.0  # new entity index remains 0
        #assert all([x in update_e_idxs for x in new_e_idxs]), 'all new entities should be updated'

        """
        e_idx_tmp = e_idx.clone()
        e_idx_tmp[e_idx < 0] = 0
        """
        curr_e = E[(np.arange(E.shape[0]), e_idx)]  # [N x entity_dim]

        #assert not np.any(np.logical_and(e_idx.cpu() == -1, e_mask).numpy())

        # update null_context
        null_context = self._update_null_context(null_context, curr_e, h, e_mask, final_tok)  # [N x entity_dim]

        # predict next e_t
        out_e_t = self.R_op(h)

        # predict next e_idx
        proj_e = self.E_context_op(h)  # [N x entity_dim]
        #out_e_idx = torch.matmul(proj_e[None], E) - torch.exp(e_dists * self.lambda_dist) # [N x max_entities] TODO debug
        out_e_idx = (E * proj_e[:,None]).sum(dim=2) - torch.exp(e_dists * self.lambda_dist)  # [N x max_entities] TODO debug

        # set dims corresponding to not-yet-seen entities to -inf
        out_e_idx[~e_idx_lookup[:, :-1]] = float('-inf')

        #assert not np.any(np.isnan(out_e_idx.data.cpu().numpy()))

        # predict next mention lengths
        length_input = torch.cat([h, torch.where(e_mask[:, None].to(curr_e.device),
                                                 curr_e, null_context)], dim=1)  # [N x hidden_dim + entity_dim]
        out_e_len = self.L_op(length_input)  # [N x max_ent_length]

        # generate conditioning vectors for word sampling
        out_x = torch.where(e_mask[:, None].to(curr_e.device), self.X_op(curr_e),
                            self.X_null_op(null_context))  # [N x hidden_dim]

        # build next iteration's entity cache
        next_cache = E, n_entities, e_dists, null_context, e_idx_lookup

        # TODO validate ordering - probably good
        return out_e_t, out_e_idx, out_e_len, out_x, next_cache


class EntitiyNLM(nn.Module):

    def __init__(self, vocab_size, embed_dim=256, hidden_dim=256, entity_dim=256, max_ent_length=25, max_entities=64,
                 break_tok_idx=None):
        super(EntitiyNLM, self).__init__()
        self.break_tok_idx = break_tok_idx
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.entity_encoder = EntityContext(hidden_dim=hidden_dim, entity_dim=entity_dim, max_ent_length=max_ent_length,
                                            max_entities=max_entities)
        self.lstm = nn.LSTMCell(embed_dim, hidden_dim)
        self.out_layer = nn.Linear(hidden_dim, vocab_size)

    def cell_forward(self, x, states, entity_cache, entity_annotations, final_tok=None, debug_var=None):
        h, c = states
        embed = self.embed(x)
        h, c = self.lstm(embed, (h, c))
        out_e_t, out_e_idx, out_e_len, cond_x, next_e_cache = self.entity_encoder.cell_forward(h, entity_cache,
                                                                                               entity_annotations,
                                                                                               final_tok=final_tok,
                                                                                               debug_var=debug_var)
        out_x = self.out_layer(h + cond_x)
        return (h, c), (out_e_t, out_e_idx, out_e_len, out_x), next_e_cache

    def forward(self, xs, states, entity_annotations, default_context=None):
        h, c = states
        seq_len, batch = xs.shape

        # get sentence completion mask
        if self.break_tok_idx is not None:
            final_toks = xs[1:] == self.break_tok_idx

        # initialize entity cache
        entity_cache = self.entity_encoder.initialize_e_cache(batch, device=xs.device)

        # todo debug
        debug_var = None

        e_ts, e_idxs, e_lens = entity_annotations
        out_e_ts, out_e_idxs, out_e_lens, out_xs = [], [], [], []
        for i, (x, e_t, e_idx, e_len) in enumerate(zip(xs[:-1], e_ts[:-1], e_idxs[:-1], e_lens[:-1])):
            final_tok = final_toks[i] if self.break_tok_idx is not None else None
            (h, c), (out_e_t, out_e_idx, out_e_len, out_x), entity_cache = self.cell_forward(x, (h, c), entity_cache,
                                                                                             (e_t, e_idx, e_len),
                                                                                             final_tok=final_tok,
                                                                                             debug_var=debug_var)

            out_e_ts += [out_e_t]
            out_e_idxs += [out_e_idx]
            out_e_lens += [out_e_len]
            out_xs += [out_x]

        return (
            h,
            (
                torch.stack(out_e_ts),
                torch.stack(out_e_idxs),
                torch.stack(out_e_lens),
                torch.stack(out_xs)
            )
        )

    def predict_entity(self, xs, states, default_context=None):
        h, c = states
        seq_len, batch = xs.shape

        # get sentence completion mask
        if self.break_tok_idx is not None:
            final_toks = xs[1:] == self.break_tok_idx

        # initialize entity cache
        if default_context is None:
            default_context = self.entity_encoder.default_context[None].repeat(batch, 1).to(xs.device)
        entity_cache = (
            torch.zeros(batch, self.entity_encoder.max_entities, self.entity_encoder.entity_dim).to(xs.device),
            torch.zeros(batch).long(),  # keep n_entities on cpu
            torch.zeros(batch, self.entity_encoder.max_entities).to(xs.device),  # keep e_dists as float
            default_context,
            torch.zeros(batch, self.entity_encoder.max_entities).to(xs.device).long() - 1  # e idx lookup
        )

        device = xs.device
        null_et = torch.zeros(batch).long() - 1
        null_eidx = torch.zeros(batch).long() - 1
        null_elen = torch.ones(batch).long()

        # set initial entity annotations
        e_t = null_et.clone().to(device)
        e_idx = null_eidx.clone().to(device)
        e_len = null_elen.clone().to(device)

        pred_e_ts, pred_e_idxs, pred_e_lens, out_xs = [], [], [], []
        with torch.no_grad():
            for i, x in enumerate(xs[:-1]):
                final_tok = final_toks[i] if self.break_tok_idx is not None else None
                (h, c), (out_e_t, out_e_idx, out_e_len, out_x), entity_cache = self.cell_forward(x, (h, c), entity_cache,
                                                                                                 (e_t, e_idx, e_len),
                                                                                                 final_tok=final_tok)
                # TODO predicting new entity???
                # check if we are still decoding the previous entity
                e_t = torch.where(e_len > 1, null_et.clone().to(device), out_e_t.argmax(dim=1))

                e_idx = torch.where(e_len > 1, e_idx,
                                    torch.where(e_t > 0, out_e_idx.argmax(dim=1), null_eidx.clone().to(device)))

                e_len = torch.where(e_len > 1, e_len - 1,
                                    torch.where(e_t > 0, out_e_len.argmax(dim=1), null_elen.clone().to(device)))

                pred_e_ts += [e_t]
                pred_e_idxs += [e_idx]
                pred_e_lens += [e_len]
                out_xs += [out_x]

        return (
            torch.stack(pred_e_ts),
            torch.stack(pred_e_idxs),
            torch.stack(pred_e_lens),
            torch.stack(out_xs)
        )
