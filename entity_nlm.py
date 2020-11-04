# adapted from https://github.com/jiyfeng/entitynlm/blob/master/entitynlm.h
import numpy as np
import torch
import torch.nn as nn


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
        self.entity_embed = nn.Embedding(n_entities, entity_dim)  # entity embeddings
        self.entity_embed.weight.requires_grad = False  # we dont optimize entity embeddings with SGD
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

    def _sample_embeddings(self, n):
        z = self.entity_init_mean[None] + torch.randn(n, self.entity_dim) * self.entity_init_var
        norm = torch.sqrt((z ** 2).sum(dim=1))  # [N x 1]
        return z / norm[:,None]

    def _add_embeddings(self, E, sample_idxs, entity_idxs):
        E[(sample_idxs, entity_idxs)] = self._sample_embeddings(len(sample_idxs))

    def _update_embeddings(self, E, h, sample_idxs, entity_idxs):
        proj_e = self.E_forget_op(h)  # [n x entity_dim]
        f = (E[(sample_idxs, entity_idxs)] * proj_e).sum(dim=1)  # [n]
        i = self.E_input_op(h)  # [n x entity_dim]
        E[(sample_idxs, entity_idxs)] = (1 - f) * E[(sample_idxs, entity_idxs)] + f * i

    # TODO replace all inplace operations
    def cell_forward(self, h, entity_cache, current_annotations):
        e_t, e_idx, e_len = current_annotations  # e_t: entity label {0, 1}, e_idx: entity id, e_len: mention length
        E, n_entities, e_dists, null_context = entity_cache  # E: [N x max_entities x entity_dim], n_entities: [N x 1]
                                                             # e_dists: [N x 1], null_context: [N x entity_dim]

        e_mask = e_t == 1  # get samples with entities present for this time step
        new_e_mask = e_idx >= n_entities  # get samples whose current entity is new
        max_e_reached = n_entities == self.max_entities  # get samples where the entity max has been reached
        if max_e_reached.sum() > 0:
            print("WARNING: Maximum entity threshold reached")

        # add new embeddings for new entities
        new_e_idxs, = np.where(np.logical_and(new_e_mask, ~max_e_reached))
        self._add_embeddings(E, new_e_idxs, n_entities[new_e_idxs])

        # update embeddings
        update_e_idxs, = np.where(e_mask)
        self._update_embeddings(E, h[update_e_idxs], update_e_idxs, e_idx[update_e_idxs])

        # TODO check  whether referencing over all e_idxs is problematic
        e_idx[e_idx < 0] = 0
        curr_e = E[(np.arange(E.shape[0]), e_idx)]  # [N x entity_dim]

        # update null_context
        null_context[e_mask] = curr_e[e_mask]  # [N x entity_dim]

        # predict next e_t
        out_e_t = self.R_op(h)

        # predict next e_idx
        proj_e = self.E_context_op(h)  # [N x entity_dim]
        #out_e_idx = torch.matmul(E, proj_e[:,None])  # [N x max_entities] TODO debug
        out_e_idx = (E * proj_e[:,None]).sum(dim=2)  # [N x max_entities] TODO debug (equivalent ^)
        out_e_idx += torch.exp(e_dists * self.lambda_dist)  # [N x max_entities] TODO debug

        # predict next mention lengths
        length_input = torch.cat([h, curr_e], dim=1)  # [N x hidden_dim + entity_dim]
        out_e_len = self.L_op(length_input)  # [N x max_ent_length]

        # generate conditioning vectors for word sampling
        out_x = torch.where(e_mask, self.X_op(curr_e), self.X_null_op(null_context))  # [N x hidden_dim]

        return out_e_t, out_e_idx, out_e_len, out_x

    def forward(self, h):
        # TODO set null_context to final hidden state when sentence is finished
        pass
