import torch
import torch.nn as nn
import torch.nn.functional as F


def _L2_loss_mean(x):
    return torch.mean(torch.sum(torch.pow(x, 2), dim=1, keepdim=False) / 2.)


class Aggregator(nn.Module):

    def __init__(self, in_dim, out_dim, dropout, aggregator_type, is_complex_embedding=False):
        super(Aggregator, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.dropout = dropout
        self.aggregator_type = aggregator_type

        self.is_complex_embedding = is_complex_embedding
        if self.is_complex_embedding:
            self.in_dim = 2 * in_dim
            self.out_dim = 2 * out_dim

        self.message_dropout = nn.Dropout(dropout)
        self.activation = nn.LeakyReLU()

        if self.aggregator_type == 'gcn':
            self.linear = nn.Linear(self.in_dim, self.out_dim)  # W in Equation (6)
            nn.init.xavier_uniform_(self.linear.weight)

        elif self.aggregator_type == 'graphsage':
            self.linear = nn.Linear(self.in_dim * 2, self.out_dim)  # W in Equation (7)
            nn.init.xavier_uniform_(self.linear.weight)

        elif self.aggregator_type == 'bi-interaction':
            self.linear1 = nn.Linear(self.in_dim, self.out_dim)  # W1 in Equation (8)
            self.linear2 = nn.Linear(self.in_dim, self.out_dim)  # W2 in Equation (8)
            nn.init.xavier_uniform_(self.linear1.weight)
            nn.init.xavier_uniform_(self.linear2.weight)

        else:
            raise NotImplementedError

    def forward(self, ego_embeddings, A_in):
        """
        ego_embeddings:  (n_users + n_entities, in_dim)
        A_in:            (n_users + n_entities, n_users + n_entities), torch.sparse.FloatTensor
        """
        # Equation (3)
        # ego_embeddings are complex vectors, multiply A_in individually
        re_ego_embeddings, im_ego_embeddings = torch.chunk(ego_embeddings, 2, dim=1)
        re_side_embeddings, im_side_embeddings = torch.matmul(A_in, re_ego_embeddings), torch.matmul(A_in,
                                                                                                     im_ego_embeddings)

        # concat side_embeddings
        side_embeddings = torch.cat([re_side_embeddings, im_side_embeddings], dim=1)

        if self.aggregator_type == 'gcn':
            # Equation (6) & (9)
            embeddings = ego_embeddings + side_embeddings
            embeddings = self.activation(self.linear(embeddings))

        elif self.aggregator_type == 'graphsage':
            # Equation (7) & (9)
            embeddings = torch.cat([ego_embeddings, side_embeddings], dim=1)
            embeddings = self.activation(self.linear(embeddings))

        elif self.aggregator_type == 'bi-interaction':
            # Equation (8) & (9)
            sum_embeddings = self.activation(self.linear1(ego_embeddings + side_embeddings))
            bi_embeddings = self.activation(self.linear2(ego_embeddings * side_embeddings))
            embeddings = bi_embeddings + sum_embeddings

        embeddings = self.message_dropout(embeddings)  # (n_users + n_entities, out_dim)
        return embeddings


class KGAT(nn.Module):

    def __init__(self, args,
                 n_users, n_entities, n_relations, A_in=None,
                 user_pre_embed=None, item_pre_embed=None):

        super(KGAT, self).__init__()
        self.use_pretrain = args.use_pretrain

        self.n_users = n_users
        self.n_entities = n_entities
        self.n_relations = n_relations

        self.embed_dim = args.embed_dim
        self.relation_dim = args.relation_dim

        # Check what type of KG Embedding
        self.is_complex_embedding = False
        self.kg_embedding_type = args.kg_embedding_type

        if self.kg_embedding_type.lower() in ["complex", "rotate"]:
            self.is_complex_embedding = True

        self.conv_dim_list = [args.embed_dim] + eval(args.conv_dim_list)
        self.aggregation_type = args.aggregation_type

        self.mess_dropout = eval(args.mess_dropout)
        self.n_layers = len(eval(args.conv_dim_list))

        self.kg_l2loss_lambda = args.kg_l2loss_lambda
        self.cf_l2loss_lambda = args.cf_l2loss_lambda

        if self.is_complex_embedding:
            self.entity_user_embed = nn.Embedding(self.n_entities + self.n_users, self.embed_dim * 2)

            if self.kg_embedding_type == "rotate":
                self.relation_embed = nn.Embedding(self.n_relations, self.relation_dim)
            else:
                self.relation_embed = nn.Embedding(self.n_relations, self.relation_dim * 2)
        else:
            self.entity_user_embed = nn.Embedding(self.n_entities + self.n_users, self.embed_dim)
            self.relation_embed = nn.Embedding(self.n_relations, self.relation_dim)
            self.trans_M = nn.Parameter(torch.Tensor(self.n_relations, self.embed_dim, self.relation_dim))
            nn.init.xavier_uniform_(self.trans_M)

        if (self.use_pretrain == 1) and (user_pre_embed is not None) and (item_pre_embed is not None) and \
                not self.is_complex_embedding:
            other_entity_embed = nn.Parameter(torch.Tensor(self.n_entities - item_pre_embed.shape[0], self.embed_dim))
            nn.init.xavier_uniform_(other_entity_embed)
            entity_user_embed = torch.cat([item_pre_embed, other_entity_embed, user_pre_embed], dim=0)
            self.entity_user_embed.weight = nn.Parameter(entity_user_embed)
        else:
            nn.init.xavier_uniform_(self.entity_user_embed.weight)

        nn.init.xavier_uniform_(self.relation_embed.weight)

        self.aggregator_layers = nn.ModuleList()
        for k in range(self.n_layers):
            self.aggregator_layers.append(
                Aggregator(self.conv_dim_list[k], self.conv_dim_list[k + 1], self.mess_dropout[k],
                           self.aggregation_type, is_complex_embedding=self.is_complex_embedding))

        self.A_in = nn.Parameter(
            torch.sparse.FloatTensor(self.n_users + self.n_entities, self.n_users + self.n_entities))
        if A_in is not None:
            self.A_in.data = A_in
        self.A_in.requires_grad = False

    def calc_cf_embeddings(self):
        # We need to concat all the real and img parts in different vectors
        ego_embed = self.entity_user_embed.weight
        re_ini, im_ini = torch.chunk(ego_embed, 2, dim=1)

        all_embed_re, all_embed_im = [re_ini], [im_ini]

        for idx, layer in enumerate(self.aggregator_layers):
            ego_embed = layer(ego_embed, self.A_in)

            re_ego_embed, im_ego_embed = torch.chunk(ego_embed, 2, dim=1)
            all_embed_re.append(F.normalize(re_ego_embed, p=2, dim=1))
            all_embed_im.append(F.normalize(im_ego_embed, p=2, dim=1))

        # Equation (11)
        re_all_embed = torch.cat(all_embed_re, dim=1)  # (n_users + n_entities, concat_dim)
        im_all_embed = torch.cat(all_embed_im, dim=1)  # (n_users + n_entities, concat_dim)

        return re_all_embed, im_all_embed

    def calc_cf_loss(self, user_ids, item_pos_ids, item_neg_ids):
        """
        user_ids:       (cf_batch_size)
        item_pos_ids:   (cf_batch_size)
        item_neg_ids:   (cf_batch_size)
        """
        re_all_embed, im_all_embed = self.calc_cf_embeddings()  # (n_users + n_entities, concat_dim)
        re_user_embed = re_all_embed[user_ids]  # (cf_batch_size, concat_dim)
        re_item_pos_embed = re_all_embed[item_pos_ids]  # (cf_batch_size, concat_dim)
        re_item_neg_embed = re_all_embed[item_neg_ids]  # (cf_batch_size, concat_dim)

        im_user_embed = im_all_embed[user_ids]  # (cf_batch_size, concat_dim)
        im_item_pos_embed = im_all_embed[item_pos_ids]  # (cf_batch_size, concat_dim)
        im_item_neg_embed = im_all_embed[item_neg_ids]  # (cf_batch_size, concat_dim)

        # Equation (12)
        re_pos_score = torch.sum(re_user_embed * re_item_pos_embed, dim=1)  # (cf_batch_size)
        im_pos_score = torch.sum(im_user_embed * im_item_pos_embed, dim=1)  # (cf_batch_size)

        re_neg_score = torch.sum(re_user_embed * re_item_neg_embed, dim=1)  # (cf_batch_size)
        im_neg_score = torch.sum(im_user_embed * im_item_neg_embed, dim=1)  # (cf_batch_size)

        pos_score = re_pos_score + im_pos_score
        neg_score = re_neg_score + im_neg_score

        # Equation (13)
        # cf_loss = F.softplus(neg_score - pos_score)
        cf_loss = (-1.0) * F.logsigmoid(pos_score - neg_score)
        cf_loss = torch.mean(cf_loss)

        l2_loss = _L2_loss_mean(re_user_embed) + _L2_loss_mean(re_item_pos_embed) + _L2_loss_mean(re_item_neg_embed)
        loss = cf_loss + self.cf_l2loss_lambda * l2_loss
        return loss

    def calc_kg_loss(self, h, r, pos_t, neg_t):
        """
        h:      (kg_batch_size)
        r:      (kg_batch_size)
        pos_t:  (kg_batch_size)
        neg_t:  (kg_batch_size)
        """
        if self.kg_embedding_type == "transr":
            return self.calc_kg_loss_transr(h, r, pos_t, neg_t)
        if self.kg_embedding_type == "rotate":
            return self.calc_kg_loss_rotate(h, r, pos_t, neg_t)
        if self.kg_embedding_type == "complex":
            return self.calc_kg_loss_complex(h, r, pos_t, neg_t)

    def _get_rotate_embedding_range(self):
        gamma, epsilon = 12.0, 2.0

        embedding_range = nn.Parameter(
            torch.Tensor([(gamma + epsilon) / 64]),
            requires_grad=False
        )
        return embedding_range

    def calc_kg_loss_rotate(self, h, r, pos_t, neg_t):
        pi = 3.14159265358979323846
        embedding_range = self._get_rotate_embedding_range()

        r_embed = self.relation_embed(r)  # (kg_batch_size, relation_dim)
        phase_relation = r_embed / (embedding_range.item() / pi)

        h_embed = self.entity_user_embed(h)  # (kg_batch_size, embed_dim)
        pos_t_embed = self.entity_user_embed(pos_t)  # (kg_batch_size, embed_dim)
        neg_t_embed = self.entity_user_embed(neg_t)  # (kg_batch_size, embed_dim)

        re_h_embed, im_h_embed = torch.chunk(h_embed, 2, dim=1)
        re_pos_t, im_pos_t = torch.chunk(pos_t_embed, 2, dim=1)
        re_neg_t, im_neg_t = torch.chunk(neg_t_embed, 2, dim=1)

        re_relation = torch.cos(phase_relation)  # 2 x 64
        im_relation = torch.sin(phase_relation)

        re_pos_score = (re_relation * re_pos_t + im_relation * im_pos_t) - re_h_embed
        im_pos_score = (re_relation * im_pos_t - im_relation * re_pos_t) - im_h_embed
        pos_score = torch.stack([re_pos_score, im_pos_score], dim=0)
        pos_score = pos_score.norm(dim=0)
        pos_score = pos_score.sum(dim=1)

        re_neg_score = (re_relation * re_neg_t + im_relation * im_neg_t) - re_h_embed
        im_neg_score = (re_relation * im_neg_t - im_relation * re_neg_t) - im_h_embed
        neg_score = torch.stack([re_neg_score, im_neg_score], dim=0)
        neg_score = neg_score.norm(dim=0)
        neg_score = neg_score.sum(dim=1)

        kg_loss = (-1.0) * F.logsigmoid(neg_score - pos_score)
        kg_loss = torch.mean(kg_loss)

        l2_loss = _L2_loss_mean(re_h_embed) + _L2_loss_mean(r_embed) + _L2_loss_mean(re_pos_t) + _L2_loss_mean(
            re_neg_t)
        loss = kg_loss + self.kg_l2loss_lambda * l2_loss
        return loss

    def calc_kg_loss_transr(self, h, r, pos_t, neg_t):
        """
        h:      (kg_batch_size)
        r:      (kg_batch_size)
        pos_t:  (kg_batch_size)
        neg_t:  (kg_batch_size)
        """
        r_embed = self.relation_embed(r)  # (kg_batch_size, relation_dim)
        W_r = self.trans_M[r]  # (kg_batch_size, embed_dim, relation_dim)

        h_embed = self.entity_user_embed(h)  # (kg_batch_size, embed_dim)
        pos_t_embed = self.entity_user_embed(pos_t)  # (kg_batch_size, embed_dim)
        neg_t_embed = self.entity_user_embed(neg_t)  # (kg_batch_size, embed_dim)

        r_mul_h = torch.bmm(h_embed.unsqueeze(1), W_r).squeeze(1)  # (kg_batch_size, relation_dim)
        r_mul_pos_t = torch.bmm(pos_t_embed.unsqueeze(1), W_r).squeeze(1)  # (kg_batch_size, relation_dim)
        r_mul_neg_t = torch.bmm(neg_t_embed.unsqueeze(1), W_r).squeeze(1)  # (kg_batch_size, relation_dim)

        # Trans R

        # Equation (1)
        pos_score = torch.sum(torch.pow(r_mul_h + r_embed - r_mul_pos_t, 2), dim=1)  # (kg_batch_size)
        neg_score = torch.sum(torch.pow(r_mul_h + r_embed - r_mul_neg_t, 2), dim=1)  # (kg_batch_size)

        # Equation (2)
        # kg_loss = F.softplus(pos_score - neg_score)
        kg_loss = (-1.0) * F.logsigmoid(neg_score - pos_score)
        kg_loss = torch.mean(kg_loss)

        l2_loss = _L2_loss_mean(r_mul_h) + _L2_loss_mean(r_embed) + _L2_loss_mean(r_mul_pos_t) + _L2_loss_mean(
            r_mul_neg_t)
        loss = kg_loss + self.kg_l2loss_lambda * l2_loss
        return loss

    def calc_kg_loss_complex(self, h, r, pos_t, neg_t):
        r_embed = self.relation_embed(r)  # (kg_batch_size, relation_dim)
        re_r_embed, im_r_embed = torch.chunk(r_embed, 2, dim=1)

        h_embed = self.entity_user_embed(h)  # (kg_batch_size, embed_dim)
        re_h_embed, im_h_embed = torch.chunk(h_embed, 2, dim=1)

        pos_t_embed = self.entity_user_embed(pos_t)  # (kg_batch_size, embed_dim)
        re_pos_t, im_pos_t = torch.chunk(pos_t_embed, 2, dim=1)

        neg_t_embed = self.entity_user_embed(neg_t)  # (kg_batch_size, embed_dim)
        re_neg_t, im_neg_t = torch.chunk(neg_t_embed, 2, dim=1)

        pos_re_score = re_r_embed * re_pos_t + im_r_embed * im_pos_t
        pos_im_score = re_r_embed * im_pos_t - im_r_embed * re_pos_t
        pos_score = re_h_embed * pos_re_score + im_h_embed * pos_im_score
        # pos_score_sum = pos_score.sum(dim=1)

        neg_re_score = re_r_embed * re_neg_t + im_r_embed * im_neg_t
        neg_im_score = re_r_embed * im_neg_t - im_r_embed * re_neg_t
        neg_score = re_h_embed * neg_re_score + im_h_embed * neg_im_score
        # neg_score_sum = neg_score.sum(dim=1)

        # Equation (2)
        # kg_loss = F.softplus(pos_score - neg_score)
        # TODO Check if same loss function as that of complex paper
        kg_loss = (-1.0) * F.logsigmoid(neg_score - pos_score)
        kg_loss = torch.mean(kg_loss)

        l2_loss = _L2_loss_mean(re_h_embed) + _L2_loss_mean(im_h_embed) + _L2_loss_mean(re_pos_t) + _L2_loss_mean(
            im_pos_t)
        loss = kg_loss + self.kg_l2loss_lambda * l2_loss
        return loss

    def update_attention_batch_rotate(self, h_list, t_list, r_idx):
        r_embed = self.relation_embed.weight[r_idx]
        # W_r = self.trans_M[r_idx]

        pi = 3.14159265358979323846
        embedding_range = self._get_rotate_embedding_range()
        phase_relation = r_embed / (embedding_range.item() / pi)

        h_embed = self.entity_user_embed.weight[h_list]
        t_embed = self.entity_user_embed.weight[t_list]
        re_t_embed, im_t_embed = torch.chunk(t_embed, 2, dim=1)

        re_relation = torch.cos(phase_relation)  # 2 x 64
        im_relation = torch.sin(phase_relation)

        rel_tail = torch.cat([(re_relation * re_t_embed + im_relation * im_t_embed),
                              (re_relation * im_t_embed - im_relation * re_t_embed)], dim=1)

        # TODO Check if 'r_mul_h' shape is same as 't_embed'
        # TODO: change h_embed + r_embed to h_embed . r_embed
        v_list = torch.sum(h_embed * torch.tanh(rel_tail), dim=1)
        return v_list

    def update_attention_batch_complex(self, h_list, t_list, r_idx):
        r_embed = self.relation_embed.weight[r_idx]

        h_embed = self.entity_user_embed.weight[h_list]
        t_embed = self.entity_user_embed.weight[t_list]

        v_list = torch.sum(t_embed * torch.tanh(h_embed + r_embed), dim=1)
        return v_list

    def update_attention(self, h_list, t_list, r_list, relations):
        device = self.A_in.device

        rows = []
        cols = []
        values = []

        for r_idx in relations:
            index_list = torch.where(r_list == r_idx)
            batch_h_list = h_list[index_list]
            batch_t_list = t_list[index_list]

            if self.kg_embedding_type == "complex":
                batch_v_list = self.update_attention_batch_complex(batch_h_list, batch_t_list, r_idx)
            elif self.kg_embedding_type == "rotate":
                batch_v_list = self.update_attention_batch_rotate(batch_h_list, batch_t_list, r_idx)

            rows.append(batch_h_list)
            cols.append(batch_t_list)
            values.append(batch_v_list)

        rows = torch.cat(rows)
        cols = torch.cat(cols)
        values = torch.cat(values)

        indices = torch.stack([rows, cols])
        shape = self.A_in.shape
        A_in = torch.sparse.FloatTensor(indices, values, torch.Size(shape))

        # Equation (5)
        A_in = torch.sparse.softmax(A_in.cpu(), dim=1)
        self.A_in.data = A_in.to(device)

    def calc_score(self, user_ids, item_ids):
        """
        user_ids:  (n_users)
        item_ids:  (n_items)
        """
        re_all_embed, im_all_embed = self.calc_cf_embeddings()  # (n_users + n_entities, concat_dim)

        re_user_embed = re_all_embed[user_ids]  # (cf_batch_size, concat_dim)
        im_user_embed = im_all_embed[user_ids]  # (cf_batch_size, concat_dim)

        re_item_embed = re_all_embed[item_ids]  # (cf_batch_size, concat_dim)
        im_item_embed = im_all_embed[item_ids]  # (cf_batch_size, concat_dim)

        # Concat both to create the user and item embedding
        user_embed = torch.cat([re_user_embed, im_user_embed], dim=1)
        item_embed = torch.cat([re_item_embed, im_item_embed], dim=1)

        # Equation (12)
        cf_score = torch.matmul(user_embed, item_embed.transpose(0, 1))  # (n_users, n_items)
        return cf_score

    def forward(self, *input, mode):
        if mode == 'train_cf':
            return self.calc_cf_loss(*input)
        if mode == 'train_kg':
            return self.calc_kg_loss(*input)
        if mode == 'update_att':
            return self.update_attention(*input)
        if mode == 'predict':
            return self.calc_score(*input)
