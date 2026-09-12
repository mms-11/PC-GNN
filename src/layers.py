import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from torch.autograd import Variable

from operator import itemgetter
import math

"""
	PC-GNN Layers
	Paper: Pick and Choose: A GNN-based Imbalanced Learning Approach for Fraud Detection
	Modified from https://github.com/YingtongDou/CARE-GNN

	TCC Extension (PC-GNN-ATT):
	IntraAggAtt replaces the mean aggregator (Eq. 8) with a single-head GAT-style
	attention mechanism applied AFTER the Choose step. This tests the hypothesis that
	attention over already-filtered neighbors (class imbalance already handled by Choose)
	outperforms plain mean aggregation.
"""


class InterAgg(nn.Module):

	def __init__(self, features, feature_dim, embed_dim, 
				 train_pos, adj_lists, intraggs, inter='GNN', cuda=True):
		"""
		Initialize the inter-relation aggregator
		:param features: the input node features or embeddings for all nodes
		:param feature_dim: the input dimension
		:param embed_dim: the embed dimension
		:param train_pos: positive samples in training set
		:param adj_lists: a list of adjacency lists for each single-relation graph
		:param intraggs: the intra-relation aggregators used by each single-relation graph
		:param inter: NOT used in this version, the aggregator type: 'Att', 'Weight', 'Mean', 'GNN'
		:param cuda: whether to use GPU
		"""
		super(InterAgg, self).__init__()

		self.features = features
		self.dropout = 0.6
		self.adj_lists = adj_lists
		self.intra_agg1 = intraggs[0]
		self.intra_agg2 = intraggs[1]
		self.intra_agg3 = intraggs[2]
		self.embed_dim = embed_dim
		self.feat_dim = feature_dim
		self.inter = inter
		self.cuda = cuda
		self.intra_agg1.cuda = cuda
		self.intra_agg2.cuda = cuda
		self.intra_agg3.cuda = cuda
		self.train_pos = train_pos

		# initial filtering thresholds
		self.thresholds = [0.5, 0.5, 0.5]

		# parameter used to transform node embeddings before inter-relation aggregation
		self.weight = nn.Parameter(torch.FloatTensor(self.embed_dim*len(intraggs)+self.feat_dim, self.embed_dim))
		init.xavier_uniform_(self.weight)

		# label predictor for similarity measure
		self.label_clf = nn.Linear(self.feat_dim, 2)

		# initialize the parameter logs
		self.weights_log = []
		self.thresholds_log = [self.thresholds]
		self.relation_score_log = []

	def forward(self, nodes, labels, train_flag=True):
		"""
		:param nodes: a list of batch node ids
		:param labels: a list of batch node labels
		:param train_flag: indicates whether in training or testing mode
		:return combined: the embeddings of a batch of input node features
		:return center_scores: the label-aware scores of batch nodes
		"""

		# extract 1-hop neighbor ids from adj lists of each single-relation graph
		to_neighs = []
		for adj_list in self.adj_lists:
			to_neighs.append([set(adj_list[int(node)]) for node in nodes])

		# find unique nodes and their neighbors used in current batch
		unique_nodes = set.union(set.union(*to_neighs[0]), set.union(*to_neighs[1]),
								 set.union(*to_neighs[2], set(nodes)))

		# calculate label-aware scores
		if self.cuda:
			batch_features = self.features(torch.cuda.LongTensor(list(unique_nodes)))
			pos_features = self.features(torch.cuda.LongTensor(list(self.train_pos)))
		else:
			batch_features = self.features(torch.LongTensor(list(unique_nodes)))
			pos_features = self.features(torch.LongTensor(list(self.train_pos)))
		batch_scores = self.label_clf(batch_features)
		pos_scores = self.label_clf(pos_features)
		id_mapping = {node_id: index for node_id, index in zip(unique_nodes, range(len(unique_nodes)))}

		# the label-aware scores for current batch of nodes
		center_scores = batch_scores[itemgetter(*nodes)(id_mapping), :]

		# get neighbor node id list for each batch node and relation
		r1_list = [list(to_neigh) for to_neigh in to_neighs[0]]
		r2_list = [list(to_neigh) for to_neigh in to_neighs[1]]
		r3_list = [list(to_neigh) for to_neigh in to_neighs[2]]

		# assign label-aware scores to neighbor nodes for each batch node and relation
		r1_scores = [batch_scores[itemgetter(*to_neigh)(id_mapping), :].view(-1, 2) for to_neigh in r1_list]
		r2_scores = [batch_scores[itemgetter(*to_neigh)(id_mapping), :].view(-1, 2) for to_neigh in r2_list]
		r3_scores = [batch_scores[itemgetter(*to_neigh)(id_mapping), :].view(-1, 2) for to_neigh in r3_list]

		# count the number of neighbors kept for aggregation for each batch node and relation
		r1_sample_num_list = [math.ceil(len(neighs) * self.thresholds[0]) for neighs in r1_list]
		r2_sample_num_list = [math.ceil(len(neighs) * self.thresholds[1]) for neighs in r2_list]
		r3_sample_num_list = [math.ceil(len(neighs) * self.thresholds[2]) for neighs in r3_list]

		# intra-aggregation steps for each relation
		# Eq. (8) in the paper
		r1_feats, r1_scores = self.intra_agg1.forward(nodes, labels, r1_list, center_scores, r1_scores, pos_scores, r1_sample_num_list, train_flag)
		r2_feats, r2_scores = self.intra_agg2.forward(nodes, labels, r2_list, center_scores, r2_scores, pos_scores, r2_sample_num_list, train_flag)
		r3_feats, r3_scores = self.intra_agg3.forward(nodes, labels, r3_list, center_scores, r3_scores, pos_scores, r3_sample_num_list, train_flag)

		# get features or embeddings for batch nodes
		if self.cuda and isinstance(nodes, list):
			index = torch.LongTensor(nodes).cuda()
		else:
			index = torch.LongTensor(nodes)
		self_feats = self.features(index)

		# number of nodes in a batch
		n = len(nodes)

		# concat the intra-aggregated embeddings from each relation
		# Eq. (9) in the paper
		cat_feats = torch.cat((self_feats, r1_feats, r2_feats, r3_feats), dim=1)

		combined = F.relu(cat_feats.mm(self.weight).t())

		return combined, center_scores


class IntraAgg(nn.Module):

	def __init__(self, features, feat_dim, embed_dim, train_pos, rho, cuda=False):
		"""
		Initialize the intra-relation aggregator
		:param features: the input node features or embeddings for all nodes
		:param feat_dim: the input dimension
		:param embed_dim: the embed dimension
		:param train_pos: positive samples in training set
		:param rho: the ratio of the oversample neighbors for the minority class
		:param cuda: whether to use GPU
		"""
		super(IntraAgg, self).__init__()

		self.features = features
		self.cuda = cuda
		self.feat_dim = feat_dim
		self.embed_dim = embed_dim
		self.train_pos = train_pos
		self.rho = rho
		self.weight = nn.Parameter(torch.FloatTensor(2*self.feat_dim, self.embed_dim))
		init.xavier_uniform_(self.weight)

	def forward(self, nodes, batch_labels, to_neighs_list, batch_scores, neigh_scores, pos_scores, sample_list, train_flag):
		"""
		Code partially from https://github.com/williamleif/graphsage-simple/
		:param nodes: list of nodes in a batch
		:param to_neighs_list: neighbor node id list for each batch node in one relation
		:param batch_scores: the label-aware scores of batch nodes
		:param neigh_scores: the label-aware scores 1-hop neighbors each batch node in one relation
		:param pos_scores: the label-aware scores 1-hop neighbors for the minority positive nodes
		:param train_flag: indicates whether in training or testing mode
		:param sample_list: the number of neighbors kept for each batch node in one relation
		:return to_feats: the aggregated embeddings of batch nodes neighbors in one relation
		:return samp_scores: the average neighbor distances for each relation after filtering
		"""

		# filer neighbors under given relation in the train mode
		if train_flag:
			samp_neighs, samp_scores = choose_step_neighs(batch_scores, batch_labels, neigh_scores, to_neighs_list, pos_scores, self.train_pos, sample_list, self.rho)
		else:
			samp_neighs, samp_scores = choose_step_test(batch_scores, neigh_scores, to_neighs_list, sample_list)
		
		# find the unique nodes among batch nodes and the filtered neighbors
		unique_nodes_list = list(set.union(*samp_neighs))
		unique_nodes = {n: i for i, n in enumerate(unique_nodes_list)}

		# intra-relation aggregation only with sampled neighbors
		mask = Variable(torch.zeros(len(samp_neighs), len(unique_nodes)))
		column_indices = [unique_nodes[n] for samp_neigh in samp_neighs for n in samp_neigh]
		row_indices = [i for i in range(len(samp_neighs)) for _ in range(len(samp_neighs[i]))]
		mask[row_indices, column_indices] = 1
		if self.cuda:
			mask = mask.cuda()
		num_neigh = mask.sum(1, keepdim=True)
		mask = mask.div(num_neigh)  # mean aggregator
		if self.cuda:
			self_feats = self.features(torch.LongTensor(nodes).cuda())
			embed_matrix = self.features(torch.LongTensor(unique_nodes_list).cuda())
		else:
			self_feats = self.features(torch.LongTensor(nodes))
			embed_matrix = self.features(torch.LongTensor(unique_nodes_list))
		agg_feats = mask.mm(embed_matrix)  # single relation aggregator
		cat_feats = torch.cat((self_feats, agg_feats), dim=1)  # concat with last layer
		to_feats = F.relu(cat_feats.mm(self.weight))
		return to_feats, samp_scores


class IntraAggAtt(nn.Module):
	"""
	TCC Extension: Intra-relation aggregator with scaled dot-product attention.

	Replaces the mean aggregator (Eq. 8 of the PC-GNN paper) with a learnable
	scaled dot-product attention mechanism inspired by Vaswani et al. (2017).
	The Choose step is preserved EXACTLY as in the original IntraAgg — attention
	is applied only over the already-filtered neighbor set tilde_N_r(v).

	For each central node v and relation r, given filtered neighbors tilde_N_r(v):

	  (1) Query/Key projection:
	        q_v = h_v @ W_q^T              (batch  x d')
	        k_u = h_u @ W_k^T              (unique x d')
	  (2) Scaled dot-product scores:
	        e_vu = (q_v @ k_u^T) / sqrt(d')   -> (batch x unique)
	  (3) Masked softmax — non-neighbors set to -inf before softmax:
	        alpha_vu = softmax_{u in N~_r(v)}(e_vu)
	  (4) Weighted aggregation — Value = raw h_u (preserves feat_dim, no shape change):
	        h_v^(r) = alpha_v @ H_neighbors    (batch x feat_dim)

	Design note on Value vs. Key/Query:
	  Aggregating raw h_u (not the projected k_u) is a deliberate choice that keeps
	  agg_feats in feat_dim, so self.weight (shape 2*feat_dim -> embed_dim) requires
	  no modification. This mirrors the Q/K/V separation in Transformers: the score
	  space (d') is decoupled from the value space (feat_dim).

	use_choose flag:
	  When False, the Choose filter is skipped and attention runs over ALL neighbors.
	  This is used in the ablation study to isolate Choose's contribution from the
	  attention mechanism's contribution. Note: without Choose, large-degree relations
	  (e.g. R-S-R in YelpChi) will produce much larger unique_nodes sets per batch —
	  run on a subset first to calibrate memory usage on CPU.
	"""

	def __init__(self, features, feat_dim, embed_dim, train_pos, rho,
				 cuda=False, use_choose=True, attn_dim=None):
		"""
		:param features: the input node features or embeddings for all nodes
		:param feat_dim: the input feature dimension (read from feat_data.shape[1] at runtime)
		:param embed_dim: the output embedding dimension
		:param train_pos: positive samples in training set (for oversampling in Choose)
		:param rho: oversampling ratio for the minority class (used by Choose)
		:param cuda: whether to use GPU
		:param use_choose: if False, disables the Choose filter (ablation mode)
		:param attn_dim: attention projection dimension d'; if None, defaults to feat_dim // 2.
		                 Do NOT hardcode this — it is set dynamically from feat_data.shape[1]
		                 in model_handler.py so it adapts to any feature version of the dataset.
		"""
		super(IntraAggAtt, self).__init__()

		self.features = features
		self.cuda = cuda
		self.feat_dim = feat_dim
		self.embed_dim = embed_dim
		self.train_pos = train_pos
		self.rho = rho
		self.use_choose = use_choose

		# d': attention projection dimension — derived from actual feat_dim at construction time
		self.attn_dim = attn_dim if attn_dim is not None else max(feat_dim // 2, 1)

		# W_q: Query projection  (d' x feat_dim)
		self.W_q = nn.Parameter(torch.FloatTensor(self.attn_dim, self.feat_dim))
		init.xavier_uniform_(self.W_q)

		# W_k: Key projection    (d' x feat_dim)
		self.W_k = nn.Parameter(torch.FloatTensor(self.attn_dim, self.feat_dim))
		init.xavier_uniform_(self.W_k)

		# Output projection — identical shape to IntraAgg.weight: [h_v || agg] -> embed_dim
		# Shape: (2*feat_dim x embed_dim) — no change needed vs. original
		self.weight = nn.Parameter(torch.FloatTensor(2 * self.feat_dim, self.embed_dim))
		init.xavier_uniform_(self.weight)

	def forward(self, nodes, batch_labels, to_neighs_list, batch_scores, neigh_scores,
				pos_scores, sample_list, train_flag):
		"""
		:param nodes: list of nodes in a batch
		:param batch_labels: labels of the batch nodes
		:param to_neighs_list: neighbor id list per batch node for one relation
		:param batch_scores: label-aware scores of the batch nodes
		:param neigh_scores: label-aware scores of the 1-hop neighbors
		:param pos_scores: label-aware scores of the minority positive nodes
		:param sample_list: number of neighbors kept per batch node after Choose
		:param train_flag: True during training, False during inference
		:return to_feats: aggregated embeddings for the batch nodes in this relation
		:return samp_scores: average neighbor distances after filtering (for threshold update)
		"""

		# ── Choose step (identical to original IntraAgg) ─────────────────────
		# This block is deliberately left untouched — the attention mechanism only
		# replaces what happens AFTER Choose has produced samp_neighs.
		if self.use_choose:
			if train_flag:
				samp_neighs, samp_scores = choose_step_neighs(
					batch_scores, batch_labels, neigh_scores, to_neighs_list,
					pos_scores, self.train_pos, sample_list, self.rho)
			else:
				samp_neighs, samp_scores = choose_step_test(
					batch_scores, neigh_scores, to_neighs_list, sample_list)
		else:
			# Ablation mode: skip Choose, attend over ALL neighbors.
			# Warning: large-degree relations will produce big unique_nodes sets —
			# test memory usage on a small batch before full training.
			samp_neighs = [set(neighs) for neighs in to_neighs_list]
			samp_scores = [[] for _ in to_neighs_list]  # dummy scores (unused by current threshold logic)

		# ── Build binary adjacency mask (same as IntraAgg) ───────────────────
		# mask[i, j] = 1  iff unique node j is a filtered neighbor of batch node i
		# Guard against the edge case where every batch node has an empty neighbor
		# set (can happen in the no-Choose ablation on very sparse graphs).
		if all(len(s) == 0 for s in samp_neighs):
			# No neighbors at all: return zero embeddings with the right shape
			if self.cuda:
				zero_feats = self.features(torch.LongTensor(nodes).cuda()) * 0
			else:
				zero_feats = self.features(torch.LongTensor(nodes)) * 0
			return F.relu(torch.cat((zero_feats, zero_feats), dim=1).mm(self.weight)), samp_scores

		unique_nodes_list = list(set.union(*samp_neighs))
		unique_nodes = {n: i for i, n in enumerate(unique_nodes_list)}

		mask = Variable(torch.zeros(len(samp_neighs), len(unique_nodes)))
		column_indices = [unique_nodes[n] for samp_neigh in samp_neighs for n in samp_neigh]
		row_indices = [i for i in range(len(samp_neighs)) for _ in range(len(samp_neighs[i]))]
		mask[row_indices, column_indices] = 1
		if self.cuda:
			mask = mask.cuda()

		# ── Fetch feature matrices ────────────────────────────────────────────
		if self.cuda:
			self_feats = self.features(torch.LongTensor(nodes).cuda())
			embed_matrix = self.features(torch.LongTensor(unique_nodes_list).cuda())
		else:
			self_feats = self.features(torch.LongTensor(nodes))
			embed_matrix = self.features(torch.LongTensor(unique_nodes_list))

		# ── Scaled dot-product attention ──────────────────────────────────────
		# Q: queries from batch nodes     (batch    x d')
		# K: keys   from unique neighbors (unique   x d')
		# V: values = raw features h_u   (unique   x feat_dim)  [no extra projection]
		Q = self_feats.mm(self.W_q.t())        # (batch x d')
		K = embed_matrix.mm(self.W_k.t())      # (unique x d')

		# Attention scores matrix — same shape as mask: (batch x unique)
		scale = math.sqrt(self.attn_dim)
		scores = Q.mm(K.t()) / scale           # (batch x unique)

		# Mask out non-neighbor positions with -inf so softmax gives them weight 0
		scores = scores.masked_fill(mask == 0, float('-inf'))

		# Row-wise softmax: alpha[i, j] = attention weight of node i over neighbor j
		alpha = F.softmax(scores, dim=1)       # (batch x unique)

		# Replace any NaN rows (nodes with zero filtered neighbors) with zeros.
		# torch.nan_to_num requires PyTorch >= 1.8; use a mask-based guard for
		# compatibility with the repo's declared torch==1.4.0 requirement.
		nan_mask = alpha != alpha  # True where alpha is NaN (all-inf softmax row)
		alpha = alpha.masked_fill(nan_mask, 0.0)

		# Weighted aggregation — V = embed_matrix (raw h_u), preserves feat_dim
		agg_feats = alpha.mm(embed_matrix)     # (batch x feat_dim)

		# ── Concat [h_v || agg] and project to embed_dim (identical to IntraAgg) ─
		cat_feats = torch.cat((self_feats, agg_feats), dim=1)  # (batch x 2*feat_dim)
		to_feats = F.relu(cat_feats.mm(self.weight))
		return to_feats, samp_scores


def choose_step_neighs(center_scores, center_labels, neigh_scores, neighs_list, minor_scores, minor_list, sample_list, sample_rate):
    """
    Choose step for neighborhood sampling
    :param center_scores: the label-aware scores of batch nodes
    :param center_labels: the label of batch nodes
    :param neigh_scores: the label-aware scores 1-hop neighbors each batch node in one relation
    :param neighs_list: neighbor node id list for each batch node in one relation
	:param minor_scores: the label-aware scores for nodes of minority class in one relation
    :param minor_list: minority node id list for each batch node in one relation
    :param sample_list: the number of neighbors kept for each batch node in one relation
	:para sample_rate: the ratio of the oversample neighbors for the minority class
    """
    samp_neighs = []
    samp_score_diff = []
    for idx, center_score in enumerate(center_scores):
        center_score = center_scores[idx][0]
        neigh_score = neigh_scores[idx][:, 0].view(-1, 1)
        center_score_neigh = center_score.repeat(neigh_score.size()[0], 1)
        neighs_indices = neighs_list[idx]
        num_sample = sample_list[idx]

        # compute the L1-distance of batch nodes and their neighbors
        score_diff_neigh = torch.abs(center_score_neigh - neigh_score).squeeze()
        sorted_score_diff_neigh, sorted_neigh_indices = torch.sort(score_diff_neigh, dim=0, descending=False)
        selected_neigh_indices = sorted_neigh_indices.tolist()

        # top-p sampling according to distance ranking
        if len(neigh_scores[idx]) > num_sample + 1:
            selected_neighs = [neighs_indices[n] for n in selected_neigh_indices[:num_sample]]
            selected_score_diff = sorted_score_diff_neigh.tolist()[:num_sample]
        else:
            selected_neighs = neighs_indices
            selected_score_diff = score_diff_neigh.tolist()
            if isinstance(selected_score_diff, float):
                selected_score_diff = [selected_score_diff]

        if center_labels[idx] == 1:
            num_oversample = int(num_sample * sample_rate)
            center_score_minor = center_score.repeat(minor_scores.size()[0], 1)
            score_diff_minor = torch.abs(center_score_minor - minor_scores[:, 0].view(-1, 1)).squeeze()
            sorted_score_diff_minor, sorted_minor_indices = torch.sort(score_diff_minor, dim=0, descending=False)
            selected_minor_indices = sorted_minor_indices.tolist()
            selected_neighs.extend([minor_list[n] for n in selected_minor_indices[:num_oversample]])
            selected_score_diff.extend(sorted_score_diff_minor.tolist()[:num_oversample])

        samp_neighs.append(set(selected_neighs))
        samp_score_diff.append(selected_score_diff)

    return samp_neighs, samp_score_diff


def choose_step_test(center_scores, neigh_scores, neighs_list, sample_list):
	"""
	Filter neighbors according label predictor result with adaptive thresholds
	:param center_scores: the label-aware scores of batch nodes
	:param neigh_scores: the label-aware scores 1-hop neighbors each batch node in one relation
	:param neighs_list: neighbor node id list for each batch node in one relation
	:param sample_list: the number of neighbors kept for each batch node in one relation
	:return samp_neighs: the neighbor indices and neighbor simi scores
	:return samp_scores: the average neighbor distances for each relation after filtering
	"""

	samp_neighs = []
	samp_scores = []
	for idx, center_score in enumerate(center_scores):
		center_score = center_scores[idx][0]
		neigh_score = neigh_scores[idx][:, 0].view(-1, 1)
		center_score = center_score.repeat(neigh_score.size()[0], 1)
		neighs_indices = neighs_list[idx]
		num_sample = sample_list[idx]

		# compute the L1-distance of batch nodes and their neighbors
		score_diff = torch.abs(center_score - neigh_score).squeeze()
		sorted_scores, sorted_indices = torch.sort(score_diff, dim=0, descending=False)
		selected_indices = sorted_indices.tolist()

		# top-p sampling according to distance ranking and thresholds
		if len(neigh_scores[idx]) > num_sample + 1:
			selected_neighs = [neighs_indices[n] for n in selected_indices[:num_sample]]
			selected_scores = sorted_scores.tolist()[:num_sample]
		else:
			selected_neighs = neighs_indices
			selected_scores = score_diff.tolist()
			if isinstance(selected_scores, float):
				selected_scores = [selected_scores]

		samp_neighs.append(set(selected_neighs))
		samp_scores.append(selected_scores)

	return samp_neighs, samp_scores
