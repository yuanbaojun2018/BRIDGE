
import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import gc
import random
import numpy as np
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm



# =========================================================
# Config
# =========================================================

class Config:
    visual_path = "/data/gky/kgc_data/example_data/images/MKG-Y-visual.pth"
    textual_path = "/data/gky/kgc_data/example_data/text/MKG-Y-textual.pth"

    train_path = "/data/gky/kgc_data/example_data/structure/benchmarks/MKG-Y/train2id.txt"
    valid_path = "/data/gky/kgc_data/example_data/structure/benchmarks/MKG-Y/valid2id.txt"
    test_path = "/data/gky/kgc_data/example_data/structure/benchmarks/MKG-Y/test2id.txt"

    entity2id_path = "/data/gky/kgc_data/example_data/structure/benchmarks/MKG-Y/entity2id.txt"
    relation2id_path = "/data/gky/kgc_data/example_data/structure/benchmarks/MKG-Y/relation2id.txt"
    type_constrain_path = "/data/gky/kgc_data/example_data/structure/benchmarks/MKG-Y/type_constrain.txt"

    emb_dim = 512

    # 先给稳一点的默认值；如果稳定后还想提速，可再往上加
    batch_size = 96
    eval_batch_size = 64

    lr = 3e-4
    max_epoch = 200
    weight_decay = 1e-4
    dropout = 0.1

    hard_neg_k = 10
    listwise_weight = 0.0001
    listwise_temperature = 0.8
    label_smooth = 0.0

    early_stop_metric = "hit1"
    early_stop_patience = 20
    min_delta = 1e-6

    device = "cuda"
    temperature = 0.1
    use_norm = True

    # handshake
    use_handshake = True
    handshake_lambda = 0.10
    agree_lambda = 0.01
    handshake_max_candidates = 128
    max_handshake_pos_per_sample = 2

    # dynamic modality gating + modality dropout
    use_dynamic_modality_gate = True
    query_img_dropout_prob = 0.15
    query_txt_dropout_prob = 0.15
    query_struct_dropout_prob = 0.00

    entity_img_dropout_prob = 0.15
    entity_txt_dropout_prob = 0.15
    entity_struct_dropout_prob = 0.00

    gate_entropy_reg = 0.0

    # =========================
    # 稳定显存新增项
    # =========================
    use_amp = True
    amp_dtype = "float16"             # "float16" 或 "bfloat16"
    grad_clip_norm = 1.0

    # relation group candidate union 上限
    max_group_candidates = 1024

    # 单样本候选上限（训练时）
    max_train_candidates = 768

    # handshake 反向候选上限
    max_reverse_candidates = 256

    # entity_repr 分块
    entity_chunk_size = 256

    # evaluation 分块
    eval_entity_chunk_size = 1024

    # 低频清缓存
    empty_cache_every_steps = 50


# =========================================================
# Utils
# =========================================================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def maybe_empty_cache(device):
    if isinstance(device, str) and "cuda" in device:
        torch.cuda.empty_cache()


def get_amp_dtype(cfg):
    if cfg.amp_dtype == "bfloat16":
        return torch.bfloat16
    return torch.float16


def get_autocast_ctx(cfg):
    if cfg.use_amp and torch.cuda.is_available() and "cuda" in cfg.device:
        return torch.amp.autocast("cuda", dtype=get_amp_dtype(cfg))
    return nullcontext()


# =========================================================
# IO
# =========================================================

def load_ids(path):
    with open(path) as f:
        return int(f.readline())


def load_triples(path):
    triples = []
    with open(path) as f:
        next(f)
        for line in f:
            h, t, r = map(int, line.strip().split())
            triples.append((h, r, t))
    return triples


def load_type_constrain(path, n_rel):
    """
    兼容 OpenKE 风格 type_constrain.txt

    返回：
      rel2cand[r]         = 原关系 r 的合法 tail 集合
      rel2cand[r + n_rel] = 反向关系 r_rev 的合法 head 集合
    """
    with open(path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    if lines and len(lines[0].split()) == 1:
        lines = lines[1:]

    rel2cand = {}

    i = 0
    while i + 1 < len(lines):
        head_parts = list(map(int, lines[i].split()))
        tail_parts = list(map(int, lines[i + 1].split()))

        r_h = head_parts[0]
        num_h = head_parts[1]
        heads = head_parts[2:2 + num_h]

        r_t = tail_parts[0]
        num_t = tail_parts[1]
        tails = tail_parts[2:2 + num_t]

        rel2cand[r_t] = set(tails)
        rel2cand[r_h + n_rel] = set(heads)

        i += 2

    return rel2cand


# =========================================================
# Dataset
# =========================================================

class KGDataset(torch.utils.data.Dataset):
    def __init__(self, triples, n_rel):
        self.samples = []
        for h, r, t in triples:
            self.samples.append((h, r, t))
            self.samples.append((t, r + n_rel, h))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]





# =========================================================
# Model
# =========================================================

class MultiModalKGC(nn.Module):
    def __init__(self, n_ent, n_rel, visual_feat, textual_feat,
                 dim, dropout=0.1, temperature=0.1, use_norm=True, cfg=None):
        super().__init__()

        self.visual_feat = visual_feat
        self.textual_feat = textual_feat
        self.base_n_rel = n_rel
        self.n_ent = n_ent
        self.use_norm = use_norm
        self.cfg = cfg

        vis_dim = visual_feat.shape[1]
        txt_dim = textual_feat.shape[1]

        self.vis_proj = nn.Linear(vis_dim, dim)
        self.txt_proj = nn.Linear(txt_dim, dim)

        self.ent_emb = nn.Embedding(n_ent, dim)
        self.ent_residual = nn.Embedding(n_ent, dim)

        self.rel_emb = nn.Embedding(n_rel * 2, dim)
        self.rel_mod = nn.Embedding(n_rel * 2, dim)

        self.logit_scale = nn.Parameter(torch.tensor(3.0))

        self.query_gate_mlp = nn.Sequential(
            nn.Linear(dim * 4, dim),
            nn.LayerNorm(dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, 3)
        )

        self.entity_gate_mlp = nn.Sequential(
            nn.Linear(dim * 4, dim),
            nn.LayerNorm(dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, 3)
        )

        self.query_fusion = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.LayerNorm(dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim)
        )

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.vis_proj.weight)
        nn.init.zeros_(self.vis_proj.bias)

        nn.init.xavier_uniform_(self.txt_proj.weight)
        nn.init.zeros_(self.txt_proj.bias)

        nn.init.xavier_uniform_(self.ent_emb.weight)
        nn.init.zeros_(self.ent_residual.weight)

        nn.init.xavier_uniform_(self.rel_emb.weight)
        nn.init.zeros_(self.rel_mod.weight)

        for m in [self.query_gate_mlp, self.entity_gate_mlp, self.query_fusion]:
            for layer in m:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.zeros_(layer.bias)

    def _expand_rel(self, rel_vec, target_len):
        if rel_vec.dim() == 1:
            rel_vec = rel_vec.unsqueeze(0)
        if rel_vec.shape[0] != target_len:
            rel_vec = rel_vec.expand(target_len, -1)
        return rel_vec

    def _sample_keep_mask(self, batch_size, p_drop, device):
        if (not self.training) or p_drop <= 0.0:
            return torch.ones(batch_size, 1, device=device)
        keep = (torch.rand(batch_size, 1, device=device) > p_drop).float()
        return keep

    def _masked_softmax(self, logits, avail_mask, dim=-1):
        very_neg = torch.finfo(logits.dtype).min / 4.0
        masked_logits = logits.masked_fill(avail_mask <= 0, very_neg)
        weights = F.softmax(masked_logits, dim=dim)

        invalid_rows = (avail_mask.sum(dim=dim, keepdim=True) == 0)
        if invalid_rows.any():
            fallback = torch.full_like(weights, 1.0 / weights.shape[-1])
            weights = torch.where(invalid_rows, fallback, weights)

        return weights

    def _apply_modality_dropout(self, vis, txt, struct, side="query"):
        B = vis.shape[0]
        device = vis.device

        if side == "query":
            keep_vis = self._sample_keep_mask(B, self.cfg.query_img_dropout_prob, device)
            keep_txt = self._sample_keep_mask(B, self.cfg.query_txt_dropout_prob, device)
            keep_struct = self._sample_keep_mask(B, self.cfg.query_struct_dropout_prob, device)
        else:
            keep_vis = self._sample_keep_mask(B, self.cfg.entity_img_dropout_prob, device)
            keep_txt = self._sample_keep_mask(B, self.cfg.entity_txt_dropout_prob, device)
            keep_struct = self._sample_keep_mask(B, self.cfg.entity_struct_dropout_prob, device)

        all_zero = (keep_vis + keep_txt + keep_struct == 0)
        if all_zero.any():
            keep_struct = torch.where(all_zero, torch.ones_like(keep_struct), keep_struct)

        vis_d = vis * keep_vis
        txt_d = txt * keep_txt
        struct_d = struct * keep_struct

        avail_mask = torch.cat([keep_vis, keep_txt, keep_struct], dim=1)
        return vis_d, txt_d, struct_d, avail_mask

    def entity_repr(self, e, r=None):
        vis = self.vis_proj(self.visual_feat[e])
        txt = self.txt_proj(self.textual_feat[e])
        ent = self.ent_emb(e)
        residual = self.ent_residual(e)
        struct = ent + residual

        vis, txt, struct, avail_mask = self._apply_modality_dropout(
            vis, txt, struct, side="entity"
        )

        if r is not None:
            r_ctx = self.rel_emb(r)
            r_ctx = self._expand_rel(r_ctx, vis.shape[0])
        else:
            r_ctx = torch.zeros_like(vis)

        if self.cfg.use_dynamic_modality_gate:
            gate_inp = torch.cat([vis, txt, struct, r_ctx], dim=1)
            gate_logits = self.entity_gate_mlp(gate_inp)
            gate_weights = self._masked_softmax(gate_logits, avail_mask, dim=-1)

            x = (
                gate_weights[:, 0:1] * vis +
                gate_weights[:, 1:2] * txt +
                gate_weights[:, 2:3] * struct
            )
        else:
            x = (vis + txt + struct) / 3.0

        if r is not None:
            r_emb = self.rel_mod(r)
            r_emb = self._expand_rel(r_emb, x.shape[0])
            x = x * torch.sigmoid(r_emb)

        if self.use_norm:
            x = F.normalize(x, p=2, dim=-1)

        return x

    def query(self, h, r):
        vis = self.vis_proj(self.visual_feat[h])
        txt = self.txt_proj(self.textual_feat[h])
        ent = self.ent_emb(h)
        rel = self.rel_emb(r)
        struct = ent

        vis, txt, struct, avail_mask = self._apply_modality_dropout(
            vis, txt, struct, side="query"
        )

        if self.cfg.use_dynamic_modality_gate:
            gate_inp = torch.cat([vis, txt, struct, rel], dim=1)
            gate_logits = self.query_gate_mlp(gate_inp)
            gate_weights = self._masked_softmax(gate_logits, avail_mask, dim=-1)

            fused_modal = (
                gate_weights[:, 0:1] * vis +
                gate_weights[:, 1:2] * txt +
                gate_weights[:, 2:3] * struct
            )
        else:
            fused_modal = (vis + txt + struct) / 3.0

        x = torch.cat([fused_modal, rel], dim=1)
        x = self.query_fusion(x)

        if self.use_norm:
            x = F.normalize(x, p=2, dim=-1)

        return x

    def score(self, q, cand_emb):
        scale = F.softplus(self.logit_scale)
        return torch.matmul(q, cand_emb.T) * scale


# =========================================================
# Build maps
# =========================================================

def build_relation_candidates(triples, n_rel):
    rel_cand = {}
    for h, r, t in triples:
        r_rev = r + n_rel
        rel_cand.setdefault(r, set()).add(t)
        rel_cand.setdefault(r_rev, set()).add(h)

    for r in rel_cand:
        rel_cand[r] = list(rel_cand[r])
    return rel_cand


def build_sr2o(triples, n_rel):
    sr2o = {}
    for h, r, t in triples:
        r_rev = r + n_rel
        sr2o.setdefault((h, r), set()).add(t)
        sr2o.setdefault((t, r_rev), set()).add(h)
    return sr2o


def build_all_true(triples, n_rel):
    all_true = {}
    for h, r, t in triples:
        r_rev = r + n_rel
        all_true.setdefault((h, r), set()).add(t)
        all_true.setdefault((t, r_rev), set()).add(h)
    return all_true


def reverse_relation_id(r, n_rel):
    if r < n_rel:
        return r + n_rel
    return r - n_rel


# =========================================================
# Candidate helpers
# =========================================================

def limit_candidates(cand, positive_set, max_size):
    """
    对 candidate 做上限裁剪，但始终保留 positives。
    """
    cand = list(dict.fromkeys(cand))
    pos = list(dict.fromkeys(int(x) for x in positive_set))

    if len(cand) <= max_size:
        return cand

    pos_set = set(pos)
    others = [x for x in cand if x not in pos_set]
    keep_others = max(0, max_size - len(pos))

    if len(pos) >= max_size:
        return pos[:max_size]

    random.shuffle(others)
    new_cand = pos + others[:keep_others]
    random.shuffle(new_cand)
    return new_cand


def sample_candidates_multi(r, positive_set, cfg, rel_cand, rel2cand_type, n_ent):
    """
    训练时保留 candidate-aware 逻辑，但加入上限控制。
    """
    type_set = rel2cand_type.get(r, None)

    if type_set is not None and len(type_set) > 0:
        cand = set(type_set)
    else:
        cand = set(range(n_ent))

    cand.update(positive_set)
    cand = list(cand)
    random.shuffle(cand)

    cand = limit_candidates(cand, positive_set, cfg.max_train_candidates)
    return cand

def cap_group_union(group_cand_union, group_positive_sets, max_size):
    """
    裁剪 relation group 的 union，尽量保留所有样本的 positives。
    """
    union_list = list(group_cand_union)
    if len(union_list) <= max_size:
        return union_list

    must_keep = set()
    for pos_set in group_positive_sets.values():
        must_keep.update(int(x) for x in pos_set)

    must_keep = list(must_keep)
    if len(must_keep) >= max_size:
        return must_keep[:max_size]

    others = [x for x in union_list if x not in set(must_keep)]
    random.shuffle(others)
    keep = must_keep + others[:max_size - len(must_keep)]
    random.shuffle(keep)
    return keep


def chunked_entity_repr(model, cand_tensor, r_tensor, chunk_size):
    outs = []
    for st in range(0, cand_tensor.shape[0], chunk_size):
        ed = st + chunk_size
        outs.append(model.entity_repr(cand_tensor[st:ed], r_tensor[st:ed]))
    return torch.cat(outs, dim=0)


# =========================================================
# Loss
# =========================================================

# def classification_loss(scores, labels):
#     return F.binary_cross_entropy_with_logits(scores, labels)

def listwise_loss(scores, labels, temperature=1.0):
    if scores.dim() == 2:
        scores = scores.squeeze(0)
    if labels.dim() == 2:
        labels = labels.squeeze(0)

    pos_mask = labels > 0.5
    if pos_mask.sum() == 0:
        return torch.tensor(0.0, device=scores.device)

    target = pos_mask.float()
    target = target / target.sum()

    log_probs = F.log_softmax(scores / temperature, dim=0)
    return -(target * log_probs).sum()

# =========================================================
# Handshake / Cycle helpers
# =========================================================

def compute_reverse_handshake_loss(
    model,
    current_sample,
    current_forward_scores,
    current_cand_list,
    train_sr2o,
    rel_cand,
    rel2cand_type,
    n_ent,
    n_rel,
    device,
    cfg,
):
    s = int(current_sample["s"])
    r = int(current_sample["r"])
    pos_list = list(current_sample["positives"])

    if len(pos_list) == 0:
        return None, None, 0

    if len(pos_list) > cfg.max_handshake_pos_per_sample:
        pos_list = random.sample(pos_list, cfg.max_handshake_pos_per_sample)

    r_rev = reverse_relation_id(r, n_rel)

    cycle_losses = []
    agree_losses = []
    used_pairs = 0

    cand2idx = {eid: i for i, eid in enumerate(current_cand_list)}

    for g in pos_list:
        g = int(g)
        if g not in cand2idx:
            continue

        rev_key = (g, r_rev)
        rev_pos_set = set(int(x) for x in train_sr2o.get(rev_key, set()))
        rev_pos_set.add(s)


        rev_cand = sample_candidates_multi(
            r=r_rev,
            positive_set=rev_pos_set,
            cfg=cfg,
            rel_cand=rel_cand,
            rel2cand_type=rel2cand_type,
            n_ent=n_ent
        )

        # 显式保证 s 一定属于“必须保留集合”
        must_keep_rev = set(rev_pos_set)
        must_keep_rev.add(s)

        if s not in rev_cand:
            rev_cand.append(s)

        rev_cand = limit_candidates(rev_cand, must_keep_rev, cfg.max_reverse_candidates)

        # 再做一次兜底，防止任何极端情况
        if s not in rev_cand:
            if len(rev_cand) >= cfg.max_reverse_candidates:
                rev_cand[-1] = s
            else:
                rev_cand.append(s)

        rev_h_tensor = torch.tensor([g], dtype=torch.long, device=device)
        rev_r_tensor = torch.tensor([r_rev], dtype=torch.long, device=device)
        q_rev = model.query(rev_h_tensor, rev_r_tensor)

        rev_cand_tensor = torch.tensor(rev_cand, dtype=torch.long, device=device)
        rev_r_full = torch.full((len(rev_cand),), r_rev, dtype=torch.long, device=device)
        rev_cand_emb = chunked_entity_repr(
            model,
            rev_cand_tensor,
            rev_r_full,
            cfg.entity_chunk_size
        )
        rev_scores = model.score(q_rev, rev_cand_emb).squeeze(0)

        # 不再直接用 list.index，改成更稳的映射查找
        rev_id2local = {eid: idx for idx, eid in enumerate(rev_cand)}
        if s not in rev_id2local:
            del rev_cand_tensor, rev_r_full, rev_cand_emb, rev_scores
            continue
        s_local = rev_id2local[s]

        rev_log_probs = F.log_softmax(rev_scores, dim=0)
        cycle_loss = -rev_log_probs[s_local]
        cycle_losses.append(cycle_loss)

        forward_logit_g = current_forward_scores[cand2idx[g]]
        reverse_logit_s = rev_scores[s_local]

        forward_conf = torch.sigmoid(forward_logit_g)
        reverse_conf = torch.sigmoid(reverse_logit_s)
        agree_loss = F.mse_loss(forward_conf, reverse_conf)
        agree_losses.append(agree_loss)

        used_pairs += 1

        del rev_cand_tensor, rev_r_full, rev_cand_emb, rev_scores, rev_log_probs

    if used_pairs == 0:
        return None, None, 0

    cycle_loss = torch.stack(cycle_losses).mean() if len(cycle_losses) > 0 else None
    agree_loss = torch.stack(agree_losses).mean() if len(agree_losses) > 0 else None
    return cycle_loss, agree_loss, used_pairs


# =========================================================
# Evaluation
# =========================================================

@torch.no_grad()
def evaluate(model, triples, n_ent, all_true, rel2cand_type, device, batch_size=64, eval_entity_chunk_size=1024):
    model.eval()
    ranks = []

    for i in tqdm(range(0, len(triples), batch_size), desc="Evaluating"):
        batch = triples[i:i + batch_size]

        h = torch.tensor([x[0] for x in batch], dtype=torch.long, device=device)
        r = torch.tensor([x[1] for x in batch], dtype=torch.long, device=device)
        q = model.query(h, r)

        # 一个 batch 里按 relation 分组，共享 full/all candidate embedding，但采用分块避免 OOM
        rel_to_indices = {}
        for j, (_, rr, _) in enumerate(batch):
            rel_to_indices.setdefault(rr, []).append(j)

        for rr, idxs in rel_to_indices.items():
            cand_type = rel2cand_type.get(rr, None)
            if cand_type is not None and len(cand_type) > 0:
                eval_cands = list(cand_type)
            else:
                eval_cands = list(range(n_ent))

            # 补上 gold，防止 type constrain 文件把真值误排除
            golds = [batch[j][2] for j in idxs]
            eval_cands_set = set(eval_cands)
            for g in golds:
                if g not in eval_cands_set:
                    eval_cands.append(g)
                    eval_cands_set.add(g)

            cand_tensor = torch.tensor(eval_cands, dtype=torch.long, device=device)
            rr_tensor = torch.full((len(eval_cands),), rr, dtype=torch.long, device=device)
            cand_emb = chunked_entity_repr(model, cand_tensor, rr_tensor, eval_entity_chunk_size)

            id2local = {eid: k for k, eid in enumerate(eval_cands)}

            for j in idxs:
                hh, _, tt = batch[j]
                score = model.score(q[j:j+1], cand_emb).squeeze(0)

                if (hh, rr) in all_true:
                    true_set = all_true[(hh, rr)]
                    for e in true_set:
                        if e != tt and e in id2local:
                            score[id2local[e]] = -1e9

                rank = (torch.argsort(score, descending=True) == id2local[tt]).nonzero(as_tuple=True)[0].item() + 1
                ranks.append(rank)

            del cand_tensor, rr_tensor, cand_emb

        del h, r, q

    ranks = np.asarray(ranks, dtype=np.int64)
    hit1 = np.mean(ranks <= 1)
    hit3 = np.mean(ranks <= 3)
    hit10 = np.mean(ranks <= 10)
    mrr = np.mean(1.0 / ranks)
    return hit1, hit3, hit10, mrr


# =========================================================
# Train
# =========================================================

def train(cfg):
    device = cfg.device

    n_ent = load_ids(cfg.entity2id_path)
    n_rel = load_ids(cfg.relation2id_path)

    train_triples = load_triples(cfg.train_path)
    valid_triples = load_triples(cfg.valid_path)
    test_triples = load_triples(cfg.test_path)

    visual_feat = torch.load(cfg.visual_path, map_location=device).float().to(device)
    textual_feat = torch.load(cfg.textual_path, map_location=device).float().to(device)

    dataset = KGDataset(train_triples, n_rel)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=False,
        pin_memory=True if "cuda" in device else False
    )

    rel_cand = build_relation_candidates(train_triples, n_rel)
    train_sr2o = build_sr2o(train_triples, n_rel)
    all_true_test = build_all_true(train_triples + valid_triples + test_triples, n_rel)
    rel2cand_type = load_type_constrain(cfg.type_constrain_path, n_rel)

    model = MultiModalKGC(
        n_ent=n_ent,
        n_rel=n_rel,
        visual_feat=visual_feat,
        textual_feat=textual_feat,
        dim=cfg.emb_dim,
        dropout=cfg.dropout,
        temperature=cfg.temperature,
        use_norm=cfg.use_norm,
        cfg=cfg
    ).to(device)


    opt = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay
    )

    scaler = torch.amp.GradScaler("cuda", enabled=(cfg.use_amp and torch.cuda.is_available() and "cuda" in device))

    test_eval_triples = (
        [(h, r, t) for (h, r, t) in test_triples] +
        [(t, r + n_rel, h) for (h, r, t) in test_triples]
    )

    best_hit1 = -1.0
    best_hit3 = -1.0
    best_hit10 = -1.0
    best_mrr = -1.0

    best_hit1_epoch = -1
    best_hit3_epoch = -1
    best_hit10_epoch = -1
    best_mrr_epoch = -1

    best_stop_metric = -1.0
    no_improve_epochs = 0
    global_step = 0

    for epoch in range(cfg.max_epoch):
        model.train()

        total = 0.0
        total_listwise_loss = 0.0
        total_cycle_loss = 0.0
        total_agree_loss = 0.0
        total_handshake_pairs = 0

        num_steps = 0

        pbar = tqdm(loader, desc=f"[Epoch {epoch:03d}] Training")

        for h, r, t in pbar:
            h = h.to(device, non_blocking=True)
            r = r.to(device, non_blocking=True)
            t = t.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)

            with get_autocast_ctx(cfg):
                q_all = model.query(h, r)

                batch_listwise_loss = 0.0
                batch_cycle_loss = 0.0
                batch_agree_loss = 0.0
                batch_handshake_pairs = 0


                cycle_count = 0
                agree_count = 0
                valid_sample_count = 0

                unique_relations = torch.unique(r)

                for rel_id_tensor in unique_relations:
                    rel_id = rel_id_tensor.item()
                    group_idx_tensor = (r == rel_id).nonzero(as_tuple=True)[0]
                    group_indices = group_idx_tensor.tolist()

                    group_cand_union = set()
                    group_positive_sets = {}
                    group_cands_list = {}

                    for i in group_indices:
                        key = (h[i].item(), r[i].item())
                        positive_set = train_sr2o.get(key, {t[i].item()})
                        group_positive_sets[i] = positive_set

                        cand = sample_candidates_multi(
                            r=r[i].item(),
                            positive_set=positive_set,
                            cfg=cfg,
                            rel_cand=rel_cand,
                            rel2cand_type=rel2cand_type,
                            n_ent=n_ent
                        )

                        cand = limit_candidates(cand, positive_set, cfg.max_train_candidates)
                        group_cands_list[i] = cand
                        group_cand_union.update(cand)

                    if len(group_cand_union) == 0:
                        continue

                    group_cand_union = cap_group_union(
                        group_cand_union,
                        group_positive_sets,
                        cfg.max_group_candidates
                    )

                    group_cand_tensor = torch.tensor(group_cand_union, dtype=torch.long, device=device)
                    group_r_tensor = torch.full(
                        (len(group_cand_union),),
                        rel_id,
                        dtype=torch.long,
                        device=device
                    )

                    group_cand_emb = chunked_entity_repr(
                        model,
                        group_cand_tensor,
                        group_r_tensor,
                        cfg.entity_chunk_size
                    )

                    id2local = {eid: idx for idx, eid in enumerate(group_cand_union)}

                    for i in group_indices:
                        cand = group_cands_list[i]
                        positive_set = group_positive_sets[i]

                        # 若 union 裁剪后把某些元素裁掉了，需要同步修正 cand
                        filtered_cand = [eid for eid in cand if eid in id2local]
                        filtered_pos = set(eid for eid in positive_set if eid in id2local)

                        # 理论上 positives 不应该丢，但为了绝对稳妥加个兜底
                        if len(filtered_pos) == 0:
                            filtered_cand = list(dict.fromkeys(list(filtered_cand) + list(positive_set)))
                            filtered_cand = [eid for eid in filtered_cand if eid in id2local]
                            filtered_pos = set(eid for eid in positive_set if eid in id2local)

                        if len(filtered_cand) == 0 or len(filtered_pos) == 0:
                            continue

                        local_idx = [id2local[eid] for eid in filtered_cand]
                        local_idx_tensor = torch.tensor(local_idx, dtype=torch.long, device=device)

                        cand_emb = group_cand_emb[local_idx_tensor]
                        q = q_all[i:i+1]
                        scores = model.score(q, cand_emb).squeeze(0)

                        label = torch.zeros(len(filtered_cand), dtype=torch.float32, device=device)
                        cand_tensor_for_label = torch.tensor(filtered_cand, dtype=torch.long, device=device)

                        for pos_ent in filtered_pos:
                            pos_idx = (cand_tensor_for_label == pos_ent).nonzero(as_tuple=True)[0]
                            if pos_idx.numel() > 0:
                                label[pos_idx[0]] = 1.0

                        if cfg.label_smooth > 0:
                            label = label * (1.0 - cfg.label_smooth) + cfg.label_smooth / len(filtered_cand)

                        loss_listwise_i = listwise_loss(
                            scores=scores,
                            labels=label,
                            temperature=cfg.listwise_temperature
                        )
                        batch_listwise_loss = batch_listwise_loss + loss_listwise_i

                        if cfg.use_handshake:
                            current_sample = {
                                "s": h[i].item(),
                                "r": r[i].item(),
                                "positives": list(filtered_pos),
                            }
                            cycle_loss_i, agree_loss_i, used_pairs = compute_reverse_handshake_loss(
                                model=model,
                                current_sample=current_sample,
                                current_forward_scores=scores,
                                current_cand_list=filtered_cand,
                                train_sr2o=train_sr2o,
                                rel_cand=rel_cand,
                                rel2cand_type=rel2cand_type,
                                n_ent=n_ent,
                                n_rel=n_rel,
                                device=device,
                                cfg=cfg,
                            )

                            if cycle_loss_i is not None:
                                batch_cycle_loss = batch_cycle_loss + cycle_loss_i
                                cycle_count += 1

                            if agree_loss_i is not None:
                                batch_agree_loss = batch_agree_loss + agree_loss_i
                                agree_count += 1

                            batch_handshake_pairs += used_pairs

                        valid_sample_count += 1

                        del local_idx_tensor, cand_emb, scores, label, cand_tensor_for_label

                    del group_cand_tensor, group_r_tensor, group_cand_emb

                if valid_sample_count == 0:
                    continue

                loss_listwise = batch_listwise_loss / valid_sample_count

                if cycle_count > 0:
                    loss_cycle = batch_cycle_loss / cycle_count
                else:
                    loss_cycle = torch.zeros((), device=device)

                if agree_count > 0:
                    loss_agree = batch_agree_loss / agree_count
                else:
                    loss_agree = torch.zeros((), device=device)

                loss = cfg.listwise_weight * loss_listwise
                if cfg.use_handshake:
                    loss = loss + cfg.handshake_lambda * loss_cycle + cfg.agree_lambda * loss_agree

            scaler.scale(loss).backward()

            if cfg.grad_clip_norm is not None and cfg.grad_clip_norm > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)

            scaler.step(opt)
            scaler.update()

            # update_ema(teacher, model)

            total += loss.item()
            total_listwise_loss += loss_listwise.item()
            total_cycle_loss += float(loss_cycle)
            total_agree_loss += float(loss_agree)
            total_handshake_pairs += batch_handshake_pairs

            num_steps += 1
            global_step += 1

            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                # "cls": f"{loss_cls.item():.4f}",
                "listwise": f"{loss_listwise.item():.4f}",
                "cycle": f"{float(loss_cycle):.4f}",
                "agree": f"{float(loss_agree):.4f}",
                # "llm_aug": batch_llm_aug_samples,
                "pairs": batch_handshake_pairs,
            })

            del h, r, t, q_all, loss, loss_listwise, loss_cycle, loss_agree

            if global_step % cfg.empty_cache_every_steps == 0 and torch.cuda.is_available():
                maybe_empty_cache(device)
                gc.collect()

        avg_total = total / max(num_steps, 1)
        avg_listwise = total_listwise_loss / max(num_steps, 1)
        avg_cycle = total_cycle_loss / max(num_steps, 1)
        avg_agree = total_agree_loss / max(num_steps, 1)

        print(
            f"[Epoch {epoch:03d}] TRAIN  "
            f"Total = {avg_total:.6f} | "
            # f"Cls = {avg_cls:.6f} | "
            f"Listwsie = {avg_listwise:.6f} | "
            f"Cycle = {avg_cycle:.6f} | "
            f"Agree = {avg_agree:.6f} | "
            f"Pairs = {total_handshake_pairs} | "

        )

        hit1, hit3, hit10, mrr = evaluate(
            model,
            test_eval_triples,
            n_ent,
            all_true_test,
            rel2cand_type,
            device,
            batch_size=cfg.eval_batch_size,
            eval_entity_chunk_size=cfg.eval_entity_chunk_size
        )

        print(
            f"[Epoch {epoch:03d}] TEST   "
            f"Hit@1 = {hit1:.4f} | "
            f"Hit@3 = {hit3:.4f} | "
            f"Hit@10 = {hit10:.4f} | "
            f"MRR = {mrr:.4f}"
        )

        if hit1 > best_hit1:
            best_hit1 = hit1
            best_hit1_epoch = epoch
        if hit3 > best_hit3:
            best_hit3 = hit3
            best_hit3_epoch = epoch
        if hit10 > best_hit10:
            best_hit10 = hit10
            best_hit10_epoch = epoch
        if mrr > best_mrr:
            best_mrr = mrr
            best_mrr_epoch = epoch

        print(
            f"[Best] Hit@1 = {best_hit1:.4f} (epoch {best_hit1_epoch}) | "
            f"Hit@3 = {best_hit3:.4f} (epoch {best_hit3_epoch}) | "
            f"Hit@10 = {best_hit10:.4f} (epoch {best_hit10_epoch}) | "
            f"MRR = {best_mrr:.4f} (epoch {best_mrr_epoch})"
        )

        current_metric = hit1 if cfg.early_stop_metric.lower() == "hit1" else mrr

        if current_metric > best_stop_metric + cfg.min_delta:
            best_stop_metric = current_metric
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1

        print(
            f"[Epoch {epoch:03d}] EarlyStop monitor = {cfg.early_stop_metric}, "
            f"best = {best_stop_metric:.6f}, "
            f"no_improve = {no_improve_epochs}/{cfg.early_stop_patience}"
        )

        if no_improve_epochs >= cfg.early_stop_patience:
            print(f"[INFO] Early stopping triggered at epoch {epoch}.")
            break

        if torch.cuda.is_available():
            maybe_empty_cache(device)
            gc.collect()


if __name__ == "__main__":
    cfg = Config()
    set_seed(42)
    train(cfg)
