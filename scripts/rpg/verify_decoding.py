#!/usr/bin/env python3
"""Verify the large-catalogue decoding pipeline against the original RPG code
and measure how often graph decoding reaches the items the model ranks highest.

Checks, on a trained checkpoint:
  A  graph:       the row-blocked GPU graph (decoding.graph.build_chunked_exact_adjacency)
                  equals the upstream graph. Catalogues up to 70k items: upstream
                  build_ii_sim_mat + topk on every row. Larger catalogues, where the
                  upstream dense matrix does not fit: an independent per-digit
                  computation of the similarity for --sample-rows sampled rows.
  B  exhaustive:  chunked exhaustive top-10 equals upstream RPG.generate without the
                  graph (same scores; ids may differ only between tied items).
  C  decoder:     upstream RPG.generate with the graph equals the traced decoder used
                  for reachability (same predictions and visited counts, same RNG).
Measures, at the config's decoding budget, on --max-users test users (the same
fixed subset as scripts/rpg/eval_seeds.py --max-users):
  D  reachability: share of the catalogue visited, share of targets reached, and
                  how many of the exhaustive top-10 hits graph decoding keeps, loses
                  without reaching, or loses after reaching.

Usage:
  python scripts/rpg/verify_decoding.py --name books_m32 \
      --checkpoint artifacts/rpg/ckpt/<run>.pth \
      --config configs/rpg/books.yaml --n_codebook 32
Writes artifacts/rpg/verification/<name>.json.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
from decoding.bruteforce import _chunked_topk_from_logits
from decoding.config import build_repo_config_files, parse_override_args
from decoding.graph import build_chunked_exact_adjacency
from decoding.harness import EvaluationHarness
from decoding.trace import compute_decoding_context, traced_generate
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--name", required=True, help="Label for the output file, e.g. books_m32.")
parser.add_argument("--checkpoint", required=True, help="Path to a trained RPG checkpoint.")
parser.add_argument("--config", action="append", default=[], help="Additional YAML config file.")
parser.add_argument("--max-users", type=int, default=100_000, help="Test users for the reachability measurement.")
parser.add_argument("--sample-rows", type=int, default=300, help="Rows checked against the independent graph computation.")
parser.add_argument("--parity-users", type=int, default=256, help="Users checked in parity checks B and C.")
parser.add_argument("--output-dir", default="artifacts/rpg/verification", help="Directory for the result JSON.")
cli, override_tokens = parser.parse_known_args()
name = cli.name
config = cli.config
overrides = parse_override_args(override_tokens)
checkpoint = str(Path(cli.checkpoint).resolve())
out = {"name": name, "checkpoint": checkpoint}


def log(key, value):
    out[key] = value
    print(f"[{name}] {key}: {value}", flush=True)


h = EvaluationHarness.build(
    checkpoint_path=checkpoint,
    config_files=build_repo_config_files(extra_configs=config, include_root_config=True, include_local_config=False),
    config_overrides=overrides,
)
model = h.model.eval()
dev = h.accelerator.device
model.to(dev)
model.item_id2tokens = model.item_id2tokens.to(dev)
n_items = int(h.dataset.n_items)
k_edges, beams, steps = int(h.config["n_edges"]), int(h.config["num_beams"]), int(h.config["propagation_steps"])
log("catalogue", {"n_items": n_items - 1, "n_digit": int(h.config["n_codebook"]), "budget_b_k_q": [beams, k_edges, steps]})

# ---------------------------------------------------------------- A: graph
t = time.time()
adjacency = build_chunked_exact_adjacency(model, topk=k_edges)
log("A_exact_gpu_build_seconds", round(time.time() - t, 1))
with torch.no_grad():
    if n_items <= 70_000:
        t = time.time()
        sim = model.build_ii_sim_mat()                       # upstream code, dense
        upstream = model.build_adjacency_list(sim).cpu()     # upstream code, topk
        log("A_upstream_dense_build_seconds", round(time.time() - t, 1))
        rows = range(1, n_items)
        same = sum(set(adjacency[i].tolist()) == set(upstream[i].tolist()) for i in rows)
        gap = max(abs(float(sim[i, adjacency[i].to(dev)].sum() - sim[i, upstream[i].to(dev)].sum())) for i in rows)
        log("A_rows_identical_to_upstream", f"{same}/{n_items - 1}")
        log("A_max_neighbour_similarity_gap_vs_upstream", gap)
        del sim
    else:
        rng = np.random.default_rng(0)
        sample = np.sort(rng.choice(np.arange(1, n_items), cli.sample_rows, replace=False))
        # Independent computation of the upstream similarity for the sampled rows:
        # per digit, the cosine between the sampled items' token embeddings and
        # every item's token embedding, mapped to [0, 1] and averaged over digits.
        n_digit, codebook_size = model.tokenizer.n_digit, model.tokenizer.codebook_size
        tok = torch.nn.functional.normalize(model.gpt2.wte.weight[1:-1].detach().float(), dim=-1).view(n_digit, codebook_size, -1)
        offsets = torch.arange(n_digit, device=dev) * codebook_size + 1
        codes = model.item_id2tokens[1:] - offsets                 # (n_real, n_digit)
        sample_t = torch.from_numpy(sample).to(dev)
        rows = torch.zeros((len(sample), n_items - 1), device=dev)
        for digit in range(n_digit):
            all_vec = tok[digit][codes[:, digit]]                  # (n_real, d)
            sample_vec = tok[digit][codes[sample_t - 1, digit]]    # (sample, d)
            rows += 0.5 * (sample_vec @ all_vec.T + 1.0)
        rows /= n_digit
        best = rows.topk(k_edges, dim=-1)
        ours = adjacency[torch.from_numpy(sample)].to(dev) - 1
        ours_sum = rows.gather(1, ours).sum(-1)
        gap = float((ours_sum - best.values.sum(-1)).abs().max())
        same = int(sum(set(ours[i].tolist()) == set(best.indices[i].tolist()) for i in range(len(sample))))
        log("A_sampled_rows_identical_to_independent_topk", f"{same}/{len(sample)}")
        log("A_max_neighbour_similarity_gap_vs_independent", gap)

model.adjacency = adjacency.to(dev)
model.init_flag = True
model.num_beams, model.n_edges, model.propagation_steps = beams, k_edges, steps
maxk = h.trainer.evaluator.maxk

# ---------------------------------------------------------------- B: exhaustive parity
loader = DataLoader(h.test_dataloader.dataset.select(range(cli.parity_users)), batch_size=16, shuffle=False, collate_fn=h.test_dataloader.collate_fn)
ids_equal = scores_equal = total = 0
with torch.no_grad():
    for batch in loader:
        batch = {k: v.to(dev) for k, v in batch.items()}
        model.generate_w_decoding_graph = False
        upstream_ids = model.generate(batch, n_return_sequences=maxk).squeeze(-1)
        ctx = compute_decoding_context(model, batch)
        ours_ids, ours_scores = _chunked_topk_from_logits(
            token_logits=ctx.token_logits, item_id2tokens=model.item_id2tokens, n_items=n_items,
            batch_size=upstream_ids.shape[0], topk=maxk, item_chunk_size=50_000, device=dev)
        up_scores = ctx.token_logits.gather(1, (model.item_id2tokens[upstream_ids] - 1).flatten(1)).view(*upstream_ids.shape, -1).mean(-1)
        ids_equal += int((upstream_ids == ours_ids).all(-1).sum())
        scores_equal += int(torch.isclose(up_scores, ours_scores, rtol=1e-5, atol=1e-6).all(-1).sum())
        total += upstream_ids.shape[0]
log("B_exhaustive_same_top10_ids", f"{ids_equal}/{total}")
log("B_exhaustive_same_top10_scores", f"{scores_equal}/{total}")

# ---------------------------------------------------------------- C: decoder parity
same_pred = same_visit = total = 0
with torch.no_grad():
    for batch in loader:
        batch = {k: v.to(dev) for k, v in batch.items()}
        model.generate_w_decoding_graph = True
        state = torch.cuda.get_rng_state(dev) if dev.type == "cuda" else torch.get_rng_state()
        up_pred, up_visit = model.generate(batch, n_return_sequences=maxk)
        (torch.cuda.set_rng_state(state, dev) if dev.type == "cuda" else torch.set_rng_state(state))
        tr_pred, tr_visit, _ = traced_generate(model, batch, n_return_sequences=maxk)
        same_pred += int((up_pred.cpu() == tr_pred.cpu()).flatten(1).all(-1).sum())
        same_visit += int((up_visit.cpu() == tr_visit.cpu()).flatten().sum())
        total += up_pred.shape[0]
log("C_traced_decoder_same_predictions", f"{same_pred}/{total}")
log("C_traced_decoder_same_visited_counts", f"{same_visit}/{total}")

# ---------------------------------------------------------------- D: reachability and ranking
test = h.test_dataloader.dataset
n_users = len(test)
subset = np.sort(np.random.default_rng(2024).choice(n_users, min(cli.max_users, n_users), replace=False)) if n_users > cli.max_users else np.arange(n_users)
loader = DataLoader(test.select(subset.tolist()), batch_size=int(h.config["eval_batch_size"]), shuffle=False, collate_fn=h.test_dataloader.collate_fn)
torch.manual_seed(2024)
torch.cuda.manual_seed_all(2024)
c = dict(users=0, reached=0, graph_hit=0, ex_hit=0, ex_hit_reached=0, ex_hit_graph_hit=0, graph_ndcg=0.0, ex_ndcg=0.0, visited=0.0,
         ex_hit_missed_unreached=0, ex_hit_missed_reached=0, reached_not_graph_hit=0)
with torch.no_grad():
    for batch in loader:
        batch = {k: v.to(dev) for k, v in batch.items()}
        labels = batch["labels"][:, 0]
        g_pred, g_visit, trace = traced_generate(model, batch, n_return_sequences=maxk)
        g_top = g_pred.squeeze(-1).cpu()
        ctx = compute_decoding_context(model, batch)
        e_top, _ = _chunked_topk_from_logits(token_logits=ctx.token_logits, item_id2tokens=model.item_id2tokens, n_items=n_items,
                                             batch_size=labels.shape[0], topk=maxk, item_chunk_size=50_000, device=dev)
        e_top = e_top.cpu()
        for row, label in enumerate(labels.cpu().tolist()):
            reached = label in set(trace.final_visited_items[row])
            g_hit_pos = (g_top[row] == label).nonzero()
            e_hit_pos = (e_top[row] == label).nonzero()
            g_hit, e_hit = len(g_hit_pos) > 0, len(e_hit_pos) > 0
            c["users"] += 1
            c["reached"] += reached
            c["graph_hit"] += g_hit
            c["ex_hit"] += e_hit
            c["visited"] += float(g_visit[row])
            c["reached_not_graph_hit"] += reached and not g_hit
            if g_hit:
                c["graph_ndcg"] += 1.0 / np.log2(int(g_hit_pos[0]) + 2)
            if e_hit:
                c["ex_ndcg"] += 1.0 / np.log2(int(e_hit_pos[0]) + 2)
                c["ex_hit_reached"] += reached
                c["ex_hit_graph_hit"] += g_hit
                if not g_hit:
                    c["ex_hit_missed_unreached" if not reached else "ex_hit_missed_reached"] += 1
n = c["users"]
log("D_users", n)
log("D_mean_items_visited", round(c["visited"] / n))
log("D_share_of_catalogue_visited", round(c["visited"] / n / (n_items - 1), 4))
log("D_target_reached_rate", round(c["reached"] / n, 4))
log("D_graph_recall@10", round(c["graph_hit"] / n, 4))
log("D_graph_ndcg@10", round(c["graph_ndcg"] / n, 4))
log("D_exhaustive_recall@10", round(c["ex_hit"] / n, 4))
log("D_exhaustive_ndcg@10", round(c["ex_ndcg"] / n, 4))
log("D_reached_but_not_in_graph_top10_rate", round(c["reached_not_graph_hit"] / n, 4))
if c["ex_hit"]:
    log("D_of_exhaustive_hits_reached_by_graph", round(c["ex_hit_reached"] / c["ex_hit"], 4))
    log("D_of_exhaustive_hits_kept_by_graph", round(c["ex_hit_graph_hit"] / c["ex_hit"], 4))
    log("D_of_exhaustive_hits_lost_because_never_reached", round(c["ex_hit_missed_unreached"] / c["ex_hit"], 4))
    log("D_of_exhaustive_hits_lost_although_reached", round(c["ex_hit_missed_reached"] / c["ex_hit"], 4))
out_dir = Path(cli.output_dir)
if not out_dir.is_absolute():
    out_dir = REPO_ROOT / out_dir
out_dir.mkdir(parents=True, exist_ok=True)
(out_dir / f"{name}.json").write_text(json.dumps(out, indent=2))
print(out_dir / f"{name}.json")
