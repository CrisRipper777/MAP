from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader as TorchDataLoader
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
from sklearn.metrics import f1_score

from src.data import MAGData
from src.models import build_model
from src.tasks.common import (
    build_optimizer,
    clone_state_dict,
    format_aux_info_stats,
    load_state_dict_cpu,
    resolve_num_neighbors,
    scheduler_step,
    summarize_aux_info_stats,
    update_aux_info_stats,
)
from src.tasks.analysis import export_edge_aux_stats, export_node_aux_stats
from src.tasks.inference import infer_all_embeddings, resolve_inference_mode
from src.tasks.modality import iter_masked_data, modality_drop_key, modality_mask_key, modality_mask_eval_enabled
from src.tasks.prototype_analysis import export_node_prototype_stats
from src.utils.metrics import format_pct
from src.utils.seeds import set_seed
from src.utils.summary import count_parameters, mean_std


def _uses_graph_encoder(cfg) -> bool:
    return str(cfg.model.name).lower() != "mlp"


def _resolve_training_mode(cfg, model) -> str:
    """Use the 0901 NC protocol: full graph by default."""
    if getattr(model, "requires_full_graph_training", False):
        return "full_graph"
    mode = str(cfg.task.get("training_mode", "full_graph")).strip().lower()
    if mode not in {"full_graph", "sampled"}:
        raise ValueError(f"task.training_mode must be full_graph|sampled, got {mode!r}")
    return mode


@torch.no_grad()
def _evaluate_split(
    classifier,
    z: torch.Tensor,
    labels: torch.Tensor,
    idx: torch.Tensor,
    device: torch.device,
    batch_size: int,
) -> dict[str, float]:
    classifier.eval()
    preds: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    for batch_idx in TorchDataLoader(idx.cpu(), batch_size=batch_size, shuffle=False):
        logits = classifier(z[batch_idx].to(device))
        preds.append(logits.argmax(dim=-1).cpu())
        targets.append(labels[batch_idx].cpu())

    pred = torch.cat(preds, dim=0)
    target = torch.cat(targets, dim=0)
    return {
        "acc": float((pred == target).float().mean().item()),
        "macro_f1": float(f1_score(target.numpy(), pred.numpy(), average="macro", zero_division=0)),
    }


def _evaluate_modality_masks(
    cfg,
    model,
    classifier,
    data: MAGData,
    device: torch.device,
    uses_graph: bool,
    inference_mode: str,
    inference_batch_size: int,
    seed: int,
    full_test: dict[str, float],
    logger: logging.Logger,
) -> dict[str, float]:
    if not modality_mask_eval_enabled(cfg):
        return {}

    output: dict[str, float] = {}
    for mode, masked_data in iter_masked_data(cfg, data, seed):
        mask_key = modality_mask_key(mode)
        drop_key = modality_drop_key(mode)
        z = infer_all_embeddings(model, masked_data, device, uses_graph, inference_batch_size, inference_mode)
        metrics = _evaluate_split(classifier, z, data.y, data.test_idx, device, inference_batch_size)
        output[f"{mask_key}_test_acc"] = metrics["acc"]
        output[f"{mask_key}_test_macro_f1"] = metrics["macro_f1"]
        output[f"{drop_key}_acc"] = full_test["test_acc"] - metrics["acc"]
        output[f"{drop_key}_macro_f1"] = full_test["test_macro_f1"] - metrics["macro_f1"]
        logger.info(
            "Modality mask [%s] Test Acc %.2f | Test F1 %.2f | Drop Acc %.2f | Drop F1 %.2f",
            mode,
            format_pct(metrics["acc"]),
            format_pct(metrics["macro_f1"]),
            format_pct(output[f"{drop_key}_acc"]),
            format_pct(output[f"{drop_key}_macro_f1"]),
        )
        del z
        torch.cuda.empty_cache()
    return output


def _run_single_nc(
    cfg,
    data: MAGData,
    device: torch.device,
    logger: logging.Logger,
    run_id: int,
    output_dir: str | Path | None,
) -> dict[str, float]:
    seed = int(cfg.seed) + run_id
    set_seed(seed)
    inference_mode = resolve_inference_mode(cfg)

    data_info = {
        "input_dim": data.input_dim,
        "num_nodes": data.num_nodes,
        "num_classes": data.num_classes,
        "text_dim": int(data.x_t.shape[1]) if data.x_t is not None else 0,
        "visual_dim": int(data.x_i.shape[1]) if data.x_i is not None else 0,
    }
    model = build_model(cfg, data_info).to(device)
    if hasattr(model, "initialize_prototypes"):
        initialized = model.initialize_prototypes(data.x, device=device, seed=seed)
        if initialized:
            logger.info("Initialized modality prototypes with K-means from frozen node features")
    classifier = nn.Linear(model.out_dim, int(data.num_classes)).to(device)
    optimizer = build_optimizer(list(model.parameters()) + list(classifier.parameters()), cfg, model=model)
    criterion = nn.CrossEntropyLoss()
    uses_graph = _uses_graph_encoder(cfg)
    training_mode = _resolve_training_mode(cfg, model)
    full_graph_training = training_mode == "full_graph"

    num_neighbors = None
    if full_graph_training:
        loader = None
        loader_name = "FullGraph"
    elif uses_graph:
        pyg_data = Data(x=data.x, edge_index=data.edge_index, y=data.y)
        num_neighbors = resolve_num_neighbors(cfg)
        loader = NeighborLoader(
            pyg_data,
            input_nodes=data.train_idx,
            num_neighbors=num_neighbors,
            batch_size=int(cfg.task.batch_size),
            shuffle=True,
        )
        loader_name = "NeighborLoader"
    else:
        loader = TorchDataLoader(
            data.train_idx,
            batch_size=int(cfg.task.batch_size),
            shuffle=True,
        )
        loader_name = "NodeDataLoader"
    x_all = data.x.to(device) if (not uses_graph or full_graph_training) else None
    y_all = data.y.to(device) if (not uses_graph or full_graph_training) else None
    edge_index_all = data.edge_index.to(device) if (uses_graph and full_graph_training) else None
    train_idx_all = data.train_idx.to(device) if full_graph_training else None

    logger.info(
        "[Run %d/%d] seed=%d | model params=%d",
        run_id + 1,
        int(cfg.num_runs),
        seed,
        count_parameters(model) + count_parameters(classifier),
    )
    logger.info("Loader: %s", loader_name)
    logger.info("Protocol: %s", str(cfg.task.get("protocol_version", "unified_full_graph_nc_v1")))
    if full_graph_training:
        logger.info("Train full graph: enabled by the unified NC protocol")
    elif uses_graph:
        logger.info("Train neighbor sampling: %s", num_neighbors)
    logger.info("Inference mode: %s", inference_mode)
    logger.info("Training...")

    best_val = -1.0
    best_test: dict[str, float] = {}
    best_model_state = None
    best_head_state = None
    best_epoch = 0
    early_stop_epoch = 0
    patience_total = int(cfg.task.patience)
    patience_left = patience_total
    early_stop_min_epoch = int(cfg.task.get("early_stop_min_epoch", 1))
    early_stop_min_delta = float(cfg.task.get("early_stop_min_delta", 0.0))
    grad_clip = cfg.task.get("grad_clip")
    aux_weight = float(cfg.task.loss.aux_weight)
    max_train_batches = cfg.task.get("max_train_batches")
    inference_batch_size = int(cfg.task.inference_batch_size)
    evaluate_test = bool(cfg.task.get("evaluate_test", True))

    for epoch in range(1, int(cfg.task.epochs) + 1):
        model.train()
        if hasattr(model, "set_epoch"):
            model.set_epoch(epoch)
        classifier.train()
        total_loss = 0.0
        total_examples = 0
        aux_sums: dict[str, float] = {}
        aux_counts: dict[str, float] = {}
        if full_graph_training:
            optimizer.zero_grad(set_to_none=True)
            z, _, _, aux_loss, aux_info = model(x_all, edge_index_all)
            labels = y_all[train_idx_all]
            logits = classifier(z[train_idx_all])
            loss = criterion(logits, labels) + aux_weight * aux_loss
            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(
                    list(model.parameters()) + list(classifier.parameters()), max_norm=float(grad_clip)
                )
            optimizer.step()
            total_loss += float(loss.item()) * int(labels.numel())
            total_examples += int(labels.numel())
            update_aux_info_stats(aux_sums, aux_counts, aux_info, weight=float(labels.numel()))
            del z
        else:
            for step, batch in enumerate(loader):
                if max_train_batches is not None and step >= int(max_train_batches):
                    break
                optimizer.zero_grad(set_to_none=True)
                if uses_graph:
                    batch = batch.to(device)
                    if hasattr(model, "_batch_n_id"):
                        model._batch_n_id = batch.n_id
                    z, _, _, aux_loss, aux_info = model(batch.x, batch.edge_index)
                    logits = classifier(z[: batch.batch_size])
                    labels = batch.y[: batch.batch_size]
                else:
                    batch_idx = batch.to(device)
                    x_batch = x_all[batch_idx]
                    labels = y_all[batch_idx]
                    z, _, _, aux_loss, aux_info = model(x_batch, None)
                    logits = classifier(z)
                loss = criterion(logits, labels) + aux_weight * aux_loss
                loss.backward()
                if grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(
                        list(model.parameters()) + list(classifier.parameters()), max_norm=float(grad_clip)
                    )
                optimizer.step()
                total_loss += float(loss.item()) * int(labels.numel())
                total_examples += int(labels.numel())
                update_aux_info_stats(aux_sums, aux_counts, aux_info, weight=float(labels.numel()))
                if hasattr(model, "_batch_n_id"):
                    model._batch_n_id = None

        train_loss = total_loss / max(total_examples, 1)
        aux_stats = summarize_aux_info_stats(aux_sums, aux_counts)
        scheduler_step(cfg, optimizer, epoch, int(cfg.task.epochs))
        if epoch % int(cfg.task.eval_every) != 0:
            logger.info("Epoch %05d | Train Loss %.4f", epoch, train_loss)
            if aux_stats:
                logger.info("Aux %s", format_aux_info_stats(aux_stats))
            continue

        z = infer_all_embeddings(model, data, device, uses_graph, inference_batch_size, inference_mode)
        val_metrics = _evaluate_split(classifier, z, data.y, data.val_idx, device, inference_batch_size)
        logger.info("Epoch %05d | Train Loss %.4f", epoch, train_loss)
        if aux_stats:
            logger.info("Aux %s", format_aux_info_stats(aux_stats))
        logger.info(
            "Val Acc %.2f | Val F1 %.2f",
            format_pct(val_metrics["acc"]),
            format_pct(val_metrics["macro_f1"]),
        )

        stop_early = False
        if val_metrics["acc"] > best_val + early_stop_min_delta:
            best_val = val_metrics["acc"]
            best_test = {
                "val_acc": val_metrics["acc"],
                "val_macro_f1": val_metrics["macro_f1"],
            }
            best_epoch = epoch
            best_model_state = clone_state_dict(model)
            best_head_state = clone_state_dict(classifier)
            patience_left = patience_total
        elif epoch >= early_stop_min_epoch:
            patience_left -= 1
            patience_used = patience_total - patience_left
            logger.info("Patience %d/%d | Best Val Acc %.2f", patience_used, patience_total, format_pct(best_val))
            if patience_left <= 0:
                logger.info("Early stopping at epoch %03d", epoch)
                early_stop_epoch = epoch
                stop_early = True
        del z
        torch.cuda.empty_cache()
        if stop_early:
            break

    if output_dir is not None:
        export_node_aux_stats(
            cfg=cfg,
            model=model,
            data=data,
            device=device,
            output_dir=output_dir,
            run_id=run_id,
            tag="final_epoch",
            task_name="nc",
            logger=logger,
        )
        export_edge_aux_stats(
            cfg=cfg,
            model=model,
            data=data,
            device=device,
            output_dir=output_dir,
            run_id=run_id,
            tag="final_epoch",
            task_name="nc",
            logger=logger,
        )
        export_node_prototype_stats(
            cfg=cfg,
            model=model,
            data=data,
            device=device,
            output_dir=output_dir,
            run_id=run_id,
            tag="final_epoch",
            task_name="nc",
            logger=logger,
            classifier=classifier,
            uses_graph=uses_graph,
            inference_batch_size=inference_batch_size,
            inference_mode=inference_mode,
        )

    if best_model_state is not None and best_head_state is not None:
        load_state_dict_cpu(model, best_model_state)
        load_state_dict_cpu(classifier, best_head_state)
        if output_dir is not None:
            export_node_aux_stats(
                cfg=cfg,
                model=model,
                data=data,
                device=device,
                output_dir=output_dir,
                run_id=run_id,
                tag="best_val",
                task_name="nc",
                logger=logger,
            )
            export_edge_aux_stats(
                cfg=cfg,
                model=model,
                data=data,
                device=device,
                output_dir=output_dir,
                run_id=run_id,
                tag="best_val",
                task_name="nc",
                logger=logger,
            )
            export_node_prototype_stats(
                cfg=cfg,
                model=model,
                data=data,
                device=device,
                output_dir=output_dir,
                run_id=run_id,
                tag="best_val",
                task_name="nc",
                logger=logger,
                classifier=classifier,
                uses_graph=uses_graph,
                inference_batch_size=inference_batch_size,
                inference_mode=inference_mode,
            )
        if evaluate_test:
            z = infer_all_embeddings(model, data, device, uses_graph, inference_batch_size, inference_mode)
            test_metrics = _evaluate_split(classifier, z, data.y, data.test_idx, device, inference_batch_size)
            best_test["test_acc"] = test_metrics["acc"]
            best_test["test_macro_f1"] = test_metrics["macro_f1"]
            del z
            best_test.update(
                _evaluate_modality_masks(
                    cfg,
                    model,
                    classifier,
                    data,
                    device,
                    uses_graph,
                    inference_mode,
                    inference_batch_size,
                    seed,
                    best_test,
                    logger,
                )
            )

    if not best_test:
        z = infer_all_embeddings(model, data, device, uses_graph, inference_batch_size, inference_mode)
        val_metrics = _evaluate_split(classifier, z, data.y, data.val_idx, device, inference_batch_size)
        best_test = {
            "val_acc": val_metrics["acc"],
            "val_macro_f1": val_metrics["macro_f1"],
        }
        best_epoch = epoch
        if evaluate_test:
            test_metrics = _evaluate_split(classifier, z, data.y, data.test_idx, device, inference_batch_size)
            best_test["test_acc"] = test_metrics["acc"]
            best_test["test_macro_f1"] = test_metrics["macro_f1"]
        del z
        if evaluate_test:
            best_test.update(
                _evaluate_modality_masks(
                    cfg,
                    model,
                    classifier,
                    data,
                    device,
                    uses_graph,
                    inference_mode,
                    inference_batch_size,
                    seed,
                    best_test,
                    logger,
                )
            )

    best_test["best_epoch"] = float(best_epoch)
    best_test["early_stop_epoch"] = float(early_stop_epoch or epoch)

    if hasattr(model, "_batch_n_id"):
        model._batch_n_id = None

    save_ckpt_path = cfg.task.get("save_ckpt_path")
    if save_ckpt_path:
        path = Path(str(save_ckpt_path))
        if int(cfg.num_runs) > 1:
            path = path.with_name(f"{path.stem}_run{run_id + 1}{path.suffix}")
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "task": "nc",
                "seed": seed,
                "model_state": clone_state_dict(model),
                "head_state": clone_state_dict(classifier),
                "data_info": data_info,
            },
            path,
        )
        logger.info("Saved checkpoint: %s", path)

    logger.info(
        "[Run %d] Best Val Acc %.2f | Best Val F1 %.2f | Best Epoch %03d | Early Stop %03d",
        run_id + 1,
        format_pct(best_test["val_acc"]),
        format_pct(best_test["val_macro_f1"]),
        int(best_test["best_epoch"]),
        int(best_test["early_stop_epoch"]),
    )
    if evaluate_test:
        logger.info(
            "[Run %d] Final Test Acc %.2f | Final Test F1 %.2f",
            run_id + 1,
            format_pct(best_test["test_acc"]),
            format_pct(best_test["test_macro_f1"]),
        )
    return best_test


def run_nc(
    cfg,
    data: MAGData,
    device: torch.device,
    logger: logging.Logger,
    output_dir: str | Path | None = None,
) -> dict[str, tuple[float, float]]:
    evaluate_test = bool(cfg.task.get("evaluate_test", True))
    if data.y is None or data.train_idx is None or data.val_idx is None:
        raise ValueError("NC data must contain y/train_idx/val_idx")
    if evaluate_test and data.test_idx is None:
        raise ValueError("NC data must contain test_idx when task.evaluate_test=true")

    run_results = [
        _run_single_nc(cfg, data, device, logger, run_id, output_dir)
        for run_id in range(int(cfg.num_runs))
    ]
    val_acc = [item["val_acc"] for item in run_results]
    val_macro_f1 = [item["val_macro_f1"] for item in run_results]
    val_mean, val_std = mean_std(val_acc)
    f1_mean, f1_std = mean_std(val_macro_f1)
    logger.info("============================================================")
    logger.info("Final Results over %d runs", int(cfg.num_runs))
    logger.info("============================================================")
    logger.info("Highest Valid Acc: %.2f ± %.2f", format_pct(val_mean), format_pct(val_std))
    output = {
        "val_acc": (val_mean, val_std),
        "val_macro_f1": (f1_mean, f1_std),
    }
    logger.info("Val Macro-F1: %.2f ± %.2f", format_pct(f1_mean), format_pct(f1_std))
    if evaluate_test:
        test_acc = [item["test_acc"] for item in run_results]
        test_f1 = [item["test_macro_f1"] for item in run_results]
        acc_mean, acc_std = mean_std(test_acc)
        f1_mean, f1_std = mean_std(test_f1)
        logger.info("Test Acc: %.2f ± %.2f", format_pct(acc_mean), format_pct(acc_std))
        logger.info("Test Macro-F1: %.2f ± %.2f", format_pct(f1_mean), format_pct(f1_std))
        output.update({"test_acc": (acc_mean, acc_std), "test_macro_f1": (f1_mean, f1_std)})
    primary_keys = set(output)
    for key in sorted(set().union(*(item.keys() for item in run_results)) - primary_keys):
        values = [item[key] for item in run_results if key in item]
        if len(values) != len(run_results):
            continue
        mean, std = mean_std(values)
        output[key] = (mean, std)
        if key in {"best_epoch", "early_stop_epoch"}:
            logger.info("%s: %.2f ± %.2f", key, mean, std)
        else:
            logger.info("%s: %.2f ± %.2f", key, format_pct(mean), format_pct(std))
    logger.info("============================================================")
    return output
