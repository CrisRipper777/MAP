from __future__ import annotations

import torch
from omegaconf import OmegaConf
from torch_geometric.data import Data

from src.data import EdgeSplit
from src.tasks.lp import (
    _build_epoch_train_labels,
    _build_forbidden_edge_keys,
    _build_link_loader,
    _build_message_edge_lookup,
    _exclude_positive_label_edges_from_message_graph,
    _exclude_positive_label_edges_by_global_eid,
)


def _pack(src: list[int], dst: list[int]) -> dict[str, torch.Tensor]:
    return {
        "source_node": torch.tensor(src, dtype=torch.long),
        "target_node": torch.tensor(dst, dtype=torch.long),
    }


def test_epoch_train_negative_edges_are_globally_filtered() -> None:
    edge_split = EdgeSplit(
        train=_pack([0, 2], [1, 3]),
        valid=_pack([0], [2]) | {"target_node_neg": torch.tensor([[4, 5]])},
        test=_pack([4], [5]) | {"target_node_neg": torch.tensor([[0, 1]])},
    )
    num_nodes = 6
    forbidden = _build_forbidden_edge_keys(edge_split, num_nodes=num_nodes, undirected=True)
    edge_label_index, edge_label = _build_epoch_train_labels(
        edge_split,
        num_nodes=num_nodes,
        num_neg=2,
        forbidden_keys=forbidden,
        generator=torch.Generator().manual_seed(11),
    )

    assert edge_label_index.size(1) == 6
    assert edge_label.tolist() == [1.0, 1.0, 0.0, 0.0, 0.0, 0.0]
    assert edge_label_index[:, :2].tolist() == [[0, 2], [1, 3]]

    positive_pairs = {(0, 1), (1, 0), (2, 3), (3, 2), (0, 2), (2, 0), (4, 5), (5, 4)}
    sampled_pairs = list(zip(edge_label_index[0, 2:].tolist(), edge_label_index[1, 2:].tolist(), strict=True))
    per_source: dict[int, list[int]] = {}
    for src, dst in sampled_pairs:
        assert src != dst
        assert (src, dst) not in positive_pairs
        per_source.setdefault(src, []).append(dst)
    assert all(len(targets) == len(set(targets)) for targets in per_source.values())


def test_positive_label_edges_are_excluded_from_message_graph() -> None:
    message_edge_index = torch.tensor(
        [
            [0, 1, 0, 2, 3, 4, 5],
            [1, 0, 2, 0, 4, 3, 0],
        ],
        dtype=torch.long,
    )
    edge_label_index = torch.tensor(
        [
            [0, 3],
            [1, 5],
        ],
        dtype=torch.long,
    )
    edge_label = torch.tensor([1.0, 0.0])

    filtered = _exclude_positive_label_edges_from_message_graph(
        message_edge_index,
        edge_label_index,
        edge_label,
        num_nodes=6,
    )

    assert filtered.tolist() == [[0, 2, 3, 4, 5], [2, 0, 4, 3, 0]]


def test_global_eid_mask_is_exactly_equivalent_to_local_key_mask() -> None:
    global_edge_index = torch.tensor(
        [[0, 1, 0, 2, 3, 4, 5], [1, 0, 2, 0, 4, 3, 0]],
        dtype=torch.long,
    )
    # Local nodes map to global IDs through n_id.
    batch = Data(
        edge_index=torch.tensor(
            [[0, 1, 2, 3, 2, 4, 5], [1, 0, 3, 2, 4, 2, 2]],
            dtype=torch.long,
        ),
        e_id=torch.tensor([4, 5, 0, 1, 2, 3, 6]),
        n_id=torch.tensor([3, 4, 0, 1, 2, 5]),
        edge_label_index=torch.tensor([[2, 0], [3, 5]], dtype=torch.long),
        edge_label=torch.tensor([1.0, 0.0]),
        num_nodes=6,
    )
    expected = _exclude_positive_label_edges_from_message_graph(
        batch.edge_index,
        batch.edge_label_index,
        batch.edge_label,
        num_nodes=6,
    )

    removed = _exclude_positive_label_edges_by_global_eid(
        batch,
        _build_message_edge_lookup(global_edge_index, num_nodes=6),
    )

    assert removed == 2
    assert torch.equal(batch.edge_index, expected)
    assert batch.e_id.tolist() == [4, 5, 2, 3, 6]


def test_global_eid_mask_removes_all_duplicate_directed_edges() -> None:
    global_edge_index = torch.tensor(
        [[0, 0, 1, 1, 2], [1, 1, 0, 0, 3]],
        dtype=torch.long,
    )
    batch = Data(
        edge_index=global_edge_index.clone(),
        e_id=torch.arange(global_edge_index.size(1)),
        n_id=torch.arange(4),
        edge_label_index=torch.tensor([[0], [1]], dtype=torch.long),
        edge_label=torch.ones(1),
        num_nodes=4,
    )

    removed = _exclude_positive_label_edges_by_global_eid(
        batch,
        _build_message_edge_lookup(global_edge_index, num_nodes=4),
    )

    assert removed == 4
    assert batch.edge_index.tolist() == [[2], [3]]
    assert batch.e_id.tolist() == [4]


def test_global_eid_mask_rejects_misaligned_edge_ids() -> None:
    global_edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    batch = Data(
        edge_index=global_edge_index.clone(),
        e_id=torch.tensor([0]),
        n_id=torch.arange(2),
        edge_label_index=torch.tensor([[0], [1]], dtype=torch.long),
        edge_label=torch.ones(1),
        num_nodes=2,
    )

    try:
        _exclude_positive_label_edges_by_global_eid(
            batch,
            _build_message_edge_lookup(global_edge_index, num_nodes=2),
        )
    except ValueError as error:
        assert "align one-to-one" in str(error)
    else:
        raise AssertionError("misaligned sampled edge IDs were unexpectedly accepted")


def test_global_eid_mask_matches_local_keys_on_real_link_neighbor_batch() -> None:
    global_edge_index = torch.tensor(
        [
            [0, 1, 1, 2, 2, 3, 3, 4, 4, 5],
            [1, 0, 2, 1, 3, 2, 4, 3, 5, 4],
        ],
        dtype=torch.long,
    )
    graph = Data(x=torch.randn(6, 4), edge_index=global_edge_index)
    cfg = OmegaConf.create(
        {
            "task": {
                "num_neighbors": [-1, -1],
                "batch_size": 2,
                "subgraph_type": "bidirectional",
                "loader_num_workers": 0,
            }
        }
    )
    loader = _build_link_loader(
        cfg,
        graph,
        edge_label_index=torch.tensor([[1, 4], [2, 5]], dtype=torch.long),
        edge_label=torch.tensor([1.0, 0.0]),
        batch_generator=torch.Generator().manual_seed(7),
        neighbor_seed=9,
    )
    batch = next(iter(loader))
    expected = _exclude_positive_label_edges_from_message_graph(
        batch.edge_index,
        batch.edge_label_index,
        batch.edge_label,
        num_nodes=int(batch.num_nodes),
    )

    _exclude_positive_label_edges_by_global_eid(
        batch,
        _build_message_edge_lookup(global_edge_index, num_nodes=6),
    )

    assert torch.equal(batch.edge_index, expected)
