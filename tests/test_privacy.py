"""Tests for accounting, noise, secure aggregation, ledgers, bounding, and the audit."""
import numpy as np
import pandas as pd
import pytest

from urbanepi.aggregator.stream import BudgetRefused, MonitoringStream, fast_noisy_table
from urbanepi.config import load_config
from urbanepi.gate.c2 import alert_onsets, c2_scores
from urbanepi.institution.bounding import apply_person_cap
from urbanepi.institution.ledger import BudgetLedger, reserve_all
from urbanepi.institution.node import InstitutionNode
from urbanepi.privacy.accounting import (StreamBudget, epsilon_from_rho, rho_for, rho_from_epsilon, sensitivity_for,
                                         sigma_for)
from urbanepi.privacy.audit import empirical_epsilon
from urbanepi.privacy.secure_agg import add_mask, aggregate, decode, encode, pairwise_masks

CFG = load_config()
BITS = CFG["privacy"]["noise_scale_bits"]


def make_nodes(count: int, total: float = 1.0, seed: int = 1) -> list[InstitutionNode]:
    seeds = np.random.SeedSequence(seed).spawn(count)
    return [InstitutionNode(str(i), BudgetLedger(total), np.random.default_rng(s), BITS) for i, s in enumerate(seeds)]


@pytest.mark.parametrize("epsilon", [0.5, 1, 2, 4, 8])
def test_rho_and_epsilon_convert_both_ways(epsilon):
    assert epsilon_from_rho(rho_from_epsilon(epsilon, 1e-6), 1e-6) == pytest.approx(epsilon)


def test_sigma_and_rho_are_inverses():
    assert rho_for(sigma_for(0.05, 2.0), 2.0) == pytest.approx(0.05)


@pytest.mark.parametrize("epsilon, unit, expected", [
    (2, "record", 3.5), (2, "person", 7.0), (4, "record", 1.8), (4, "person", 3.6), (0.5, "person", 27.4), (8, "record", 1.0),
])
def test_borough_noise_matches_design_table(epsilon, unit, expected):
    assert StreamBudget.from_config(CFG, epsilon, unit).borough_sigma == pytest.approx(expected, abs=0.05)


def test_citywide_table_accounts_for_records_in_two_cells():
    assert sensitivity_for("record", 2, 4) == pytest.approx(2 ** 0.5)
    budget = StreamBudget.from_config(CFG, 2, "record")
    assert rho_for(budget.citywide_sigma, 2 ** 0.5) == pytest.approx(budget.citywide_rho)


def test_budget_shares_never_exceed_total():
    budget = StreamBudget.from_config(CFG, 2, "person")
    assert budget.borough_rho + budget.citywide_rho < budget.rho_total


def test_unknown_unit_is_rejected():
    with pytest.raises(ValueError):
        sensitivity_for("household", 1, 4)


def test_masks_cancel_exactly():
    masks = pairwise_masks(np.random.default_rng(1), 5, (20, 16))
    assert (aggregate(masks) == 0).all()


def test_aggregate_returns_the_exact_sum_including_negatives():
    rng = np.random.default_rng(2)
    values = [rng.normal(0, 50, (10, 4)) for _ in range(5)]
    masks = pairwise_masks(rng, 5, (10, 4))
    total = decode(aggregate([add_mask(encode(v, BITS), m) for v, m in zip(values, masks)]), BITS)
    assert np.abs(total - sum(values)).max() <= 5 / (1 << BITS)


def test_single_share_reveals_nothing_readable():
    rng = np.random.default_rng(3)
    counts = np.zeros(100_000)
    share = add_mask(encode(counts, BITS), pairwise_masks(rng, 5, counts.shape)[0]).astype(float)
    assert share.mean() == pytest.approx(2 ** 61, rel=0.01)
    other = add_mask(encode(counts + 1000, BITS), pairwise_masks(rng, 5, counts.shape)[0]).astype(float)
    assert other.mean() == pytest.approx(2 ** 61, rel=0.01)


def test_institution_path_has_configured_total_noise_on_every_cell():
    nodes = make_nodes(5)
    stream = MonitoringStream("s", nodes, 7.0, 0.05, np.random.default_rng(4), BITS)
    counts = [np.zeros((1000, 16)) for _ in nodes]
    counts[2][:, 3] = 11
    released = stream.release_window(counts)
    expected = np.zeros((1000, 16))
    expected[:, 3] = 11
    assert (released - expected).std() == pytest.approx(7.0, rel=0.02)
    assert np.allclose((released - expected).std(axis=0), 7.0, rtol=0.15), "zero-count cells must also receive noise"


def test_each_institution_adds_only_its_share_of_noise():
    node = make_nodes(1)[0]
    share = add_mask(encode(node.rng.normal(0, 7.0 / 5 ** 0.5, 50_000), BITS), np.zeros(50_000, dtype=np.uint64))
    assert decode(share, BITS).std() == pytest.approx(7.0 / 5 ** 0.5, rel=0.02)


def test_fast_path_matches_the_institution_path_distribution():
    nodes = make_nodes(5)
    stream = MonitoringStream("s", nodes, 5.0, 0.05, np.random.default_rng(5), BITS)
    truth = np.full((4000, 16), 9.0)
    slow = stream.release_window([truth / 5 for _ in nodes])
    fast = fast_noisy_table(truth, 5.0, np.random.default_rng(6))
    assert slow.mean() == pytest.approx(fast.mean(), abs=0.25)
    assert slow.std() == pytest.approx(fast.std(), rel=0.03)


def test_releases_use_fresh_noise():
    nodes = make_nodes(5)
    stream = MonitoringStream("s", nodes, 5.0, 0.05, np.random.default_rng(7), BITS)
    counts = [np.zeros((5, 5)) for _ in nodes]
    assert not np.array_equal(stream.release_window(counts), stream.release_window(counts))


def test_stream_charges_budget_once_however_many_windows():
    nodes = make_nodes(5)
    stream = MonitoringStream("s", nodes, 5.0, 0.2, np.random.default_rng(8), BITS)
    for _ in range(50):
        stream.release_window([np.zeros((1, 16)) for _ in nodes])
    assert all(node.ledger.spent == pytest.approx(0.2) for node in nodes)


def test_over_budget_stream_is_refused_and_releases_nothing():
    nodes = make_nodes(3, total=0.1)
    with pytest.raises(BudgetRefused):
        MonitoringStream("s", nodes, 5.0, 0.2, np.random.default_rng(9), BITS)
    assert all(node.ledger.spent == 0 for node in nodes)


def test_ledger_refuses_over_budget():
    ledger = BudgetLedger(1.0)
    assert ledger.reserve("a", 0.7)
    assert not ledger.reserve("b", 0.4)
    ledger.commit("a")
    assert ledger.spent == pytest.approx(0.7)
    assert ledger.remaining() == pytest.approx(0.3)


def test_ledger_release_returns_budget():
    ledger = BudgetLedger(1.0)
    ledger.reserve("a", 0.7)
    ledger.release("a")
    assert ledger.remaining() == pytest.approx(1.0)


def test_ledger_rejects_duplicates_and_negative_costs():
    ledger = BudgetLedger(1.0)
    assert ledger.reserve("a", 0.1)
    assert not ledger.reserve("a", 0.1)
    with pytest.raises(ValueError):
        ledger.reserve("b", -0.1)


def test_multi_institution_reservation_is_all_or_nothing():
    ledgers = [BudgetLedger(1.0), BudgetLedger(1.0), BudgetLedger(0.1)]
    assert not reserve_all(ledgers, "q", 0.5)
    assert all(ledger.remaining() == pytest.approx(ledger.total) for ledger in ledgers)
    assert reserve_all(ledgers[:2], "q", 0.5)


def test_person_cap_limits_cells_in_time_order():
    records = pd.DataFrame({
        "person": ["a"] * 6 + ["b"],
        "admitted": pd.to_datetime(["2024-01-06", "2024-01-01", "2024-01-01", "2024-01-03", "2024-01-04", "2024-01-05", "2024-01-01"]),
        "cell": [6, 1, 1, 3, 4, 5, 1],
    })
    kept = apply_person_cap(records, "person", ["cell"], 3)
    assert kept[kept["person"] == "a"]["cell"].tolist() == [1, 3, 4]
    assert kept[kept["person"] == "b"]["cell"].tolist() == [1]


def test_audit_stays_below_claimed_epsilon():
    epsilon, sigma = 4.0, sigma_for(rho_from_epsilon(4.0, 1e-6), 1.0)
    trials = 400_000
    nodes = make_nodes(5, seed=11)
    stream = MonitoringStream("audit", nodes, sigma, rho_from_epsilon(4.0, 1e-6), np.random.default_rng(12), BITS)
    base = [np.zeros((trials, 1)) for _ in nodes]
    added = [b.copy() for b in base]
    added[0][:, 0] = 1
    without, with_record = stream.release_window(base).ravel(), stream.release_window(added).ravel()
    audit = empirical_epsilon(without, with_record, 1e-6, np.linspace(0, 4 * sigma, 25))
    assert 0 < audit <= epsilon


def test_audit_catches_too_little_noise():
    claimed = 1.0
    sigma = sigma_for(rho_from_epsilon(claimed, 1e-6), 1.0) * 0.1
    rng = np.random.default_rng(13)
    without, with_record = rng.normal(0, sigma, 200_000), rng.normal(1, sigma, 200_000)
    assert empirical_epsilon(without, with_record, 1e-6, np.linspace(0, 4 * sigma, 25)) > claimed


def test_c2_flags_a_clear_jump_once():
    table = np.random.default_rng(14).normal(20, 2, (40, 2))
    table[30:33, 1] += 60
    scores = c2_scores(table, 20, 2)
    assert np.isnan(scores[:22]).all()
    onsets = alert_onsets(scores, 5.0)
    assert onsets[30, 1] and onsets[:, 1].sum() == 1
    assert onsets[:, 0].sum() == 0
