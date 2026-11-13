"""
Tests for the Financial Spreading Pipeline.

These tests verify structural correctness and key behaviors. They are NOT
exhaustive — you are expected to add your own tests. A good submission will
have at least as many candidate-written tests as provided tests.

Run:  pytest test_pipeline.py -v
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pipeline import (
    AccountMapping,
    MergedLineItem,
    ProvenanceRecord,
    SourceLineItem,
    ValidationResult,
    audit_trail,
    load_rules,
    load_source,
    load_taxonomy,
    map_to_accounts,
    reconcile,
    validate,
)

DATA_DIR = Path(__file__).parent / "data"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def source_a() -> list[SourceLineItem]:
    return load_source(DATA_DIR / "source_a_llm.json")


@pytest.fixture(scope="module")
def source_b() -> list[SourceLineItem]:
    return load_source(DATA_DIR / "source_b_ocr.json")


@pytest.fixture(scope="module")
def taxonomy() -> list[dict]:
    return load_taxonomy(DATA_DIR / "taxonomy.json")


@pytest.fixture(scope="module")
def rules() -> list[dict]:
    return load_rules(DATA_DIR / "validation_rules.json")


@pytest.fixture(scope="module")
def merged(source_a, source_b) -> list[MergedLineItem]:
    return reconcile(source_a, source_b)


@pytest.fixture(scope="module")
def mappings(merged, taxonomy) -> list[AccountMapping]:
    return map_to_accounts(merged, taxonomy)


@pytest.fixture(scope="module")
def validation_results(mappings, rules) -> list[ValidationResult]:
    return validate(mappings, rules)


@pytest.fixture(scope="module")
def provenance(mappings) -> list[ProvenanceRecord]:
    return audit_trail(mappings)


# ---------------------------------------------------------------------------
# 1. Reconciliation tests
# ---------------------------------------------------------------------------


class TestReconcile:
    """Tests for the reconcile() function."""

    def test_returns_list_of_merged_items(self, merged):
        assert isinstance(merged, list)
        assert all(isinstance(m, MergedLineItem) for m in merged)

    def test_not_just_concatenation(self, merged, source_a, source_b):
        """Merged count should be less than the sum of both sources
        (because matched items are combined, not duplicated)."""
        assert len(merged) < len(source_a) + len(source_b)

    def test_product_sales_conflict_detected(self, merged):
        """Source A has Product Sales = 12,450,000; Source B = 12,456,000.
        The conflict must be detected."""
        product = [m for m in merged if "product" in m.label.lower() and "sales" in m.label.lower()
                   or ("product" in m.label.lower() and "revenue" in m.label.lower())]
        assert len(product) >= 1, "Product Sales should appear in merged output"
        item = product[0]
        assert item.has_conflict is True
        assert len(item.sources) == 2

    def test_freight_conflict_detected(self, merged):
        """Source A has Freight = 388,000; Source B = 380,000."""
        freight = [m for m in merged if "freight" in m.label.lower()]
        assert len(freight) >= 1, "Freight & Shipping should appear in merged output"
        item = freight[0]
        assert item.has_conflict is True

    def test_items_from_single_source_are_included(self, merged):
        """Travel & Entertainment is only in Source A; Office Supplies only
        in Source B. Both should appear."""
        labels_lower = {m.label.lower() for m in merged}
        # At least one of these should match (label might vary)
        has_travel = any("travel" in l for l in labels_lower)
        has_office = any("office" in l for l in labels_lower)
        assert has_travel, "Travel & Entertainment (Source A only) should be in merged"
        assert has_office, "Office Supplies (Source B only) should be in merged"

    def test_every_merged_item_has_at_least_one_source(self, merged):
        for m in merged:
            assert len(m.sources) >= 1, f"'{m.label}' has no source references"

    def test_hallucination_candidates_are_flagged(self, merged):
        """Items like 'Warranty Reserve Adjustment' and 'Earnings Per Share'
        appear only in Source A at low confidence. They should be flagged."""
        suspicious_keywords = ["warranty", "earnings per share"]
        for kw in suspicious_keywords:
            matches = [m for m in merged if kw in m.label.lower()]
            if matches:
                assert matches[0].flagged is True, (
                    f"'{matches[0].label}' should be flagged as suspicious"
                )

    def test_employee_benefits_split_recognized(self, merged):
        """Source A splits Employee Benefits into Health Insurance (420K) and
        Retirement Benefits (225K). Source B has Employee Benefits (645K).
        The reconciliation should handle this — either keep the split or
        merge — but the total value should be ~645,000."""
        benefit_keywords = ["benefit", "health insurance", "retirement"]
        benefit_items = [
            m for m in merged
            if any(kw in m.label.lower() for kw in benefit_keywords)
        ]
        total = sum(m.value for m in benefit_items)
        assert abs(total - 645000) < 1000, (
            f"Employee benefits total should be ~$645,000, got ${total:,.0f}"
        )

    def test_pretax_income_conflict_detected(self, merged):
        """Source A: 638,000 vs Source B: 636,000 (OCR error)."""
        pretax = [
            m for m in merged
            if any(kw in m.label.lower() for kw in ["pre-tax", "pretax", "before tax"])
        ]
        assert len(pretax) >= 1
        assert pretax[0].has_conflict is True


# ---------------------------------------------------------------------------
# 2. Account mapping tests
# ---------------------------------------------------------------------------


class TestMapToAccounts:
    """Tests for the map_to_accounts() function."""

    def test_returns_list_of_account_mappings(self, mappings):
        assert isinstance(mappings, list)
        assert all(isinstance(m, AccountMapping) for m in mappings)

    def test_core_accounts_have_values(self, mappings):
        """Key accounts must be populated."""
        mapped_ids = {m.account_id for m in mappings if m.value != 0}
        required = {"REV_PRODUCT", "TOTAL_COGS", "GROSS_PROFIT", "NET_INCOME"}
        missing = required - mapped_ids
        assert not missing, f"These accounts should have non-zero values: {missing}"

    def test_interest_expense_mapped_correctly(self, mappings):
        """Interest Expense should map to INT_EXPENSE, not SGA."""
        ie = [m for m in mappings if m.account_id == "INT_EXPENSE"]
        assert len(ie) == 1
        assert abs(ie[0].value - 340000) < 100

    def test_net_income_mapped(self, mappings):
        ni = [m for m in mappings if m.account_id == "NET_INCOME"]
        assert len(ni) == 1
        assert abs(ni[0].value - 478500) < 100

    def test_rnd_mapped(self, mappings):
        rnd = [m for m in mappings if m.account_id == "RND"]
        assert len(rnd) == 1
        assert abs(rnd[0].value - 1280000) < 100

    def test_depreciation_amortization_mapped(self, mappings):
        """Whether from Source A's combined D&A or Source B's separate entries,
        DEPR_AMORT should end up at ~700,000."""
        da = [m for m in mappings if m.account_id == "DEPR_AMORT"]
        assert len(da) == 1
        assert abs(da[0].value - 700000) < 100

    def test_no_earnings_per_share_in_any_account(self, mappings):
        """EPS is not a financial line item for spreading — it should not
        appear in any account mapping."""
        for m in mappings:
            for item in m.mapped_items:
                assert "earnings per share" not in item.label.lower(), (
                    f"EPS should not be mapped to {m.account_id}"
                )

    def test_every_mapping_has_items(self, mappings):
        """Every AccountMapping with value > 0 should have mapped_items."""
        for m in mappings:
            if abs(m.value) > 0:
                assert len(m.mapped_items) > 0, (
                    f"{m.account_id} has value {m.value} but no mapped items"
                )


# ---------------------------------------------------------------------------
# 3. Validation tests
# ---------------------------------------------------------------------------


class TestValidate:
    """Tests for the validate() function."""

    def test_returns_results_for_all_rules(self, validation_results, rules):
        assert len(validation_results) == len(rules)
        result_ids = {r.rule_id for r in validation_results}
        rule_ids = {r["id"] for r in rules}
        assert result_ids == rule_ids

    def test_every_result_has_required_fields(self, validation_results):
        for r in validation_results:
            assert isinstance(r, ValidationResult)
            assert isinstance(r.passed, bool)
            assert isinstance(r.gap, (int, float))
            if r.passed:
                assert abs(r.gap) <= 100  # within tolerance

    def test_net_income_rule_check(self, validation_results):
        """R7: Pre-Tax Income - Tax = Net Income. This should pass if the
        candidate's reconciliation picked correct values."""
        r7 = [r for r in validation_results if r.rule_id == "R7"]
        assert len(r7) == 1
        # We don't assert r7[0].passed because it depends on the candidate's
        # reconciliation choices. But the gap should be computed.
        assert isinstance(r7[0].gap, (int, float))


# ---------------------------------------------------------------------------
# 4. Audit trail tests
# ---------------------------------------------------------------------------


class TestAuditTrail:
    """Tests for the audit_trail() function."""

    def test_returns_provenance_records(self, provenance):
        assert isinstance(provenance, list)
        assert all(isinstance(p, ProvenanceRecord) for p in provenance)

    def test_every_nonzero_account_has_provenance(self, mappings, provenance):
        mapped_with_values = {m.account_id for m in mappings if abs(m.value) > 0}
        provenance_ids = {p.account_id for p in provenance}
        missing = mapped_with_values - provenance_ids
        assert not missing, f"Accounts with values but no provenance: {missing}"

    def test_provenance_chain_has_source_refs(self, provenance):
        """Every chain entry must trace back to at least one source."""
        for p in provenance:
            assert len(p.chain) > 0, f"{p.account_id} has empty chain"
            for entry in p.chain:
                assert "source_values" in entry, (
                    f"Chain entry in {p.account_id} missing 'source_values'"
                )
                assert len(entry["source_values"]) > 0, (
                    f"Chain entry '{entry.get('merged_label', '?')}' in "
                    f"{p.account_id} has no source references"
                )

    def test_provenance_has_page_and_bbox(self, provenance):
        """Source references must include page and bounding_box for
        document traceability."""
        for p in provenance:
            for entry in p.chain:
                for sv in entry["source_values"]:
                    assert "page" in sv, f"Missing 'page' in source ref for {p.account_id}"
                    assert "bounding_box" in sv, (
                        f"Missing 'bounding_box' in source ref for {p.account_id}"
                    )

    def test_provenance_value_matches_mapping(self, mappings, provenance):
        """The final_value in provenance should match the mapping value."""
        mapping_values = {m.account_id: m.value for m in mappings}
        for p in provenance:
            if p.account_id in mapping_values:
                assert abs(p.final_value - mapping_values[p.account_id]) < 0.01, (
                    f"{p.account_id}: provenance value {p.final_value} != "
                    f"mapping value {mapping_values[p.account_id]}"
                )

    def test_conflict_noted_in_provenance(self, provenance):
        """At least one provenance record should note a conflict in its
        merge_decision (since the data has known conflicts)."""
        has_conflict_note = False
        for p in provenance:
            for entry in p.chain:
                decision = entry.get("merge_decision", "")
                if "conflict" in decision.lower() or "disagree" in decision.lower():
                    has_conflict_note = True
                    break
            if has_conflict_note:
                break
        assert has_conflict_note, (
            "No provenance entries mention conflicts in merge_decision, "
            "but the data has known value disagreements"
        )


# ---------------------------------------------------------------------------
# ADD YOUR OWN TESTS BELOW THIS LINE
# ---------------------------------------------------------------------------

# class TestMyReconcileEdgeCases:
#     ...
