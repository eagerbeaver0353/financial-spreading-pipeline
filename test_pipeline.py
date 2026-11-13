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


class TestReconcileEdgeCases:
    """Candidate-added reconciliation edge cases."""

    def test_license_label_mismatch_matched(self, merged):
        """License Revenue (A) and Licensing Fees (B) are the same concept."""
        matches = [
            m for m in merged
            if "license" in m.label.lower() or "licensing" in m.label.lower()
        ]
        assert len(matches) == 1
        item = matches[0]
        assert len(item.sources) == 2
        assert abs(item.value - 890000) < 1
        assert item.conflict_type in ("label_only", None) or item.has_conflict is True

    def test_product_sales_resolution_retains_both_sources(self, merged):
        """Conflict resolution must keep both source values for auditability."""
        product = [
            m for m in merged
            if "product" in m.label.lower() and ("sales" in m.label.lower() or "revenue" in m.label.lower())
        ]
        assert len(product) >= 1
        item = product[0]
        source_values = sorted(s.value for s in item.sources)
        assert source_values == [12450000.0, 12456000.0]
        assert item.resolution_note
        assert str(int(item.value))[:5] in item.resolution_note.replace(",", "") or (
            f"{item.value:,.0f}" in item.resolution_note or f"{item.value:,.2f}" in item.resolution_note
        )

    def test_da_not_double_counted(self, merged):
        """Combined LLM D&A and OCR Depreciation+Amortization → one ~700k concept."""
        da_items = [
            m for m in merged
            if any(k in m.label.lower() for k in ("depreciation", "amortization", "d&a"))
        ]
        total = sum(m.value for m in da_items)
        assert abs(total - 700000) < 1000
        assert len(da_items) == 1

    def test_packaging_flagged_as_double_count_risk(self, merged):
        packaging = [m for m in merged if "packaging" in m.label.lower()]
        assert len(packaging) == 1
        assert packaging[0].flagged is True
        assert packaging[0].conflict_type == "single_source"

    def test_net_other_income_not_paired_with_net_income(self, merged):
        """a_34 (Net Other Income) must not steal b_29 from true Net Income."""
        for item in merged:
            ids = {s.id for s in item.sources}
            assert not ({"a_34", "b_29"} <= ids), (
                "Net Other Income (Expense) must not pair with Net Income"
            )

    def test_net_income_sources_are_a37_and_b29(self, merged):
        """True Net Income lines a_37 and b_29 should be reconciled together."""
        net_income = [
            m for m in merged
            if m.label.lower() == "net income"
            or {s.id for s in m.sources} == {"a_37", "b_29"}
        ]
        assert len(net_income) == 1
        ids = {s.id for s in net_income[0].sources}
        assert ids == {"a_37", "b_29"}
        assert abs(net_income[0].value - 478500) < 1


class TestMappingEdgeCases:
    """Candidate-added mapping edge cases."""

    def test_payroll_production_maps_to_cogs_labor(self, mappings):
        labor = [m for m in mappings if m.account_id == "COGS_LABOR"]
        assert len(labor) == 1
        assert abs(labor[0].value - 2890000) < 100
        labels = " ".join(i.label.lower() for i in labor[0].mapped_items)
        assert "labor" in labels or "payroll" in labels

    def test_sga_excludes_interest_expense(self, mappings):
        sga = [m for m in mappings if m.account_id == "SGA"]
        assert len(sga) == 1
        for item in sga[0].mapped_items:
            assert "interest" not in item.label.lower()

    def test_totals_not_inflated_by_leaf_children(self, mappings):
        """TOTAL_COGS should reflect the extracted total (~8.84M), not sum of leaves."""
        cogs = [m for m in mappings if m.account_id == "TOTAL_COGS"]
        assert len(cogs) == 1
        assert abs(cogs[0].value - 8840000) < 100


class TestValidateEdgeCases:
    def test_failed_rule_reports_nonzero_gap(self, validation_results):
        """When a rule fails, gap must be actual - expected (not silently zeroed)."""
        failures = [r for r in validation_results if not r.passed]
        for r in failures:
            assert abs(r.gap) > 100
            assert abs((r.actual - r.expected) - r.gap) < 0.01

    def test_gap_formula_consistency(self, validation_results):
        for r in validation_results:
            if r.passed:
                assert abs(r.actual - r.expected) <= 100
            else:
                assert abs(r.gap - (r.actual - r.expected)) < 0.01


class TestAuditEdgeCases:
    def test_product_sales_provenance_lists_both_source_ids(self, provenance):
        rev = [p for p in provenance if p.account_id == "REV_PRODUCT"]
        assert len(rev) == 1
        ids = set()
        for entry in rev[0].chain:
            for sv in entry["source_values"]:
                ids.add(sv["id"])
        assert "a_01" in ids and "b_01" in ids

    def test_employee_benefits_provenance_lists_three_sources_once(self, merged, provenance):
        """Benefits must be one $645K merged item; OCR total appears once in provenance."""
        benefit_items = [
            m for m in merged
            if any(kw in m.label.lower() for kw in ("benefit", "health insurance", "retirement"))
        ]
        assert len(benefit_items) == 1
        item = benefit_items[0]
        assert abs(item.value - 645000) < 1
        assert len(item.sources) == 3
        source_ids = [s.id for s in item.sources]
        assert source_ids.count("a_15") == 1  # Health Insurance (LLM)
        assert source_ids.count("a_16") == 1  # Retirement Benefits (LLM)
        assert source_ids.count("b_12") == 1  # Employee Benefits (OCR)

        sga = [p for p in provenance if p.account_id == "SGA"]
        assert len(sga) == 1
        benefit_entries = [
            e for e in sga[0].chain
            if "benefit" in e.get("merged_label", "").lower()
        ]
        assert len(benefit_entries) == 1
        sv_ids = [sv["id"] for sv in benefit_entries[0]["source_values"]]
        assert sorted(sv_ids) == ["a_15", "a_16", "b_12"]
        assert sv_ids.count("b_12") == 1

    def test_fx_loss_contribution_traceable_in_other_net(self, mappings, provenance):
        """FX loss stays positive at source/merge but contributes negatively."""
        other = [m for m in mappings if m.account_id == "OTHER_NET"]
        assert len(other) == 1
        assert abs(other[0].value - 120000) < 1

        prov = [p for p in provenance if p.account_id == "OTHER_NET"]
        assert len(prov) == 1
        assert abs(prov[0].final_value - other[0].value) < 0.01

        fx = [
            e for e in prov[0].chain
            if "foreign exchange" in e["merged_label"].lower()
        ]
        assert len(fx) == 1
        entry = fx[0]
        assert abs(entry["merged_value"] - 15000) < 0.01
        assert any(abs(sv["value"] - 15000) < 0.01 for sv in entry["source_values"])
        assert abs(entry["account_contribution"] - (-15000)) < 0.01
        assert "negat" in entry["mapping_decision"].lower()

        contrib_sum = sum(e["account_contribution"] for e in prov[0].chain)
        assert abs(contrib_sum - prov[0].final_value) < 0.01


# ---------------------------------------------------------------------------
# Additional candidate tests (behavior-locked; do not change implementation)
# ---------------------------------------------------------------------------

_BBOX = {"x": 0.0, "y": 0.0, "width": 1.0, "height": 1.0}


def _src(
    item_id: str,
    label: str,
    value: float,
    *,
    confidence: float = 0.9,
    source: str = "llm",
    section_hint: str = "",
    page: int = 1,
) -> SourceLineItem:
    return SourceLineItem(
        id=item_id,
        label=label,
        value=value,
        confidence=confidence,
        page=page,
        bounding_box=dict(_BBOX),
        source=source,
        section_hint=section_hint,
    )


class TestTaxLabelReconciliation:
    """Regression: Provision for Income Taxes ↔ Income Tax Expense as one TAX concept."""

    def test_tax_lines_merge_a36_and_b28_once(self, merged):
        tax_items = [
            m for m in merged
            if {s.id for s in m.sources} & {"a_36", "b_28"}
            or any(
                k in m.label.lower()
                for k in ("income tax", "provision for income", "provision income")
            )
        ]
        assert len(tax_items) == 1
        item = tax_items[0]
        source_ids = [s.id for s in item.sources]
        assert sorted(source_ids) == ["a_36", "b_28"]
        assert source_ids.count("a_36") == 1
        assert source_ids.count("b_28") == 1
        assert abs(item.value - 159_500.0) < 0.01

    def test_tax_account_is_single_159500(self, mappings):
        tax = [m for m in mappings if m.account_id == "TAX"]
        assert len(tax) == 1
        assert abs(tax[0].value - 159_500.0) < 0.01
        assert len(tax[0].mapped_items) == 1
        ids = [s.id for s in tax[0].mapped_items[0].sources]
        assert sorted(ids) == ["a_36", "b_28"]

    def test_r7_passes_with_zero_gap(self, validation_results):
        r7 = [r for r in validation_results if r.rule_id == "R7"]
        assert len(r7) == 1
        assert r7[0].passed is True
        assert r7[0].gap == 0.0


class TestLowConfidenceAndFlags:
    """Single-source flagging: low confidence and known-risk keys."""

    def test_low_confidence_single_source_is_flagged(self, merged):
        """Confidence < 0.60 on a single-source item triggers flagged=True."""
        bad_debt = [m for m in merged if "bad debt" in m.label.lower()]
        assert len(bad_debt) == 1
        item = bad_debt[0]
        assert item.conflict_type == "single_source"
        assert item.confidence < 0.60
        assert item.flagged is True

    def test_known_risk_key_flagged_even_when_confidence_above_threshold(self, merged):
        """EPS is flagged via known-risk key even though confidence is 0.62 (>= 0.60)."""
        eps = [m for m in merged if "earnings per share" in m.label.lower()]
        assert len(eps) == 1
        assert eps[0].conflict_type == "single_source"
        assert eps[0].confidence >= 0.60
        assert eps[0].flagged is True

    def test_high_confidence_ordinary_single_source_not_flagged(self, merged):
        """Ordinary OCR-only Office Supplies stays unflagged at high confidence."""
        office = [m for m in merged if "office" in m.label.lower()]
        assert len(office) == 1
        assert office[0].conflict_type == "single_source"
        assert office[0].confidence >= 0.60
        assert office[0].flagged is False


class TestDefaultConflictResolution:
    """Fallback when no statement-identity rule applies: higher confidence wins."""

    def test_default_conflict_prefers_higher_confidence(self):
        """Unspecialized concept: OCR 0.90 beats LLM 0.70; both sources retained."""
        merged = reconcile(
            [_src("a1", "Insurance", 100_000.0, confidence=0.70, source="llm")],
            [_src("b1", "Insurance", 105_000.0, confidence=0.90, source="ocr")],
        )
        assert len(merged) == 1
        item = merged[0]
        assert item.has_conflict is True
        assert item.conflict_type == "value_mismatch"
        assert abs(item.value - 105_000.0) < 0.01
        assert len(item.sources) == 2
        assert "higher confidence" in item.resolution_note.lower()

    def test_default_conflict_ocr_wins_on_confidence_tie(self):
        """When confidences are equal, current policy selects OCR (>= comparison)."""
        merged = reconcile(
            [_src("a1", "Insurance", 100_000.0, confidence=0.85, source="llm")],
            [_src("b1", "Insurance", 110_000.0, confidence=0.85, source="ocr")],
        )
        assert len(merged) == 1
        assert abs(merged[0].value - 110_000.0) < 0.01
        assert "Selected OCR" in merged[0].resolution_note


class TestMatchingBoundaries:
    """Pairing requires label similarity >= 0.45 or an alias; value alone is insufficient."""

    def test_value_proximity_alone_does_not_match_dissimilar_labels(self):
        """Jaccard 0.4 (< 0.45) with identical values → two single-source items."""
        merged = reconcile(
            [_src("a1", "Alpha Beta Gamma", 1_000.0, source="llm")],
            [_src("b1", "Alpha Beta Delta Epsilon", 1_000.0, source="ocr")],
        )
        assert len(merged) == 2
        assert all(m.conflict_type == "single_source" for m in merged)
        assert all(len(m.sources) == 1 for m in merged)

    def test_token_jaccard_at_half_matches_across_sources(self):
        """Jaccard 0.5 (>= 0.45) with same value → one label_only matched item."""
        merged = reconcile(
            [_src("a1", "Alpha Beta Gamma", 1_000.0, source="llm")],
            [_src("b1", "Alpha Beta Delta", 1_000.0, source="ocr")],
        )
        assert len(merged) == 1
        item = merged[0]
        assert len(item.sources) == 2
        assert item.conflict_type == "label_only"
        assert item.has_conflict is True
        assert abs(item.value - 1_000.0) < 0.01


class TestUnmappedAndAmbiguousAccounts:
    """Explicit skip paths and section-disambiguated payroll."""

    def test_returns_and_net_other_are_unmapped(self, merged, mappings):
        """Returns & Allowances and Net Other Income (Expense) are intentionally skipped."""
        unmapped_labels = {
            m.label
            for m in merged
            if not any(m in acct.mapped_items for acct in mappings)
        }
        assert any("returns" in l.lower() for l in unmapped_labels)
        assert any("net other income" in l.lower() for l in unmapped_labels)
        for acct in mappings:
            for item in acct.mapped_items:
                assert "earnings per share" not in item.label.lower()
                assert "returns & allowances" not in item.label.lower()
                assert item.label.lower() != "net other income (expense)"

    def test_ambiguous_payroll_in_operating_section_maps_to_sga(self):
        """Payroll with Operating Expenses section hint → SGA, not COGS_LABOR."""
        taxonomy = [
            {
                "id": "COGS_LABOR",
                "name": "Direct Labor",
                "type": "leaf",
                "category": "Cost of Goods Sold",
                "keywords": ["labor", "payroll", "wages"],
            },
            {
                "id": "SGA",
                "name": "SG&A",
                "type": "leaf",
                "category": "Operating Expenses",
                "keywords": ["salaries", "wages", "sga"],
            },
        ]
        item = MergedLineItem(
            label="Payroll - Admin",
            value=50_000.0,
            confidence=0.9,
            sources=[
                _src(
                    "a1",
                    "Payroll - Admin",
                    50_000.0,
                    section_hint="Operating Expenses",
                )
            ],
            conflict_type="single_source",
        )
        result = map_to_accounts([item], taxonomy)
        assert len(result) == 1
        assert result[0].account_id == "SGA"
        assert abs(result[0].value - 50_000.0) < 0.01


class TestTotalAndAccumulation:
    """Total-line authority and multi-item leaf accumulation."""

    def test_total_account_keeps_first_line_skips_duplicate_total(self):
        """Second extracted total for the same total account is skipped."""
        taxonomy = [
            {
                "id": "TOTAL_COGS",
                "name": "Total COGS",
                "type": "total",
                "category": "Cost of Goods Sold",
                "keywords": ["total cost of goods sold"],
            }
        ]
        first = MergedLineItem(
            label="Total Cost of Goods Sold",
            value=8_840_000.0,
            confidence=0.99,
            sources=[_src("a1", "Total Cost of Goods Sold", 8_840_000.0)],
        )
        second = MergedLineItem(
            label="Total Cost of Goods Sold",
            value=9_000_000.0,
            confidence=0.50,
            sources=[_src("b1", "Total Cost of Goods Sold", 9_000_000.0, source="ocr")],
        )
        result = map_to_accounts([first, second], taxonomy)
        assert len(result) == 1
        assert abs(result[0].value - 8_840_000.0) < 0.01
        assert len(result[0].mapped_items) == 1

    def test_cogs_other_accumulates_freight_and_packaging(self, mappings):
        """COGS_OTHER sums Freight + Packaging (multiple leaf contributors)."""
        other = [m for m in mappings if m.account_id == "COGS_OTHER"]
        assert len(other) == 1
        assert len(other[0].mapped_items) == 2
        labels = {i.label.lower() for i in other[0].mapped_items}
        assert any("freight" in l for l in labels)
        assert any("packaging" in l for l in labels)
        assert abs(other[0].value - 590_000.0) < 1

    def test_other_net_accumulates_multiple_non_operating_items(self, mappings):
        """OTHER_NET holds Other Income, Warranty, FX loss, and Gain on Asset Sale."""
        other = [m for m in mappings if m.account_id == "OTHER_NET"]
        assert len(other) == 1
        assert len(other[0].mapped_items) >= 4
        assert abs(other[0].value - 120_000.0) < 1


class TestValidateToleranceAndMissing:
    """Tolerance boundary (inclusive) and missing-account-as-zero policy."""

    def test_validate_passes_exactly_at_tolerance_with_zero_stored_gap(self):
        """|actual - expected| == tolerance → passed; stored gap is 0.0."""
        rules = [
            {
                "id": "T_AT",
                "name": "at tolerance",
                "add": ["A"],
                "subtract": [],
                "equals": "B",
                "tolerance": 100.0,
            }
        ]
        mappings = [
            AccountMapping("A", "A", 200.0, []),
            AccountMapping("B", "B", 100.0, []),
        ]
        results = validate(mappings, rules)
        assert len(results) == 1
        assert results[0].passed is True
        assert results[0].gap == 0.0
        assert abs(results[0].actual - results[0].expected) == 100.0

    def test_validate_fails_just_beyond_tolerance(self):
        """|actual - expected| == tolerance + 1 → failed; gap = actual - expected."""
        rules = [
            {
                "id": "T_OVER",
                "name": "over tolerance",
                "add": ["A"],
                "subtract": [],
                "equals": "B",
                "tolerance": 100.0,
            }
        ]
        mappings = [
            AccountMapping("A", "A", 201.0, []),
            AccountMapping("B", "B", 100.0, []),
        ]
        results = validate(mappings, rules)
        assert len(results) == 1
        assert results[0].passed is False
        assert abs(results[0].gap - 101.0) < 0.01
        assert abs(results[0].gap - (results[0].actual - results[0].expected)) < 0.01

    def test_validate_missing_account_treated_as_zero(self):
        """Absent account IDs contribute 0.0 to add/subtract/equals."""
        rules = [
            {
                "id": "T_MISS",
                "name": "missing equals",
                "add": ["A"],
                "subtract": [],
                "equals": "B",
                "tolerance": 100.0,
            }
        ]
        # B is missing → expected 0; |50 - 0| <= 100 → pass
        results = validate([AccountMapping("A", "A", 50.0, [])], rules)
        assert results[0].expected == 0.0
        assert results[0].actual == 50.0
        assert results[0].passed is True
        assert results[0].gap == 0.0


class TestAuditNonFxProvenance:
    """Provenance for label-only / ordinary mappings (no FX sign flip)."""

    def test_label_only_provenance_contribution_equals_merged_value(self, provenance):
        """License/Licensing label_only: contribution == merged value; both source IDs."""
        rev_other = [p for p in provenance if p.account_id == "REV_OTHER"]
        assert len(rev_other) == 1
        entries = [
            e for e in rev_other[0].chain if e.get("conflict_type") == "label_only"
        ]
        assert len(entries) == 1
        entry = entries[0]
        assert abs(entry["account_contribution"] - entry["merged_value"]) < 0.01
        assert abs(entry["merged_value"] - 890_000.0) < 0.01
        assert "label mismatch" in entry["merge_decision"].lower()
        ids = {sv["id"] for sv in entry["source_values"]}
        assert "a_03" in ids and "b_03" in ids
        # Non-FX path: mapping_decision must not claim negation
        assert "negat" not in entry["mapping_decision"].lower()

    def test_single_source_unflagged_item_still_in_provenance(self, provenance):
        """High-confidence OCR-only Office Supplies remains traceable in SGA chain."""
        sga = [p for p in provenance if p.account_id == "SGA"]
        assert len(sga) == 1
        office = [
            e for e in sga[0].chain if "office" in e["merged_label"].lower()
        ]
        assert len(office) == 1
        entry = office[0]
        assert entry.get("flagged") is False
        assert entry.get("conflict_type") == "single_source"
        assert len(entry["source_values"]) == 1
        assert entry["source_values"][0]["id"] == "b_18"
        assert "page" in entry["source_values"][0]
        assert "bounding_box" in entry["source_values"][0]
