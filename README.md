# Financial Spreading Pipeline — Conflict Resolution & Audit Trail

## Candidate Instructions

Read this entire document before coding. Return your implementation, tests,
and a NOTES file. We care most about **design decisions, conflict handling,
and traceability**. There will be a follow-up conversation during the
on-site interview where you defend your choices.

If you make an assumption, write it down and keep going.

---

## Overview

You are building part of a financial spreading system. Two independent
extraction sources — an LLM-based extractor and a rule-based OCR pipeline —
have processed the same income statement document. They produced overlapping,
sometimes contradictory results.

Your job: **reconcile, map, validate, and trace** the extracted data.

**Company:** Apex Manufacturing Inc., FY 2024 Income Statement
**Document:** 3-page PDF (you don't need the PDF; the extractions are provided)

---

## Data Files (in `data/`)

| File | Description |
|---|---|
| `source_a_llm.json` | ~38 line items from an LLM extractor (GPT-4o). Higher recall, occasionally hallucinates, sometimes splits or combines items, variable confidence. |
| `source_b_ocr.json` | ~29 line items from a template-based OCR pipeline. Higher precision, misses some items, occasional digit-level OCR errors, high confidence scores. |
| `taxonomy.json` | 22 target accounts (15 leaf + 7 totals) that define where each line item should land. |
| `validation_rules.json` | 7 mathematical constraints the final spreading must satisfy. |

### Known data characteristics you should expect

- **Value disagreements**: The two sources sometimes extract different dollar
  amounts for the same line item.
- **Label mismatches**: Source A may call something "License Revenue" while
  Source B calls it "Licensing Fees". Same item, different labels.
- **Split vs. combined**: Source A may break one line into sub-items (e.g.,
  Employee Benefits → Health Insurance + Retirement Benefits), while Source B
  keeps it as one line.
- **Items in only one source**: Some items appear in only one source — could
  be a genuine find or a hallucination.
- **Double-count risks**: Source A may extract a sub-total AND its components,
  risking double-counting if both are mapped.
- **Internal inconsistencies**: A source may have individual items that don't
  sum to its own extracted total.

---

## Deliverables

### 1. `pipeline.py` — Your implementation (single file)

Implement these four functions. The skeleton with type definitions is provided.

#### `reconcile(source_a, source_b) → list[MergedLineItem]`
Merge the two extraction sources into a single set of line items.
- Match items across sources (by label similarity, value proximity, or both)
- When both sources have the same item but disagree on the value, choose a
  resolution and flag the conflict
- Include items found in only one source (with appropriate flagging)
- Document your conflict resolution strategy in the NOTES file

#### `map_to_accounts(merged, taxonomy) → list[AccountMapping]`
Map each merged line item to the closest target account in the taxonomy.
- Handle ambiguous mappings (e.g., "Payroll — Production" could be COGS or SGA)
- Avoid double-counting (e.g., if both a total and its components are present)
- You may use fuzzy matching, keyword rules, or any approach — but you must
  be able to explain your method

#### `validate(mappings, rules) → list[ValidationResult]`
Check the mathematical constraints defined in `validation_rules.json`.
- Report which rules pass and which fail
- For failures, compute the gap (actual − expected)
- Tolerance is specified per rule (default $100)

#### `audit_trail(mappings) → list[ProvenanceRecord]`
For every account with a mapped value, produce a provenance chain:
```
final_value → merge_decision → source_value(s) → page / bounding_box
```
This is what an auditor would follow to trace a number back to the document.

### 2. `test_pipeline.py` — Your tests

The provided test file has tests your code must pass. You are expected to
**add your own tests** covering additional edge cases and scenarios. The
provided tests are not exhaustive.

### 3. `NOTES.md` (use the template)

Must include:
- Your conflict resolution policy and why you chose it
- One scenario where your reconciliation produces a wrong answer and how
  you would detect it
- The time/space complexity of your mapping step
- How you would adapt if Source A's confidence scores were poorly calibrated

---

## Constraints

- **Standard library + pydantic** (v2). No other external packages.
- No LLM API calls at runtime. No database. All in-memory.
- No global mutable state.
- Prefer correctness over cleverness.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest test_pipeline.py -v
```

## Evaluation Criteria

| Criterion | Weight | What we look for |
|---|---|---|
| Conflict resolution design | High | Thoughtful policy with explicit tradeoffs, not just "pick higher confidence" |
| Correctness | High | Passes provided tests; handles edge cases; no double-counting |
| Audit trail quality | High | Every number traceable to source evidence; conflicts noted in provenance |
| Mapping accuracy | Medium | Ambiguous items mapped with justifiable reasoning |
| Code quality | Medium | Readable, well-structured, idiomatic Python with type hints |
| NOTES depth | Medium | Shows genuine understanding, not surface-level descriptions |
| Test quality | Medium | Tests that would catch real bugs, not just happy-path |

## Time Expectation

**4–6 hours** of focused work. This is not meant to take a full weekend.

