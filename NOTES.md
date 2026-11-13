# NOTES

## 1. Conflict Resolution Policy

I match labels with normalized aliases and token similarity. Close values can strengthen a possible match, but value proximity alone does not make two different labels the same concept. I handle known split and combined lines first: the two LLM benefits lines become one $645,000 Employee Benefits item, and the combined LLM D&A line becomes one $700,000 item with the two OCR components.

I don’t average conflicting amounts. An average would create a value neither source reported and would be harder to trace back to the document. When a statement identity clearly supports one value, I select that value and keep both source readings in the resolution note and audit trail. For other conflicts, I use the higher confidence score as a fallback.

| Concept | Selected value | Reason |
|---|---:|---|
| Product Sales | LLM: $12,450,000 | Product Sales plus Service Revenue and License Revenue equals the shared $16,620,000 Total Revenue. The OCR value is $6,000 higher. |
| Freight | OCR: $380,000 | Raw Materials, Direct Labor, Manufacturing Overhead, and OCR Freight equal the shared $8,840,000 Total COGS. |
| Pre-Tax Income | LLM: $638,000 | The test data identifies the OCR value as a digit error. The LLM value also matches the listed operating and non-operating components. |
| Other value conflicts | Higher confidence | I retain both readings and record the choice. |

Single-source items remain in the merged output. I flag low-confidence items and known risks such as Earnings Per Share, Warranty Reserve Adjustment, and Packaging Costs. EPS is not mapped to an account. A single-source item’s provenance remains available even when it is not flagged.

## 2. A Scenario Where Your Reconciliation Produces a Wrong Answer

Packaging Costs is an LLM-only $210,000 line. The pipeline flags it as a possible double-count, but still maps it to Other Direct Costs. If the extracted $8,840,000 Total COGS is authoritative and already includes packaging, adding the line makes the COGS components total $9,050,000. Rule R2 reports a $210,000 gap. The gap and the flagged source record give a reviewer a reason to check the document.

An alias guard now blocks known distinct concepts such as “Net Other Income (Expense)” and “Net Income” from pairing through shared generic tokens, and regression tests cover that case. A remaining risk is incomplete alias coverage: an unfamiliar label can still match the wrong counterpart through token similarity alone, leave the true peer unmatched, and send the wrong source IDs into mapping. In production I would flag low-evidence fuzzy matches for review and extend the alias table when new near-misses appear.

On the current fixture, R2 fails with a $210,000 gap, and R4 fails with a $587,000 gap because mapped detail exceeds the extracted totals. R6 fails with a $92,000 gap, which is greater than its $100 tolerance. R7 passes. The pipeline reports these gaps instead of changing extracted values to force the statement to balance.

## 3. Mapping Step — Time & Space Complexity

Let **M** be the number of merged items, **A** the number of taxonomy accounts, and **K** the keyword work per account.

The common explicit-mapping path scans the taxonomy to find the selected account, so it takes **O(M · A)**. The keyword fallback scores accounts and takes **O(M · A · K)** in the worst case. The mapping result stores account records and references to the merged items, so space is **O(A + M)**.

At 10,000 items and 500 accounts, the taxonomy scans and keyword checks are still manageable for an in-memory batch, though the fallback does more work. Indexing accounts by ID and keywords by account would reduce lookup work. The current bucket accumulation also recalculates a bucket’s sum after each addition. If many items land in one account, that can reach **O(M²)**. A running total would avoid repeated summation.

## 4. Adapting to Poorly Calibrated Confidence Scores

If Source A’s confidence scores were unreliable, I would stop using them as a deciding vote. I’d give more weight to agreement between sources, known label relationships, statement identities, and evidence from the source image, such as the bounding box and page.

I’d also track how often each source is right for different line types and compare its confidence scores with those observed results. Until the scores were recalibrated, I’d flag more single-source items and unresolved conflicts for review rather than treating a high score as proof.