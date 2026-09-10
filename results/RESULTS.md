# Evaluation Results

Golden set size: 200

| System | Intent Acc | Intent Macro-F1 | Action Acc | Escalate P/R/F1 | Reply Score |
|---|---:|---:|---:|---:|---:|
| trivial | 0.090 | 0.015 | 0.730 | 0.00/0.00/0.00 | 0.978 |
| simple_retrieval | 0.445 | 0.457 | 0.775 | 0.63/0.41/0.49 | 0.707 |
| agent | 0.825 | 0.825 | 0.820 | 0.68/0.63/0.65 | 0.858 |

## Judge calibration

- Primary task: human_vs_auto_judge_on_agent_drafts
- Human↔auto agreement on agent drafts: 0.389 (n=36)
- Note: 36 agent drafts were scored by an independent human rubric focused on issue-specific helpfulness (not the keyword checklist used by the auto judge). Disagreement usually means the auto judge over-rates generic DM/diagnostic templates.
- Secondary reproducibility on references: 1.000
