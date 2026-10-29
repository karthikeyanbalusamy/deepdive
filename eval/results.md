# DeepDive self-evaluation

Models (via Mesh API): writer/researcher `openai/gpt-4o-mini`, verifier `google/gemini-2.5-flash`, judge `anthropic/claude-haiku-4.5`.  
Run on 2026-10-29: 12 of 12 questions completed.

| metric | average |
|---|---|
| verification pass rate | 0.82 |
| grounded sentences (independent judge) | 0.97 |
| sub-question coverage | 1.00 |
| extra research rounds | 0.25 |
| known-answer fact recall | 1.00 |
| seconds per question | 118.83 |

| id | category | claims | pass_rate | grounded | coverage | retries | fact_recall | seconds |
|---|---|---|---|---|---|---|---|---|
| ev-risks | analytical | 24 | 0.92 | 0.96 | 1.0 | 0 | - | 117 |
| battery-share | factual | 30 | 0.9 | 1.0 | 1.0 | 1 | - | 142 |
| sodium-ion | analytical | 30 | 0.83 | 0.96 | 1.0 | 1 | - | 143 |
| pli-acc | factual | 24 | 0.83 | 1.0 | 1.0 | 0 | - | 118 |
| lithium-price | factual | 24 | 0.75 | 1.0 | 1.0 | 0 | - | 120 |
| solid-state | contradictory | 24 | 0.88 | 1.0 | 1.0 | 0 | - | 105 |
| eu-ai-act | factual | 24 | 0.67 | 0.94 | 1.0 | 0 | - | 105 |
| glp1 | analytical | 24 | 1.0 | 0.95 | 1.0 | 0 | - | 102 |
| remote-work | contradictory | 24 | 0.83 | 0.95 | 1.0 | 0 | - | 97 |
| small-modular | analytical | 24 | 0.83 | 1.0 | 1.0 | 0 | - | 116 |
| niche-thin | thin-evidence | 24 | 0.83 | 0.93 | 1.0 | 0 | - | 103 |
| known-fact | known-answer | 29 | 0.59 | 1.0 | 1.0 | 1 | 1.0 | 158 |
