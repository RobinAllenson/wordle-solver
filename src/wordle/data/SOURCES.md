# Word-list provenance

`answers.txt` is Alex Selby's 3,158-word July 2023 candidate snapshot:

- Source: https://github.com/alex1770/wordle/blob/main/wordlist_nyt20230701_hidden
- Source Git blob: `7ad97bc5aa2289cbc799985b771e2aaa9b2dea24`
- Retrieved: 2026-09-19
- Background: https://sonorouschocolate.com/notes/index.php/The_best_strategies_for_Wordle,_part_3_(July_2023)

This is a reproducible baseline, **not** an authoritative current NYT answer
schedule or a guarantee of completeness. Keep the snapshot date distinct from
the retrieval date. Review future updates against known-answer regressions and
record their provenance here. Do not remove previous answers merely because
they have already appeared.

`guesses.txt` remains the separately vendored tabatkins/wordle-list accepted-guess
list (14,855 words). It is a broader diagnostic pool, not a claim that every
entry is equally plausible as a daily answer.

For the 2026-09-19 regression, this snapshot leaves 141 candidates after
SLATE:bbyby (WordleBot reported 140), 10 after BROND:bbbyb, and MAVEN/WAKEN/WAXEN
after AHEAP:ybybb. The first count difference is intentional evidence that the
snapshot is not identical to today's WordleBot dictionary.
