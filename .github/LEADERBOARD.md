# Leaderboard maintenance

How a submission flows from issue to leaderboard, and what the maintainer must do.

## Flow

1. A submitter runs `python -m relbench.submit <pred_dir>`, which validates their
   prediction tables locally and writes one zip per validated leaderboard family.
2. They open an issue from the **submit** form
   (`.github/ISSUE_TEMPLATE/submit.yml`), which fills in method metadata and
   attaches the zip(s). The form applies the `submit` label.
3. The **Validate leaderboard submission** workflow runs on every open/edit of a
   `submit`-labeled issue. It downloads the attachments, re-scores them with
   `relbench.submit.evaluate_submission` (no submitter code ever runs — attachments
   are only parsed as CSV), posts the report as an issue comment, and labels the
   issue `ok` or `invalid`.
4. **Maintainer action — this is the manual step:** review an `ok` issue (sanity of
   scores, method name/URL, in-context claim) and add the **`accept`** label.
   Only users with triage permission can label, so `accept` is the approval gate.
5. The **Publish leaderboard entry** workflow triggers on the `accept` label:
   it re-validates from scratch, writes `leaderboard/entries/<issue>.json`,
   regenerates `leaderboard/leaderboard.json`, commits to `main`, and closes the
   issue. On failure it posts the report and removes the `accept` label.

## Labels

| Label     | Meaning                                              | Applied by |
|-----------|------------------------------------------------------|------------|
| `submit`  | Leaderboard submission issue (workflow trigger gate) | issue form |
| `ok`      | Passed validation                                    | CI         |
| `invalid` | Failed validation                                    | CI         |
| `accept`  | Maintainer approval — publishes the entry            | maintainer |

## Notes

- Re-running validation: edit the issue (or re-add the `submit` label).
- Entries are keyed by issue number; to amend a published entry, edit and
  re-`accept` the issue (the entry JSON is overwritten and the aggregate rebuilt).
- To remove an entry, delete `leaderboard/entries/<issue>.json` and rebuild
  `leaderboard/leaderboard.json` (see `rebuild_aggregate` in
  `.github/scripts/leaderboard_submission.py`).
- All scoring goes through `relbench.submit` — the CI script
  (`.github/scripts/leaderboard_submission.py`) only parses the issue form,
  downloads attachments, and renders `relbench.submit`'s own report.
