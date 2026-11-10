"""Complete the saved review-sheet caller contract without rerunning inference."""
import argparse
from pathlib import Path

import qwen140_v2_shots_only_report_20261105 as report

original_sheets = report.review.review_sheets


def sheets(scored, episodes):
    report.write(report.review.OUT / 'scored_crops.json', scored)
    return original_sheets(scored, episodes)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['auto', 'final'])
    args = parser.parse_args()
    report.review.review_sheets = sheets
    report.write(report.OUT / 'report_resume_contract.json', {
        'entry': str(Path(__file__).resolve()), 'entry_sha256': report.sha(__file__),
        'original_report_sha256': report.sha(report.__file__),
        'change': 'Save scored_crops.json before the existing score-hidden sheet generator hashes it; no inference/metric changes.'})
    report.compute(args.action == 'final')
    report.summarize(args.action == 'final')
