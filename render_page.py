"""
Assemble the final static HTML page from template.html + the pipeline's
outputs (predictions.json, metrics.json). Run this after predict.py.

Writes nfl_predictions.html -- publish that file with the Artifact tool,
passing the existing artifact URL so it updates in place rather than
minting a new one each run.
"""

import json
from datetime import datetime, timezone

TEMPLATE_PATH = "template.html"
PREDICTIONS_PATH = "predictions.json"
METRICS_PATH = "metrics.json"
INJURIES_PATH = "injuries.json"
WEATHER_PATH = "weather.json"
RANKINGS_PATH = "team_rankings.json"
OUTPUT_PATH = "nfl_predictions.html"


def main():
    with open(TEMPLATE_PATH) as f:
        html = f.read()
    with open(PREDICTIONS_PATH) as f:
        predictions_json = f.read()
    with open(METRICS_PATH) as f:
        metrics = json.load(f)
    with open(INJURIES_PATH) as f:
        injuries_json = f.read()
    with open(WEATHER_PATH) as f:
        weather_json = f.read()
    with open(RANKINGS_PATH) as f:
        rankings_json = f.read()

    generated_at = datetime.fromisoformat(metrics["generated_at"])
    generated_label = generated_at.strftime("%b %-d, %Y")

    replacements = {
        "__DATA_JSON__": predictions_json,
        "__INJURIES_JSON__": injuries_json,
        "__WEATHER_JSON__": weather_json,
        "__RANKINGS_JSON__": rankings_json,
        "__GENERATED_AT__": generated_label,
        "__VAL_ACC__": f"{metrics['validation']['model_accuracy'] * 100:.1f}",
        "__VAL_SEASON__": str(metrics["validation_season"]),
        "__HOLDOUT_ACC__": f"{metrics['holdout']['model_accuracy'] * 100:.1f}",
        "__HOLDOUT_SEASON__": str(metrics["holdout_season"]),
        "__VEGAS_ACC__": f"{metrics['holdout']['vegas_line_baseline'] * 100:.1f}",
    }
    for placeholder, value in replacements.items():
        html = html.replace(placeholder, value)

    remaining = [p for p in replacements if p in html]
    if remaining:
        raise ValueError(f"template placeholders not replaced: {remaining}")

    with open(OUTPUT_PATH, "w") as f:
        f.write(html)
    print(f"Wrote {OUTPUT_PATH} ({len(html)} bytes), generated_at={generated_label}")


if __name__ == "__main__":
    main()
