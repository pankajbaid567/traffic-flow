"""Build and execute the verified Phase 1 evaluation notebook without sockets."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import nbformat
from scripts.execute_notebooks import execute_in_process


def main():
    md = nbformat.v4.new_markdown_cell
    code = nbformat.v4.new_code_cell
    cells = [
        md(
            "# Phase 1 verified model evaluation\n\nAll numerical outputs below are loaded from independently scored forecast rows. No metric or geographic conclusion is entered manually."
        ),
        code(
            "from pathlib import Path\nimport json\nimport pandas as pd\nfrom IPython.display import display, Image\nROOT=Path.cwd()\nif not (ROOT/'data/metadata').exists(): ROOT=ROOT.parent\nMETA=ROOT/'data/metadata'\nresults=json.loads((META/'person_e_results.json').read_text())\nmetrics=pd.read_csv(META/'phase1_metrics.csv')\ndisplay(results['metadata'])"
        ),
        md(
            "## Shared test benchmark\n\nEvery model uses identical valid positive targets and forecast origins. RF and plain XGBoost use 32 features; the historical-average extension uses 36."
        ),
        code(
            "headline=metrics[(metrics.split=='test') & (metrics.period=='all')]\ndisplay(headline[['horizon','model','n','mae','rmse','mape']])\nassert headline.groupby('horizon').n.nunique().eq(1).all()\ndisplay(Image(filename=str(ROOT/'figures/person_e_horizon_metrics_comparison.png')))"
        ),
        md(
            "## Temporal slices include Random Forest\n\nMorning and evening groups use weekday origin times on the stored clock. The timezone remains unverified."
        ),
        code(
            "display(metrics[(metrics.split=='test') & (metrics.horizon=='15min')][['model','period','n','mae','rmse','mape']])\ndisplay(Image(filename=str(ROOT/'figures/person_e_peak_vs_offpeak_analysis.png')))"
        ),
        md(
            "## Actual forecast errors on the sensor map\n\nColors compare observed speed, predicted speed and MAE on identical test targets. The maps do not infer verified freeway membership or traffic direction."
        ),
        code(
            "for model in ['random_forest_32','xgboost_ha']:\n    for period in ['morning_peak','evening_peak']:\n        display(Image(filename=str(ROOT/f'figures/person_e_forecast_map_{model}_{period}.png')))"
        ),
        md(
            "## Predicted and observed congestion frequency\n\nCongestion is an operational proxy: speed below 60% of the sensor training 85th percentile. The threshold is fitted on train only."
        ),
        code(
            "display(Image(filename=str(ROOT/'figures/person_e_predicted_vs_actual_congestion.png')))\ndisplay(pd.read_csv(META/'corridor_performance.csv'))"
        ),
        md(
            "## Per-sensor and geographic error evidence\n\nRankings below are forecast MAE rather than speed volatility. Regions are reproducible coordinate bins. Weighted error sums and valid counts produce region scores."
        ),
        code(
            "ranking=pd.read_csv(META/'sensor_error_rankings.csv')\ndisplay(ranking[(ranking.horizon=='15min')&(ranking.model=='xgboost_ha')].head(10)[['sensor_id','region','n','mae','rmse','actual_mean','predicted_mean']])\nregion=pd.read_csv(META/'region_forecast_metrics.csv')\ndisplay(region[(region.split=='test')&(region.horizon=='15min')&(region.model=='xgboost_ha')&(region.period=='all')])"
        ),
        md(
            "## Paired day-block uncertainty\n\nTwo consecutive stored days are resampled together, with all sensors retained; 2,000 paired circular moving-block draws use seed 42. Positive MAE differences favor the candidate. These are conditional intervals for this held-out period."
        ),
        code("display(pd.read_csv(META/'paired_bootstrap.csv'))"),
        md(
            "## Feature ablations and a retrospective example\n\nThe RF groups contain 12, 19, 27 and 32 features. The worst-error example is selected retrospectively for diagnosis and never used to tune the model."
        ),
        code(
            "display(Image(filename=str(ROOT/'figures/person_e_rf_ablation_progression.png')))\ndisplay(results['example'])\ndisplay(Image(filename=str(ROOT/'figures/person_e_prediction_example.png')))"
        ),
        md(
            "## Reproduction checks and interpretation limits\n\nThe input hashes, package versions and model-to-evaluation comparisons are recorded below. Units and timezone are unverified. Missing targets are not imputed. This study does not establish a causal propagation mechanism or a guaranteed improvement from graph learning."
        ),
        code(
            "display(pd.DataFrame(results['metric_reproduction_checks']))\nprint('Hashed inputs:',len(results['input_sha256']))\nassert all(r['maximum_metric_difference'] <= .002 for r in results['metric_reproduction_checks'])"
        ),
        md(
            "## Future experiment\n\nA graph model may be compared on the same partitions and observed targets in Phase 2. The existing results motivate this experiment but do not evaluate GAT, DCRNN, or any live traffic control."
        ),
    ]
    notebook = nbformat.v4.new_notebook(
        cells=cells,
        metadata={
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            }
        },
    )
    notebook = execute_in_process(notebook, ROOT)
    output = ROOT / "notebooks/03_model_evaluation_and_congestion_maps.ipynb"
    nbformat.write(notebook, output)
    print("Saved executed notebook", output)


if __name__ == "__main__":
    main()
