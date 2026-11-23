"""Four basic Matplotlib charts showing the selected model building IAH -> LAX.

Run: python scripts/plot_model_steps.py --data data/pax_forecasting.csv
Optional: --step 1 (or 2, 3, 4) to make only one chart; --no-show to save only.
Requires numpy, pandas, matplotlib. All outputs stay beside this script.
"""
import argparse
import calendar
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter


from route_recovery.data import read_data, build_history
from route_recovery.model import Config, recovery_components, directional_adjustment, forecast_history

def components(raw):
    history = build_history(raw, "2024-07")
    vol = history.to_numpy()
    daily = vol / history.columns.days_in_month.to_numpy()
    base = daily[:, history.columns.year == 2019]
    recent = history.columns[-3:].month - 1
    observed = daily[:, -3:].sum(axis=1)
    expected = base[:, recent].sum(axis=1)
    ratio = np.clip((observed + 2)/(expected + 2), .05, 8)
    network_ratio = observed.sum()/expected.sum()
    recovery_weight = np.clip((1.06-network_ratio)/.20, 0, 1)
    last = daily[:, -12:][:, np.argsort(history.columns[-12:].month)]
    structural = np.minimum(1, base.mean(axis=1)/np.maximum(.25*last.mean(axis=1), .1))
    q = (1-.7*(1-recovery_weight))*structural
    # Forecast values come from the package's authoritative numerical kernels.
    own = recovery_components(history, 3, 12., .7)
    paired = directional_adjustment(history, own)
    prediction = forecast_history(history, Config()).to_numpy()
    targets = pd.period_range("2024-08", periods=24, freq="M")
    return dict(history=history, base=base, last=last, ratio=ratio, network_ratio=network_ratio,
                recovery_weight=recovery_weight, q=q, own=own, paired=paired,
                prediction=prediction, targets=targets)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--step', type=int, choices=[1, 2, 3, 4])
    parser.add_argument('--no-show', action='store_true')
    parser.add_argument('--data', required=True)
    parser.add_argument('--output', default='outputs/model_steps')
    parser.add_argument('--origin', default='IAH')
    parser.add_argument('--destination', default='LAX')
    args = parser.parse_args()
    origin, destination = args.origin.upper().strip(), args.destination.upper().strip()
    if not (len(origin) == len(destination) == 3 and origin.isalpha() and destination.isalpha()):
        parser.error('Use three-letter airport codes.')
    route_label = f'{origin} â†’ {destination}'
    output_dir = Path(args.output)
    if output_dir.exists():
        parser.error("Output directory exists; choose a fresh path.")
    output_dir.mkdir(parents=True)
    raw = read_data(args.data)
    c = components(raw)
    # Check future demand cannot change the displayed calculations.
    poisoned = raw.copy()
    poisoned.loc[(poisoned.DEP_YEAR * 100 + poisoned.DEP_MONTH) > 202407, 'OD_PAX'] = 9e12
    np.testing.assert_array_equal(c['prediction'], components(poisoned)['prediction'])
    i = c['history'].index.get_loc((origin, destination))
    if c['recovery_weight'] != 0:
        raise ValueError('These calendar-profile illustrations require the July 2024 inactive recovery gate.')
    b, last, r, q = c['base'][i], c['last'][i], c['ratio'][i], c['q'][i]
    scaled = b * r
    blended = q * scaled + (1 - q) * last
    # Check the daily illustration matches all 24 actual own-route calculations.
    np.testing.assert_allclose(blended[c['targets'].month - 1] * c['targets'].days_in_month,
                               c['own'][i], rtol=1e-12)
    ymax = 1.15 * max(b.max(), last.max(), scaled.max(), blended.max())
    months = np.arange(1, 13)
    plt.style.use('default')
    for step in ([args.step] if args.step else [1, 2, 3, 4]):
        fig, ax = plt.subplots(figsize=(8, 4.5))
        if step <= 3:
            ax.set_xticks(months, list(calendar.month_abbr)[1:])
            ax.set_xlabel('Calendar month')
            ax.set_ylabel('Average passengers per day')
            ax.set_ylim(0, ymax)
            if step == 1:
                ax.plot(months, b, marker='o', label='2019 profile')
                ax.plot(months, last, marker='o', label='Latest 12-month profile')
                ax.set_title(f'1. {route_label}: starting seasonal profiles')
                caption = 'Latest profile: Aug 2023â€“Jul 2024, arranged by calendar month.'
            elif step == 2:
                ax.plot(months, b, color='gray', linestyle=':', label='Original 2019 profile')
                ax.plot(months, scaled, color='tab:blue', marker='o', label=f'2019 profile Ã— {r:.4f}')
                ax.set_title(f'2. {route_label}: adjust the 2019 demand level')
                caption = (f'Mayâ€“Jul route ratio: {r:.4f}. Network ratio: {c["network_ratio"]:.2%}.\n'
                           'Recovery weight is 0, so there is no gradual return toward 2019.')
            else:
                ax.plot(months, scaled, color='tab:blue', linestyle='--', label='Scaled 2019 profile')
                ax.plot(months, last, color='tab:orange', linestyle='--', label='Latest profile')
                ax.plot(months, blended, color='tab:green', marker='o', label='Blended daily forecast')
                ax.set_title(f'3. {route_label}: blend both seasonal profiles')
                caption = f'Own-route daily forecast = {q:.0%} scaled 2019 + {1-q:.0%} latest profile.'
            ax.legend(fontsize=9)
        else:
            j = 4  # December 2024, the fifth forecast month.
            own, paired, final = (float(c[k][i, j]) for k in ['own', 'paired', 'prediction'])
            bars = ax.bar(['Own route', 'Adjusted reverse', 'Final forecast'], [own, paired, final])
            ax.bar_label(bars, labels=[f'{v:,.0f}' for v in [own, paired, final]], padding=4)
            ax.set_ylim(0, max(own, paired, final) * 1.2)
            ax.set_ylabel('Monthly passengers')
            ax.set_title(f'4. {route_label}: final December 2024 forecast')
            caption = ('Convert daily averages to monthly PAX using Decemberâ€™s 31 days.\n'
                       f'75% Ã— {own:,.2f} + 25% Ã— {paired:,.2f} = {final:,.2f} passengers.')
        ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
        fig.text(.10, .035, caption, fontsize=9)
        fig.tight_layout(rect=(0, .13, 1, 1))
        output = output_dir / f'step_{step}.png'
        fig.savefig(output, dpi=300)
        print(f'Saved: {output}')
        if args.no_show:
            plt.close(fig)
    table = pd.DataFrame({'month': c['targets'].astype(str), 'own_forecast': c['own'][i],
                          'adjusted_reverse_forecast': c['paired'][i], 'final_forecast': c['prediction'][i]})
    table.to_csv(output_dir / 'forecast_values.csv', index=False)
    (output_dir / 'calculation_details.json').write_text(json.dumps({
        'route_ratio': float(r), 'network_ratio': float(c['network_ratio']),
        'recovery_weight': float(c['recovery_weight']), '2019_weight': float(q),
        'future_poisoning_check': 'passed; predictions exactly unchanged',
        'historical_cutoff': '2024-07', 'latest_profile': '2023-08 through 2024-07'}, indent=2))
    if not args.no_show:
        plt.show()


if __name__ == '__main__':
    main()
