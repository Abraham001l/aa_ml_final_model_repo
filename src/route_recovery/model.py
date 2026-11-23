"""Audited seasonal kernels, adapted from Keshav's seasonal_challenger.py.

See docs/PROVENANCE.md for attribution and the source hash. The equations retain
the original numerical ordering. New data/model wrappers enforce as-of boundaries.
"""
from dataclasses import asdict, dataclass
import numpy as np
import pandas as pd

HORIZONS = np.arange(1, 25)

@dataclass(frozen=True)
class Config:
    family: str = "route recovery"
    window: int = 3
    recent: float = 0.7
    recovery: float = 12.0
    pool: float = 0.25

    def __post_init__(self):
        if self.family not in {"route recovery", "network recovery"}:
            raise ValueError("Unknown model family.")
        if not isinstance(self.window, int) or isinstance(self.window, bool) or not 1 <= self.window <= 12:
            raise ValueError("window must be an integer from 1 to 12.")
        if not all(np.isfinite(x) for x in [self.recent, self.recovery, self.pool]):
            raise ValueError("Parameters must be finite.")
        if not 0 <= self.recent <= 1 or not 0 <= self.pool <= 1 or self.recovery <= 0:
            raise ValueError("recent/pool must lie in [0,1]; recovery must be positive.")

    def to_dict(self):
        return asdict(self)

def _network_ratio(observed, expected):
    # Explicit extension for degenerate all-zero reference networks, absent from
    # the audited data. A positive new network uses latest-history seasonality.
    return observed / expected if expected > 0 else (np.inf if observed > 0 else 0.0)

def seasonal_components(history, window, recent_weight, recovery):
    """Forecast using only a routes x months frame ending at the forecast origin."""
    calendar = history.columns
    origin = calendar[-1]
    targets = pd.period_range(origin + 1, periods=24, freq="M")
    # Static route universe is supplied to match the report. Only known volumes enter.
    volume = history.fillna(0).to_numpy()
    daily = volume / calendar.days_in_month.to_numpy()
    reference = daily[:, calendar.year == 2019]
    if reference.shape[1] != 12:
        raise ValueError("A full 2019 seasonal reference is required.")
    # Bound the effect of a sparse or unusually low month on the inferred level.
    def scaled(shape, dates):
        expected = shape[:, dates.month - 1].sum(axis=1)
        observed = daily[:, -len(dates):].sum(axis=1)
        return np.clip((observed + 1) / (expected + 1), 0.1, 6)

    annual = daily[:, -12:][:, np.argsort(calendar[-12:].month)]
    # In a recovery, the last year may include closed borders or suspended routes.
    # Weight recent seasonality more heavily as network volume approaches 2019 levels.
    recent = calendar[-window:]
    network_recovery = _network_ratio(daily[:, -window:].sum(), reference[:, recent.month - 1].sum())
    trust_recent = recent_weight * np.clip((network_recovery - .65) / .35, 0, 1)
    ratio = scaled(reference, recent)
    # Keep structural route gains; allow a temporary, network-wide deficit to recover.
    uplift = max(0, 1 / max(network_recovery, .25) - 1)
    growth = 1 + recovery * uplift * (1 - np.exp(-HORIZONS / 8))
    pre = reference[:, targets.month - 1] * ratio[:, None]
    current = annual[:, targets.month - 1] * scaled(annual, recent)[:, None]
    own = ((1 - trust_recent) * pre + trust_recent * current) * growth * targets.days_in_month.to_numpy()
    # Estimate a second forecast using the reverse route, scaled by recent directional
    # volumes after removing the 2019 directional seasonality (important in winter).
    reverse = history.index.get_indexer(history.index.swaplevel())
    valid = reverse >= 0
    backward = own.copy()
    # Smooth the directional seasonal ratio with the annual ratio on thin routes.
    totals = volume[:, -12:].sum(axis=1)
    annual_ratio = (totals + 100) / (totals[reverse] + 100)
    historical_ratio = (100 * annual_ratio[:, None] + reference[:, targets.month - 1] * 30) / (
        100 + reference[reverse][:, targets.month - 1] * 30)
    level_ratio = (ratio + .1) / (ratio[reverse] + .1)
    adjustment = np.clip(historical_ratio * level_ratio[:, None], .25, 4)
    backward[valid] = own[reverse[valid]] * adjustment[valid]
    return own, backward


def recovery_components(history, window, damping, recent_weight):
    """Blend 2019 and recent seasonality, allowing recovery to differ by route."""
    calendar = history.columns
    daily = history.fillna(0).to_numpy() / calendar.days_in_month.to_numpy()
    base = daily[:, calendar.year == 2019]
    recent = calendar[-window:]
    targets = pd.period_range(calendar[-1] + 1, periods=24, freq="M")
    target_months = targets.month - 1
    expected = base[:, recent.month - 1].sum(axis=1)
    observed = daily[:, -window:].sum(axis=1)
    ratio = np.clip((observed + 2) / (expected + 2), .05, 8)
    network_ratio = _network_ratio(observed.sum(), expected.sum())
    # A broad traffic deficit provides evidence for a temporary disruption.
    recovery_weight = np.clip((1.06 - network_ratio) / .20, 0, 1)
    # Use route-level shrinkage towards the pre-disruption monthly volume. Smooth
    # towards a current seasonal shape as the network returns to normal.
    pull = (1 - np.exp(-HORIZONS / damping)) * recovery_weight
    level = ratio[:, None] * (1 - pull) + pull
    pre = base[:, target_months] * level
    last = daily[:, -12:][:, np.argsort(calendar[-12:].month)]
    recent_profile = last[:, target_months]
    factor = np.clip((observed + 2) / (last[:, recent.month - 1].sum(axis=1) + 2), .5, 2)
    latest = recent_profile * factor[:, None]
    trust = recent_weight * (1 - recovery_weight)
    # Include the recent profile for routes with little or no 2019 service.
    structural_share = np.minimum(1., base.mean(axis=1) / np.maximum(.25 * daily[:, -12:].mean(axis=1), .1))
    weight = (1 - trust) * structural_share[:, None]
    own = (weight * pre + (1 - weight) * latest) * targets.days_in_month.to_numpy()
    return own


def directional_adjustment(history, own):
    """Use the opposite forecast, translating it to this direction's seasonal share."""
    origin = history.columns[-1]
    months = pd.period_range(origin + 1, periods=24, freq="M").month
    vol = history.fillna(0).to_numpy()
    rev = history.index.get_indexer(history.index.swaplevel())
    both = rev >= 0
    prior = vol[:, :12][:, months - 1]
    last_year = vol[:, -12:][:, np.argsort(history.columns[-12:].month)][:, months - 1]
    # Preserve persistent flow imbalances and average noisy direction-specific shares.
    annual = (vol[:, -12:].sum(axis=1) + 100) / (vol[rev, -12:].sum(axis=1) + 100)
    def share(reference):
        return (reference + 100 * annual[:, None]) / (reference[rev] + 100)
    multiplier = np.sqrt(share(prior) * share(last_year)).clip(.25, 4)
    paired = own.copy()
    paired[both] = own[rev[both]] * multiplier[both]
    return paired

def forecast_history(history, config=Config()):
    """Return 24 route/month forecasts from an already history-only monthly panel."""
    if not isinstance(history.index, pd.MultiIndex) or history.index.nlevels != 2:
        raise ValueError("Expected a two-level directional-route index.")
    if history.empty or history.index.has_duplicates:
        raise ValueError("History must be nonempty with unique routes.")
    if not isinstance(history.columns, pd.PeriodIndex):
        raise ValueError("History columns must be monthly pandas Periods.")
    expected = pd.period_range("2019-01", history.columns[-1], freq="M")
    if not history.columns.equals(expected) or len(expected) < 12:
        raise ValueError("Expected consecutive monthly history beginning January 2019.")
    values = history.to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("History must be finite and nonnegative; fill absent months first.")
    if config.family == "route recovery":
        own = recovery_components(history, config.window, config.recovery, config.recent)
        paired = directional_adjustment(history, own)
    else:
        own, paired = seasonal_components(history, config.window, config.recent, config.recovery)
    result = own + config.pool * (paired - own)
    if not np.isfinite(result).all() or (result < 0).any():
        raise ValueError("Model produced invalid passenger forecasts.")
    months = pd.period_range(history.columns[-1] + 1, periods=24, freq="M")
    return pd.DataFrame(result, index=history.index, columns=months)
