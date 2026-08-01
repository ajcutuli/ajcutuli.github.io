from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from pandas_datareader import data as web
from sklearn.decomposition import PCA


OUTPUT_DIR = Path(__file__).resolve().parents[1] / "images" / "PCA_geometry"
SERIES = {
    "1Y": "DGS1",
    "2Y": "DGS2",
    "3Y": "DGS3",
    "5Y": "DGS5",
    "7Y": "DGS7",
    "10Y": "DGS10",
    "20Y": "DGS20",
    "30Y": "DGS30",
}
MATURITIES = np.array([1, 2, 3, 5, 7, 10, 20, 30], dtype=float)


def download_treasury_yields(start="2005-01-01"):
    """Download constant-maturity Treasury yields from FRED."""
    yields = web.DataReader(list(SERIES.values()), "fred", start=start)
    yields.columns = list(SERIES.keys())
    return yields


def orient_pc2(pc2):
    """Orient PC2 so its long-end loading exceeds its short-end loading."""
    pc2 = pc2.copy()
    if pc2[-1] < pc2[0]:
        pc2 *= -1
    return pc2


def count_sign_changes(values, tolerance=1e-8):
    """Count crossings after removing values that are effectively zero."""
    values = np.asarray(values).copy()
    values[np.abs(values) < tolerance] = 0
    signs = np.sign(values)
    signs = signs[signs != 0]
    return int(np.sum(signs[1:] != signs[:-1])) if len(signs) >= 2 else 0


def second_difference_roughness(values):
    """Measure deviation from a locally linear loading shape."""
    return float(np.sum(np.diff(values, n=2) ** 2))


def make_segments(x, y):
    """Convert a curve into adjacent line segments for a LineCollection."""
    points = np.column_stack([x, y])
    return np.stack([points[:-1], points[1:]], axis=1)


def plot_treasury_yield_pca(daily_changes):
    """Plot the first three PCs of daily Treasury yield changes."""
    pca = PCA(n_components=3).fit(daily_changes)
    loadings = pca.components_.copy()

    if loadings[0].mean() < 0:
        loadings[0] *= -1
    if loadings[1, -1] < loadings[1, 0]:
        loadings[1] *= -1
    if loadings[2, 4] > 0:
        loadings[2] *= -1

    fig, ax = plt.subplots(figsize=(8, 5))
    labels = ["PC1 (Level)", "PC2 (Slope)", "PC3 (Curvature)"]
    for loading, label in zip(loadings, labels):
        ax.plot(MATURITIES, loading, marker="o", linewidth=2, label=label)

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set(
        xlabel="Maturity (Years)",
        ylabel="Loading",
        title="Principal Components of Daily Treasury Yield Changes",
    )
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "treasury_yield_pca.png", dpi=300)
    plt.close(fig)

    print("Explained variance:")
    for index, ratio in enumerate(pca.explained_variance_ratio_, start=1):
        print(f"PC{index}: {100 * ratio:.2f}%")

    return loadings


def plot_yield_curve_animation(recent_yields):
    """Animate the recent Treasury yield curve."""
    fig, ax = plt.subplots(figsize=(8, 5))
    (line,) = ax.plot([], [], linewidth=3, marker="o")
    date_text = ax.text(0.02, 0.95, "", transform=ax.transAxes, fontsize=12)

    ax.set_xlim(1, 30)
    ax.set_ylim(recent_yields.min().min() - 0.5, recent_yields.max().max() + 0.5)
    ax.set(
        xlabel="Maturity (Years)",
        ylabel="Yield (%)",
        title="US Treasury Yield Curve",
    )
    ax.grid(alpha=0.3)

    def init():
        line.set_data([], [])
        date_text.set_text("")
        return line, date_text

    def update(frame):
        line.set_data(MATURITIES, recent_yields.iloc[frame].values)
        date_text.set_text(recent_yields.index[frame].strftime("%Y-%m-%d"))
        return line, date_text

    animation = FuncAnimation(
        fig,
        update,
        frames=len(recent_yields),
        init_func=init,
        interval=30,
        blit=True,
    )
    animation.save(OUTPUT_DIR / "yield_curve.gif", writer=PillowWriter(fps=30))
    plt.close(fig)


def plot_yield_curve_motion(weekly_yields, trail_length=52):
    """Animate weekly yield-curve slopes alongside recent 2Y and 10Y yields."""
    slopes = np.diff(weekly_yields.values, axis=1) / np.diff(MATURITIES)
    slope_limit = np.nanpercentile(np.abs(slopes), 98)
    norm = Normalize(vmin=-slope_limit, vmax=slope_limit)

    fig, (ax_curve, ax_history) = plt.subplots(
        1, 2, figsize=(12, 5), gridspec_kw={"width_ratios": [1.5, 1]}
    )
    initial_curve = weekly_yields.iloc[0].values
    line_collection = LineCollection(
        make_segments(MATURITIES, initial_curve),
        cmap="coolwarm",
        norm=norm,
        linewidth=4,
    )
    ax_curve.add_collection(line_collection)
    (points,) = ax_curve.plot(MATURITIES, initial_curve, "o", markersize=6)
    date_text = ax_curve.text(0.03, 0.94, "", transform=ax_curve.transAxes, fontsize=12)

    ax_curve.set_xlim(0.5, 31)
    ax_curve.set_ylim(weekly_yields.min().min() - 0.5, weekly_yields.max().max() + 0.5)
    ax_curve.set_xticks(MATURITIES)
    ax_curve.set(
        xlabel="Maturity (years)",
        ylabel="Yield (%)",
        title="U.S. Treasury Yield Curve",
    )
    ax_curve.grid(alpha=0.25)
    fig.colorbar(line_collection, ax=ax_curve, pad=0.02).set_label(
        "Local slope: percentage points per year"
    )

    (two_year_line,) = ax_history.plot([], [], linewidth=2, label="2Y yield")
    (ten_year_line,) = ax_history.plot([], [], linewidth=2, label="10Y yield")
    (current_2y,) = ax_history.plot([], [], "o", markersize=7)
    (current_10y,) = ax_history.plot([], [], "o", markersize=7)
    ax_history.set_xlim(weekly_yields.index.min(), weekly_yields.index.max())
    ax_history.set_ylim(
        min(weekly_yields["2Y"].min(), weekly_yields["10Y"].min()) - 0.5,
        max(weekly_yields["2Y"].max(), weekly_yields["10Y"].max()) + 0.5,
    )
    ax_history.set(
        xlabel="Date",
        ylabel="Yield (%)",
        title="Recent 2Y and 10Y Movement",
    )
    ax_history.legend()
    ax_history.grid(alpha=0.25)

    def init():
        line_collection.set_segments(make_segments(MATURITIES, initial_curve))
        points.set_data(MATURITIES, initial_curve)
        two_year_line.set_data([], [])
        ten_year_line.set_data([], [])
        current_2y.set_data([], [])
        current_10y.set_data([], [])
        date_text.set_text("")
        return (
            line_collection,
            points,
            two_year_line,
            ten_year_line,
            current_2y,
            current_10y,
            date_text,
        )

    def update(frame):
        curve = weekly_yields.iloc[frame].values
        line_collection.set_segments(make_segments(MATURITIES, curve))
        line_collection.set_array(np.diff(curve) / np.diff(MATURITIES))
        points.set_data(MATURITIES, curve)

        date = weekly_yields.index[frame]
        date_text.set_text(date.strftime("%B %d, %Y"))
        history = weekly_yields.iloc[max(0, frame - trail_length) : frame + 1]
        two_year_line.set_data(history.index, history["2Y"])
        ten_year_line.set_data(history.index, history["10Y"])
        current_2y.set_data([date], [weekly_yields.iloc[frame]["2Y"]])
        current_10y.set_data([date], [weekly_yields.iloc[frame]["10Y"]])
        if len(history) > 1:
            ax_history.set_xlim(history.index[0], history.index[-1])
        return (
            line_collection,
            points,
            two_year_line,
            ten_year_line,
            current_2y,
            current_10y,
            date_text,
        )

    animation = FuncAnimation(
        fig, update, frames=len(weekly_yields), init_func=init, interval=60, blit=False
    )
    fig.tight_layout()
    animation.save(
        OUTPUT_DIR / "yield_curve_motion.gif",
        writer=PillowWriter(fps=10),
        dpi=120,
    )
    plt.close(fig)


def plot_random_covariance_pca(seed=42, n_assets=10, n_samples=5000):
    """Plot PCA loadings from a reproducible random covariance matrix."""
    rng = np.random.default_rng(seed)
    matrix = rng.normal(size=(n_assets, n_assets))
    covariance = matrix @ matrix.T
    samples = rng.multivariate_normal(
        mean=np.zeros(n_assets), cov=covariance, size=n_samples
    )
    components = PCA(n_components=3).fit(samples).components_.copy()
    for component in components:
        if component.mean() < 0:
            component *= -1

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(1, n_assets + 1)
    for index, component in enumerate(components, start=1):
        ax.plot(x, component, marker="o", linewidth=2, label=f"PC{index}")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set(
        xlabel="Variable",
        ylabel="Loading",
        title="PCA of a Random Covariance Matrix",
    )
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "random_covariance_pca.png", dpi=300)
    plt.close(fig)


def plot_smooth_vs_oscillatory_loadings(daily_changes, loadings):
    """Compare actual PC2 with an orthogonal oscillatory alternative."""
    covariance = daily_changes.cov().to_numpy()
    pc1 = loadings[0] / np.linalg.norm(loadings[0])
    pc2 = orient_pc2(loadings[1] / np.linalg.norm(loadings[1]))

    alternating_signs = (-1.0) ** np.arange(len(MATURITIES))
    oscillatory = np.abs(pc2) * alternating_signs
    oscillatory -= np.dot(oscillatory, pc1) * pc1
    oscillatory /= np.linalg.norm(oscillatory)

    def rayleigh_quotient(vector):
        return float(vector.T @ covariance @ vector) / float(vector.T @ vector)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    x = np.arange(len(MATURITIES))
    maturity_labels = [f"{m:g}Y" for m in MATURITIES]
    panels = [
        (pc2, "Actual PC2: smooth, one sign change", rayleigh_quotient(pc2)),
        (oscillatory, "Oscillatory alternative", rayleigh_quotient(oscillatory)),
    ]
    for ax, (values, title, quotient) in zip(axes, panels):
        ax.bar(x, values)
        ax.axhline(0, linewidth=0.8)
        ax.set_xticks(x, maturity_labels)
        ax.set_xlabel("Maturity")
        ax.set_title(f"{title}\nRayleigh quotient = {quotient:.4f}")
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Loading")
    fig.suptitle("Alternating Signs Waste Positive Local Covariance", fontsize=14)
    fig.tight_layout()
    fig.savefig(
        OUTPUT_DIR / "smooth_vs_oscillatory_loadings.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_vibrating_string_modes():
    """Plot the first three eigenfunctions of a vibrating string."""
    x = np.linspace(0, 1, 1000)
    fig, axes = plt.subplots(3, 1, figsize=(8, 6), sharex=True)

    for mode, ax in enumerate(axes, start=1):
        ax.plot(x, np.sin(mode * np.pi * x), linewidth=2)
        ax.axhline(0, color="black", linewidth=0.8)
        nodes = np.arange(1, mode) / mode
        if len(nodes):
            ax.scatter(nodes, np.zeros_like(nodes), s=35, zorder=5)
        ax.set_ylim(-1.15, 1.15)
        ax.set_ylabel(rf"$n={mode}$")
        ax.set_title(rf"$\phi_{mode}(x)=\sin({mode}\pi x)$", fontsize=12)
        ax.grid(alpha=0.25)

    axes[-1].set_xlabel(r"$x$")
    fig.suptitle("Normal Modes of a Vibrating String", fontsize=15)
    fig.tight_layout()
    fig.savefig(
        OUTPUT_DIR / "vibrating_string_modes.png", dpi=250, bbox_inches="tight"
    )
    plt.close(fig)


def calculate_rolling_pc2(daily_changes, window=252 * 3, step=1):
    """Calculate consistently oriented PC2 loadings and their metrics."""
    records = []
    for end in range(window, len(daily_changes), step):
        pca = PCA(n_components=3).fit(daily_changes.iloc[end - window : end])
        pc2 = orient_pc2(pca.components_[1])
        records.append(
            {
                "date": daily_changes.index[end],
                "pc2": pc2,
                "first_difference_roughness": float(np.sum(np.diff(pc2) ** 2)),
                "roughness": second_difference_roughness(pc2),
                "sign_changes": count_sign_changes(pc2),
                "explained_variance": pca.explained_variance_ratio_[1],
            }
        )
    if not records:
        raise ValueError("Not enough observations for the selected rolling window.")
    return records


def plot_rolling_pc2_roughness(records):
    """Plot first-difference roughness of PC2 over rolling windows."""
    roughness = pd.Series(
        [record["first_difference_roughness"] for record in records],
        index=[record["date"] for record in records],
    )

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(roughness.index, roughness.values, linewidth=2)
    ax.set(
        title="Rolling Roughness of the Second PCA Loading",
        xlabel="Date",
        ylabel=r"$\sum_i (v_{i+1}-v_i)^2$",
    )
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(
        OUTPUT_DIR / "rolling_pc2_roughness.png", dpi=250, bbox_inches="tight"
    )
    plt.close(fig)


def plot_rolling_pc2_animation(records, fps=12):
    """Animate rolling PC2 loadings and second-difference roughness."""
    dates = pd.DatetimeIndex([record["date"] for record in records])
    pc2_history = np.vstack([record["pc2"] for record in records])
    metrics = pd.DataFrame(
        {
            key: [record[key] for record in records]
            for key in ("roughness", "sign_changes", "explained_variance")
        },
        index=dates,
    )

    loading_limit = 1.10 * np.max(np.abs(pc2_history))
    fig = plt.figure(figsize=(11, 7))
    grid = fig.add_gridspec(2, 1, height_ratios=[2, 1], hspace=0.35)
    ax_loading = fig.add_subplot(grid[0])
    ax_metric = fig.add_subplot(grid[1])

    (loading_line,) = ax_loading.plot(
        MATURITIES, pc2_history[0], marker="o", linewidth=2.5
    )
    ax_loading.axhline(0, linewidth=0.8)
    date_text = ax_loading.text(
        0.02, 0.94, "", transform=ax_loading.transAxes, fontsize=12, va="top"
    )
    metric_text = ax_loading.text(
        0.02, 0.84, "", transform=ax_loading.transAxes, fontsize=10, va="top"
    )
    ax_loading.set_xlim(MATURITIES.min() - 0.5, MATURITIES.max() + 1)
    ax_loading.set_ylim(-loading_limit, loading_limit)
    ax_loading.set_xticks(MATURITIES, [f"{m:g}Y" for m in MATURITIES])
    ax_loading.set(
        xlabel="Maturity",
        ylabel="PC2 loading",
        title="Rolling Second Principal Component of Treasury Yield Changes",
    )
    ax_loading.grid(alpha=0.25)

    ax_metric.plot(metrics.index, metrics["roughness"], linewidth=1.5)
    (current_metric,) = ax_metric.plot([], [], marker="o", markersize=7)
    current_date_line = ax_metric.axvline(dates[0], linewidth=1, linestyle="--")
    ax_metric.set_xlim(dates.min(), dates.max())
    ax_metric.set_ylim(0, metrics["roughness"].max() * 1.05)
    ax_metric.set(
        xlabel="Rolling-window end date",
        ylabel=r"$\sum_i (\Delta^2 v_i)^2$",
        title="Second-Difference Roughness of PC2",
    )
    ax_metric.grid(alpha=0.25)

    def init():
        loading_line.set_data(MATURITIES, pc2_history[0])
        current_metric.set_data([dates[0]], [metrics.iloc[0]["roughness"]])
        current_date_line.set_xdata([dates[0], dates[0]])
        date_text.set_text("")
        metric_text.set_text("")
        return loading_line, current_metric, current_date_line, date_text, metric_text

    def update(frame):
        date = dates[frame]
        row = metrics.iloc[frame]
        loading_line.set_data(MATURITIES, pc2_history[frame])
        current_metric.set_data([date], [row["roughness"]])
        current_date_line.set_xdata([date, date])
        date_text.set_text(date.strftime("%B %d, %Y"))
        metric_text.set_text(
            f"Sign changes: {int(row['sign_changes'])}\n"
            f"Roughness: {row['roughness']:.4f}\n"
            f"PC2 variance share: {100 * row['explained_variance']:.1f}%"
        )
        return loading_line, current_metric, current_date_line, date_text, metric_text

    animation = FuncAnimation(
        fig,
        update,
        frames=len(records),
        init_func=init,
        interval=1000 / fps,
        blit=False,
    )
    animation.save(
        OUTPUT_DIR / "rolling_pc2_geometry.gif",
        writer=PillowWriter(fps=fps),
        dpi=140,
    )
    plt.close(fig)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    raw_yields = download_treasury_yields()
    complete_yields = raw_yields.dropna()
    daily_changes = complete_yields.diff().dropna()
    recent_yields = raw_yields.loc["2025-01-01":].dropna().iloc[-400:]
    weekly_yields = (
        raw_yields.loc["2025-01-01":]
        .ffill()
        .dropna()
        .resample("W-FRI")
        .last()
        .dropna()
    )

    loadings = plot_treasury_yield_pca(daily_changes)
    plot_yield_curve_animation(recent_yields)
    plot_yield_curve_motion(weekly_yields)
    plot_random_covariance_pca()
    plot_smooth_vs_oscillatory_loadings(daily_changes, loadings)
    plot_vibrating_string_modes()
    rolling_records = calculate_rolling_pc2(daily_changes)
    plot_rolling_pc2_roughness(rolling_records)
    plot_rolling_pc2_animation(rolling_records[::10])


if __name__ == "__main__":
    main()
