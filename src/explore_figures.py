from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

FIG_DIR = "report/figures"

def _finish(fig, name, show):
    os.makedirs(FIG_DIR, exist_ok=True)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, name + ".png"), dpi=150)
    if show:
        plt.show()
    plt.close(fig)


def plot_price_distribution(df, name="explore_price_distribution", show=False):
    price = df["price"].dropna()
    price = price[price > 0]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].hist(price, bins=60, color="C0", alpha=0.85)
    axes[0].set_xlabel("price")
    axes[0].set_ylabel("count")
    axes[0].set_title(f"price (skew={price.skew():.2f})")
    logp = np.log1p(price)
    axes[1].hist(logp, bins=60, color="C1", alpha=0.85)
    axes[1].set_xlabel("log1p(price)")
    axes[1].set_ylabel("count")
    axes[1].set_title(f"log1p(price) (skew={logp.skew():.2f})")
    for ax in axes:
        ax.grid(alpha=0.3)
    _finish(fig, name, show)


def plot_area_vs_price(df, name="explore_area_vs_price", show=False):
    d = df[(df["area"] > 0) & (df["price"] > 0)]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    if "category" in d.columns:
        for i, (cat, g) in enumerate(d.groupby("category")):
            axes[0].scatter(g["area"], g["price"], s=8, alpha=0.35, color=f"C{i}", label=str(cat))
            axes[1].scatter(g["area"], g["price"], s=8, alpha=0.35, color=f"C{i}", label=str(cat))
        axes[1].legend(markerscale=2, fontsize=8)
    else:
        axes[0].scatter(d["area"], d["price"], s=8, alpha=0.35)
        axes[1].scatter(d["area"], d["price"], s=8, alpha=0.35)
    axes[0].set_xlabel("area")
    axes[0].set_ylabel("price")
    axes[0].set_title("area vs price")
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("area (log)")
    axes[1].set_ylabel("price (log)")
    axes[1].set_title("area vs price (log-log)")
    corr = np.corrcoef(np.log(d["area"]), np.log(d["price"]))[0, 1]
    axes[1].text(0.03, 0.95, f"corr(log area, log price)={corr:.2f}", transform=axes[1].transAxes, va="top")
    for ax in axes:
        ax.grid(alpha=0.3)
    _finish(fig, name, show)


def plot_rooms(df, max_rooms=8, name="explore_rooms", show=False):
    d = df[df["rooms"].notna() & df["price"].notna()].copy()
    d["_r"] = d["rooms"].astype(float).round().clip(upper=max_rooms).astype(int)
    counts = d["_r"].value_counts().sort_index()
    med = d.groupby("_r")["price"].median()
    labels = [f"{r}+" if r == max_rooms else str(r) for r in counts.index]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].bar(labels, counts.values, color="C0")
    axes[0].set_xlabel("rooms")
    axes[0].set_ylabel("count")
    axes[0].set_title("room count distribution")
    axes[1].bar(labels, med.loc[counts.index].values, color="C1")
    axes[1].set_xlabel("rooms")
    axes[1].set_ylabel("median price")
    axes[1].set_title("median price by rooms")
    for ax in axes:
        ax.grid(alpha=0.3, axis="y")
    _finish(fig, name, show)


def plot_location(df, lat_col="lat", lng_col="lng", name="explore_location", show=False):
    d = df[df["price"] > 0].copy()
    d["_lp"] = np.log1p(d["price"])
    panels = 1 + int("city" in d.columns) + int("category" in d.columns)
    fig, axes = plt.subplots(1, panels, figsize=(5.5 * panels, 4.8), squeeze=False)
    axes = axes.ravel()
    g = d.dropna(subset=[lat_col, lng_col])
    lo, hi = g["_lp"].quantile([0.05, 0.95])
    sc = axes[0].scatter(g[lng_col], g[lat_col], c=g["_lp"], s=8, alpha=0.6, vmin=lo, vmax=hi, cmap="viridis")
    fig.colorbar(sc, ax=axes[0], label="log1p(price)")
    axes[0].set_xlabel("lng")
    axes[0].set_ylabel("lat")
    axes[0].set_title("location vs log price")
    k = 1
    for col in ["city", "category"]:
        if col in d.columns:
            med = d.groupby(col)["_lp"].median().sort_values()
            cnt = d[col].value_counts()
            axes[k].barh([f"{i} (n={cnt[i]})" for i in med.index], med.values, color="C1")
            axes[k].set_xlabel("median log1p(price)")
            axes[k].set_title(f"median log price by {col}")
            axes[k].grid(alpha=0.3, axis="x")
            k += 1
    _finish(fig, name, show)


def plot_class_balance(y_tr, y_va, y_te, name="explore_class_balance", show=False):
    splits = {"train": y_tr, "val": y_va, "test": y_te}
    names = list(splits)
    premium = np.array([int(np.sum(np.asarray(v) == 1)) for v in splits.values()])
    standard = np.array([int(np.sum(np.asarray(v) == 0)) for v in splits.values()])
    total = premium + standard
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar(names, standard, label="standard (0)", color="C0")
    axes[0].bar(names, premium, bottom=standard, label="premium (1)", color="C1")
    axes[0].set_ylabel("count")
    axes[0].set_title("tier counts per split")
    axes[0].legend()
    share = premium / total
    axes[1].bar(names, share, color="C1")
    axes[1].axhline(0.5, color="gray", linestyle=":")
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel("premium share")
    axes[1].set_title("premium share per split")
    for i, s in enumerate(share):
        axes[1].text(i, s + 0.01, f"{s:.3f}", ha="center")
    for ax in axes:
        ax.grid(alpha=0.3, axis="y")
    _finish(fig, name, show)

def print_explore_stats(df, y_tr, y_va, y_te, lat_col="lat", lng_col="lng"):
    price = df["price"].dropna()
    price = price[price > 0]
    logp = np.log1p(price)
    print("\n  [Explore] Price distribution (train):")
    print(f"    n={len(price):,} | mean={price.mean():,.0f} | median={price.median():,.0f} | min={price.min():,.0f} | max={price.max():,.0f}")
    print(f"    skew(price)={price.skew():.3f} | skew(log1p(price))={logp.skew():.3f}")

    d = df[(df["area"] > 0) & (df["price"] > 0)]
    if len(d) > 2:
        corr = np.corrcoef(np.log(d["area"]), np.log(d["price"]))[0, 1]
        print(f"  [Explore] area vs price: corr(log area, log price)={corr:.3f} (n={len(d):,}, area>0 only)")

    r = df[df["rooms"].notna() & (df["price"] > 0)]
    if len(r) > 2:
        rooms = r["rooms"].astype(float)
        rc = np.corrcoef(rooms, np.log1p(r["price"]))[0, 1]
        counts = rooms.round().clip(upper=8).astype(int).value_counts().sort_index()
        print(f"  [Explore] rooms: corr(rooms, log price)={rc:.3f}")
        print("    counts: " + ", ".join(f"{k}{'+' if k == 8 else ''}: {v}" for k, v in counts.items()))

    if lat_col in df.columns and lng_col in df.columns:
        g = df[[lat_col, lng_col]].assign(_lp=np.log1p(df["price"].where(df["price"] > 0))).dropna()
        print(f"  [Explore] location: corr(lat, log price)={g[lat_col].corr(g['_lp']):.3f} | "
              f"corr(lng, log price)={g[lng_col].corr(g['_lp']):.3f} (n={len(g):,})")

    for col in ["city", "category"]:
        if col in df.columns:
            s = (df[df["price"] > 0].groupby(col)["price"]
                 .agg(["count", "median"]).sort_values("median", ascending=False))
            print(f"  [Explore] median price by {col}:")
            for k, row in s.iterrows():
                print(f"    {str(k):<24} n={int(row['count']):>6,} | median={row['median']:>12,.0f}")

    print("  [Explore] Task B class balance:")
    for nm, y in [("train", y_tr), ("val", y_va), ("test", y_te)]:
        y = np.asarray(y)
        print(f"    {nm:<6} n={len(y):>6,} | premium={int((y == 1).sum()):>6,} | "
              f"standard={int((y == 0).sum()):>6,} | premium share={y.mean():.3f}")


def run_explore(df, y_tr, y_va, y_te, lat_col="lat", lng_col="lng"):
    plot_price_distribution(df)
    plot_area_vs_price(df)
    plot_rooms(df)
    if lat_col in df.columns and lng_col in df.columns:
        plot_location(df, lat_col=lat_col, lng_col=lng_col)
    else:
        print(f"  [Explore] '{lat_col}'/'{lng_col}' not found, skipping location plot. Columns: {list(df.columns)}")
    plot_class_balance(y_tr, y_va, y_te)
    print_explore_stats(df, y_tr, y_va, y_te, lat_col=lat_col, lng_col=lng_col)

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n = 2000
    area = rng.lognormal(4.3, 0.5, n)
    rooms = np.clip((area / 35 + rng.normal(0, 0.6, n)).round(), 1, 10)
    lat = rng.normal(40.4, 0.05, n)
    lng = rng.normal(49.85, 0.07, n)
    price = np.exp(8.5 + 0.9 * np.log(area) + 2 * (lat - 40.4) + rng.normal(0, 0.3, n))
    demo = pd.DataFrame(dict(
        price=price, area=area, rooms=rooms, lat=lat, lng=lng,
        city=rng.choice(["bakı", "sumqayıt", "xırdalan"], n, p=[0.8, 0.12, 0.08]),
        category=rng.choice(["Yeni tikili", "Köhnə tikili", "Həyət evi/Bağ evi"], n),
    ))
    plot_price_distribution(demo)
    plot_area_vs_price(demo)
    plot_rooms(demo)
    plot_location(demo)
    thr = demo["price"].iloc[:1400].median()
    tier = (demo["price"] > thr).astype(int)
    plot_class_balance(tier.iloc[:1400], tier.iloc[1400:1700], tier.iloc[1700:])