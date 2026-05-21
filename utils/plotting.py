# plik: plotting.py — importuj w każdym notebooku

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import numpy as np
import os
from pathlib import Path

# ─── Styl globalny ───────────────────────────────────────────────────
plt.rcParams.update({
    'figure.dpi': 150,
    'figure.figsize': (10, 6),
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'font.size': 11,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
})
PALETTE = ['#2196F3', '#F44336', '#4CAF50', '#FF9800', '#9C27B0']

PLOT_DIR = Path("plots")
PLOT_DIR.mkdir(exist_ok=True)

def save(fig, name):
    """Zapisz wykres do katalogu plots/"""
    path = PLOT_DIR / f"{name}.png"
    fig.savefig(path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"  ✓ Saved: {path}")
    return path


# ─── EDA ─────────────────────────────────────────────────────────────

def plot_class_distribution(y, class_names=None, title="Rozkład klas", save_name=None):
    """Wykres słupkowy rozkładu klas."""
    unique, counts = np.unique(y, return_counts=True)
    labels = class_names if class_names else [str(u) for u in unique]
    
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(labels, counts, color=PALETTE[:len(unique)], edgecolor='white', linewidth=0.8)
    
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{count}\n({count/len(y)*100:.1f}%)', ha='center', va='bottom', fontsize=10)
    
    ax.set_title(title)
    ax.set_ylabel("Liczba próbek")
    fig.tight_layout()
    
    if save_name:
        save(fig, save_name)
    return fig


def plot_histograms(df, columns, n_cols=3, title="Rozkłady cech", save_name=None):
    """Histogramy dla wielu cech numerycznych."""
    n_rows = (len(columns) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]
    
    for i, col in enumerate(columns):
        ax = axes[i]
        ax.hist(df[col].dropna(), bins=30, color=PALETTE[0], edgecolor='white', alpha=0.8)
        ax.set_title(col)
        ax.set_xlabel("Wartość")
        ax.set_ylabel("Liczba")
    
    # Ukryj puste wykresy
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    
    fig.suptitle(title, fontsize=16, y=1.02)
    fig.tight_layout()
    
    if save_name:
        save(fig, save_name)
    return fig


def plot_correlation_heatmap(df, columns=None, title="Macierz korelacji", save_name=None):
    """Heatmapa korelacji Pearsona."""
    if columns:
        df = df[columns]
    
    corr = df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))  # ukryj górny trójkąt
    
    fig, ax = plt.subplots(figsize=(max(8, len(corr) * 0.7), max(6, len(corr) * 0.6)))
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
                center=0, vmin=-1, vmax=1, ax=ax,
                annot_kws={'size': 9}, linewidths=0.5)
    ax.set_title(title)
    fig.tight_layout()
    
    if save_name:
        save(fig, save_name)
    return fig


def plot_missing_values(df, title="Braki danych", save_name=None):
    """Wykres braków danych."""
    missing = df.isnull().sum().sort_values(ascending=True)
    missing = missing[missing > 0]
    
    if len(missing) == 0:
        print("Brak brakujących wartości!")
        return None
    
    fig, ax = plt.subplots(figsize=(8, max(4, len(missing) * 0.4)))
    bars = ax.barh(missing.index, missing.values / len(df) * 100, color=PALETTE[1], alpha=0.8)
    
    for bar, val in zip(bars, missing.values):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                f'{val} ({val/len(df)*100:.1f}%)', va='center', fontsize=10)
    
    ax.set_xlabel("% brakujących wartości")
    ax.set_title(title)
    fig.tight_layout()
    
    if save_name:
        save(fig, save_name)
    return fig


def plot_boxplots(df, columns, by=None, title="Boxploty", save_name=None):
    """Boxploty dla wykrywania outlierów."""
    n_cols = min(3, len(columns))
    n_rows = (len(columns) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]
    
    for i, col in enumerate(columns):
        ax = axes[i]
        if by:
            groups = [df[df[by] == v][col].dropna() for v in df[by].unique()]
            ax.boxplot(groups, labels=df[by].unique(), patch_artist=True)
        else:
            ax.boxplot(df[col].dropna(), patch_artist=True,
                       boxprops=dict(facecolor=PALETTE[0], alpha=0.7))
        ax.set_title(col)
    
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    
    fig.suptitle(title, fontsize=16, y=1.02)
    fig.tight_layout()
    
    if save_name:
        save(fig, save_name)
    return fig


# ─── Trening sieci ────────────────────────────────────────────────────

def plot_training_history(history, title="Historia uczenia", save_name=None):
    """
    Wykres błędów w obu warstwach + accuracy.
    Oczekuje słownika z kluczami: loss, loss_layer1, loss_layer2, accuracy
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    epochs = range(1, len(history['loss']) + 1)
    
    # ── Strata globalna
    axes[0].plot(epochs, history['loss'], color=PALETTE[0], linewidth=2)
    axes[0].set_title("Strata (globalna)")
    axes[0].set_xlabel("Epoka")
    axes[0].set_ylabel("Loss")
    axes[0].set_yscale('log')
    
    # ── Gradienty (błędy) warstw
    if 'loss_layer1' in history and 'loss_layer2' in history:
        axes[1].plot(epochs, history['loss_layer1'], label='Warstwa 1', color=PALETTE[0], linewidth=2)
        axes[1].plot(epochs, history['loss_layer2'], label='Warstwa 2', color=PALETTE[1], linewidth=2)
        axes[1].set_title("Norma gradientu w warstwach")
        axes[1].set_xlabel("Epoka")
        axes[1].set_ylabel("||∇W||")
        axes[1].legend()
        axes[1].set_yscale('log')
    
    # ── Accuracy
    if 'accuracy' in history:
        axes[2].plot(epochs, history['accuracy'], color=PALETTE[2], linewidth=2)
        if 'val_accuracy' in history:
            axes[2].plot(epochs, history['val_accuracy'], '--', color=PALETTE[1], linewidth=2, label='Val')
            axes[2].legend()
        axes[2].set_title("Dokładność")
        axes[2].set_xlabel("Epoka")
        axes[2].set_ylabel("Accuracy")
        axes[2].set_ylim(0, 1)
    
    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    
    if save_name:
        save(fig, save_name)
    return fig


def plot_confusion_matrix(y_true, y_pred, class_names=None, 
                           title="Macierz pomyłek", save_name=None):
    """Macierz konfuzji jako heatmapa."""
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)  # normalizacja po wierszach
    
    n = len(cm)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    for ax, data, fmt, subtitle in zip(axes, [cm, cm_norm], ['d', '.2f'],
                                        ['Liczby bezwzględne', 'Znormalizowana']):
        sns.heatmap(data, annot=True, fmt=fmt, cmap='Blues', ax=ax,
                    xticklabels=class_names or range(n),
                    yticklabels=class_names or range(n),
                    linewidths=0.5)
        ax.set_xlabel("Predykcja")
        ax.set_ylabel("Rzeczywistość")
        ax.set_title(subtitle)
    
    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    
    if save_name:
        save(fig, save_name)
    return fig


def plot_optimizer_comparison(histories, labels, metric='loss',
                               title="Porównanie optymalizatorów", save_name=None):
    """Porównanie krzywych uczenia dla różnych optymalizatorów."""
    fig, ax = plt.subplots(figsize=(10, 5))
    
    for history, label, color in zip(histories, labels, PALETTE):
        epochs = range(1, len(history[metric]) + 1)
        ax.plot(epochs, history[metric], label=label, color=color, linewidth=2)
    
    ax.set_title(title)
    ax.set_xlabel("Epoka")
    ax.set_ylabel(metric.capitalize())
    ax.legend()
    if metric == 'loss':
        ax.set_yscale('log')
    
    fig.tight_layout()
    if save_name:
        save(fig, save_name)
    return fig


def plot_decision_boundary(model_fn, X, y, title="Granica decyzyjna", save_name=None):
    """
    Wizualizacja granicy decyzyjnej (dla 2D danych).
    model_fn: funkcja przyjmująca (N, 2) array, zwracająca predykcje
    """
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                          np.linspace(y_min, y_max, 200))
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = model_fn(grid).reshape(xx.shape)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.contourf(xx, yy, Z, alpha=0.3, cmap='RdBu')
    scatter = ax.scatter(X[:, 0], X[:, 1], c=y, cmap='RdBu',
                          edgecolors='k', linewidth=0.5, s=50)
    ax.set_title(title)
    plt.colorbar(scatter, ax=ax)
    fig.tight_layout()
    
    if save_name:
        save(fig, save_name)
    return fig