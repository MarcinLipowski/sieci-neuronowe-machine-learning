# Sieci Neuronowe i Machine Learning
### Projekt zaliczeniowy | Python | PyTorch | 2024/2025

Implementacje algorytmów ML i sieci neuronowych **od zera** — bez gotowych
modeli z bibliotek. PyTorch używany wyłącznie do operacji macierzowych i
automatycznego różniczkowania w CNN.

---

## Struktura projektu

```
sieci-neuronowe-machine-learning/
├── 1_bayes/
│   └── naive_bayes.ipynb          # Naiwny Bayes + Drzewo decyzyjne (subscribers)
├── 2_two_layer_networks/
│   ├── xor.ipynb                  # XOR — sieć 2→4→1
│   ├── titanic.ipynb              # Titanic — klasyfikacja (acc: 77.65%)
│   └── boston_housing.ipynb       # Boston Housing — regresja (R²: 0.834)
├── 3_cnn_mnist/
│   └── mnist_cnn.ipynb            # CNN — MNIST (acc: 99.41%, RTX 4070 Ti)
├── 4_advanced_training/
│   └── cifar10.ipynb              # Głęboka CNN — CIFAR-10 (acc: 91.07%)
├── data/                          # Małe datasety (CSV/XLSX)
├── plots/                         # Wykresy generowane przez notebooki
├── sprawozdania/                  # Sprawozdania PDF
└── utils/
    ├── models.py                  # Implementacje od zera: NaiveBayes, DecisionTree, TwoLayerNetwork
    └── plotting.py                # Moduł wykresów (EDA, trening, macierz konfuzji)
```

---

## Wyniki

| Zadanie | Model | Metryka |
|---------|-------|---------|
| Naiwny Bayes (subscribers) | CategoricalNaiveBayes | Accuracy: 64.4% |
| Drzewo decyzyjne (subscribers) | DecisionTreeCategorical | F1: 0.25 |
| XOR | TwoLayerNetwork 2→4→1 | Accuracy: 100% |
| Titanic | TwoLayerNetwork 8→16→1 | Accuracy: 77.65% |
| Boston Housing | TwoLayerNetwork 13→32→1 | R²: 0.834, MAE: 2.27 tys. $ |
| MNIST | CNN (2 bloki conv) | Accuracy: 99.41% |
| CIFAR-10 | Głęboka CNN VGG-like | Accuracy: 91.07% |

---

## Instalacja

```bash
git clone https://github.com/MarcinLipowski/sieci-neuronowe-machine-learning.git
cd sieci-neuronowe-machine-learning

python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux/macOS

pip install -r requirements.txt
```

> **GPU (NVIDIA):** jeśli masz kartę NVIDIA, zainstaluj PyTorch z obsługą CUDA:
> ```
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu132
> ```
> Kod automatycznie wykrywa GPU przez `torch.cuda.is_available()`.

---

## Uruchomienie

```bash
# Uruchom serwer Jupyter (zalecane)
jupyter lab

# Lub otwórz w VS Code i wybierz kernel "Python (venv - projekt)"
code .
```

Notebooki uruchamiaj **od góry do dołu** (`Run All`). Datasety MNIST i CIFAR-10
pobierają się automatycznie przez `torchvision` (~180MB łącznie).

---

## Implementacje od zera (`utils/models.py`)

| Klasa | Opis |
|-------|------|
| `CategoricalNaiveBayes` | Naiwny Bayes z wygładzaniem Laplace'a, predykcja w log-przestrzeni |
| `DecisionTreeCategorical` | Drzewo dla cech kategorycznych, kryterium Gini |
| `DecisionTreeRegressor` | Drzewo dla regresji, kryterium wariancji (MSE) |
| `TwoLayerNetwork` | Sieć dwuwarstwowa, jawny forward/backward, SGD, tryby binary/regression |