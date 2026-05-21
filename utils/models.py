"""
utils/models.py — Implementacje modeli ML od zera.

Zawiera:
- CategoricalNaiveBayes: klasyfikator Bayesowski dla cech kategorycznych
- (później) DecisionTree
- (później) TwoLayerNetwork
"""
import numpy as np
from collections import Counter
import torch

class CategoricalNaiveBayes:
    """
    Naiwny Bayes dla cech kategorycznych (dyskretnych).
    
    Algorytm:
    1. Trenowanie: dla każdej klasy oblicz:
       - prior: P(c) = liczba_próbek_klasy_c / liczba_wszystkich_próbek
       - likelihoods: P(cecha=v | c) z wygładzaniem Laplace'a
    
    2. Predykcja: dla nowej próbki x policz dla każdej klasy:
       log P(c|x) ∝ log P(c) + Σ log P(x_i | c)
       Wybierz klasę z najwyższym wynikiem.
    """
    
    def __init__(self, alpha=1.0):
        self.alpha = alpha          # wygładzanie Laplace'a
        self.classes = None         # lista unikalnych klas (np. [0, 1])
        self.priors = {}            # P(c) — słownik klasa → prior
        self.likelihoods = {}       # P(cecha=v | c) — zagnieżdżone słowniki
        self.feature_values = {}    # zapamiętujemy wszystkie wartości każdej cechy
        self.feature_names = None   # nazwy kolumn
    
    def fit(self, X, y):
        """
        Trenowanie modelu.
        X: DataFrame z cechami kategorycznymi
        y: array z etykietami klas
        """
        self.feature_names = list(X.columns)
        self.classes = np.unique(y)
        n_samples = len(y)
        
        # Zapamiętaj wszystkie unikalne wartości każdej cechy
        # (potrzebne do wygładzania Laplace'a — znamy mianownik)
        for feature in self.feature_names:
            self.feature_values[feature] = X[feature].unique().tolist()
        
        # Dla każdej klasy oblicz prior i likelihood dla każdej cechy
        for c in self.classes:
            # ── Prior: P(c) ──
            mask = (y == c)                     # logiczna maska próbek tej klasy
            n_class = mask.sum()                 # ile próbek w klasie c
            self.priors[c] = n_class / n_samples
            
            # ── Likelihoods: P(cecha=v | c) dla każdej cechy i wartości ──
            self.likelihoods[c] = {}
            X_class = X[mask]                    # tylko próbki klasy c
            
            for feature in self.feature_names:
                n_unique = len(self.feature_values[feature])  # liczba unikalnych wartości tej cechy
                value_counts = X_class[feature].value_counts().to_dict()
                
                self.likelihoods[c][feature] = {}
                for value in self.feature_values[feature]:
                    count = value_counts.get(value, 0)  # 0 jeśli wartość nie występuje
                    # Wygładzanie Laplace'a
                    self.likelihoods[c][feature][value] = (count + self.alpha) / (n_class + self.alpha * n_unique)
        
        return self
    
    def predict_log_proba(self, X):
        """
        Zwraca log P(c|x) dla każdej próbki i klasy.
        Używamy logarytmów żeby uniknąć underflow (mnożenia małych liczb).
        """
        results = []
        for _, row in X.iterrows():
            log_probs = {}
            for c in self.classes:
                # log P(c)
                log_prob = np.log(self.priors[c])
                # + Σ log P(cecha_i = wartość_i | c)
                for feature in self.feature_names:
                    value = row[feature]
                    # Jeśli wartość nieznana (np. nie była w treningu), używamy wygładzania
                    if value in self.likelihoods[c][feature]:
                        log_prob += np.log(self.likelihoods[c][feature][value])
                    else:
                        # Wartość nigdy nie widziana — przypisujemy bardzo małe prawdopodobieństwo
                        n_unique = len(self.feature_values[feature])
                        n_class_total = sum(self.likelihoods[c][feature].values()) * (1 + self.alpha * n_unique)
                        log_prob += np.log(self.alpha / (n_class_total + self.alpha * n_unique))
                log_probs[c] = log_prob
            results.append(log_probs)
        return results
    
    def predict(self, X):
        """Predykcja klasy — wybieramy klasę z największym log P(c|x)."""
        log_probs = self.predict_log_proba(X)
        return np.array([max(lp, key=lp.get) for lp in log_probs])
    
    def predict_proba(self, X):
        """
        Zwraca prawdopodobieństwa klas (znormalizowane).
        Używamy log-sum-exp trick dla stabilności.
        """
        log_probs = self.predict_log_proba(X)
        probas = []
        for lp in log_probs:
            # log-sum-exp trick
            log_values = np.array([lp[c] for c in self.classes])
            log_values_shifted = log_values - log_values.max()  # stabilność numeryczna
            exp_values = np.exp(log_values_shifted)
            probas.append(exp_values / exp_values.sum())
        return np.array(probas)
    
    def score(self, X, y):
        """Accuracy na zbiorze X, y."""
        return np.mean(self.predict(X) == y)
    

class Node:
    """Pojedynczy węzeł drzewa decyzyjnego."""
    
    def __init__(self, feature=None, branches=None, prediction=None, fallback=None):
        self.feature = feature           # nazwa cechy do podziału (tylko węzły wewnętrzne)
        self.branches = branches or {}   # słownik: {wartość_cechy: Node}
        self.prediction = prediction     # klasa (TYLKO w liściu, None w węzłach wewnętrznych)
        self.fallback = fallback         # większościowa klasa w węźle (dla nieznanych wartości)
    
    def is_leaf(self):
        return len(self.branches) == 0


class DecisionTreeCategorical:
    """
    Drzewo decyzyjne dla cech kategorycznych.
    
    Używa miary Gini Impurity do wyboru najlepszej cechy do podziału.
    Dla cech kategorycznych każdy podział tworzy tyle gałęzi ile jest unikalnych wartości.
    
    Warunki stopu:
    - osiągnięto max_depth
    - liczba próbek < min_samples_split  
    - wszystkie próbki w węźle mają tę samą klasę
    - skończyły się cechy do podziału
    - podział nie daje information gain (gain <= 0)
    """
    
    def __init__(self, max_depth=10, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None
        self.feature_names = None
    
    def _gini(self, y):
        """Gini Impurity: 1 - Σ p_i²"""
        if len(y) == 0:
            return 0
        counts = Counter(y)
        n = len(y)
        return 1 - sum((c / n) ** 2 for c in counts.values())
    
    def _weighted_gini_after_split(self, X, y, feature):
        """Średnia ważona Gini po podziale po danej cesze."""
        n = len(y)
        weighted_gini = 0
        for value in X[feature].unique():
            mask = (X[feature] == value).values
            subset_y = y[mask]
            weighted_gini += (len(subset_y) / n) * self._gini(subset_y)
        return weighted_gini
    
    def _best_feature(self, X, y, available_features):
        """Wybierz cechę dającą największy spadek Gini (information gain)."""
        parent_gini = self._gini(y)
        best_gain = -float('inf')
        best_feature = None
        
        for feature in available_features:
            child_gini = self._weighted_gini_after_split(X, y, feature)
            gain = parent_gini - child_gini
            if gain > best_gain:
                best_gain = gain
                best_feature = feature
        
        return best_feature, best_gain
    
    def _build_tree(self, X, y, available_features, depth):
        """Rekurencyjne budowanie drzewa."""
        majority_class = Counter(y).most_common(1)[0][0]
        
        # Warunki stopu — utwórz liść
        if (depth >= self.max_depth 
            or len(y) < self.min_samples_split 
            or len(np.unique(y)) == 1
            or len(available_features) == 0):
            return Node(prediction=majority_class)
        
        # Znajdź najlepszą cechę do podziału
        best_feature, gain = self._best_feature(X, y, available_features)
        
        # Jeśli podział nie poprawia jakości — liść
        if gain <= 1e-10:
            return Node(prediction=majority_class)
        
        # Tworzymy węzeł wewnętrzny (prediction=None, fallback=majority)
        node = Node(feature=best_feature, fallback=majority_class)
        remaining_features = [f for f in available_features if f != best_feature]
        
        for value in X[best_feature].unique():
            mask = (X[best_feature] == value).values
            subset_X = X[mask]
            subset_y = y[mask]
            
            if len(subset_y) == 0:
                node.branches[value] = Node(prediction=majority_class)
            else:
                node.branches[value] = self._build_tree(
                    subset_X, subset_y, remaining_features, depth + 1
                )
        
        return node
    
    def fit(self, X, y):
        self.feature_names = list(X.columns)
        self.root = self._build_tree(X, y, self.feature_names, depth=0)
        return self
    
    def _predict_one(self, row, node):
        """Predykcja dla jednej próbki — przejście drzewem od korzenia do liścia."""
        if node.is_leaf():
            return node.prediction
        
        value = row[node.feature]
        if value in node.branches:
            return self._predict_one(row, node.branches[value])
        else:
            # Wartość nieznana w treningu — użyj fallback (większościowej klasy tego węzła)
            return node.fallback
    
    def predict(self, X):
        return np.array([self._predict_one(row, self.root) for _, row in X.iterrows()])
    
    def score(self, X, y):
        return np.mean(self.predict(X) == y)
    
    def print_tree(self, node=None, indent=""):
        """Wyświetlenie drzewa w czytelnej formie tekstowej."""
        if node is None:
            node = self.root
        
        if node.is_leaf():
            print(f"{indent}→ Predykcja: {node.prediction}")
            return
        
        print(f"{indent}[{node.feature}?]  (fallback: {node.fallback})")
        for value, child in node.branches.items():
            print(f"{indent}  = {value}:")
            self.print_tree(child, indent + "    ")


class TwoLayerNetwork:
    """
    Sieć neuronowa z jedną warstwą ukrytą, zaimplementowana od zera w PyTorch.

    Architektura: wejście → [warstwa ukryta: ReLU] → [wyjście: Sigmoid/Linear]

    Wszystkie obliczenia forward i backward są jawne — bez nn.Module.
    PyTorch używany tylko do operacji macierzowych i przechowywania tensorów.

    Tryby pracy (mode):
    - 'binary'      → wyjście Sigmoid + Binary Cross-Entropy  (klasyfikacja 0/1)
    - 'regression'  → wyjście liniowe + MSE                   (regresja)
    """

    def __init__(self, n_input, n_hidden, n_output=1, lr=0.01, mode='binary'):
        self.lr = lr
        self.mode = mode

        # Xavier initialization — skaluje wagi tak żeby wariancja aktywacji
        # była stabilna przez warstwy (zapobiega zanikającemu/eksplodującemu gradientowi)
        self.W1 = torch.randn(n_hidden, n_input)  * np.sqrt(2.0 / n_input)
        self.b1 = torch.zeros(n_hidden)
        self.W2 = torch.randn(n_output, n_hidden) * np.sqrt(2.0 / n_hidden)
        self.b2 = torch.zeros(n_output)

        # Historia błędów — do wykresów
        self.history = {
            'loss':           [],
            'grad_norm_W1':   [],   # norma gradientu wag warstwy 1
            'grad_norm_W2':   [],   # norma gradientu wag warstwy 2
            'accuracy':       []    # tylko dla trybu binary
        }

    # ── Funkcje aktywacji ─────────────────────────────────────────────

    def _relu(self, z):
        return torch.clamp(z, min=0)

    def _relu_derivative(self, z):
        return (z > 0).float()

    def _sigmoid(self, z):
        return 1 / (1 + torch.exp(-torch.clamp(z, -50, 50)))  # clamp → stabilność

    # ── Forward pass ──────────────────────────────────────────────────

    def forward(self, X):
        """
        Propagacja sygnału od wejścia do wyjścia.
        Zapamiętujemy z1, a1, z2 — potrzebne w backpropie.

        X: (batch_size, n_input)
        zwraca: (batch_size, n_output)
        """
        self.X_cache = X

        self.z1 = X @ self.W1.T + self.b1          # (batch, n_hidden)
        self.a1 = self._relu(self.z1)               # (batch, n_hidden)

        self.z2 = self.a1 @ self.W2.T + self.b2    # (batch, n_output)

        if self.mode == 'binary':
            self.a2 = self._sigmoid(self.z2)        # (batch, n_output) ∈ (0,1)
        else:
            self.a2 = self.z2                       # liniowe dla regresji

        return self.a2

    # ── Funkcje straty ────────────────────────────────────────────────

    def _bce_loss(self, y_pred, y_true, eps=1e-9):
        """Binary Cross-Entropy"""
        y_pred = torch.clamp(y_pred, eps, 1 - eps)
        return -torch.mean(y_true * torch.log(y_pred) + (1 - y_true) * torch.log(1 - y_pred))

    def _mse_loss(self, y_pred, y_true):
        """Mean Squared Error"""
        return torch.mean((y_pred - y_true) ** 2)

    def compute_loss(self, y_pred, y_true):
        if self.mode == 'binary':
            return self._bce_loss(y_pred, y_true)
        return self._mse_loss(y_pred, y_true)

    # ── Backward pass ─────────────────────────────────────────────────

    def backward(self, y_true):
        """
        Backpropagation — obliczenie gradientów przez regułę łańcuchową.

        Dla BCE + Sigmoid gradient upraszcza się do: δ2 = (ŷ - y) / N
        Dla MSE + Linear:                             δ2 = 2*(ŷ - y) / N
        """
        n = self.X_cache.shape[0]

        # ── Gradient warstwy wyjściowej ──
        if self.mode == 'binary':
            delta2 = (self.a2 - y_true) / n          # (batch, n_output)
        else:
            delta2 = 2 * (self.a2 - y_true) / n

        dW2 = delta2.T @ self.a1                      # (n_output, n_hidden)
        db2 = delta2.sum(dim=0)                       # (n_output,)

        # ── Propagacja wstecz przez warstwę ukrytą ──
        delta1 = (delta2 @ self.W2) * self._relu_derivative(self.z1)  # (batch, n_hidden)

        dW1 = delta1.T @ self.X_cache                 # (n_hidden, n_input)
        db1 = delta1.sum(dim=0)                       # (n_hidden,)

        # ── Aktualizacja wag (SGD) ──
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2

        return dW1.norm().item(), dW2.norm().item()

    # ── Pętla treningowa ──────────────────────────────────────────────

    def train_network(self, X_train, y_train, epochs=1000, batch_size=32, verbose=True):
        """
        Mini-batch SGD.

        X_train, y_train: numpy arrays lub torch tensors
        """
        X = torch.FloatTensor(np.array(X_train))
        y = torch.FloatTensor(np.array(y_train))


        # y musi być (N, 1) dla operacji macierzowych
        if y.dim() == 1:
            y = y.unsqueeze(1)

        n = X.shape[0]

        for epoch in range(epochs):
            # Losowa kolejność mini-batchy
            perm = torch.randperm(n)
            epoch_loss, epoch_g1, epoch_g2 = 0.0, 0.0, 0.0
            n_batches = 0

            for start in range(0, n, batch_size):
                idx = perm[start:start + batch_size]
                X_batch, y_batch = X[idx], y[idx]

                y_pred  = self.forward(X_batch)
                loss    = self.compute_loss(y_pred, y_batch)
                # Wykryj NaN zanim zepsuje wagi
                if torch.isnan(loss):
                    print(f"   NaN w epoce {epoch}, batch {start//batch_size}!")
                    print(f"   X_batch stats: min={X_batch.min():.3f}, max={X_batch.max():.3f}")
                    print(f"   y_pred stats:  min={y_pred.min():.3f}, max={y_pred.max():.3f}")
                    break
                g1, g2  = self.backward(y_batch)

                epoch_loss += loss.item()
                epoch_g1   += g1
                epoch_g2   += g2
                n_batches  += 1

            # ── Zapis historii ──
            avg_loss = epoch_loss / n_batches
            self.history['loss'].append(avg_loss)
            self.history['grad_norm_W1'].append(epoch_g1 / n_batches)
            self.history['grad_norm_W2'].append(epoch_g2 / n_batches)

            if self.mode == 'binary':
                with torch.no_grad():
                    preds = self.forward(X)
                    acc = ((preds > 0.5).float() == y).float().mean().item()
                    self.history['accuracy'].append(acc)

            if verbose and epoch % max(1, epochs // 10) == 0:
                acc_str = f" | Acc: {self.history['accuracy'][-1]:.4f}" if self.mode == 'binary' else ""
                print(f"Epoch {epoch:5d}/{epochs} | Loss: {avg_loss:.6f}{acc_str}")

        return self.history

    # ── Predykcja ─────────────────────────────────────────────────────

    def predict(self, X):
        X_t = torch.FloatTensor(np.array(X))
        with torch.no_grad():
            out = self.forward(X_t)
        if self.mode == 'binary':
            return (out > 0.5).float().numpy().flatten()
        return out.numpy().flatten()

    def predict_proba(self, X):
        """Tylko dla trybu binary — zwraca prawdopodobieństwa."""
        X_t = torch.FloatTensor(np.array(X))
        with torch.no_grad():
            return self.forward(X_t).numpy().flatten()