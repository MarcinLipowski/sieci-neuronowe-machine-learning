"""
utils/models.py — Implementacje modeli ML od zera.

Zawiera:
- CategoricalNaiveBayes: klasyfikator Bayesowski dla cech kategorycznych
- (później) DecisionTree
- (później) TwoLayerNetwork
"""
import numpy as np
from collections import Counter


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