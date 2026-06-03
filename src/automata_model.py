"""
Otomata tabanlı zaman serisi modelleme bileşenleri.

Dönüşüm zinciri: ham/PC1 seri → PAA (sayısal indirgeme) → SAX (sembolik pattern).
"""

from __future__ import annotations

from typing import Union

import numpy as np
import pandas as pd

# Ham/PCA girdisi ve PAA çıktısı için ortak sayısal tip
SeriesInput = Union[np.ndarray, pd.Series, list[float]]
NumericInput = Union[np.ndarray, pd.Series, list[float]]


class PAA:
    """
    Piecewise Aggregate Approximation (PAA).

    Zaman serisini ardışık, örtüşmeyen pencerelere böler; her pencerenin
    ortalaması indirgenmiş serinin bir elemanı olur.
    """

    def __init__(self, window_size: int) -> None:
        """
        Args:
            window_size: Her dilimin (pencere) uzunluğu. Dışarıdan verilir;
                         config.yaml bağlantısı sonraki commit'lerde yapılacaktır.
        """
        if window_size < 1:
            raise ValueError(f"window_size en az 1 olmalıdır; verilen: {window_size}")
        self.window_size = window_size

    def transform(self, series: SeriesInput) -> np.ndarray:
        """
        Tek boyutlu zaman serisine PAA uygular.

        Args:
            series: PCA sonrası PC1 (numpy 1D, pandas Series veya float listesi).

        Returns:
            Her pencere ortalamasından oluşan 1D numpy dizisi.
        """
        values = self._to_1d_array(series)
        if values.size == 0:
            raise ValueError("PAA için boş zaman serisi verilemez.")

        # Ardışık dilimler: [0:ws), [ws:2*ws), ... — son dilim kısa kalabilir
        segment_means: list[float] = []
        for start in range(0, values.size, self.window_size):
            segment = values[start : start + self.window_size]
            segment_means.append(float(np.mean(segment)))

        return np.asarray(segment_means, dtype=np.float64)

    @staticmethod
    def _to_1d_array(series: SeriesInput) -> np.ndarray:
        """Girdiyi tek boyutlu float numpy dizisine çevirir."""
        if isinstance(series, pd.Series):
            arr = series.to_numpy(dtype=np.float64, copy=False)
        else:
            arr = np.asarray(series, dtype=np.float64)

        if arr.ndim != 1:
            arr = arr.ravel()
        return arr


def apply_paa(series: SeriesInput, window_size: int) -> np.ndarray:
    """
    PAA dönüşümü için kısayol fonksiyonu.

    Args:
        series: Tek boyutlu zaman serisi (PC1 çıktısı).
        window_size: Pencere (dilim) boyutu.

    Returns:
        PAA ile indirgenmiş seri.
    """
    return PAA(window_size=window_size).transform(series)


def gaussian_breakpoints(alphabet_size: int) -> np.ndarray:
    """
    SAX için standart normal dağılımda eşit olasılıklı bölgelere ayrıran kesim noktaları.

    k sembollü alfabe için k-1 breakpoint üretilir: Φ⁻¹(i/k), i = 1 … k-1.

    Args:
        alphabet_size: Alfabe (sembol) sayısı.

    Returns:
        Artan sırada (k-1,) şeklinde breakpoint dizisi.
    """
    if alphabet_size < 2:
        raise ValueError(
            f"alphabet_size en az 2 olmalıdır; verilen: {alphabet_size}"
        )
    quantiles = np.arange(1, alphabet_size, dtype=np.float64) / alphabet_size
    # Φ⁻¹(p) = √2 · erfinv(2p − 1)
    return (np.sqrt(2.0) * np.erfinv(2.0 * quantiles - 1.0)).astype(np.float64)


class SAX:
    """
    Symbolic Aggregate Approximation (SAX).

    PAA ile indirgenmiş sayısal değerleri, Gaussian çan eğrisi breakpoint'lerine
    göre harf/sembol dizisine (pattern) dönüştürür.
    """

    def __init__(self, alphabet_size: int) -> None:
        """
        Args:
            alphabet_size: Sembol sayısı (örn. 3 → 'a', 'b', 'c'). Dışarıdan verilir;
                           config.yaml bağlantısı sonraki commit'lerde yapılacaktır.
        """
        if alphabet_size < 2:
            raise ValueError(
                f"alphabet_size en az 2 olmalıdır; verilen: {alphabet_size}"
            )
        if alphabet_size > 26:
            raise ValueError(
                f"alfabe en fazla 26 harf (a-z) desteklenir; verilen: {alphabet_size}"
            )
        self.alphabet_size = alphabet_size
        self.breakpoints: np.ndarray = gaussian_breakpoints(alphabet_size)
        self._symbols: tuple[str, ...] = tuple(
            chr(ord("a") + i) for i in range(alphabet_size)
        )

    def transform(self, paa_values: NumericInput) -> str:
        """
        PAA çıktısını sembolik pattern dizgesine dönüştürür.

        Değerler önce z-skoru ile normalize edilir; ardından breakpoint bölgelerine
        göre harf eşlemesi yapılır.

        Args:
            paa_values: PAA.transform çıktısı veya eşdeğer 1D sayısal dizi.

        Returns:
            Sembolik pattern (örn. "abcba"). Boş girdi için "".
        """
        return "".join(self.transform_symbols(paa_values))

    def transform_symbols(self, paa_values: NumericInput) -> list[str]:
        """
        PAA çıktısını sembol listesine dönüştürür (pattern / Levenshtein için).

        Args:
            paa_values: PAA çıktısı.

        Returns:
            Her segment için bir harf (örn. ['a', 'b', 'c']).
        """
        arr = self._to_1d_array(paa_values)
        if arr.size == 0:
            return []

        normalized = self._z_normalize(arr)
        # digitize: (-∞, b1], (b1, b2], … → 0 … alphabet_size-1
        indices = np.digitize(normalized, self.breakpoints)
        return [self._symbols[i] for i in indices]

    @staticmethod
    def _to_1d_array(values: NumericInput) -> np.ndarray:
        """Girdiyi tek boyutlu float numpy dizisine çevirir."""
        if isinstance(values, pd.Series):
            arr = values.to_numpy(dtype=np.float64, copy=False)
        else:
            arr = np.asarray(values, dtype=np.float64)
        return arr.ravel() if arr.ndim != 1 else arr

    @staticmethod
    def _z_normalize(arr: np.ndarray) -> np.ndarray:
        """Breakpoint'ler N(0,1) uzayında tanımlı olduğu için seriyi z-skorlar."""
        std = float(np.std(arr))
        if std == 0.0:
            return np.zeros_like(arr, dtype=np.float64)
        mean = float(np.mean(arr))
        return (arr - mean) / std


def apply_sax(paa_values: NumericInput, alphabet_size: int) -> str:
    """
    SAX dönüşümü için kısayol fonksiyonu.

    Args:
        paa_values: PAA ile indirgenmiş sayısal dizi.
        alphabet_size: Alfabe boyutu.

    Returns:
        Sembolik pattern dizgesi.
    """
    return SAX(alphabet_size=alphabet_size).transform(paa_values)
