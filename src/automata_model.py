"""
Otomata tabanlı zaman serisi modelleme bileşenleri.

Bu modül, ham/ön işlenmiş tek boyutlu seriyi sembolik temsile dönüştürmek için
kullanılan adımları içerir. İlk adım: PAA (Piecewise Aggregate Approximation).
"""

from __future__ import annotations

from typing import Union

import numpy as np
import pandas as pd

# Arkadaşın veri boru hattından gelecek tip: 1D numpy dizisi veya pandas Serisi
SeriesInput = Union[np.ndarray, pd.Series, list[float]]


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
