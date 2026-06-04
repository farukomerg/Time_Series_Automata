import os
import glob
import pandas as pd
import numpy as np
import yaml
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import GroupKFold

class DataPipeline:
    """Veri okuma, temizleme, bölme (split) ve ön işleme (scaler, pca) işlemlerini yürütür."""
    
    def __init__(self, config_path=None):
        if config_path is None:
            # Otomatik olarak bu dosyanın bulunduğu src klasöründeki config.yaml'ı bul
            config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
            
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)
            
        # Data klasörünü projenin ana dizinine göre ayarla
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        # config.yaml içindeki data_dir "../data/raw" şeklindedir
        self.data_dir = os.path.normpath(os.path.join(project_root, "src", self.config['project']['data_dir']))
        
    def load_skab(self):
        """SKAB verilerini birleştirir ve özellikleri/hedefleri döner."""
        skab_dir = os.path.join(self.data_dir, "SKAB")
        all_dfs = []
        for group in ["valve1", "valve2"]:
            folder_path = os.path.join(skab_dir, group)
            csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
            for file in csv_files:
                df = pd.read_csv(file, sep=';', index_col=False)
                df['source_group'] = group
                df['source_file'] = os.path.basename(file)
                all_dfs.append(df)
                
        full_df = pd.concat(all_dfs, ignore_index=True)
        # Zaman sıralamasını garanti altına alalım
        full_df = full_df.sort_values(by=['source_group', 'source_file', 'datetime']).reset_index(drop=True)
        
        # Model girdisi olmayacak kolonlar
        drop_cols = ['datetime', 'changepoint', 'source_group', 'source_file', 'anomaly']
        feature_cols = [c for c in full_df.columns if c not in drop_cols]
        target_col = 'anomaly'
        
        return full_df, feature_cols, target_col

    def load_batadal(self):
        """BATADAL verisini okur ve özellikleri/hedefleri döner."""
        batadal_path = os.path.join(self.data_dir, "BATADAL_dataset04.csv")
        # BATADAL virgülden sonra boşluk içerebilir, bu yüzden skipinitialspace=True
        df = pd.read_csv(batadal_path, skipinitialspace=True)
        
        # Model girdisi olmayacak kolonlar
        drop_cols = ['DATETIME', 'ATT_FLAG']
        feature_cols = [c for c in df.columns if c not in drop_cols]
        target_col = 'ATT_FLAG'
        
        return df, feature_cols, target_col

    def get_skab_splits(self, df, n_splits=5):
        """SKAB için dosya bazlı GroupKFold jeneratörü döner."""
        gkf = GroupKFold(n_splits=n_splits)
        groups = df['source_file'].values
        # test_idx aslında o fold için test, train_idx ise eğitim+val içindir.
        return gkf.split(df, groups=groups)

    def split_batadal(self, df):
        """BATADAL için %60 Train, %20 Val, %20 Test zaman sıralı bölme yapar."""
        n = len(df)
        train_end = int(n * 0.6)
        val_end = int(n * 0.8)
        
        train_idx = np.arange(0, train_end)
        val_idx = np.arange(train_end, val_end)
        test_idx = np.arange(val_end, n)
        
        return train_idx, val_idx, test_idx

    def preprocess_features(self, df, train_idx, val_idx, test_idx, feature_cols):
        """Train verisine fit edilerek Scaler ve PCA uygular (Data Leakage önleme kuralı)."""
        X_train = df.loc[train_idx, feature_cols].values
        X_test = df.loc[test_idx, feature_cols].values
        
        if val_idx is not None and len(val_idx) > 0:
            X_val = df.loc[val_idx, feature_cols].values
        else:
            X_val = np.array([])
            
        # 1. Normalizasyon (Deep Learning için)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        X_val_scaled = scaler.transform(X_val) if len(X_val) > 0 else X_val
        
        # 2. PCA ile Tek Boyuta İndirme (Otomata için)
        pca = PCA(n_components=1)
        X_train_pca = pca.fit_transform(X_train_scaled)
        X_test_pca = pca.transform(X_test_scaled)
        X_val_pca = pca.transform(X_val_scaled) if len(X_val_scaled) > 0 else X_val_scaled
        
        return {
            'deep_learning': {
                'train': X_train_scaled,
                'val': X_val_scaled,
                'test': X_test_scaled,
                'scaler': scaler
            },
            'automata': {
                'train': X_train_pca.flatten(),
                'val': X_val_pca.flatten() if len(X_val_pca) > 0 else np.array([]),
                'test': X_test_pca.flatten(),
                'pca': pca
            }
        }
