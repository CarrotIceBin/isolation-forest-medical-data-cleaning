"""
离群值检测模块 - 负责训练和使用异常检测模型
"""
import numpy as np
import time
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix


class OutlierDetectionSystem:
    """离群值检测系统"""
    
    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
        self.X_scaled = None
    
    def train_models(self, X, contamination=0.1):
        """训练多种异常检测模型"""
        print("\n" + "="*60)
        print("2. 训练多种异常检测模型")
        print("="*60)
        print(f"参与训练的样本数: {len(X)}")
        
        self.X_scaled = self.scaler.fit_transform(X)
        print(f"✓ 数据标准化完成")
        
        models_info = {}
        
        # 1. 孤立森林
        print("\n训练孤立森林模型...")
        try:
            start_time = time.time()
            if_model = IsolationForest(
                n_estimators=500,
                max_samples=len(X),
                contamination=contamination,
                random_state=42,
                n_jobs=-1
            )
            if_predictions = if_model.fit_predict(self.X_scaled)
            if_scores = if_model.decision_function(self.X_scaled)
            if_time = time.time() - start_time
            
            models_info['Isolation Forest'] = {
                'model': if_model,
                'predictions': if_predictions,
                'scores': if_scores,
                'time': if_time,
                'trained_samples': len(X)
            }
            print(f"✓ 孤立森林训练完成，时间: {if_time:.2f}秒")
        except Exception as e:
            print(f"✗ 孤立森林训练失败: {e}")
        
        # 2. 局部离群因子
        print("训练LOF模型...")
        try:
            start_time = time.time()
            lof_model = LocalOutlierFactor(
                n_neighbors=min(20, len(X)-1),
                contamination=contamination,
                novelty=False,
                n_jobs=-1
            )
            lof_predictions = lof_model.fit_predict(self.X_scaled)
            lof_scores = -lof_model.negative_outlier_factor_
            lof_time = time.time() - start_time
            
            models_info['LOF'] = {
                'model': lof_model,
                'predictions': lof_predictions,
                'scores': lof_scores,
                'time': lof_time,
                'trained_samples': len(X)
            }
            print(f"✓ LOF训练完成，时间: {lof_time:.2f}秒")
        except Exception as e:
            print(f"✗ LOF训练失败: {e}")
        
        # 3. One-Class SVM
        print("训练One-Class SVM模型...")
        try:
            start_time = time.time()
            gamma = 1 / (X.shape[1] * self.X_scaled.var()) if self.X_scaled.var() > 0 else 'scale'
            ocsvm_model = OneClassSVM(nu=contamination, kernel='rbf', gamma=gamma, tol=1e-3)
            ocsvm_predictions = ocsvm_model.fit_predict(self.X_scaled)
            ocsvm_scores = ocsvm_model.decision_function(self.X_scaled)
            ocsvm_time = time.time() - start_time
            
            models_info['One-Class SVM'] = {
                'model': ocsvm_model,
                'predictions': ocsvm_predictions,
                'scores': ocsvm_scores,
                'time': ocsvm_time,
                'trained_samples': len(X)
            }
            print(f"✓ One-Class SVM训练完成，时间: {ocsvm_time:.2f}秒")
        except Exception as e:
            print(f"✗ One-Class SVM训练失败: {e}")
        
        self.models = models_info
        return self.models
    
    def evaluate_models(self, X, true_labels=None):
        """评估模型性能"""
        print("\n" + "="*60)
        print("3. 模型性能评估")
        print("="*60)
        
        evaluation_results = {}
        
        for model_name, model_data in self.models.items():
            predictions = model_data['predictions']
            scores = model_data['scores']
            trained_samples = model_data['trained_samples']
            
            outlier_count = np.sum(predictions == -1)
            outlier_ratio = outlier_count / trained_samples
            
            result = {
                'trained_samples': trained_samples,
                'outlier_count': int(outlier_count),
                'outlier_ratio': float(outlier_ratio),
                'training_time': float(model_data['time']),
                'score_statistics': {
                    'mean': float(np.mean(scores)),
                    'std': float(np.std(scores)),
                    'min': float(np.min(scores)),
                    'max': float(np.max(scores))
                }
            }
            
            if true_labels is not None and len(true_labels) == trained_samples:
                pred_binary = np.where(predictions == 1, 0, 1)
                try:
                    result['accuracy'] = float(accuracy_score(true_labels, pred_binary))
                    result['precision'] = float(precision_score(true_labels, pred_binary, zero_division=0))
                    result['recall'] = float(recall_score(true_labels, pred_binary, zero_division=0))
                    result['f1_score'] = float(f1_score(true_labels, pred_binary, zero_division=0))
                    if len(np.unique(true_labels)) > 1:
                        result['roc_auc'] = float(roc_auc_score(true_labels, scores))
                    cm = confusion_matrix(true_labels, pred_binary)
                    result['confusion_matrix'] = cm.tolist()
                except Exception as e:
                    print(f"评估{model_name}时出错: {e}")
            
            evaluation_results[model_name] = result
            print(f"\n{model_name}: 检测到{outlier_count}个离群值 ({outlier_ratio:.1%})")
        
        return evaluation_results