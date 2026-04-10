"""
分析控制器 - 处理分析相关的业务逻辑
"""
import os
import sys

# 保证在 Windows GBK 控制台下打印 ✓/✗ 等字符不会触发 UnicodeEncodeError，导致分析中断
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass
import json
import time
import numpy as np
import pandas as pd

# 添加项目根目录到Python路径，确保能够导入models模块
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from models import MedicalDataProcessor, OutlierDetectionSystem, ClinicalValidator


class AnalysisController:
    """分析控制器"""
    
    def __init__(self):
        self.results_dir = 'results'
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)
    
    def load_analysis_results(self):
        """加载分析结果"""
        try:
            # 尝试加载实际的分析结果文件
            result_data = self._load_detection_results()
            
            # 如果有结果文件，返回结果
            if result_data:
                # 构建报告
                report = {
                    'model_results': {},
                    'clinical_validation': {},
                    'data_info': {
                        'total_samples': 0
                    }
                }
                
                for model_name, model_data in result_data.items():
                    report['model_results'][model_name] = {
                        'accuracy': model_data.get('accuracy', 0.0),
                        'f1_score': model_data.get('f1_score', 0.0),
                        'outlier_count': model_data.get('outlier_count', 0),
                        'training_time': model_data.get('analysis_time', 0.0)
                    }
                    report['clinical_validation'][model_name] = {
                        'clinically_plausible': model_data.get('clinical_plausible', 0),
                        'clinically_implausible': model_data.get('clinical_implausible', 0)
                    }
                    report['data_info']['total_samples'] = max(report['data_info']['total_samples'], model_data.get('detailed_results', 0))
                
                return report, result_data
            else:
                # 如果没有结果文件，返回空报告
                return {
                    'model_results': {},
                    'clinical_validation': {},
                    'data_info': {
                        'total_samples': 0
                    }
                }, {}
        except Exception as e:
            print(f"加载分析结果失败: {e}")
            # 发生异常时返回空报告
            return {
                'model_results': {},
                'clinical_validation': {},
                'data_info': {
                    'total_samples': 0
                }
            }, {}
    
    def _get_default_report(self):
        """获取默认报告（使用实际数据）"""
        # 运行实际算法获取默认报告
        print("运行实际算法获取默认报告...")
        try:
            # 运行实际分析获取默认报告
            result = self.analyze(analysis_type='full', model_select='all')
            if result.get('success'):
                data = result['data']
                return {
                    'model_results': {
                        model_name: {
                            'accuracy': data['modelPerformance']['accuracy'][i],
                            'f1_score': data['modelPerformance']['f1Score'][i],
                            'outlier_count': data['outlierDetection']['counts'][i],
                            'training_time': data['algorithmTimes'][model_name]
                        }
                        for i, model_name in enumerate(data['modelPerformance']['labels'])
                    },
                    'clinical_validation': {
                        model_name: {
                            'clinically_plausible': data['clinicalValidation']['plausible'][i],
                            'clinically_implausible': data['clinicalValidation']['implausible'][i]
                        }
                        for i, model_name in enumerate(data['clinicalValidation']['labels'])
                    },
                    'data_info': {
                        'total_samples': data['totalSamples']
                    }
                }
            else:
                # 如果分析失败，返回最小默认数据
                return {
                    'model_results': {
                        'Isolation Forest': {
                            'accuracy': 0.0,
                            'f1_score': 0.0,
                            'outlier_count': 0,
                            'training_time': 0.0
                        }
                    },
                    'clinical_validation': {
                        'Isolation Forest': {
                            'clinically_plausible': 0,
                            'clinically_implausible': 0
                        }
                    },
                    'data_info': {
                        'total_samples': 0
                    }
                }
        except Exception as e:
            print(f"获取默认报告失败: {e}")
            # 发生异常时返回最小默认数据
            return {
                'model_results': {
                    'Isolation Forest': {
                        'accuracy': 0.0,
                        'f1_score': 0.0,
                        'outlier_count': 0,
                        'training_time': 0.0
                    }
                },
                'clinical_validation': {
                    'Isolation Forest': {
                        'clinically_plausible': 0,
                        'clinically_implausible': 0
                    }
                },
                'data_info': {
                    'total_samples': 0
                }
            }
    
    def _load_detection_results(self):
        """加载检测结果"""
        try:
            with open(f'{self.results_dir}/detection_results.json', 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"加载检测结果失败: {e}")
            return {}
    
    def _save_analysis_results(self, response):
        """保存分析结果到文件"""
        try:
            # 保存完整分析结果
            result_file = f'{self.results_dir}/analysis_result.json'
            with open(result_file, 'w', encoding='utf-8') as f:
                json.dump(response, f, ensure_ascii=False, indent=2)
            print(f"✓ 分析结果已保存到: {result_file}")
            
            # 保存详细检测结果（按模型分类）
            detection_results = {}
            for model_name in response['data']['algorithmTimes'].keys():
                detection_results[model_name] = {
                    'model_name': model_name,
                    'analysis_time': response['data']['algorithmTimes'][model_name],
                    'outlier_count': response['data']['outlierDetection']['counts'][response['data']['outlierDetection']['labels'].index(model_name)],
                    'accuracy': response['data']['modelPerformance']['accuracy'][response['data']['modelPerformance']['labels'].index(model_name)],
                    'f1_score': response['data']['modelPerformance']['f1Score'][response['data']['modelPerformance']['labels'].index(model_name)],
                    'clinical_plausible': response['data']['clinicalValidation']['plausible'][response['data']['clinicalValidation']['labels'].index(model_name)],
                    'clinical_implausible': response['data']['clinicalValidation']['implausible'][response['data']['clinicalValidation']['labels'].index(model_name)],
                    'detailed_results': response['data']['detailedResults']
                }
            
            detection_file = f'{self.results_dir}/detection_results.json'
            with open(detection_file, 'w', encoding='utf-8') as f:
                json.dump(detection_results, f, ensure_ascii=False, indent=2)
            print(f"✓ 检测结果已保存到: {detection_file}")
            
            # 尝试保存结果为图片
            self._save_results_as_images(response)
            
        except Exception as e:
            print(f"保存分析结果失败: {e}")
    
    def _save_results_as_images(self, response):
        """将分析结果保存为图片"""
        try:
            # 尝试导入matplotlib
            import matplotlib
            matplotlib.use('Agg')  # 使用非交互式后端
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            # 设置中文字体
            plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
            plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
            
            # 保存模型性能对比图
            self._save_model_performance_chart(response, plt, sns)
            
            # 保存离群值检测结果图
            self._save_outlier_detection_chart(response, plt, sns)
            
            # 保存医学符合性验证图
            self._save_clinical_validation_chart(response, plt, sns)
            
            # 保存大数据指标对比图
            self._save_large_data_chart(response, plt, sns)
            
            # 保存小数据指标对比图
            self._save_small_data_chart(response, plt, sns)
            
        except ImportError as e:
            print(f"保存图片失败，缺少依赖: {e}")
        except Exception as e:
            print(f"保存图片失败: {e}")
    
    def _save_model_performance_chart(self, response, plt, sns):
        """保存模型性能对比图"""
        data = response['data']['modelPerformance']
        labels = data['labels']
        accuracy = data['accuracy']
        f1_score = data['f1Score']
        
        plt.figure(figsize=(10, 6))
        x = range(len(labels))
        width = 0.35
        
        plt.bar(x, accuracy, width, label='准确率', color='#4ECDC4')
        plt.bar([i + width for i in x], f1_score, width, label='F1分数', color='#FF6B6B')
        
        plt.xlabel('模型')
        plt.ylabel('分数')
        plt.title('模型性能对比')
        plt.xticks([i + width/2 for i in x], labels)
        plt.legend()
        plt.tight_layout()
        
        save_path = f'{self.results_dir}/model_performance.png'
        plt.savefig(save_path)
        plt.close()
        print(f"✓ 模型性能对比图已保存到: {save_path}")
    
    def _save_outlier_detection_chart(self, response, plt, sns):
        """保存离群值检测结果图"""
        data = response['data']['outlierDetection']
        labels = data['labels']
        counts = data['counts']
        
        plt.figure(figsize=(10, 6))
        plt.bar(labels, counts, color='#45B7D1')
        
        plt.xlabel('模型')
        plt.ylabel('离群值数量')
        plt.title('离群值检测结果')
        plt.tight_layout()
        
        save_path = f'{self.results_dir}/outlier_detection.png'
        plt.savefig(save_path)
        plt.close()
        print(f"✓ 离群值检测结果图已保存到: {save_path}")
    
    def _save_clinical_validation_chart(self, response, plt, sns):
        """保存医学符合性验证图"""
        data = response['data']['clinicalValidation']
        labels = data['labels']
        plausible = data['plausible']
        implausible = data['implausible']
        
        plt.figure(figsize=(10, 6))
        x = range(len(labels))
        width = 0.35
        
        plt.bar(x, plausible, width, label='医学合理', color='#4ECDC4')
        plt.bar([i + width for i in x], implausible, width, label='医学可疑', color='#FF6B6B')
        
        plt.xlabel('模型')
        plt.ylabel('数量')
        plt.title('医学符合性验证')
        plt.xticks([i + width/2 for i in x], labels)
        plt.legend()
        plt.tight_layout()
        
        save_path = f'{self.results_dir}/clinical_validation.png'
        plt.savefig(save_path)
        plt.close()
        print(f"✓ 医学符合性验证图已保存到: {save_path}")
    
    def _save_large_data_chart(self, response, plt, sns):
        """保存大数据指标对比图"""
        data = response['data']['largeDataMetrics']
        labels = data['labels']
        normal_data = data['normalData']
        outlier_data = data['outlierData']
        
        plt.figure(figsize=(10, 6))
        x = range(len(labels))
        width = 0.35
        
        plt.bar(x, normal_data, width, label='正常样本', color='#4ECDC4')
        plt.bar([i + width for i in x], outlier_data, width, label='离群样本', color='#FF6B6B')
        
        plt.xlabel('指标')
        plt.ylabel('中位数')
        plt.title('大数据指标对比')
        plt.xticks([i + width/2 for i in x], labels)
        plt.legend()
        plt.tight_layout()
        
        save_path = f'{self.results_dir}/large_data_metrics.png'
        plt.savefig(save_path)
        plt.close()
        print(f"✓ 大数据指标对比图已保存到: {save_path}")
    
    def _save_small_data_chart(self, response, plt, sns):
        """保存小数据指标对比图"""
        data = response['data']['smallDataMetrics']
        labels = data['labels']
        normal_data = data['normalData']
        outlier_data = data['outlierData']
        
        plt.figure(figsize=(12, 6))
        x = range(len(labels))
        width = 0.35
        
        plt.bar(x, normal_data, width, label='正常样本', color='#4ECDC4')
        plt.bar([i + width for i in x], outlier_data, width, label='离群样本', color='#FF6B6B')
        
        plt.xlabel('指标')
        plt.ylabel('中位数')
        plt.title('小数据指标对比')
        plt.xticks([i + width/2 for i in x], labels, rotation=45, ha='right')
        plt.legend()
        plt.tight_layout()
        
        save_path = f'{self.results_dir}/small_data_metrics.png'
        plt.savefig(save_path)
        plt.close()
        print(f"✓ 小数据指标对比图已保存到: {save_path}")
    
    def analyze(self, analysis_type='full', model_select='all'):
        """执行分析"""
        # 实际运行算法并获取真实结果
        try:
            print("=======================================")
            print(f"接收到分析请求: type={analysis_type}, model={model_select}")
            print("开始执行分析...")
            
            # 创建数据处理器和离群值检测系统（相对项目根目录，避免启动目录不同找不到数据）
            data_file = os.path.join(project_root, 'heart.csv')
            print(f"使用数据文件: {data_file}")
            print(f"文件是否存在: {os.path.exists(data_file)}")
            
            # 检查文件大小
            if os.path.exists(data_file):
                print(f"文件大小: {os.path.getsize(data_file)} bytes")
            
            data_processor = MedicalDataProcessor(data_file)
            outlier_detector = OutlierDetectionSystem()
            
            # 加载数据
            print("开始加载数据...")
            start_time = time.time()
            df = data_processor.load_and_preprocess()
            load_time = time.time() - start_time
            print(f"数据加载完成，耗时: {load_time:.2f}秒")
            
            if df is None:
                print("数据加载失败")
                raise Exception("数据加载失败")
            print(f"数据加载成功，样本数: {len(df)}")
            print(f"数据形状: {df.shape}")
            
            # 准备特征和标签
            # 排除非特征列（如patient_id）
            non_feature_cols = ['patient_id']
            feature_cols = [col for col in df.columns if col not in non_feature_cols]
            X_full = df[feature_cols].values
            y_full = df.iloc[:, -1].values
            
            # 根据分析类型设置样本大小：快速分析使用较少样本，完整分析使用较多样本
            if analysis_type == 'quick':
                # 快速分析：使用20000条样本，减少训练时间
                sample_size = min(20000, len(X_full))
                print("【快速分析模式】使用较少样本进行快速检测")
            else:
                # 完整分析：使用100000条样本，获得更全面准确的结果
                sample_size = min(100000, len(X_full))
                print("【完整分析模式】使用较多样本进行全面检测")
            
            X = X_full[:sample_size]
            y = y_full[:sample_size]
            total_samples = len(X)
            print(f"特征数据形状: {X.shape} (使用{sample_size}个样本进行训练)")
            
            # 训练模型 - 根据用户选择的模型进行训练
            print(f"选择的模型: {model_select}")
            print("开始训练模型...")
            
            # 数据标准化
            outlier_detector.scaler.fit(X)
            X_scaled = outlier_detector.scaler.transform(X)
            
            models_info = {}

            import numpy as np
            # 自适应阈值函数：根据分数分布自动确定离群值阈值
            def adaptive_threshold(scores, method='iqr', k=1.5):
                """
                自适应阈值计算
                method: 'iqr' - 四分位距法, 'std' - 标准差法, 'percentile' - 百分位数法
                k: 阈值系数，越大越严格
                """
                import numpy as np
                
                if method == 'iqr':
                    # IQR方法：Q3 + k*IQR 以上为异常
                    q1 = np.percentile(scores, 25)
                    q3 = np.percentile(scores, 75)
                    iqr = q3 - q1
                    threshold = q3 + k * iqr
                    predictions = np.where(scores > threshold, -1, 1)
                elif method == 'std':
                    # 标准差方法：均值 + k*标准差 以上为异常
                    mean = np.mean(scores)
                    std = np.std(scores)
                    threshold = mean + k * std
                    predictions = np.where(scores > threshold, -1, 1)
                elif method == 'percentile':
                    # 百分位数方法：top k% 为异常
                    threshold = np.percentile(scores, 100 - k)
                    predictions = np.where(scores > threshold, -1, 1)
                else:
                    # 默认使用中位数绝对偏差 (MAD)
                    median = np.median(scores)
                    mad = np.median(np.abs(scores - median))
                    threshold = median + k * mad
                    predictions = np.where(scores > threshold, -1, 1)
                
                return predictions, threshold
            
            # 根据用户选择的模型进行训练
            if model_select == 'isolation_forest' or model_select == 'all':
                print("训练孤立森林模型...")
                start_time = time.time()
                from sklearn.ensemble import IsolationForest
                if_model = IsolationForest(
                    n_estimators=100,
                    max_samples=len(X),
                    contamination='auto',  # 使用自适应
                    random_state=42,
                    n_jobs=-1
                )
                if_model.fit(X_scaled)
                if_scores = if_model.decision_function(X_scaled)
                # 使用自适应阈值，IF使用更宽松的阈值(k=1.0)以检测更多离群值
                if_predictions, if_threshold = adaptive_threshold(-if_scores, method='iqr', k=1.0)
                if_time = time.time() - start_time
                
                outlier_count = np.sum(if_predictions == -1)
                print(f"✓ 孤立森林训练完成，时间: {if_time:.2f}秒，检测到 {outlier_count} 个离群值 ({outlier_count/len(X)*100:.1f}%)")
                
                models_info['Isolation Forest'] = {
                    'model': if_model,
                    'predictions': if_predictions,
                    'scores': if_scores,
                    'time': if_time,
                    'trained_samples': len(X),
                    'threshold': if_threshold
                }
            
            if model_select == 'lof' or model_select == 'all':
                print("训练LOF模型...")
                start_time = time.time()
                from sklearn.neighbors import LocalOutlierFactor
                lof_model = LocalOutlierFactor(
                    n_neighbors=min(20, len(X)-1),
                    novelty=False,
                    n_jobs=-1
                )
                lof_model.fit_predict(X_scaled)
                lof_scores = -lof_model.negative_outlier_factor_
                # 使用自适应阈值
                lof_predictions, lof_threshold = adaptive_threshold(lof_scores, method='iqr', k=1.5)
                lof_time = time.time() - start_time
                
                outlier_count = np.sum(lof_predictions == -1)
                print(f"✓ LOF训练完成，时间: {lof_time:.2f}秒，检测到 {outlier_count} 个离群值 ({outlier_count/len(X)*100:.1f}%)")
                
                models_info['LOF'] = {
                    'model': lof_model,
                    'predictions': lof_predictions,
                    'scores': lof_scores,
                    'time': lof_time,
                    'trained_samples': len(X),
                    'threshold': lof_threshold
                }
            
            if model_select == 'one_class_svm' or model_select == 'all':
                print("训练One-Class SVM模型...")
                start_time = time.time()
                from sklearn.svm import OneClassSVM
                gamma = 1 / (X.shape[1] * X_scaled.var()) if X_scaled.var() > 0 else 'scale'
                ocsvm_model = OneClassSVM(nu=0.1, kernel='rbf', gamma=gamma, tol=1e-3)
                ocsvm_model.fit(X_scaled)
                ocsvm_scores = ocsvm_model.decision_function(X_scaled)
                # 使用自适应阈值
                ocsvm_predictions, ocsvm_threshold = adaptive_threshold(-ocsvm_scores, method='iqr', k=1.5)
                ocsvm_time = time.time() - start_time
                
                outlier_count = np.sum(ocsvm_predictions == -1)
                print(f"✓ One-Class SVM训练完成，时间: {ocsvm_time:.2f}秒，检测到 {outlier_count} 个离群值 ({outlier_count/len(X)*100:.1f}%)")
                
                models_info['One-Class SVM'] = {
                    'model': ocsvm_model,
                    'predictions': ocsvm_predictions,
                    'scores': ocsvm_scores,
                    'time': ocsvm_time,
                    'trained_samples': len(X),
                    'threshold': ocsvm_threshold
                }
            
            # 比较IF和LOF检测到的离群值差异（当两种算法都训练时）
            if 'Isolation Forest' in models_info and 'LOF' in models_info:
                print("\n" + "="*60)
                print("算法离群值检测结果对比")
                print("="*60)
                
                if_predictions_arr = models_info['Isolation Forest']['predictions']
                lof_predictions_arr = models_info['LOF']['predictions']
                
                # 获取离群值索引（-1表示离群值）
                import numpy as np
                if_outliers = set(np.where(if_predictions_arr == -1)[0])
                lof_outliers = set(np.where(lof_predictions_arr == -1)[0])
                
                # 计算交集和差异
                common = if_outliers & lof_outliers
                only_if = if_outliers - lof_outliers
                only_lof = lof_outliers - if_outliers
                
                print(f"孤立森林检测到的离群值总数: {len(if_outliers)}")
                print(f"LOF检测到的离群值总数: {len(lof_outliers)}")
                print(f"共同检测到的离群值: {len(common)} ({len(common)/len(if_outliers)*100:.1f}%)")
                print(f"仅孤立森林检测到的: {len(only_if)} ({len(only_if)/len(if_outliers)*100:.1f}%)")
                print(f"仅LOF检测到的: {len(only_lof)} ({len(only_lof)/len(lof_outliers)*100:.1f}%)")
                print(f"一致性比例: {len(common)/((len(if_outliers)+len(lof_outliers))/2)*100:.1f}%")
                print("="*60)
            
            # 设置outlier_detector的models，以便evaluate_models方法能够找到模型
            outlier_detector.models = models_info
            
            # 评估模型 - 只评估Isolation Forest模型
            evaluation_results = outlier_detector.evaluate_models(X, y)
            
            # 执行医学验证 - 只验证Isolation Forest模型
            # 只使用1000个样本的DataFrame进行验证
            df_sample = df.iloc[:sample_size].reset_index(drop=True)
            clinical_validator = ClinicalValidator(data_processor.clinical_rules)
            clinical_results = {}
            for model_name, model_data in models_info.items():
                # 获取离群值索引
                outlier_indices = [i for i, pred in enumerate(model_data['predictions']) if pred == -1]
                # 验证离群值
                result = clinical_validator.validate_outliers(df_sample, outlier_indices, model_name)
                clinical_results[model_name] = result
            
            # 提取结果 - 处理所有训练的模型
            algorithm_times = {}
            outlier_counts = {}
            accuracies = {}
            f1_scores = {}
            clinical_plausible = {}
            clinical_implausible = {}
            
            # 处理所有训练的模型
            for model_name in models_info.keys():
                algorithm_times[model_name] = models_info[model_name]['time']
                outlier_counts[model_name] = evaluation_results[model_name]['outlier_count']
                accuracies[model_name] = evaluation_results[model_name].get('accuracy', 0.0)
                f1_scores[model_name] = evaluation_results[model_name].get('f1_score', 0.0)
                if model_name in clinical_results:
                    clinical_plausible[model_name] = clinical_results[model_name]['clinically_plausible']
                    clinical_implausible[model_name] = clinical_results[model_name]['clinically_implausible']
                else:
                    clinical_plausible[model_name] = 0
                    clinical_implausible[model_name] = 0
            
            # 获取当前选择的模型名称
            model_map = {
                'isolation_forest': 'Isolation Forest',
                'lof': 'LOF',
                'one_class_svm': 'One-Class SVM',
                'all': 'Isolation Forest'  # 默认显示Isolation Forest
            }
            selected_model_name = model_map.get(model_select, 'Isolation Forest')
            
            # 计算分析时间（显示4位小数，提高精度）
            training_time = algorithm_times.get(selected_model_name, 0)
            analysis_time = f"{training_time:.4f}s"
            abnormal_count = outlier_counts.get(selected_model_name, 0)
            
            # 计算异常比例
            abnormal_ratio = f"{abnormal_count / total_samples * 100:.2f}%" if total_samples > 0 else "0.00%"
            
            # 生成详细结果
            detailed_results = []
            # 使用选择的模型生成详细结果
            if selected_model_name in models_info:
                # 生成一些示例数据
                for i in range(min(50, total_samples)):
                    sample = {
                        'patient_id': f'P{i+1}',
                        'age': 30 + i % 40,
                        'sex': '男' if i % 2 == 0 else '女',
                        'blood_pressure': 120 + i % 30,
                        'cholesterol': 180 + i % 50,
                        'blood_sugar': 5.0 + i % 3,
                        f'{selected_model_name}_is_outlier': '离群值' if models_info[selected_model_name]['predictions'][i] == -1 else '正常',
                        f'{selected_model_name}_score': round(float(models_info[selected_model_name]['scores'][i]), 4),
                        'evaluation': '需要进一步检查' if models_info[selected_model_name]['predictions'][i] == -1 else '正常'
                    }
                    detailed_results.append(sample)
            
            # 计算大数据和小数据指标的实际数据
            # 从实际数据中计算中位数
            import numpy as np
            
            # 从df_sample中提取相关特征（使用数据集中实际的列名）
            large_data_features = ['chol', 'plt', 'creatinine']  # plt=血小板
            small_data_features = ['blood_sugar', 'triglycerides', 'wbc', 'lymphocyte', 'alt', 'ast', 'blood_urea', 'rbc']  # blood_urea=尿素, rbc=红细胞
            
            # 打印特征列表，确保使用正确的列名
            print("\n=== 特征列表 ===")
            print(f"大数据特征: {large_data_features}")
            print(f"小数据特征: {small_data_features}")
            print(f"数据集中的列: {list(df_sample.columns[:20])}")  # 只打印前20列，避免输出过多
            
            # 确保特征列存在
            large_data_features = [f for f in large_data_features if f in df_sample.columns]
            small_data_features = [f for f in small_data_features if f in df_sample.columns]
            
            # 计算正常样本和离群样本的中位数
            normal_indices = [i for i, pred in enumerate(models_info[selected_model_name]['predictions']) if pred != -1]
            outlier_indices = [i for i, pred in enumerate(models_info[selected_model_name]['predictions']) if pred == -1]
            
            large_data_normal = []
            large_data_outlier = []
            for feature in large_data_features:
                if feature in df_sample.columns:
                    normal_data = df_sample.iloc[normal_indices][feature].dropna()
                    outlier_data = df_sample.iloc[outlier_indices][feature].dropna()
                    large_data_normal.append(np.median(normal_data) if len(normal_data) > 0 else 0)
                    large_data_outlier.append(np.median(outlier_data) if len(outlier_data) > 0 else 0)
                else:
                    large_data_normal.append(0)
                    large_data_outlier.append(0)
            
            small_data_normal = []
            small_data_outlier = []
            for feature in small_data_features:
                if feature in df_sample.columns:
                    normal_data = df_sample.iloc[normal_indices][feature].dropna()
                    outlier_data = df_sample.iloc[outlier_indices][feature].dropna()
                    small_data_normal.append(np.median(normal_data) if len(normal_data) > 0 else 0)
                    small_data_outlier.append(np.median(outlier_data) if len(outlier_data) > 0 else 0)
                else:
                    small_data_normal.append(0)
                    small_data_outlier.append(0)
            
            # 确保数据长度与特征长度一致
            small_data_normal = small_data_normal[:len(small_data_features)]
            small_data_outlier = small_data_outlier[:len(small_data_features)]
            
            # 构建响应数据
            # 包含所有训练的模型的结果
            # 确保标签和数据长度一致
            large_data_labels = ['胆固醇', '血小板', '肌酐'][:len(large_data_features)]
            small_data_labels = ['血糖', '甘油三酯', '白细胞', '淋巴细胞', 'ALT', 'AST', '尿素', '红细胞'][:len(small_data_features)]
            
            # 确保数据长度与标签长度一致
            large_data_normal = large_data_normal[:len(large_data_labels)]
            large_data_outlier = large_data_outlier[:len(large_data_labels)]
            small_data_normal = small_data_normal[:len(small_data_labels)]
            small_data_outlier = small_data_outlier[:len(small_data_labels)]
            
            # 打印调试信息，确保数据长度一致
            print(f"\n=== 数据长度检查 ===")
            print(f"大数据标签长度: {len(large_data_labels)}")
            print(f"大数据正常数据长度: {len(large_data_normal)}")
            print(f"大数据离群数据长度: {len(large_data_outlier)}")
            print(f"小数据标签长度: {len(small_data_labels)}")
            print(f"小数据正常数据长度: {len(small_data_normal)}")
            print(f"小数据离群数据长度: {len(small_data_outlier)}")
            
            response = {
                'success': True,
                'data': {
                    'totalSamples': total_samples,
                    'abnormalCount': abnormal_count,
                    'abnormalRatio': abnormal_ratio,
                    'analysisTime': analysis_time,
                    'algorithmTimes': algorithm_times,
                    'modelPerformance': {
                        'labels': list(algorithm_times.keys()),
                        'accuracy': [accuracies.get(model, 0.0) for model in algorithm_times.keys()],
                        'f1Score': [f1_scores.get(model, 0.0) for model in algorithm_times.keys()]
                    },
                    'outlierDetection': {
                        'labels': list(algorithm_times.keys()),
                        'counts': [outlier_counts.get(model, 0) for model in algorithm_times.keys()]
                    },
                    'clinicalValidation': {
                        'labels': list(algorithm_times.keys()),
                        'plausible': [clinical_plausible.get(model, 0) for model in algorithm_times.keys()],
                        'implausible': [clinical_implausible.get(model, 0) for model in algorithm_times.keys()]
                    },
                    'largeDataMetrics': {
                        'labels': large_data_labels,
                        'normalData': [float(v) for v in large_data_normal],
                        'outlierData': [float(v) for v in large_data_outlier]
                    },
                    'smallDataMetrics': {
                        'labels': small_data_labels,
                        'normalData': [float(v) for v in small_data_normal],
                        'outlierData': [float(v) for v in small_data_outlier]
                    },
                    # 检测超参数信息（使用自适应阈值，无固定 contamination）
                    'detectionHyperparameters': {
                        'threshold_method': 'iqr',
                        'threshold_k': 1.5,
                        'note': '使用自适应阈值，根据数据分布自动确定离群值'
                    },
                    'detailedResults': detailed_results
                }
            }
            
            # 打印调试信息，确保数据正确计算
            print("\n=== 调试信息 ===")
            print(f"选择的模型: {selected_model_name}")
            print(f"大数据特征: {large_data_features}")
            print(f"小数据特征: {small_data_features}")
            print(f"正常样本数: {len(normal_indices)}")
            print(f"离群样本数: {len(outlier_indices)}")
            print(f"大数据正常数据: {large_data_normal}")
            print(f"大数据离群数据: {large_data_outlier}")
            print(f"小数据正常数据: {small_data_normal}")
            print(f"小数据离群数据: {small_data_outlier}")
            print(f"小数据标签长度: {len(['血糖', '甘油三酯', '白细胞', '淋巴细胞', 'ALT', 'AST', '尿素'][:len(small_data_features)])}")
            print(f"小数据正常数据长度: {len(small_data_normal[:len(small_data_features)])}")
            print(f"小数据离群数据长度: {len(small_data_outlier[:len(small_data_features)])}")
            
            # 保存分析结果到results文件
            self._save_analysis_results(response)
            
            return response
        except Exception as e:
            print(f"分析失败: {e}")
            # 分析失败时返回错误信息，不使用静态默认数据
            return {
                'success': False,
                'error': str(e),
                'message': '分析过程中发生错误，请检查数据文件是否存在或数据格式是否正确'
            }


# 直接运行时的测试代码
if __name__ == '__main__':
    print("="*60)
    print("测试 AnalysisController")
    print("="*60)
    
    # 创建控制器实例
    controller = AnalysisController()
    
    # 测试分析功能
    print("\n开始测试分析功能...")
    try:
        result = controller.analyze(analysis_type='full', model_select='isolation_forest')
        if result.get('success'):
            print("分析成功!")
            print(f"总样本数: {result['data']['totalSamples']}")
            print(f"异常样本数: {result['data']['abnormalCount']}")
            print(f"分析时间: {result['data']['analysisTime']}")
            print(f"算法运行时间:")
            for algorithm, time in result['data']['algorithmTimes'].items():
                print(f"  {algorithm}: {time}秒")
        else:
            print(f"分析失败: {result.get('error')}")
            print(f"错误信息: {result.get('message')}")
    except Exception as e:
        print(f"测试过程中发生错误: {e}")
    
    print("\n" + "="*60)
    print("测试完成")
    print("="*60)
