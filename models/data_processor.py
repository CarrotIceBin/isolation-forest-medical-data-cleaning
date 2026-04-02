"""
数据处理器模块 - 负责数据加载与预处理
"""
import pandas as pd
import numpy as np


class MedicalDataProcessor:
    """医疗数据处理类 - 修复整数列NA值问题"""
    
    def __init__(self, filepath):
        self.filepath = filepath
        self.df = None
        self.clinical_rules = self._load_clinical_rules()
        
    def _load_clinical_rules(self):
        """加载临床规则"""
        rules = {
            'age': {'min': 18, 'max': 90, 'critical_min': 10, 'critical_max': 120},
            'trestbps': {'min': 90, 'max': 140, 'critical_min': 60, 'critical_max': 200},
            'chol': {'min': 120, 'max': 240, 'critical_min': 80, 'critical_max': 400},
            'thalach': {'min': 60, 'max': 180, 'critical_min': 40, 'critical_max': 220},
            'oldpeak': {'min': 0, 'max': 2.0, 'critical_min': -1, 'critical_max': 4.0},
            'blood_sugar': {'min': 3.9, 'max': 6.1, 'critical_min': 2.5, 'critical_max': 10.0},
            'blood_urea': {'min': 2.5, 'max': 7.1, 'critical_min': 1.0, 'critical_max': 15.0},
            'creatinine': {'min': 53, 'max': 106, 'critical_min': 20, 'critical_max': 200},
            'bilirubin': {'min': 5.1, 'max': 20.5, 'critical_min': 1.0, 'critical_max': 35.0},
            'alt': {'min': 5, 'max': 40, 'critical_min': 1, 'critical_max': 100},
            'ast': {'min': 5, 'max': 40, 'critical_min': 1, 'critical_max': 100},
            'albumin': {'min': 35, 'max': 55, 'critical_min': 20, 'critical_max': 65},
            'globulin': {'min': 20, 'max': 35, 'critical_min': 10, 'critical_max': 50},
            'a_g_ratio': {'min': 1.0, 'max': 2.5, 'critical_min': 0.8, 'critical_max': 3.0},
            'wbc': {'min': 4.0, 'max': 10.0, 'critical_min': 1.0, 'critical_max': 30.0},
            'rbc': {'min': 4.3, 'max': 5.8, 'critical_min': 2.0, 'critical_max': 7.0},
            'hb': {'min': 120, 'max': 160, 'critical_min': 60, 'critical_max': 200},
            'plt': {'min': 100, 'max': 300, 'critical_min': 50, 'critical_max': 500},
            'lymphocyte': {'min': 1.0, 'max': 3.0, 'critical_min': 0.5, 'critical_max': 5.0},
            'neutrophil': {'min': 2.0, 'max': 7.0, 'critical_min': 1.0, 'critical_max': 10.0},
            'ldl': {'min': 1.0, 'max': 3.4, 'critical_min': 0.5, 'critical_max': 5.0},
            'hdl': {'min': 1.0, 'max': 2.0, 'critical_min': 0.5, 'critical_max': 3.0},
            'triglycerides': {'min': 0.5, 'max': 1.7, 'critical_min': 0.3, 'critical_max': 5.0},
            'potassium': {'min': 3.5, 'max': 5.3, 'critical_min': 3.0, 'critical_max': 6.0},
            'sodium': {'min': 135, 'max': 145, 'critical_min': 120, 'critical_max': 160},
            'chloride': {'min': 96, 'max': 106, 'critical_min': 80, 'critical_max': 120},
            'calcium': {'min': 2.1, 'max': 2.6, 'critical_min': 1.8, 'critical_max': 3.0}
        }
        return rules
    
    def load_and_preprocess(self):
        """加载并预处理数据"""
        import os
        
        print("\n" + "="*60)
        print("1. 数据加载与预处理")
        print("="*60)
        
        if not os.path.exists(self.filepath):
            raise FileNotFoundError(f"文件不存在: {self.filepath}")
        
        try:
            print("第一步：加载原始数据")
            self.df = pd.read_csv(self.filepath)
            print(f"✓ 成功加载原始数据，形状: {self.df.shape}")
        except Exception as e:
            print(f"✗ 加载原始数据失败: {e}")
            return None
        
        # 处理缺失值
        print("\n第二步：处理缺失值")
        missing_values = self.df.isnull().sum()
        if missing_values.sum() > 0:
            print(f"发现缺失值: \n{missing_values[missing_values > 0]}")
            for col in self.df.columns:
                if self.df[col].isnull().sum() > 0:
                    fill_value = self.df[col].median()
                    self.df[col] = self.df[col].fillna(fill_value)
                    print(f"  - {col}列缺失值已用中位数({fill_value})填充")
            print("✓ 所有缺失值已填充完成")
        else:
            print("✓ 无缺失值")
        
        # 转换数据类型
        print("\n第三步：转换数据类型")
        dtype_map = {
            'patient_id': str, 'age': int, 'sex': int, 'smoking': int, 'alcohol': int, 'exercise_frequency': int,
            'cp': int, 'trestbps': int, 'chol': int, 'fbs': int, 'restecg': int, 
            'thalach': int, 'exang': int, 'oldpeak': float, 'slope': int, 'ca': int, 'thal': int,
            'blood_sugar': float, 'blood_urea': float, 'creatinine': int, 'bilirubin': float, 
            'alt': int, 'ast': int, 'albumin': int, 'globulin': int, 'a_g_ratio': float,
            'wbc': float, 'rbc': float, 'hb': int, 'plt': int, 'lymphocyte': float, 'neutrophil': float,
            'ldl': float, 'hdl': float, 'triglycerides': float,
            'urine_protein': int, 'urine_glucose': int,
            'potassium': float, 'sodium': int, 'chloride': int, 'calcium': float,
            'hypertension': int, 'diabetes': int, 'heart_disease': int, 'kidney_disease': int,
            'abnormal_flag': int
        }
        
        for col in self.df.columns:
            if col in dtype_map:
                try:
                    self.df[col] = self.df[col].astype(dtype_map[col])
                except Exception as e:
                    print(f"  ⚠ {col}列类型转换失败: {e}，自动清理异常值")
                    self.df[col] = pd.to_numeric(self.df[col], errors='coerce').fillna(self.df[col].median()).astype(dtype_map[col])
        
        # 处理重复值
        duplicate_count = self.df.duplicated().sum()
        if duplicate_count > 0:
            self.df = self.df.drop_duplicates()
            print(f"\n✓ 移除{duplicate_count}条重复数据")
        
        print(f"\n最终数据验证:")
        print(f"可用样本数: {len(self.df)}")
        print(f"特征数: {len(self.df.columns)}")
        
        return self.df