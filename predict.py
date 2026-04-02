"""
医疗数据离群值检测系统 - 一键整合版
1. 自动生成10000条心脏病模拟数据
2. 替换原有heart.csv文件
3. 验证数据量（确保10000条）
4. 执行完整的离群值检测分析
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# ===================== 核心库导入 =====================
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, roc_auc_score, confusion_matrix)

import matplotlib.pyplot as plt
import seaborn as sns
import json
import time
import os

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ===================== 0. 一键生成100000条模拟数据 =====================
def generate_professional_medical_data(save_path, sample_size=100000):
    """生成专业的综合医疗检查数据并保存"""
    print("="*60)
    print(f"0. 生成{sample_size}条专业医疗检查数据")
    print("="*60)
    
    # 定义专业医疗检查数据集字段
    columns = [
        # 基本信息
        'patient_id', 'age', 'sex', 'smoking', 'alcohol', 'exercise_frequency',
        # 心脏病相关
        'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal',
        # 其他常规检查
        'blood_sugar', 'blood_urea', 'creatinine', 'bilirubin', 'alt', 'ast', 'albumin', 'globulin', 'a_g_ratio',
        # 血常规
        'wbc', 'rbc', 'hb', 'plt', 'lymphocyte', 'neutrophil',
        # 血脂
        'ldl', 'hdl', 'triglycerides',
        # 肾功能
        'urine_protein', 'urine_glucose',
        # 其他生化指标
        'potassium', 'sodium', 'chloride', 'calcium',
        # 诊断信息
        'hypertension', 'diabetes', 'heart_disease', 'kidney_disease',
        # 目标变量
        'abnormal_flag'
    ]
    
    # 字段约束：分类变量取值范围 + 连续变量取值范围
    constraints = {
        # 基本信息
        'age': (18, 90),  # 年龄
        'sex': [0, 1],    # 0=女,1=男
        'smoking': [0, 1, 2],  # 0=不吸烟,1=以前吸烟,2=现在吸烟
        'alcohol': [0, 1, 2],  # 0=不饮酒,1=偶尔饮酒,2=经常饮酒
        'exercise_frequency': [0, 1, 2, 3],  # 0=不运动,1=每周1-2次,2=每周3-4次,3=每周5次以上
        
        # 心脏病相关
        'cp': [0, 1, 2, 3],  # 胸痛类型
        'trestbps': (80, 220),  # 静息血压
        'chol': (100, 600),  # 血清胆固醇
        'fbs': [0, 1],  # 空腹血糖>120mg/dl
        'restecg': [0, 1, 2],  # 静息心电图结果
        'thalach': (70, 220),  # 最大心率
        'exang': [0, 1],  # 运动诱发心绞痛
        'oldpeak': (0.0, 6.5),  # ST段压低
        'slope': [0, 1, 2],  # ST段斜率
        'ca': [0, 1, 2, 3, 4],  # 荧光检查着色的血管数
        'thal': [0, 1, 2, 3, 7],  # 地中海贫血
        
        # 其他常规检查
        'blood_sugar': (3.9, 15.0),  # 血糖 (mmol/L)
        'blood_urea': (2.5, 15.0),  # 血尿素 (mmol/L)
        'creatinine': (53, 200),  # 肌酐 (μmol/L)
        'bilirubin': (5.1, 35.0),  # 胆红素 (μmol/L)
        'alt': (5, 100),  # 丙氨酸转氨酶 (U/L)
        'ast': (5, 100),  # 天门冬氨酸转氨酶 (U/L)
        'albumin': (35, 55),  # 白蛋白 (g/L)
        'globulin': (20, 50),  # 球蛋白 (g/L)
        'a_g_ratio': (1.0, 2.5),  # 白蛋白/球蛋白比值
        
        # 血常规
        'wbc': (3.5, 30.0),  # 白细胞计数 (10^9/L)
        'rbc': (3.5, 6.5),  # 红细胞计数 (10^12/L)
        'hb': (100, 180),  # 血红蛋白 (g/L)
        'plt': (80, 400),  # 血小板计数 (10^9/L)
        'lymphocyte': (0.8, 4.0),  # 淋巴细胞计数 (10^9/L)
        'neutrophil': (1.8, 8.0),  # 中性粒细胞计数 (10^9/L)
        
        # 血脂
        'ldl': (1.0, 5.0),  # 低密度脂蛋白 (mmol/L)
        'hdl': (0.8, 2.0),  # 高密度脂蛋白 (mmol/L)
        'triglycerides': (0.5, 5.0),  # 甘油三酯 (mmol/L)
        
        # 肾功能
        'urine_protein': [0, 1, 2, 3],  # 尿蛋白: 0=阴性,1=+,2=++,3=+++
        'urine_glucose': [0, 1, 2, 3],  # 尿糖: 0=阴性,1=+,2=++,3=+++
        
        # 其他生化指标
        'potassium': (3.5, 5.5),  # 血钾 (mmol/L)
        'sodium': (135, 145),  # 血钠 (mmol/L)
        'chloride': (96, 106),  # 血氯 (mmol/L)
        'calcium': (2.1, 2.6),  # 血钙 (mmol/L)
        
        # 诊断信息
        'hypertension': [0, 1],  # 高血压
        'diabetes': [0, 1],  # 糖尿病
        'heart_disease': [0, 1],  # 心脏病
        'kidney_disease': [0, 1],  # 肾病
        
        # 目标变量
        'abnormal_flag': [0, 1]  # 1=异常,0=正常
    }

    # 基于原始样本的统计特征（均值、标准差）
    stats = {
        # 基本信息
        'age': (54, 15),
        
        # 心脏病相关
        'trestbps': (120, 15),
        'chol': (200, 40),
        'thalach': (140, 20),
        'oldpeak': (0.8, 1.0),
        
        # 其他常规检查
        'blood_sugar': (5.0, 0.8),
        'blood_urea': (5.0, 1.5),
        'creatinine': (75, 15),
        'bilirubin': (12.0, 3.0),
        'alt': (25, 10),
        'ast': (25, 10),
        'albumin': (42, 3),
        'globulin': (28, 3),
        'a_g_ratio': (1.5, 0.2),
        
        # 血常规
        'wbc': (6.5, 1.5),
        'rbc': (4.8, 0.3),
        'hb': (145, 10),
        'plt': (220, 50),
        'lymphocyte': (1.8, 0.5),
        'neutrophil': (4.0, 1.0),
        
        # 血脂
        'ldl': (2.5, 0.8),
        'hdl': (1.2, 0.3),
        'triglycerides': (1.5, 0.8),
        
        # 其他生化指标
        'potassium': (4.2, 0.3),
        'sodium': (140, 2),
        'chloride': (100, 2),
        'calcium': (2.3, 0.1)
    }

    # 生成数据
    data = []
    for i in range(sample_size):
        row = {}
        
        # 基本信息
        row['patient_id'] = f'P{i+1:05d}'
        row['age'] = int(max(18, min(90, np.random.normal(54, 15))))
        row['sex'] = np.random.choice([0, 1], p=[0.5, 0.5])
        row['smoking'] = np.random.choice([0, 1, 2], p=[0.6, 0.2, 0.2])
        row['alcohol'] = np.random.choice([0, 1, 2], p=[0.7, 0.2, 0.1])
        row['exercise_frequency'] = np.random.choice([0, 1, 2, 3], p=[0.3, 0.3, 0.25, 0.15])
        
        # 心脏病相关指标（添加相关性）
        age_factor = row['age'] / 100
        smoking_factor = row['smoking'] * 0.1
        
        row['cp'] = np.random.choice([0, 1, 2, 3], p=[0.6, 0.2, 0.15, 0.05])
        row['trestbps'] = int(max(80, min(220, np.random.normal(120 + age_factor*20 + smoking_factor*10, 15))))
        row['chol'] = int(max(100, min(600, np.random.normal(200 + age_factor*30 + smoking_factor*20, 40))))
        row['fbs'] = 1 if np.random.normal(5.0, 1.0) > 6.1 else 0
        row['restecg'] = np.random.choice([0, 1, 2], p=[0.7, 0.2, 0.1])
        row['thalach'] = int(max(70, min(220, np.random.normal(140 - age_factor*20, 20))))
        row['exang'] = np.random.choice([0, 1], p=[0.8, 0.2])
        row['oldpeak'] = round(max(0.0, min(6.5, np.random.normal(0.8, 1.0))), 1)
        row['slope'] = np.random.choice([0, 1, 2], p=[0.1, 0.6, 0.3])
        row['ca'] = np.random.choice([0, 1, 2, 3, 4], p=[0.6, 0.2, 0.1, 0.08, 0.02])
        row['thal'] = np.random.choice([0, 1, 2, 3, 7], p=[0.1, 0.2, 0.4, 0.2, 0.1])
        
        # 其他常规检查
        row['blood_sugar'] = round(max(3.9, min(15.0, np.random.normal(5.0, 1.0))), 1)
        row['blood_urea'] = round(max(2.5, min(15.0, np.random.normal(5.0, 1.5))), 1)
        row['creatinine'] = int(max(53, min(200, np.random.normal(75, 15))))
        row['bilirubin'] = round(max(5.1, min(35.0, np.random.normal(12.0, 3.0))), 1)
        row['alt'] = int(max(5, min(100, np.random.normal(25, 10))))
        row['ast'] = int(max(5, min(100, np.random.normal(25, 10))))
        row['albumin'] = int(max(35, min(55, np.random.normal(42, 3))))
        row['globulin'] = int(max(20, min(50, np.random.normal(28, 3))))
        row['a_g_ratio'] = round(max(1.0, min(2.5, row['albumin'] / row['globulin'])), 2) if row['globulin'] > 0 else 1.5
        
        # 血常规
        row['wbc'] = round(max(3.5, min(30.0, np.random.normal(6.5, 1.5))), 1)
        row['rbc'] = round(max(3.5, min(6.5, np.random.normal(4.8, 0.3))), 1)
        row['hb'] = int(max(100, min(180, np.random.normal(145, 10))))
        row['plt'] = int(max(80, min(400, np.random.normal(220, 50))))
        row['lymphocyte'] = round(max(0.8, min(4.0, np.random.normal(1.8, 0.5))), 1)
        row['neutrophil'] = round(max(1.8, min(8.0, np.random.normal(4.0, 1.0))), 1)
        
        # 血脂
        row['ldl'] = round(max(1.0, min(5.0, np.random.normal(2.5, 0.8))), 2)
        row['hdl'] = round(max(0.8, min(2.0, np.random.normal(1.2, 0.3))), 2)
        row['triglycerides'] = round(max(0.5, min(5.0, np.random.normal(1.5, 0.8))), 2)
        
        # 肾功能
        row['urine_protein'] = np.random.choice([0, 1, 2, 3], p=[0.8, 0.15, 0.04, 0.01])
        row['urine_glucose'] = np.random.choice([0, 1, 2, 3], p=[0.9, 0.08, 0.015, 0.005])
        
        # 其他生化指标
        row['potassium'] = round(max(3.5, min(5.5, np.random.normal(4.2, 0.3))), 2)
        row['sodium'] = int(max(135, min(145, np.random.normal(140, 2))))
        row['chloride'] = int(max(96, min(106, np.random.normal(100, 2))))
        row['calcium'] = round(max(2.1, min(2.6, np.random.normal(2.3, 0.1))), 2)
        
        # 诊断信息（基于风险因素）
        risk_score = 0
        if row['age'] > 60: risk_score += 2
        if row['smoking'] == 2: risk_score += 2
        if row['trestbps'] > 140: risk_score += 2
        if row['chol'] > 240: risk_score += 1
        if row['blood_sugar'] > 7.0: risk_score += 2
        
        row['hypertension'] = 1 if row['trestbps'] > 140 else 0
        row['diabetes'] = 1 if row['blood_sugar'] > 7.0 else 0
        row['heart_disease'] = 1 if risk_score >= 5 else 0
        row['kidney_disease'] = 1 if row['creatinine'] > 120 else 0
        
        # 目标变量（基于异常指标数量）
        abnormal_count = 0
        if row['trestbps'] > 140 or row['trestbps'] < 90: abnormal_count += 1
        if row['chol'] > 240 or row['chol'] < 120: abnormal_count += 1
        if row['blood_sugar'] > 7.0 or row['blood_sugar'] < 3.9: abnormal_count += 1
        if row['creatinine'] > 120: abnormal_count += 1
        if row['alt'] > 40 or row['ast'] > 40: abnormal_count += 1
        if row['wbc'] > 10.0 or row['wbc'] < 4.0: abnormal_count += 1
        if row['hb'] > 160 or row['hb'] < 120: abnormal_count += 1
        
        row['abnormal_flag'] = 1 if abnormal_count >= 2 else 0
        
        # 转换为列表格式
        row_list = [row[col] for col in columns]
        data.append(row_list)
    
    # 转换为DataFrame
    df = pd.DataFrame(data, columns=columns)
    
    # 保存到指定路径（覆盖原有文件）
    df.to_csv(save_path, index=False, encoding='utf-8-sig')
    print(f"✓ {sample_size}条专业医疗检查数据已生成并保存到: {save_path}")
    print(f"✓ 生成数据形状: {df.shape}")
    print(f"✓ 包含检查项目: {len(columns) - 1}项")
    print(f"✓ 异常样本比例: {df['abnormal_flag'].mean():.1%}")
    
    # 验证保存结果
    if os.path.exists(save_path):
        saved_df = pd.read_csv(save_path)
        print(f"✓ 验证：保存后的文件样本量 = {len(saved_df)}")
    else:
        print("❌ 数据保存失败！")
    
    return df

# ===================== 1. 数据加载与预处理 =====================
class MedicalDataProcessor:
    """医疗数据处理类 - 修复整数列NA值问题"""
    
    def __init__(self, filepath):
        self.filepath = filepath
        self.df = None
        self.clinical_rules = self._load_clinical_rules()
        
    def _load_clinical_rules(self):
        """加载临床规则"""
        rules = {
            # 基本信息
            'age': {'min': 18, 'max': 90, 'critical_min': 10, 'critical_max': 120},
            
            # 心脏病相关
            'trestbps': {'min': 90, 'max': 140, 'critical_min': 60, 'critical_max': 200},
            'chol': {'min': 120, 'max': 240, 'critical_min': 80, 'critical_max': 400},
            'thalach': {'min': 60, 'max': 180, 'critical_min': 40, 'critical_max': 220},
            'oldpeak': {'min': 0, 'max': 2.0, 'critical_min': -1, 'critical_max': 4.0},
            
            # 其他常规检查
            'blood_sugar': {'min': 3.9, 'max': 6.1, 'critical_min': 2.5, 'critical_max': 10.0},
            'blood_urea': {'min': 2.5, 'max': 7.1, 'critical_min': 1.0, 'critical_max': 15.0},
            'creatinine': {'min': 53, 'max': 106, 'critical_min': 20, 'critical_max': 200},
            'bilirubin': {'min': 5.1, 'max': 20.5, 'critical_min': 1.0, 'critical_max': 35.0},
            'alt': {'min': 5, 'max': 40, 'critical_min': 1, 'critical_max': 100},
            'ast': {'min': 5, 'max': 40, 'critical_min': 1, 'critical_max': 100},
            'albumin': {'min': 35, 'max': 55, 'critical_min': 20, 'critical_max': 65},
            'globulin': {'min': 20, 'max': 35, 'critical_min': 10, 'critical_max': 50},
            'a_g_ratio': {'min': 1.0, 'max': 2.5, 'critical_min': 0.8, 'critical_max': 3.0},
            
            # 血常规
            'wbc': {'min': 4.0, 'max': 10.0, 'critical_min': 1.0, 'critical_max': 30.0},
            'rbc': {'min': 4.3, 'max': 5.8, 'critical_min': 2.0, 'critical_max': 7.0},
            'hb': {'min': 120, 'max': 160, 'critical_min': 60, 'critical_max': 200},
            'plt': {'min': 100, 'max': 300, 'critical_min': 50, 'critical_max': 500},
            'lymphocyte': {'min': 1.0, 'max': 3.0, 'critical_min': 0.5, 'critical_max': 5.0},
            'neutrophil': {'min': 2.0, 'max': 7.0, 'critical_min': 1.0, 'critical_max': 10.0},
            
            # 血脂
            'ldl': {'min': 1.0, 'max': 3.4, 'critical_min': 0.5, 'critical_max': 5.0},
            'hdl': {'min': 1.0, 'max': 2.0, 'critical_min': 0.5, 'critical_max': 3.0},
            'triglycerides': {'min': 0.5, 'max': 1.7, 'critical_min': 0.3, 'critical_max': 5.0},
            
            # 其他生化指标
            'potassium': {'min': 3.5, 'max': 5.3, 'critical_min': 3.0, 'critical_max': 6.0},
            'sodium': {'min': 135, 'max': 145, 'critical_min': 120, 'critical_max': 160},
            'chloride': {'min': 96, 'max': 106, 'critical_min': 80, 'critical_max': 120},
            'calcium': {'min': 2.1, 'max': 2.6, 'critical_min': 1.8, 'critical_max': 3.0}
        }
        return rules
    
    def load_and_preprocess(self):
        """加载并预处理数据（修复整数列NA值问题）"""
        print("\n" + "="*60)
        print("1. 数据加载与预处理")
        print("="*60)
        
        # 检查文件是否存在
        if not os.path.exists(self.filepath):
            raise FileNotFoundError(f"文件不存在: {self.filepath}")
        
        # 第一步：先不指定类型加载数据，避免NA值冲突
        try:
            print("第一步：加载原始数据（不指定类型，避免NA值错误）")
            self.df = pd.read_csv(self.filepath)
            print(f"✓ 成功加载原始数据，形状: {self.df.shape}")
            print(f"数据列: {list(self.df.columns)}")
            print(f"总样本数: {len(self.df)}")
        except Exception as e:
            print(f"✗ 加载原始数据失败: {e}")
            return None
        
        # 第二步：处理缺失值（优先处理，避免类型转换错误）
        print("\n第二步：处理缺失值")
        missing_values = self.df.isnull().sum()
        if missing_values.sum() > 0:
            print(f"发现缺失值: \n{missing_values[missing_values > 0]}")
            # 填充所有缺失值（先填充，再转换类型）
            for col in self.df.columns:
                if self.df[col].isnull().sum() > 0:
                    # 数值列用中位数填充
                    fill_value = self.df[col].median()
                    self.df[col] = self.df[col].fillna(fill_value)
                    print(f"  - {col}列缺失值已用中位数({fill_value})填充")
            print("✓ 所有缺失值已填充完成")
        else:
            print("✓ 无缺失值")
        
        # 第三步：转换数据类型（填充后再转换，避免NA值冲突）
        print("\n第三步：转换数据类型")
        dtype_map = {
            # 基本信息
            'patient_id': str, 'age': int, 'sex': int, 'smoking': int, 'alcohol': int, 'exercise_frequency': int,
            # 心脏病相关
            'cp': int, 'trestbps': int, 'chol': int, 'fbs': int, 'restecg': int, 
            'thalach': int, 'exang': int, 'oldpeak': float, 'slope': int, 'ca': int, 'thal': int,
            # 其他常规检查
            'blood_sugar': float, 'blood_urea': float, 'creatinine': int, 'bilirubin': float, 
            'alt': int, 'ast': int, 'albumin': int, 'globulin': int, 'a_g_ratio': float,
            # 血常规
            'wbc': float, 'rbc': float, 'hb': int, 'plt': int, 'lymphocyte': float, 'neutrophil': float,
            # 血脂
            'ldl': float, 'hdl': float, 'triglycerides': float,
            # 肾功能
            'urine_protein': int, 'urine_glucose': int,
            # 其他生化指标
            'potassium': float, 'sodium': int, 'chloride': int, 'calcium': float,
            # 诊断信息
            'hypertension': int, 'diabetes': int, 'heart_disease': int, 'kidney_disease': int,
            # 目标变量
            'abnormal_flag': int
        }
        
        # 逐个列转换类型，捕获异常
        for col in self.df.columns:
            if col in dtype_map:
                try:
                    self.df[col] = self.df[col].astype(dtype_map[col])
                    print(f"  ✓ {col}列类型转换为{dtype_map[col].__name__}成功")
                except Exception as e:
                    # 处理转换失败的情况（强制清理异常值）
                    print(f"  ⚠ {col}列类型转换失败: {e}，自动清理异常值")
                    self.df[col] = pd.to_numeric(self.df[col], errors='coerce').fillna(self.df[col].median()).astype(dtype_map[col])
        
        # 第四步：处理重复值
        duplicate_count = self.df.duplicated().sum()
        if duplicate_count > 0:
            self.df = self.df.drop_duplicates()
            print(f"\n✓ 移除{duplicate_count}条重复数据，剩余样本数: {len(self.df)}")
        else:
            print(f"\n✓ 无重复数据，样本数保持: {len(self.df)}")
        
        # 最终数据验证
        print(f"\n最终数据验证:")
        print(f"可用样本数: {len(self.df)}")
        print(f"特征数: {len(self.df.columns)}")
        print(f"是否还有缺失值: {'否' if self.df.isnull().sum().sum() == 0 else '是'}")
        
        return self.df

# ===================== 2. 异常检测模型类 =====================
class OutlierDetectionSystem:
    """离群值检测系统（确保全部数据参与训练）"""
    
    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
        self.X_scaled = None  # 保存标准化后的数据，避免重复计算
    
    def train_models(self, X, contamination=0.1):
        """训练多种异常检测模型（确保全部数据参与）"""
        print("\n" + "="*60)
        print("2. 训练多种异常检测模型")
        print("="*60)
        print(f"参与训练的样本数: {len(X)}")
        print(f"参与训练的特征数: {X.shape[1]}")
        
        # 数据标准化（确保全部数据标准化）
        self.X_scaled = self.scaler.fit_transform(X)
        print(f"✓ 数据标准化完成，标准化后形状: {self.X_scaled.shape}")
        
        models_info = {}
        
        # 1. 孤立森林 (Isolation Forest) - 适合大数据集
        print("\n训练孤立森林模型...")
        try:
            start_time = time.time()
            if_model = IsolationForest(
                n_estimators=100,
                max_samples=len(X),  # 使用全部样本训练
                contamination=contamination,
                random_state=42,
                n_jobs=-1,  # 多线程加速
                verbose=0
            )
            # 确保全部数据参与训练和预测
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
        
        # 2. 局部离群因子 (LOF) - 修复novelty参数冲突
        print("训练LOF模型...")
        try:
            start_time = time.time()
            # LOF的novelty=False时，只能对训练数据预测，适合全部数据训练
            lof_model = LocalOutlierFactor(
                n_neighbors=min(20, len(X)-1),  # 适配小样本情况
                contamination=contamination,
                novelty=False,  # 必须为False才能使用fit_predict
                n_jobs=-1,
                leaf_size=30  # 优化内存使用
            )
            # 确保全部数据参与
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
        
        # 3. 一类支持向量机 (One-Class SVM) - 优化参数适配大数据
        print("训练One-Class SVM模型...")
        try:
            start_time = time.time()
            # 优化gamma参数，避免计算量过大
            gamma = 1 / (X.shape[1] * self.X_scaled.var()) if self.X_scaled.var() > 0 else 'scale'
            
            ocsvm_model = OneClassSVM(
                nu=contamination,  # 等同于contamination
                kernel='rbf',
                gamma=gamma,
                tol=1e-3  # 放宽容忍度，加速训练
            )
            # 确保全部数据参与训练
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
        
        # 显示训练结果摘要（验证全部数据参与）
        print("\n模型训练结果摘要:")
        for name, data in self.models.items():
            outliers = np.sum(data['predictions'] == -1)
            trained_samples = data['trained_samples']
            print(f"{name}: 训练样本数={trained_samples}, 检测到{outliers}个离群值 ({outliers/trained_samples*100:.1f}%)")
        
        return self.models
    
    def evaluate_models(self, X, true_labels=None):
        """评估模型性能（基于全部训练数据）"""
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
            
            # 如果有真实标签，进行有监督评估
            if true_labels is not None and len(true_labels) == trained_samples:
                # 将预测转换为二分类标签（1=正常，-1=异常 -> 0=正常，1=异常）
                pred_binary = np.where(predictions == 1, 0, 1)
                
                try:
                    result['accuracy'] = float(accuracy_score(true_labels, pred_binary))
                    result['precision'] = float(precision_score(true_labels, pred_binary, zero_division=0))
                    result['recall'] = float(recall_score(true_labels, pred_binary, zero_division=0))
                    result['f1_score'] = float(f1_score(true_labels, pred_binary, zero_division=0))
                    
                    # 检查是否有足够的类别进行ROC-AUC计算
                    if len(np.unique(true_labels)) > 1:
                        result['roc_auc'] = float(roc_auc_score(true_labels, scores))
                    
                    # 混淆矩阵
                    cm = confusion_matrix(true_labels, pred_binary)
                    result['confusion_matrix'] = cm.tolist()
                except Exception as e:
                    print(f"评估{model_name}时出错: {e}")
            
            evaluation_results[model_name] = result
            
            # 打印结果
            print(f"\n{model_name} 结果:")
            print(f"  训练样本数: {result['trained_samples']}")
            print(f"  离群值数量: {result['outlier_count']} ({result['outlier_ratio']:.1%})")
            print(f"  训练时间: {result['training_time']:.2f}秒")
            if 'accuracy' in result:
                print(f"  准确率: {result['accuracy']:.4f}")
                print(f"  F1分数: {result['f1_score']:.4f}")
        
        return evaluation_results

# ===================== 3. 医学符合性验证类 =====================
class ClinicalValidator:
    """医学符合性验证"""
    
    def __init__(self, clinical_rules):
        self.clinical_rules = clinical_rules
        
    def validate_outliers(self, df, outlier_indices, model_name):
        """验证离群值的医学合理性"""
        print(f"\n{model_name} - 医学符合性验证:")
        print("-"*40)
        
        validation_results = {
            'total_outliers': len(outlier_indices),
            'clinically_plausible': 0,
            'clinically_implausible': 0,
            'clinically_critical': 0,
            'detailed_violations': []
        }
        
        # 确保只检查存在的样本
        valid_indices = [idx for idx in outlier_indices if idx < len(df)]
        if len(valid_indices) == 0:
            print("  无有效离群值可验证")
            return validation_results
        
        # 检查所有离群值（不限于前50个）
        for idx in valid_indices:
            sample = df.iloc[idx]
            violations = []
            
            for feature, rules in self.clinical_rules.items():
                if feature in sample:
                    value = sample[feature]
                    
                    if 'min' in rules and 'max' in rules:
                        if value < rules['min'] or value > rules['max']:
                            severity = 'CRITICAL' if (value < rules.get('critical_min', rules['min']) or 
                                                    value > rules.get('critical_max', rules['max'])) else 'WARNING'
                            violations.append({
                                'feature': feature,
                                'value': float(value) if isinstance(value, (int, float)) else value,
                                'severity': severity,
                                'normal_range': f"{rules['min']}-{rules['max']}"
                            })
            
            if violations:
                validation_results['detailed_violations'].append({
                    'sample_index': int(idx),
                    'violations': violations
                })
                
                # 统计严重程度
                if any(v['severity'] == 'CRITICAL' for v in violations):
                    validation_results['clinically_critical'] += 1
                    validation_results['clinically_implausible'] += 1
                elif any(v['severity'] == 'WARNING' for v in violations):
                    validation_results['clinically_implausible'] += 1
            else:
                validation_results['clinically_plausible'] += 1
        
        # 计算比例
        total = validation_results['total_outliers']
        if total > 0:
            validation_results['plausible_ratio'] = validation_results['clinically_plausible'] / total
            validation_results['implausible_ratio'] = validation_results['clinically_implausible'] / total
            validation_results['critical_ratio'] = validation_results['clinically_critical'] / total
            
            print(f"  总离群值数: {total}")
            print(f"  医学合理离群值: {validation_results['clinically_plausible']} ({validation_results['plausible_ratio']:.1%})")
            print(f"  医学可疑离群值: {validation_results['clinically_implausible']} ({validation_results['implausible_ratio']:.1%})")
            if validation_results['clinically_critical'] > 0:
                print(f"  其中严重异常: {validation_results['clinically_critical']} ({validation_results['critical_ratio']:.1%})")
        
        return validation_results

# ===================== 4. 可视化分析类 =====================
class VisualizationAnalyzer:
    """可视化分析 - 修复图表展示问题"""
    
    def __init__(self):
        self.figsize = (30, 24)
    
    def plot_basic_analysis(self, df, models_results):
        """绘制专业医疗数据可视化分析图表"""
        # 设置美观的颜色方案
        colors = {
            'primary': '#4ECDC4',
            'secondary': '#FF6B6B',
            'accent': '#45B7D1',
            'light': '#F7F7F7',
            'dark': '#2C3E50'
        }
        
        # 创建图表，增加子图间距
        fig, axes = plt.subplots(4, 2, figsize=self.figsize)
        fig.patch.set_facecolor(colors['light'])
        # 调整子图间距
        plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, wspace=0.3, hspace=0.35)
        
        # 1. 模型对比柱状图
        ax1 = axes[0, 0]
        model_names = list(models_results.keys())
        outlier_counts = [results['outlier_count'] for results in models_results.values()]
        
        # 美化柱状图
        bars = ax1.bar(model_names, outlier_counts, 
                      color=[colors['primary'], colors['secondary'], colors['accent']][:len(model_names)],
                      edgecolor=colors['dark'], linewidth=1.5)
        ax1.set_title('各模型检测离群值数量', fontsize=16, fontweight='bold', color=colors['dark'])
        ax1.set_ylabel('离群值数量', fontsize=14, color=colors['dark'])
        ax1.grid(axis='y', alpha=0.2, linestyle='--')
        ax1.tick_params(axis='x', rotation=15, labelsize=12, color=colors['dark'])
        ax1.tick_params(axis='y', labelsize=12, color=colors['dark'])
        ax1.set_facecolor('white')
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        
        # 添加数据标签
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height+20,
                    f'{int(height)}', ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        # 2. 异常得分分布
        ax2 = axes[0, 1]
        for idx, (model_name, results) in enumerate(models_results.items()):
            if 'scores' in results:
                scores = results['scores']
                scores_scaled = (scores - np.mean(scores)) / np.std(scores)
                ax2.hist(scores_scaled, bins=20, alpha=0.6, 
                        color=[colors['primary'], colors['secondary'], colors['accent']][idx % 3], 
                        label=f'{model_name}', density=True, edgecolor='white', linewidth=0.5)
        ax2.set_title('异常得分分布（标准化后）', fontsize=16, fontweight='bold', color=colors['dark'])
        ax2.set_xlabel('标准化异常得分', fontsize=14, color=colors['dark'])
        ax2.set_ylabel('密度', fontsize=14, color=colors['dark'])
        ax2.legend(fontsize=12, loc='upper right')
        ax2.grid(alpha=0.2, linestyle='--')
        ax2.tick_params(labelsize=12, color=colors['dark'])
        ax2.set_facecolor('white')
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        
        # 3. 关键代谢指标分布对比
        ax3 = axes[1, 0]
        if all(col in df.columns for col in ['chol', 'blood_sugar', 'triglycerides']):
            model_name = list(models_results.keys())[0]
            predictions = models_results[model_name]['predictions']
            
            valid_mask = np.arange(len(predictions)) < len(df)
            normal_mask = (predictions == 1) & valid_mask
            outlier_mask = (predictions == -1) & valid_mask
            
            # 准备数据
            params = ['chol', 'blood_sugar', 'triglycerides']
            param_names = ['胆固醇', '血糖', '甘油三酯']
            
            # 使用更对比鲜明的颜色
            normal_color = '#4ECDC4'  # 正常样本颜色
            outlier_color = '#FF6B6B'  # 离群样本颜色
            
            for i, (param, name) in enumerate(zip(params, param_names)):
                normal_data = df.loc[normal_mask, param]
                outlier_data = df.loc[outlier_mask, param]
                
                # 绘制箱线图
                box = ax3.boxplot([normal_data, outlier_data], positions=[i*2+1, i*2+2], 
                               patch_artist=True, showfliers=True, notch=True)
                # 设置颜色
                box['boxes'][0].set_facecolor(normal_color)
                box['boxes'][0].set_edgecolor('#2C3E50')
                box['boxes'][0].set_linewidth(1.5)
                box['boxes'][1].set_facecolor(outlier_color)
                box['boxes'][1].set_edgecolor('#2C3E50')
                box['boxes'][1].set_linewidth(1.5)
                # 安全设置中位数颜色
                if 'medians' in box:
                    for median in box['medians']:
                        median.set_color('#2C3E50')
                        median.set_linewidth(2)
                # 设置 whiskers 和 caps
                for whisker in box['whiskers']:
                    whisker.set_color('#2C3E50')
                    whisker.set_linewidth(1)
                for cap in box['caps']:
                    cap.set_color('#2C3E50')
                    cap.set_linewidth(1)
                # 设置 fliers
                if 'fliers' in box:
                    for flier in box['fliers']:
                        flier.set_marker('o')
                        flier.set_markersize(3)
                        flier.set_color('#2C3E50')
                
                # 计算统计信息
                normal_median = normal_data.median()
                normal_mean = normal_data.mean()
                normal_std = normal_data.std()
                outlier_median = outlier_data.median()
                outlier_mean = outlier_data.mean()
                outlier_std = outlier_data.std()
                
                # 添加详细数据标签（使用更大的字体和更明显的位置）
                y_range = ax3.get_ylim()[1] - ax3.get_ylim()[0]
                # 正常样本
                ax3.text(i*2+1, ax3.get_ylim()[1] - y_range*0.1, 
                        f'正常样本\n median: {normal_median:.1f}\n mean: {normal_mean:.1f}\n std: {normal_std:.1f}', 
                        ha='center', va='top', fontsize=11, color='#2C3E50', fontweight='bold',
                        bbox=dict(facecolor='white', edgecolor=normal_color, boxstyle='round,pad=0.5'))
                # 离群样本
                ax3.text(i*2+2, ax3.get_ylim()[1] - y_range*0.1, 
                        f'离群样本\n median: {outlier_median:.1f}\n mean: {outlier_mean:.1f}\n std: {outlier_std:.1f}', 
                        ha='center', va='top', fontsize=11, color='#2C3E50', fontweight='bold',
                        bbox=dict(facecolor='white', edgecolor=outlier_color, boxstyle='round,pad=0.5'))
            
            ax3.set_title('关键代谢指标分布对比', fontsize=18, fontweight='bold', color='#2C3E50')
            ax3.set_ylabel('数值', fontsize=16, color='#2C3E50')
            ax3.set_xticks([1.5, 3.5, 5.5])
            ax3.set_xticklabels(param_names, fontsize=14, color='#2C3E50', fontweight='bold')
            ax3.grid(alpha=0.3, linestyle='--', color='#CCCCCC')
            ax3.tick_params(labelsize=14, color='#2C3E50')
            ax3.set_facecolor('#F7F7F7')
            ax3.spines['top'].set_visible(False)
            ax3.spines['right'].set_visible(False)
            ax3.spines['left'].set_color('#2C3E50')
            ax3.spines['bottom'].set_color('#2C3E50')
            
            # 添加更明显的图例
            ax3.legend(['正常样本', '离群样本'], fontsize=14, loc='upper left', 
                      bbox_to_anchor=(0.01, 0.99), frameon=True, 
                      facecolor='white', edgecolor='#2C3E50')
        
        # 4. 血常规指标分布
        ax4 = axes[1, 1]
        if all(col in df.columns for col in ['wbc', 'hb', 'plt', 'lymphocyte']):
            model_name = list(models_results.keys())[0]
            predictions = models_results[model_name]['predictions']
            
            valid_mask = np.arange(len(predictions)) < len(df)
            normal_mask = (predictions == 1) & valid_mask
            outlier_mask = (predictions == -1) & valid_mask
            
            # 准备数据
            blood_params = ['wbc', 'hb', 'plt', 'lymphocyte']
            param_names = ['白细胞', '血红蛋白', '血小板', '淋巴细胞']
            
            for i, (param, name) in enumerate(zip(blood_params, param_names)):
                normal_data = df.loc[normal_mask, param]
                outlier_data = df.loc[outlier_mask, param]
                
                # 绘制小提琴图
                violin = ax4.violinplot([normal_data, outlier_data], positions=[i*2+1, i*2+2], 
                                     showmedians=True, showextrema=True)
                # 设置颜色
                for j, pc in enumerate(violin['bodies']):
                    color = colors['primary'] if j % 2 == 0 else colors['secondary']
                    pc.set_facecolor(color)
                    pc.set_edgecolor(colors['dark'])
                    pc.set_alpha(0.7)
                # 安全设置中位数颜色
                if 'medians' in violin:
                    for median in violin['medians']:
                        median.set_color(colors['dark'])
                
                # 添加详细数据标签
                # 正常样本
                normal_median = normal_data.median()
                normal_mean = normal_data.mean()
                ax4.text(i*2+1, normal_median + (ax4.get_ylim()[1] - ax4.get_ylim()[0])*0.02, 
                        f'med: {normal_median:.1f}\nmean: {normal_mean:.1f}', 
                        ha='center', va='bottom', fontsize=10, color=colors['dark'], fontweight='bold')
                # 离群样本
                outlier_median = outlier_data.median()
                outlier_mean = outlier_data.mean()
                ax4.text(i*2+2, outlier_median + (ax4.get_ylim()[1] - ax4.get_ylim()[0])*0.02, 
                        f'med: {outlier_median:.1f}\nmean: {outlier_mean:.1f}', 
                        ha='center', va='bottom', fontsize=10, color=colors['dark'], fontweight='bold')
            
            ax4.set_title('血常规指标分布', fontsize=16, fontweight='bold', color=colors['dark'])
            ax4.set_ylabel('数值', fontsize=14, color=colors['dark'])
            ax4.set_xticks([1.5, 3.5, 5.5, 7.5])
            ax4.set_xticklabels(param_names, fontsize=12, color=colors['dark'])
            ax4.grid(alpha=0.2, linestyle='--')
            ax4.tick_params(labelsize=12, color=colors['dark'])
            ax4.set_facecolor('white')
            ax4.spines['top'].set_visible(False)
            ax4.spines['right'].set_visible(False)
        
        # 5. 肝功能与肾功能指标
        ax5 = axes[2, 0]
        if all(col in df.columns for col in ['alt', 'ast', 'creatinine', 'blood_urea']):
            model_name = list(models_results.keys())[0]
            predictions = models_results[model_name]['predictions']
            
            valid_mask = np.arange(len(predictions)) < len(df)
            normal_mask = (predictions == 1) & valid_mask
            outlier_mask = (predictions == -1) & valid_mask
            
            # 准备数据
            liver_params = ['alt', 'ast']
            kidney_params = ['creatinine', 'blood_urea']
            
            # 肝功能
            for i, param in enumerate(liver_params):
                normal_data = df.loc[normal_mask, param]
                outlier_data = df.loc[outlier_mask, param]
                box = ax5.boxplot([normal_data, outlier_data], positions=[i*2+1, i*2+2], 
                               patch_artist=True, showfliers=False)
                box['boxes'][0].set_facecolor(colors['primary'])
                box['boxes'][0].set_edgecolor(colors['dark'])
                box['boxes'][1].set_facecolor(colors['secondary'])
                box['boxes'][1].set_edgecolor(colors['dark'])
                if 'medians' in box:
                    for median in box['medians']:
                        median.set_color(colors['dark'])
                
                # 添加详细数据标签
                # 正常样本
                normal_median = normal_data.median()
                normal_mean = normal_data.mean()
                ax5.text(i*2+1, normal_median + (ax5.get_ylim()[1] - ax5.get_ylim()[0])*0.02, 
                        f'med: {normal_median:.1f}\nmean: {normal_mean:.1f}', 
                        ha='center', va='bottom', fontsize=10, color=colors['dark'], fontweight='bold')
                # 离群样本
                outlier_median = outlier_data.median()
                outlier_mean = outlier_data.mean()
                ax5.text(i*2+2, outlier_median + (ax5.get_ylim()[1] - ax5.get_ylim()[0])*0.02, 
                        f'med: {outlier_median:.1f}\nmean: {outlier_mean:.1f}', 
                        ha='center', va='bottom', fontsize=10, color=colors['dark'], fontweight='bold')
            
            # 肾功能
            for i, param in enumerate(kidney_params):
                normal_data = df.loc[normal_mask, param]
                outlier_data = df.loc[outlier_mask, param]
                box = ax5.boxplot([normal_data, outlier_data], positions=[4 + i*2+1, 4 + i*2+2], 
                               patch_artist=True, showfliers=False)
                box['boxes'][0].set_facecolor(colors['primary'])
                box['boxes'][0].set_edgecolor(colors['dark'])
                box['boxes'][1].set_facecolor(colors['secondary'])
                box['boxes'][1].set_edgecolor(colors['dark'])
                if 'medians' in box:
                    for median in box['medians']:
                        median.set_color(colors['dark'])
                
                # 添加详细数据标签
                # 正常样本
                normal_median = normal_data.median()
                normal_mean = normal_data.mean()
                ax5.text(4 + i*2+1, normal_median + (ax5.get_ylim()[1] - ax5.get_ylim()[0])*0.02, 
                        f'med: {normal_median:.1f}\nmean: {normal_mean:.1f}', 
                        ha='center', va='bottom', fontsize=10, color=colors['dark'], fontweight='bold')
                # 离群样本
                outlier_median = outlier_data.median()
                outlier_mean = outlier_data.mean()
                ax5.text(4 + i*2+2, outlier_median + (ax5.get_ylim()[1] - ax5.get_ylim()[0])*0.02, 
                        f'med: {outlier_median:.1f}\nmean: {outlier_mean:.1f}', 
                        ha='center', va='bottom', fontsize=10, color=colors['dark'], fontweight='bold')
            
            ax5.set_title('肝肾功能指标分布', fontsize=16, fontweight='bold', color=colors['dark'])
            ax5.set_ylabel('数值', fontsize=14, color=colors['dark'])
            ax5.set_xticks([1.5, 3.5, 5.5, 7.5])
            ax5.set_xticklabels(['ALT', 'AST', '肌酐', '尿素'], fontsize=12, color=colors['dark'])
            ax5.grid(alpha=0.2, linestyle='--')
            ax5.tick_params(labelsize=12, color=colors['dark'])
            ax5.set_facecolor('white')
            ax5.spines['top'].set_visible(False)
            ax5.spines['right'].set_visible(False)
        
        # 6. 血脂谱分析
        ax6 = axes[2, 1]
        if all(col in df.columns for col in ['ldl', 'hdl', 'triglycerides']):
            model_name = list(models_results.keys())[0]
            predictions = models_results[model_name]['predictions']
            
            valid_mask = np.arange(len(predictions)) < len(df)
            normal_mask = (predictions == 1) & valid_mask
            outlier_mask = (predictions == -1) & valid_mask
            
            # 准备数据
            lipid_params = ['ldl', 'hdl', 'triglycerides']
            param_names = ['LDL', 'HDL', '甘油三酯']
            
            for i, (param, name) in enumerate(zip(lipid_params, param_names)):
                normal_data = df.loc[normal_mask, param]
                outlier_data = df.loc[outlier_mask, param]
                
                # 绘制箱线图
                box = ax6.boxplot([normal_data, outlier_data], positions=[i*2+1, i*2+2], 
                               patch_artist=True, showfliers=False)
                box['boxes'][0].set_facecolor(colors['primary'])
                box['boxes'][0].set_edgecolor(colors['dark'])
                box['boxes'][1].set_facecolor(colors['secondary'])
                box['boxes'][1].set_edgecolor(colors['dark'])
                if 'medians' in box:
                    for median in box['medians']:
                        median.set_color(colors['dark'])
                
                # 添加详细数据标签
                # 正常样本
                normal_median = normal_data.median()
                normal_mean = normal_data.mean()
                ax6.text(i*2+1, normal_median + (ax6.get_ylim()[1] - ax6.get_ylim()[0])*0.02, 
                        f'med: {normal_median:.2f}\nmean: {normal_mean:.2f}', 
                        ha='center', va='bottom', fontsize=10, color=colors['dark'], fontweight='bold')
                # 离群样本
                outlier_median = outlier_data.median()
                outlier_mean = outlier_data.mean()
                ax6.text(i*2+2, outlier_median + (ax6.get_ylim()[1] - ax6.get_ylim()[0])*0.02, 
                        f'med: {outlier_median:.2f}\nmean: {outlier_mean:.2f}', 
                        ha='center', va='bottom', fontsize=10, color=colors['dark'], fontweight='bold')
            
            ax6.set_title('血脂谱分析', fontsize=16, fontweight='bold', color=colors['dark'])
            ax6.set_ylabel('mmol/L', fontsize=14, color=colors['dark'])
            ax6.set_xticks([1.5, 3.5, 5.5])
            ax6.set_xticklabels(param_names, fontsize=12, color=colors['dark'])
            ax6.grid(alpha=0.2, linestyle='--')
            ax6.tick_params(labelsize=12, color=colors['dark'])
            ax6.set_facecolor('white')
            ax6.spines['top'].set_visible(False)
            ax6.spines['right'].set_visible(False)
        
        # 7. 电解质平衡分析
        ax7 = axes[3, 0]
        if all(col in df.columns for col in ['potassium', 'sodium', 'calcium']):
            model_name = list(models_results.keys())[0]
            predictions = models_results[model_name]['predictions']
            
            valid_mask = np.arange(len(predictions)) < len(df)
            normal_mask = (predictions == 1) & valid_mask
            outlier_mask = (predictions == -1) & valid_mask
            
            # 准备数据
            electrolyte_params = ['potassium', 'sodium', 'calcium']
            param_names = ['血钾', '血钠', '血钙']
            
            for i, (param, name) in enumerate(zip(electrolyte_params, param_names)):
                normal_data = df.loc[normal_mask, param]
                outlier_data = df.loc[outlier_mask, param]
                
                # 绘制箱线图
                box = ax7.boxplot([normal_data, outlier_data], positions=[i*2+1, i*2+2], 
                               patch_artist=True, showfliers=False)
                box['boxes'][0].set_facecolor(colors['primary'])
                box['boxes'][0].set_edgecolor(colors['dark'])
                box['boxes'][1].set_facecolor(colors['secondary'])
                box['boxes'][1].set_edgecolor(colors['dark'])
                if 'medians' in box:
                    for median in box['medians']:
                        median.set_color(colors['dark'])
                
                # 添加详细数据标签
                # 正常样本
                normal_median = normal_data.median()
                normal_mean = normal_data.mean()
                ax7.text(i*2+1, normal_median + (ax7.get_ylim()[1] - ax7.get_ylim()[0])*0.02, 
                        f'med: {normal_median:.2f}\nmean: {normal_mean:.2f}', 
                        ha='center', va='bottom', fontsize=10, color=colors['dark'], fontweight='bold')
                # 离群样本
                outlier_median = outlier_data.median()
                outlier_mean = outlier_data.mean()
                ax7.text(i*2+2, outlier_median + (ax7.get_ylim()[1] - ax7.get_ylim()[0])*0.02, 
                        f'med: {outlier_median:.2f}\nmean: {outlier_mean:.2f}', 
                        ha='center', va='bottom', fontsize=10, color=colors['dark'], fontweight='bold')
            
            ax7.set_title('电解质平衡分析', fontsize=16, fontweight='bold', color=colors['dark'])
            ax7.set_ylabel('mmol/L', fontsize=14, color=colors['dark'])
            ax7.set_xticks([1.5, 3.5, 5.5])
            ax7.set_xticklabels(param_names, fontsize=12, color=colors['dark'])
            ax7.grid(alpha=0.2, linestyle='--')
            ax7.tick_params(labelsize=12, color=colors['dark'])
            ax7.set_facecolor('white')
            ax7.spines['top'].set_visible(False)
            ax7.spines['right'].set_visible(False)
        
        # 8. 模型训练时间对比
        ax8 = axes[3, 1]
        times = [results['training_time'] for results in models_results.values()]
        bars = ax8.bar(model_names, times, 
                      color=[colors['primary'], colors['secondary'], colors['accent']][:len(model_names)],
                      edgecolor=colors['dark'], linewidth=1.5)
        ax8.set_title('模型训练时间对比', fontsize=16, fontweight='bold', color=colors['dark'])
        ax8.set_ylabel('时间(秒)', fontsize=14, color=colors['dark'])
        ax8.grid(axis='y', alpha=0.2, linestyle='--')
        ax8.tick_params(axis='x', rotation=15, labelsize=12, color=colors['dark'])
        ax8.tick_params(axis='y', labelsize=12, color=colors['dark'])
        ax8.set_facecolor('white')
        ax8.spines['top'].set_visible(False)
        ax8.spines['right'].set_visible(False)
        
        # 添加数据标签
        for bar in bars:
            height = bar.get_height()
            ax8.text(bar.get_x() + bar.get_width()/2., height+0.1,
                    f'{height:.2f}s', ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        # 主标题
        plt.suptitle('专业医疗数据离群值检测分析', 
                    fontsize=20, fontweight='bold', color=colors['dark'], y=0.98)
        
        # 副标题
        plt.figtext(0.5, 0.94, f'总样本数: {len(df):,} 条 | 检查项目: {len(df.columns)-2} 项', 
                   ha='center', fontsize=14, color=colors['dark'])
        
        plt.tight_layout(rect=[0, 0, 1, 0.92])
        
        return fig
# ===================== 5. 主程序=====================
def main(generate_data=False):
    """主函数 - 离群值检测系统"""
    print("="*60)
    print("医疗数据离群值检测系统 - 一键整合版")
    print("="*60)
    
    # 配置路径（使用相对路径）
    data_path = 'heart.csv'
    output_dir = 'results'
    
    # ========== 第一步：生成专业医疗检查数据（可选） ==========
    if generate_data:
        print("\n[第一步] 生成专业医疗检查数据")
        try:
            # 生成并替换heart.csv
            generate_professional_medical_data(data_path, sample_size=100000)
        except Exception as e:
            print(f"❌ 生成数据失败: {e}")
            return
    else:
        print("\n[第一步] 使用现有医疗数据")
        if os.path.exists(data_path):
            print(f"✓ 找到现有数据文件: {data_path}")
            existing_df = pd.read_csv(data_path)
            print(f"✓ 现有数据形状: {existing_df.shape}")
            print(f"✓ 现有数据样本数: {len(existing_df)}")
        else:
            print("❌ 未找到数据文件，请先运行 generate_data=True 生成数据")
            return
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n输出目录: {output_dir}")
    
    # ========== 第二步：数据加载与预处理 ==========
    print("\n[第二步] 数据加载与预处理")
    try:
        processor = MedicalDataProcessor(data_path)
        df = processor.load_and_preprocess()
        
        if df is None or len(df) == 0:
            print("数据加载失败或无可用样本，程序退出")
            return
        
        # 验证数据量（关键！确保是100000条）
        if len(df) != 100000:
            print(f"⚠ 警告：当前数据量={len(df)}，预期100000条！")
        else:
            print(f"✓ 数据量验证通过：{len(df)}条")
        
        # 显示数据基本统计
        print(f"\n数据前5行:")
        print(df.head())
        
    except Exception as e:
        print(f"数据加载失败: {e}")
        return
    
    # ========== 第三步：准备特征数据 ==========
    print("\n[第三步] 准备特征数据")
    # 检查是否有目标变量
    if 'abnormal_flag' in df.columns:
        # 排除非特征列
        non_feature_cols = ['patient_id', 'abnormal_flag']
        feature_cols = [col for col in df.columns if col not in non_feature_cols]
        X = df[feature_cols].copy()
        y = df['abnormal_flag'].values
        print(f"特征数据形状: {X.shape} (全部{len(X)}条样本参与训练)")
        print(f"目标变量分布: 正常={np.sum(y==0)}, 异常={np.sum(y==1)}")
        print(f"特征数量: {len(feature_cols)}")
    else:
        # 排除非特征列
        non_feature_cols = ['patient_id']
        feature_cols = [col for col in df.columns if col not in non_feature_cols]
        X = df[feature_cols].copy()
        y = None
        print(f"使用所有数据作为特征: {X.shape} (全部{len(X)}条样本参与训练)")
    
    # 确保特征数据无缺失值
    if X.isnull().sum().sum() > 0:
        print("⚠ 特征数据存在缺失值，已自动填充")
        X = X.fillna(X.median())
    
    # ========== 第四步：训练异常检测模型 ==========
    print("\n[第四步] 训练异常检测模型")
    detection_system = OutlierDetectionSystem()
    try:
        models = detection_system.train_models(X, contamination=0.1)
        if not models:
            print("没有成功训练任何模型，程序退出")
            return
    except Exception as e:
        print(f"模型训练失败: {e}")
        return
    
    # ========== 第五步：模型性能评估 ==========
    print("\n[第五步] 模型性能评估")
    try:
        evaluation_results = detection_system.evaluate_models(X, y)
    except Exception as e:
        print(f"模型评估失败: {e}")
        evaluation_results = {}
    
    # ========== 第六步：医学符合性验证 ==========
    print("\n[第六步] 医学符合性验证")
    clinical_results = {}
    try:
        clinical_validator = ClinicalValidator(processor.clinical_rules)
        for model_name, model_data in models.items():
            if 'predictions' in model_data:
                predictions = model_data['predictions']
                outlier_indices = np.where(predictions == -1)[0]
                print(f"\n{model_name} 检测到离群值索引数量: {len(outlier_indices)}")
                validation_results = clinical_validator.validate_outliers(
                    df, outlier_indices, model_name
                )
                clinical_results[model_name] = validation_results
    except Exception as e:
        print(f"医学验证失败: {e}")
    
    # ========== 第七步：可视化分析 ==========
    print("\n[第七步] 生成可视化分析")
    try:
        visualizer = VisualizationAnalyzer()
        # 准备结果数据
        models_results = {}
        for model_name, model_data in models.items():
            models_results[model_name] = {
                'predictions': model_data['predictions'],
                'scores': model_data['scores'],
                'outlier_count': np.sum(model_data['predictions'] == -1),
                'outlier_ratio': np.mean(model_data['predictions'] == -1),
                'training_time': model_data['time'],
                'trained_samples': model_data['trained_samples']
            }
        # 生成图表
        fig = visualizer.plot_basic_analysis(df, models_results)
        # 保存图表
        output_path = os.path.join(output_dir, "analysis_results.png")
        fig.savefig(output_path, dpi=600, bbox_inches='tight')
        print(f"✓ 可视化图表已保存至: {output_path}")
        # 显示图表
        plt.show()
    except Exception as e:
        print(f"可视化生成失败: {e}")
    
    # ========== 第八步：保存详细结果 ==========
    print("\n[第八步] 保存详细结果")
    try:
        # 为每个模型保存完整结果
        for model_name, model_data in models.items():
            if 'predictions' in model_data:
                result_df = df.copy()
                result_df[f'{model_name}_prediction'] = model_data['predictions']
                result_df[f'{model_name}_score'] = model_data['scores']
                result_df[f'{model_name}_is_outlier'] = result_df[f'{model_name}_prediction'].map({1: '正常', -1: '离群值'})
                # 保存结果
                output_file = os.path.join(output_dir, f"detection_results_{model_name.replace(' ', '_')}.csv")
                result_df.to_csv(output_file, index=False, encoding='utf-8-sig')
                print(f"✓ {model_name} 完整结果已保存至: {output_file}")
    except Exception as e:
        print(f"保存结果失败: {e}")
    
    # ========== 第九步：保存评估报告 ==========
    print("\n[第九步] 保存评估报告")
    try:
        # 转换numpy类型为Python原生类型
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_numpy(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            else:
                return obj
        
        report = {
            'project_info': {
                'name': '医疗数据离群值检测系统',
                'author': '范文兵',
                'student_id': '2250300284',
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            },
            'data_info': {
                'file_path': data_path,
                'total_samples': len(df),
                'used_samples': len(X),
                'shape': df.shape,
                'columns': list(df.columns)
            },
            'model_results': convert_numpy(evaluation_results),
            'clinical_validation': convert_numpy(clinical_results)
        }
        report_file = os.path.join(output_dir, "evaluation_report.json")
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"✓ 评估报告已保存至: {report_file}")
    except Exception as e:
        print(f"保存评估报告失败: {e}")
    
    # ========== 最终总结 ==========
    print("\n" + "="*60)
    print("程序运行完成！最终总结")
    print("="*60)
    print(f"✅ 生成并使用的数据量: {len(df)}条（预期100000条）")
    print(f"✅ 成功训练模型数: {len(models)}个")
    print(f"✅ 所有结果已保存至: {output_dir}")
    print(f"\n生成的文件列表:")
    for file in os.listdir(output_dir):
        if os.path.isfile(os.path.join(output_dir, file)):
            file_size = os.path.getsize(os.path.join(output_dir, file)) / 1024 / 1024
            print(f"  - {file} ({file_size:.2f} MB)")

# ===================== 程序入口 =====================
if __name__ == "__main__":
    # 强制使用多核计算（加速训练）
    os.environ["OMP_NUM_THREADS"] = "4"
    os.environ["MKL_NUM_THREADS"] = "4"
    
    # 运行主函数 - 默认使用现有数据，不重新生成
    # 如果需要重新生成数据，请将 generate_data=True
    main(generate_data=False)