"""
数据生成模块 - 负责生成模拟医疗数据
"""
import numpy as np
import pandas as pd
import os


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
        'age': (18, 90),
        'sex': [0, 1],
        'smoking': [0, 1, 2],
        'alcohol': [0, 1, 2],
        'exercise_frequency': [0, 1, 2, 3],
        
        # 心脏病相关
        'cp': [0, 1, 2, 3],
        'trestbps': (80, 220),
        'chol': (100, 600),
        'fbs': [0, 1],
        'restecg': [0, 1, 2],
        'thalach': (70, 220),
        'exang': [0, 1],
        'oldpeak': (0.0, 6.5),
        'slope': [0, 1, 2],
        'ca': [0, 1, 2, 3, 4],
        'thal': [0, 1, 2, 3, 7],
        
        # 其他常规检查
        'blood_sugar': (3.9, 15.0),
        'blood_urea': (2.5, 15.0),
        'creatinine': (53, 200),
        'bilirubin': (5.1, 35.0),
        'alt': (5, 100),
        'ast': (5, 100),
        'albumin': (35, 55),
        'globulin': (20, 50),
        'a_g_ratio': (1.0, 2.5),
        
        # 血常规
        'wbc': (3.5, 30.0),
        'rbc': (3.5, 6.5),
        'hb': (100, 180),
        'plt': (80, 400),
        'lymphocyte': (0.8, 4.0),
        'neutrophil': (1.8, 8.0),
        
        # 血脂
        'ldl': (1.0, 5.0),
        'hdl': (0.8, 2.0),
        'triglycerides': (0.5, 5.0),
        
        # 肾功能
        'urine_protein': [0, 1, 2, 3],
        'urine_glucose': [0, 1, 2, 3],
        
        # 其他生化指标
        'potassium': (3.5, 5.5),
        'sodium': (135, 145),
        'chloride': (96, 106),
        'calcium': (2.1, 2.6),
        
        # 诊断信息
        'hypertension': [0, 1],
        'diabetes': [0, 1],
        'heart_disease': [0, 1],
        'kidney_disease': [0, 1],
        
        # 目标变量
        'abnormal_flag': [0, 1]
    }

    # 基于原始样本的统计特征（均值、标准差）
    stats = {
        'age': (54, 15),
        'trestbps': (120, 15),
        'chol': (200, 40),
        'thalach': (140, 20),
        'oldpeak': (0.8, 1.0),
        'blood_sugar': (5.0, 0.8),
        'blood_urea': (5.0, 1.5),
        'creatinine': (75, 15),
        'bilirubin': (12.0, 3.0),
        'alt': (25, 10),
        'ast': (25, 10),
        'albumin': (42, 3),
        'globulin': (28, 3),
        'a_g_ratio': (1.5, 0.2),
        'wbc': (6.5, 1.5),
        'rbc': (4.8, 0.3),
        'hb': (145, 10),
        'plt': (220, 50),
        'lymphocyte': (1.8, 0.5),
        'neutrophil': (4.0, 1.0),
        'ldl': (2.5, 0.8),
        'hdl': (1.2, 0.3),
        'triglycerides': (1.5, 0.8),
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
        
        # 心脏病相关指标
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
        
        # 诊断信息
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
        
        # 目标变量
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
    
    # 保存到指定路径
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