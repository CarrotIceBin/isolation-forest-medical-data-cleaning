"""
医学验证模块 - 负责验证离群值的医学合理性
"""


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
        
        valid_indices = [idx for idx in outlier_indices if idx < len(df)]
        if len(valid_indices) == 0:
            print("  无有效离群值可验证")
            return validation_results
        
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
                
                if any(v['severity'] == 'CRITICAL' for v in violations):
                    validation_results['clinically_critical'] += 1
                    validation_results['clinically_implausible'] += 1
                elif any(v['severity'] == 'WARNING' for v in violations):
                    validation_results['clinically_implausible'] += 1
            else:
                validation_results['clinically_plausible'] += 1
        
        total = validation_results['total_outliers']
        if total > 0:
            validation_results['plausible_ratio'] = validation_results['clinically_plausible'] / total
            validation_results['implausible_ratio'] = validation_results['clinically_implausible'] / total
            validation_results['critical_ratio'] = validation_results['clinically_critical'] / total
            
            print(f"  总离群值数: {total}")
            print(f"  医学合理: {validation_results['clinically_plausible']} ({validation_results['plausible_ratio']:.1%})")
            print(f"  医学可疑: {validation_results['clinically_implausible']} ({validation_results['implausible_ratio']:.1%})")
        
        return validation_results