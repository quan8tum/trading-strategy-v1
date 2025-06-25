#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化特征重要性分析
快速分析特征重要性，不包含可视化部分
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import f_regression, mutual_info_regression
from sklearn.model_selection import train_test_split
import lightgbm as lgb
import json
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class SimplifiedFeatureAnalyzer:
    def __init__(self):
        self.feature_importance = {}
        self.feature_rankings = {}
        self.results = {}
        
    def load_data(self):
        """加载数据"""
        print("📊 加载数据...")
        data = pd.read_csv('data/processed_features.csv')
        print(f"✅ 数据加载成功: {data.shape}")
        return data
    
    def prepare_features(self, data, target_col='future_return_1'):
        """准备特征和目标变量"""
        print("🔧 准备特征数据...")
        
        # 创建目标变量
        if target_col not in data.columns:
            print(f"⚠️ 目标列 {target_col} 不存在，创建未来收益率...")
            data[target_col] = data['close'].pct_change().shift(-1)
        
        # 选择特征列
        feature_cols = [col for col in data.columns if col not in 
                       ['date', 'symbol', 'close', 'open', 'high', 'low', 'volume', target_col]]
        
        print(f"📋 可用特征数量: {len(feature_cols)}")
        
        # 准备数据
        X = data[feature_cols].fillna(0)
        y = data[target_col].fillna(0)
        
        # 移除无效数据
        valid_idx = ~(np.isinf(X).any(axis=1) | np.isinf(y) | np.isnan(y))
        X = X[valid_idx]
        y = y[valid_idx]
        feature_names = X.columns.tolist()
        
        print(f"📊 最终数据形状: X={X.shape}, y={y.shape}")
        return X, y, feature_names
    
    def analyze_lightgbm_importance(self, X, y, feature_names):
        """LightGBM特征重要性分析"""
        print("🚀 LightGBM特征重要性分析...")
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = lgb.LGBMRegressor(n_estimators=100, random_state=42, verbose=-1)
        model.fit(X_train, y_train)
        
        importance = model.feature_importances_
        feature_importance = dict(zip(feature_names, importance))
        
        return feature_importance
    
    def analyze_random_forest_importance(self, X, y, feature_names):
        """随机森林特征重要性分析"""
        print("🌲 随机森林特征重要性分析...")
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        
        importance = model.feature_importances_
        feature_importance = dict(zip(feature_names, importance))
        
        return feature_importance
    
    def analyze_statistical_importance(self, X, y, feature_names):
        """统计学特征重要性分析"""
        print("📈 F统计量特征重要性分析...")
        
        f_scores, _ = f_regression(X, y)
        feature_importance = dict(zip(feature_names, f_scores))
        
        return feature_importance
    
    def create_comprehensive_ranking(self, importance_dict):
        """创建综合排名"""
        print("🏆 创建综合特征排名...")
        
        # 获取所有特征
        all_features = set()
        for method_importance in importance_dict.values():
            all_features.update(method_importance.keys())
        
        # 计算每个特征的平均排名
        feature_avg_rank = {}
        
        for feature in all_features:
            ranks = []
            for method, importance in importance_dict.items():
                if feature in importance:
                    # 按重要性排序，获取排名
                    sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)
                    rank = next(i for i, (f, _) in enumerate(sorted_features, 1) if f == feature)
                    ranks.append(rank)
            
            if ranks:
                feature_avg_rank[feature] = np.mean(ranks)
        
        # 按平均排名排序
        sorted_features = sorted(feature_avg_rank.items(), key=lambda x: x[1])
        
        return sorted_features, feature_avg_rank
    
    def analyze_feature_categories(self, feature_rankings):
        """分析特征类别性能"""
        print("📊 分析特征类别性能...")
        
        # 定义特征类别
        categories = {
            'volatility_features': ['volatility', 'std', 'atr', 'srvi'],
            'trend_features': ['sma', 'ema', 'macd', 'rsi', 'adx'],
            'price_features': ['close', 'open', 'high', 'low', 'prev', 'price'],
            'volume_features': ['volume', 'vwap', 'obv'],
            'momentum_features': ['momentum', 'roc', 'williams'],
            'other_features': []
        }
        
        # 分类特征
        feature_category_map = {}
        for feature, rank in feature_rankings.items():
            categorized = False
            for category, keywords in categories.items():
                if category == 'other_features':
                    continue
                if any(keyword in feature.lower() for keyword in keywords):
                    feature_category_map[feature] = category
                    categorized = True
                    break
            if not categorized:
                feature_category_map[feature] = 'other_features'
        
        # 计算每个类别的平均排名
        category_performance = {}
        for category in categories.keys():
            category_features = [f for f, c in feature_category_map.items() if c == category]
            if category_features:
                avg_rank = np.mean([feature_rankings[f] for f in category_features])
                category_performance[category] = {
                    'avg_rank': avg_rank,
                    'feature_count': len(category_features),
                    'features': category_features[:5]  # 显示前5个特征
                }
        
        return category_performance
    
    def test_feature_combinations(self, X, y, top_features, feature_names):
        """测试不同特征组合的性能"""
        print("🧪 测试特征组合性能...")
        
        from sklearn.metrics import mean_squared_error
        from sklearn.linear_model import Ridge
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        results = {}
        
        # 测试不同数量的顶级特征
        for n_features in [5, 10, 15, 20, 30]:
            if n_features > len(top_features):
                continue
                
            selected_features = [f[0] for f in top_features[:n_features]]
            feature_indices = [feature_names.index(f) for f in selected_features if f in feature_names]
            
            if len(feature_indices) == 0:
                continue
            
            X_train_selected = X_train.iloc[:, feature_indices]
            X_test_selected = X_test.iloc[:, feature_indices]
            
            # 训练简单模型
            model = Ridge(alpha=1.0, random_state=42)
            model.fit(X_train_selected, y_train)
            
            y_pred = model.predict(X_test_selected)
            mse = mean_squared_error(y_test, y_pred)
            
            results[f'top_{n_features}'] = {
                'mse': mse,
                'features': selected_features,
                'feature_count': len(selected_features)
            }
        
        return results
    
    def generate_recommendations(self, top_features, category_performance, combination_results):
        """生成改进建议"""
        recommendations = []
        
        # 基于最佳特征类别的建议
        best_category = min(category_performance.items(), key=lambda x: x[1]['avg_rank'])
        recommendations.append(f"重点关注{best_category[0]}特征，平均排名最佳: {best_category[1]['avg_rank']:.2f}")
        
        # 基于顶级特征的建议
        top_5_features = [f[0] for f in top_features[:5]]
        recommendations.append(f"优先使用Top 5特征: {', '.join(top_5_features)}")
        
        # 基于组合测试的建议
        if combination_results:
            best_combo = min(combination_results.items(), key=lambda x: x[1]['mse'])
            recommendations.append(f"最佳特征组合: {best_combo[0]} (MSE: {best_combo[1]['mse']:.6f})")
        
        return recommendations
    
    def save_results(self, results):
        """保存结果"""
        print("💾 保存分析结果...")
        
        os.makedirs('output', exist_ok=True)
        
        with open('output/simplified_feature_analysis_results.json', 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print("✅ 结果已保存到 output/simplified_feature_analysis_results.json")
    
    def run_analysis(self, target_col='future_return_1'):
        """运行完整分析"""
        print("🎯 开始简化特征重要性分析")
        print("=" * 50)
        
        try:
            # 加载数据
            data = self.load_data()
            
            # 准备特征
            X, y, feature_names = self.prepare_features(data, target_col)
            
            # 多种方法分析特征重要性
            importance_methods = {
                'lightgbm': self.analyze_lightgbm_importance(X, y, feature_names),
                'random_forest': self.analyze_random_forest_importance(X, y, feature_names),
                'f_statistic': self.analyze_statistical_importance(X, y, feature_names)
            }
            
            # 创建综合排名
            top_features, feature_rankings = self.create_comprehensive_ranking(importance_methods)
            
            # 分析特征类别
            category_performance = self.analyze_feature_categories(feature_rankings)
            
            # 测试特征组合
            combination_results = self.test_feature_combinations(X, y, top_features, feature_names)
            
            # 生成建议
            recommendations = self.generate_recommendations(top_features, category_performance, combination_results)
            
            # 整理结果
            results = {
                'timestamp': datetime.now().isoformat(),
                'target_column': target_col,
                'data_shape': {'samples': X.shape[0], 'features': X.shape[1]},
                'top_features': top_features[:20],  # 前20个特征
                'category_performance': category_performance,
                'combination_results': combination_results,
                'recommendations': recommendations,
                'individual_importance': importance_methods
            }
            
            # 保存结果
            self.save_results(results)
            
            # 打印总结
            print("\n📊 分析完成! 主要发现:")
            print(f"🏆 Top 10特征: {[f[0] for f in top_features[:10]]}")
            if category_performance:
                best_cat = min(category_performance.items(), key=lambda x: x[1]['avg_rank'])
                print(f"🎯 最佳特征类别: {best_cat[0]} (平均排名: {best_cat[1]['avg_rank']:.2f})")
            if combination_results:
                best_combo = min(combination_results.items(), key=lambda x: x[1]['mse'])
                print(f"⚡ 最佳特征组合: {best_combo[0]} (MSE: {best_combo[1]['mse']:.6f})")
            
            return results
            
        except Exception as e:
            print(f"❌ 分析失败: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

def main():
    analyzer = SimplifiedFeatureAnalyzer()
    results = analyzer.run_analysis()
    
    if results:
        print("\n✅ 特征重要性分析完成!")
    else:
        print("\n❌ 分析失败!")

if __name__ == "__main__":
    main()