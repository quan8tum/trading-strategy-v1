#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化模型改进策略
基于最佳特征组合(Top 10特征，MSE: 0.000005)设计模型改进策略
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import lightgbm as lgb
import xgboost as xgb
import joblib
import json
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class SimplifiedModelStrategy:
    def __init__(self):
        self.top_features = [
            'volatility_std_20_lag20', 'volatility_std_20', 'srvi', 'sma_60', 'ema_60',
            'prev_close', 'macd_12_26_9', 'price_range_rolling_20', 'sma_60_norm', 'volatility_std_20_lag5'
        ]
        self.models = {}
        self.results = {}
        
    def load_data(self):
        """加载数据"""
        print("📊 加载数据...")
        
        # 优先使用优化特征数据
        if os.path.exists('output/optimized_features_data.csv'):
            data = pd.read_csv('output/optimized_features_data.csv')
            print(f"✅ 优化特征数据加载成功: {data.shape}")
        else:
            data = pd.read_csv('data/processed_features.csv')
            print(f"✅ 基础特征数据加载成功: {data.shape}")
            
        return data
    
    def prepare_data(self, data):
        """准备训练数据"""
        print("📊 准备训练数据...")
        
        # 创建目标变量
        if 'future_return_1' not in data.columns:
            print("⚠️ 目标列 future_return_1 不存在，创建未来收益率...")
            data['future_return_1'] = data['close'].pct_change().shift(-1)
        
        # 选择可用的特征
        available_features = [f for f in self.top_features if f in data.columns]
        print(f"✅ 使用特征数量: {len(available_features)}")
        print(f"📋 特征列表: {available_features}")
        
        # 准备特征和目标
        X = data[available_features].fillna(0)
        y = data['future_return_1'].fillna(0)
        
        # 移除无效数据
        valid_idx = ~(np.isinf(X).any(axis=1) | np.isinf(y) | np.isnan(y))
        X = X[valid_idx]
        y = y[valid_idx]
        
        print(f"📊 最终数据形状: X={X.shape}, y={y.shape}")
        return X, y, available_features
    
    def create_models(self):
        """创建模型配置"""
        return {
            'ridge': Ridge(alpha=1.0, random_state=42),
            'lasso': Lasso(alpha=0.01, random_state=42),
            'elastic_net': ElasticNet(alpha=0.01, l1_ratio=0.5, random_state=42),
            'random_forest': RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1),
            'gradient_boosting': GradientBoostingRegressor(n_estimators=50, max_depth=6, random_state=42),
            'lightgbm': lgb.LGBMRegressor(n_estimators=50, max_depth=6, random_state=42, verbose=-1),
            'xgboost': xgb.XGBRegressor(n_estimators=50, max_depth=6, random_state=42, verbosity=0)
        }
    
    def train_and_evaluate(self, X, y, features):
        """训练和评估模型"""
        print("🚀 开始模型训练和评估...")
        
        # 时间序列分割
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, shuffle=False
        )
        
        models = self.create_models()
        results = {}
        
        for name, model in models.items():
            print(f"🔧 训练模型: {name}")
            try:
                # 训练模型
                model.fit(X_train, y_train)
                
                # 预测
                y_pred = model.predict(X_test)
                
                # 评估
                mse = mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                
                results[name] = {
                    'mse': mse,
                    'r2': r2,
                    'model': model
                }
                
                print(f"✅ {name} - MSE: {mse:.6f}, R²: {r2:.4f}")
                
            except Exception as e:
                print(f"❌ {name} 训练失败: {str(e)}")
                continue
        
        return results, X_test, y_test
    
    def create_ensemble(self, results, X_test, y_test):
        """创建集成模型"""
        print("🔗 创建集成模型...")
        
        # 选择表现最好的3个模型
        sorted_models = sorted(results.items(), key=lambda x: x[1]['mse'])
        top_models = sorted_models[:3]
        
        print(f"📊 选择Top 3模型: {[name for name, _ in top_models]}")
        
        # 集成预测
        ensemble_pred = np.zeros(len(X_test))
        weights = []
        
        for name, result in top_models:
            model = result['model']
            pred = model.predict(X_test)
            weight = 1 / (result['mse'] + 1e-8)  # 基于MSE的权重
            ensemble_pred += weight * pred
            weights.append(weight)
        
        # 归一化权重
        total_weight = sum(weights)
        ensemble_pred /= total_weight
        
        # 评估集成模型
        ensemble_mse = mean_squared_error(y_test, ensemble_pred)
        ensemble_r2 = r2_score(y_test, ensemble_pred)
        
        print(f"🎯 集成模型 - MSE: {ensemble_mse:.6f}, R²: {ensemble_r2:.4f}")
        
        return {
            'mse': ensemble_mse,
            'r2': ensemble_r2,
            'weights': [w/total_weight for w in weights],
            'models': [name for name, _ in top_models]
        }
    
    def save_results(self, results, ensemble_result, features):
        """保存结果"""
        print("💾 保存结果...")
        
        # 创建输出目录
        os.makedirs('output', exist_ok=True)
        
        # 准备报告数据
        report = {
            'timestamp': datetime.now().isoformat(),
            'features_used': features,
            'feature_count': len(features),
            'individual_models': {},
            'ensemble_model': ensemble_result,
            'best_model': None,
            'improvement_suggestions': [
                "基于波动率特征的进一步优化",
                "增加更多滞后特征",
                "考虑非线性特征组合",
                "实施在线学习策略"
            ]
        }
        
        # 添加个别模型结果
        for name, result in results.items():
            report['individual_models'][name] = {
                'mse': result['mse'],
                'r2': result['r2']
            }
        
        # 找到最佳模型
        best_model = min(results.items(), key=lambda x: x[1]['mse'])
        report['best_model'] = {
            'name': best_model[0],
            'mse': best_model[1]['mse'],
            'r2': best_model[1]['r2']
        }
        
        # 保存报告
        with open('output/model_improvement_report.json', 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        # 保存最佳模型
        joblib.dump(best_model[1]['model'], 'output/best_model.pkl')
        
        print(f"✅ 结果已保存到 output/ 目录")
        return report

def main():
    print("🎯 简化模型改进策略实施")
    print("=" * 50)
    
    strategy = SimplifiedModelStrategy()
    
    try:
        # 加载数据
        data = strategy.load_data()
        
        # 准备数据
        X, y, features = strategy.prepare_data(data)
        
        # 训练和评估
        results, X_test, y_test = strategy.train_and_evaluate(X, y, features)
        
        if not results:
            print("❌ 没有成功训练的模型")
            return
        
        # 创建集成模型
        ensemble_result = strategy.create_ensemble(results, X_test, y_test)
        
        # 保存结果
        report = strategy.save_results(results, ensemble_result, features)
        
        # 打印总结
        print("\n📊 模型改进策略总结:")
        print(f"✅ 最佳单模型: {report['best_model']['name']} (MSE: {report['best_model']['mse']:.6f})")
        print(f"🎯 集成模型: MSE: {ensemble_result['mse']:.6f}, R²: {ensemble_result['r2']:.4f}")
        print(f"📋 使用特征数量: {len(features)}")
        
    except Exception as e:
        print(f"❌ 执行失败: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()