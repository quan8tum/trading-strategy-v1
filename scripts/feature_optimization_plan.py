#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
特征工程优化方案
基于特征重要性分析结果，重点开发波动率和趋势特征
"""

import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class AdvancedFeatureOptimizer:
    def __init__(self):
        self.optimization_results = {}
        
    def load_data(self):
        """加载数据"""
        print("📊 加载原始特征数据...")
        data = pd.read_csv('data/processed_features.csv')
        print(f"✅ 数据加载成功: {data.shape}")
        return data
    
    def create_enhanced_volatility_features(self, data):
        """创建增强波动率特征"""
        print("🌊 创建增强波动率特征...")
        
        # 多窗口波动率
        for window in [5, 10, 20, 60]:
            data[f'volatility_std_{window}'] = data['close'].rolling(window).std()
            data[f'volatility_range_{window}'] = (data['high'] - data['low']).rolling(window).mean()
        
        # 滞后波动率特征
        for lag in [1, 5, 10, 20]:
            data[f'volatility_std_20_lag{lag}'] = data['volatility_std_20'].shift(lag)
        
        # 波动率比率
        data['volatility_ratio_short_long'] = data['volatility_std_5'] / (data['volatility_std_20'] + 1e-8)
        data['volatility_ratio_medium_long'] = data['volatility_std_10'] / (data['volatility_std_60'] + 1e-8)
        
        # 波动率制度检测
        data['volatility_regime'] = self._detect_volatility_regime(data['volatility_std_20'])
        
        # 波动率突破
        data['volatility_breakout'] = (data['volatility_std_5'] > data['volatility_std_20'].rolling(20).quantile(0.8)).astype(int)
        
        return data
    
    def _detect_volatility_regime(self, volatility):
        """检测波动率制度"""
        # 计算分位数
        q33 = volatility.rolling(252).quantile(0.33)
        q67 = volatility.rolling(252).quantile(0.67)
        
        regime = pd.Series(1, index=volatility.index)  # 中等波动率
        regime[volatility <= q33] = 0  # 低波动率
        regime[volatility >= q67] = 2  # 高波动率
        
        return regime
    
    def create_enhanced_trend_features(self, data):
        """创建增强趋势特征"""
        print("📈 创建增强趋势特征...")
        
        # 多时间框架移动平均
        for window in [10, 20, 50, 100, 200]:
            data[f'sma_{window}'] = data['close'].rolling(window).mean()
            data[f'ema_{window}'] = data['close'].ewm(span=window).mean()
            data[f'sma_{window}_norm'] = data['close'] / data[f'sma_{window}'] - 1
        
        # 趋势斜率
        for window in [10, 20, 50]:
            data[f'trend_slope_{window}'] = data[f'sma_{window}'].diff(5) / data[f'sma_{window}'].shift(5)
        
        # 趋势一致性
        data['trend_consistency'] = (
            (data['sma_10'] > data['sma_20']).astype(int) +
            (data['sma_20'] > data['sma_50']).astype(int) +
            (data['sma_50'] > data['sma_100']).astype(int)
        )
        
        # 增强MACD
        data['macd_histogram_change'] = data['macd_12_26_9'].diff()
        data['macd_signal_cross'] = ((data['macd_12_26_9'] > 0) & (data['macd_12_26_9'].shift(1) <= 0)).astype(int)
        
        # 反转检测
        data['trend_reversal'] = self._detect_trend_reversal(data)
        
        return data
    
    def _detect_trend_reversal(self, data):
        """检测趋势反转"""
        # 基于价格和移动平均的反转信号
        reversal = pd.Series(0, index=data.index)
        
        # 上升趋势反转
        up_reversal = (
            (data['close'] < data['sma_20']) & 
            (data['close'].shift(1) >= data['sma_20'].shift(1)) &
            (data['sma_20'] > data['sma_20'].shift(5))
        )
        
        # 下降趋势反转
        down_reversal = (
            (data['close'] > data['sma_20']) & 
            (data['close'].shift(1) <= data['sma_20'].shift(1)) &
            (data['sma_20'] < data['sma_20'].shift(5))
        )
        
        reversal[up_reversal] = -1  # 向下反转
        reversal[down_reversal] = 1   # 向上反转
        
        return reversal
    
    def create_enhanced_price_features(self, data):
        """创建增强价格特征"""
        print("💰 创建增强价格特征...")
        
        # 价格位置
        for window in [20, 50, 100]:
            high_roll = data['high'].rolling(window).max()
            low_roll = data['low'].rolling(window).min()
            data[f'price_position_{window}'] = (data['close'] - low_roll) / (high_roll - low_roll + 1e-8)
        
        # 价格区间
        for window in [5, 10, 20]:
            data[f'price_range_rolling_{window}'] = (data['high'] - data['low']).rolling(window).mean()
            data[f'price_range_norm_{window}'] = data[f'price_range_rolling_{window}'] / data['close']
        
        # 跳空检测
        data['gap_up'] = ((data['open'] > data['high'].shift(1)) & (data['open'] > data['close'].shift(1))).astype(int)
        data['gap_down'] = ((data['open'] < data['low'].shift(1)) & (data['open'] < data['close'].shift(1))).astype(int)
        
        # 价格动量
        for period in [5, 10, 20]:
            data[f'price_momentum_{period}'] = data['close'].pct_change(period)
            data[f'price_acceleration_{period}'] = data[f'price_momentum_{period}'].diff()
        
        return data
    
    def create_market_microstructure_features(self, data):
        """创建市场微观结构特征"""
        print("🔬 创建市场微观结构特征...")
        
        # VWAP相关
        if 'volume' in data.columns:
            data['vwap'] = (data['close'] * data['volume']).rolling(20).sum() / data['volume'].rolling(20).sum()
            data['vwap_deviation'] = (data['close'] - data['vwap']) / data['vwap']
            
            # 成交量比率
            data['volume_ratio'] = data['volume'] / data['volume'].rolling(20).mean()
            data['volume_price_trend'] = data['volume'].rolling(5).corr(data['close'].rolling(5))
        
        # 价格-成交量背离
        if 'volume' in data.columns:
            price_change = data['close'].pct_change(5)
            volume_change = data['volume'].pct_change(5)
            data['price_volume_divergence'] = np.sign(price_change) != np.sign(volume_change)
        
        # 价格效率
        for window in [10, 20]:
            price_change = abs(data['close'].diff(window))
            path_length = abs(data['close'].diff()).rolling(window).sum()
            data[f'price_efficiency_{window}'] = price_change / (path_length + 1e-8)
        
        # Amihud非流动性指标
        if 'volume' in data.columns:
            daily_return = abs(data['close'].pct_change())
            dollar_volume = data['close'] * data['volume']
            data['amihud_illiquidity'] = daily_return / (dollar_volume + 1e-8)
        
        return data
    
    def optimize_features(self, data):
        """执行特征优化"""
        print("🚀 开始特征优化...")
        print("=" * 50)
        
        original_shape = data.shape
        
        # 创建各类增强特征
        data = self.create_enhanced_volatility_features(data)
        data = self.create_enhanced_trend_features(data)
        data = self.create_enhanced_price_features(data)
        data = self.create_enhanced_microstructure_features(data)
        
        # 移除无限值和NaN
        data = data.replace([np.inf, -np.inf], np.nan)
        
        final_shape = data.shape
        new_features = final_shape[1] - original_shape[1]
        
        print(f"✅ 特征优化完成!")
        print(f"📊 原始特征数: {original_shape[1]}")
        print(f"📊 新增特征数: {new_features}")
        print(f"📊 最终特征数: {final_shape[1]}")
        
        return data
    
    def generate_optimization_report(self, original_shape, final_shape):
        """生成优化报告"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'optimization_summary': {
                'original_features': original_shape[1],
                'new_features': final_shape[1] - original_shape[1],
                'total_features': final_shape[1],
                'samples': final_shape[0]
            },
            'feature_categories': {
                'volatility_features': {
                    'description': '增强波动率特征，包括多窗口、滞后、比率和制度检测',
                    'expected_improvement': '提高对市场波动的敏感性和预测能力'
                },
                'trend_features': {
                    'description': '增强趋势特征，包括多时间框架移动平均和趋势分析',
                    'expected_improvement': '更好地捕捉市场趋势变化和反转信号'
                },
                'price_features': {
                    'description': '增强价格特征，包括位置、区间、跳空和动量指标',
                    'expected_improvement': '提高对价格行为模式的识别能力'
                },
                'microstructure_features': {
                    'description': '市场微观结构特征，包括VWAP、成交量和流动性指标',
                    'expected_improvement': '增强对市场微观结构的理解和预测'
                }
            },
            'next_steps': [
                '1. 波动率特征优化（1周内）',
                '2. 趋势特征增强（1周内）',
                '3. 价格特征改进（2周内）',
                '4. 微观结构特征（3周内）'
            ]
        }
        
        return report
    
    def save_results(self, data, report):
        """保存优化结果"""
        print("💾 保存优化结果...")
        
        os.makedirs('output', exist_ok=True)
        
        # 保存优化后的数据
        data.to_csv('output/optimized_features_data.csv', index=False)
        
        # 保存优化报告
        with open('output/feature_optimization_report.json', 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print("✅ 结果已保存到 output/ 目录")

def main():
    print("🎯 特征工程优化方案实施")
    print("=" * 50)
    
    optimizer = AdvancedFeatureOptimizer()
    
    try:
        # 加载数据
        data = optimizer.load_data()
        original_shape = data.shape
        
        # 执行优化
        optimized_data = optimizer.optimize_features(data)
        final_shape = optimized_data.shape
        
        # 生成报告
        report = optimizer.generate_optimization_report(original_shape, final_shape)
        
        # 保存结果
        optimizer.save_results(optimized_data, report)
        
        # 打印总结
        print("\n📊 特征优化总结:")
        for category, info in report['feature_categories'].items():
            print(f"✅ {category}: {info['expected_improvement']}")
        
        print("\n🎯 下一步计划:")
        for step in report['next_steps']:
            print(f"  {step}")
        
    except Exception as e:
        print(f"❌ 优化失败: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()