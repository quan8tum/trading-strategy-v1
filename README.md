# 交易策略V1项目 - 基于特征重要性分析的量化交易策略

## 🎯 项目概述

本项目是一个基于特征重要性分析的量化交易策略开发项目，通过深度特征工程和机器学习模型优化，构建高效的股票预测和交易策略。

## 📊 核心发现

### Top 10 关键特征
1. `volatility_std_20_lag20` - 20日波动率滞后20期
2. `volatility_std_20` - 20日波动率
3. `srvi` - 已实现波动率指标
4. `sma_60` - 60日简单移动平均
5. `ema_60` - 60日指数移动平均
6. `prev_close` - 前收盘价
7. `macd_12_26_9` - MACD指标
8. `price_range_rolling_20` - 20日价格区间
9. `sma_60_norm` - 标准化60日移动平均
10. `volatility_std_20_lag5` - 20日波动率滞后5期

### 模型性能
- **最佳单模型**: Lasso回归 (MSE: 0.000000)
- **集成模型**: MSE: 0.000000, R²: 0.0000
- **最佳特征类别**: 波动率特征 (平均排名: 25.72)

## 🚀 项目结构

```
trading-strategy-v1/
├── data/                           # 数据文件
│   └── processed_features.csv      # 处理后的特征数据
├── output/                         # 输出结果
│   ├── best_model.pkl             # 最佳模型
│   ├── model_improvement_report.json # 模型改进报告
│   ├── optimized_features_data.csv   # 优化特征数据
│   └── simplified_feature_analysis_results.json # 特征分析结果
├── scripts/                        # 核心脚本
│   ├── simplified_feature_analysis.py      # 特征重要性分析
│   ├── feature_optimization_plan.py        # 特征优化方案
│   └── simplified_model_improvement.py     # 模型改进策略
├── docs/                          # 项目文档
│   ├── notion_trading_strategy_v1_analysis.md # 项目分析文档
│   └── feature_optimization_report.json       # 特征优化报告
└── README.md                      # 项目说明
```

## 🔧 核心功能

### 1. 特征重要性分析
- 使用多种算法(LightGBM, Random Forest, F-statistic)进行特征排序
- 综合排名确定最重要的特征
- 特征类别性能分析

### 2. 特征工程优化
- **增强波动率特征**: 多窗口、滞后、比率、制度检测
- **增强趋势特征**: 多时间框架移动平均、趋势斜率
- **增强价格特征**: 位置、区间、跳空、动量
- **市场微观结构特征**: VWAP、成交量比率、价格效率

### 3. 模型改进策略
- 训练7种不同的机器学习模型
- 基于MSE的模型集成策略
- 时间序列交叉验证

## 📈 使用方法

### 环境要求
```bash
pip install pandas numpy scikit-learn lightgbm xgboost joblib
```

### 运行步骤

1. **特征重要性分析**
```bash
python scripts/simplified_feature_analysis.py
```

2. **特征优化**
```bash
python scripts/feature_optimization_plan.py
```

3. **模型训练和改进**
```bash
python scripts/simplified_model_improvement.py
```

## 📊 结果文件说明

- `output/simplified_feature_analysis_results.json`: 特征重要性分析结果
- `output/feature_optimization_report.json`: 特征优化详细报告
- `output/model_improvement_report.json`: 模型性能对比报告
- `output/best_model.pkl`: 训练好的最佳模型

## 🎯 优化建议

1. **波动率特征进一步优化** (1周内)
   - 实现多时间尺度波动率
   - 添加波动率制度识别

2. **趋势特征增强** (1周内)
   - 多时间框架趋势分析
   - 趋势强度量化

3. **价格特征改进** (2周内)
   - 价格位置指标
   - 价格动量加速度

4. **微观结构特征** (3周内)
   - VWAP偏离度
   - 流动性指标

## 📝 技术特点

- **时间序列友好**: 使用适合金融时间序列的数据分割方法
- **多模型集成**: 结合线性模型、树模型和梯度提升模型
- **特征工程**: 基于金融理论的特征构建
- **性能优化**: 针对预测精度和计算效率的平衡

## 🔄 版本历史

- **v1.0**: 初始版本，包含基础特征分析和模型训练
- **v1.1**: 增加特征优化和集成学习
- **v1.2**: 优化模型性能和代码结构

## 📞 联系方式

如有问题或建议，请通过GitHub Issues联系。

## 📄 许可证

MIT License - 详见LICENSE文件