use milvuso::*;
use uuid::Uuid;
use chrono::Utc;
use std::time::Duration;
use tokio::time::sleep;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize tracing
    init_tracing().await;
    
    println!("🚀 MilRustRec 推荐系统基础使用示例");
    
    // 1. 创建配置
    let config = Config::default();
    println!("✅ 配置加载完成");
    
    // 2. 初始化应用状态 (在实际环境中需要连接到真实的服务)
    // 这里我们演示基本的数据结构和算法使用
    
    // 3. 创建用户和物品
    let user_id = Uuid::new_v4();
    let item_ids: Vec<Uuid> = (0..10).map(|_| Uuid::new_v4()).collect();
    
    println!("👤 创建用户: {}", user_id);
    println!("📦 创建 {} 个物品", item_ids.len());
    
    // 4. 创建用户画像
    let mut user_profile = UserProfile::new(user_id, config.recommendation.embedding_dim);
    user_profile.preferences = vec!["电子产品".to_string(), "智能手机".to_string()];
    
    println!("🎯 用户画像创建完成，偏好: {:?}", user_profile.preferences);
    
    // 5. 创建物品特征
    let mut item_features = Vec::new();
    let categories = vec!["电子产品", "服装", "图书", "食品", "运动"];
    
    for (i, &item_id) in item_ids.iter().enumerate() {
        let category = categories[i % categories.len()];
        let embedding = algorithms::initializer::xavier_uniform(config.recommendation.embedding_dim);
        
        let feature = ItemFeature::new(item_id, embedding, category.to_string())
            .with_tags(vec![format!("标签{}", i), "热门".to_string()])
            .with_popularity(0.5 + (i as f32 * 0.05));
        
        item_features.push(feature);
    }
    
    println!("📊 物品特征创建完成");
    
    // 6. 模拟用户行为
    println!("\n📱 模拟用户行为...");
    
    let actions = vec![
        UserAction::new(user_id, item_ids[0], ActionType::View),
        UserAction::new(user_id, item_ids[1], ActionType::Click),
        UserAction::new(user_id, item_ids[2], ActionType::Like),
        UserAction::new(user_id, item_ids[0], ActionType::Purchase),
    ];
    
    for action in &actions {
        println!("  🔄 用户行为: {:?} -> 物品 {}", action.action_type, action.item_id);
    }
    
    // 7. 创建协同过滤算法
    println!("\n🧠 初始化推荐算法...");
    
    let mut cf_algorithm = algorithms::CollaborativeFiltering::new(
        config.recommendation.embedding_dim,
        config.training.learning_rate,
        0.01, // 正则化参数
    );
    
    // 8. 创建训练样本
    let mut training_examples = Vec::new();
    
    for action in &actions {
        let item_feature = item_features.iter()
            .find(|f| f.item_id == action.item_id)
            .unwrap();
        
        let label = match action.action_type {
            ActionType::View => 0.1,
            ActionType::Click => 0.3,
            ActionType::Like => 0.7,
            ActionType::Purchase => 1.0,
            ActionType::Share => 0.8,
            ActionType::Convert => 1.0,
        };
        
        let example = TrainingExample {
            user_id: action.user_id,
            item_id: action.item_id,
            label,
            user_features: user_profile.embedding.clone(),
            item_features: item_feature.embedding.clone(),
            context_features: vec![0.0; 10], // 简单的上下文特征
            timestamp: action.timestamp,
        };
        
        training_examples.push(example);
    }
    
    println!("📚 创建了 {} 个训练样本", training_examples.len());
    
    // 9. 训练模型
    println!("\n🎓 开始训练模型...");
    
    cf_algorithm.train(&training_examples).await?;
    
    println!("✅ 模型训练完成");
    
    // 10. 生成推荐
    println!("\n🎯 生成推荐结果...");
    
    let mut recommendations = Vec::new();
    
    for item_feature in &item_features {
        if !actions.iter().any(|a| a.item_id == item_feature.item_id) {
            // 只推荐用户没有交互过的物品
            let score = cf_algorithm.predict(&user_profile.embedding, &item_feature.embedding).await?;
            
            if score > 0.3 { // 阈值过滤
                recommendations.push((item_feature.item_id, score, &item_feature.category));
            }
        }
    }
    
    // 按分数排序
    recommendations.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    
    println!("🏆 推荐结果 (Top 5):");
    for (i, (item_id, score, category)) in recommendations.iter().take(5).enumerate() {
        println!("  {}. 物品: {} | 分数: {:.3} | 类别: {}", 
                i + 1, item_id, score, category);
    }
    
    // 11. 计算推荐指标
    println!("\n📊 计算推荐指标...");
    
    use utils::metrics::MetricsCalculator;
    
    let metrics_calc = MetricsCalculator::new(5);
    let recommended_items: Vec<Uuid> = recommendations.iter().take(5).map(|(id, _, _)| *id).collect();
    let relevant_items: Vec<Uuid> = actions.iter()
        .filter(|a| matches!(a.action_type, ActionType::Like | ActionType::Purchase))
        .map(|a| a.item_id)
        .collect();
    
    let precision = metrics_calc.calculate_precision_at_k(&recommended_items, &relevant_items);
    let recall = metrics_calc.calculate_recall_at_k(&recommended_items, &relevant_items);
    let f1_score = metrics_calc.calculate_f1_score(precision, recall);
    
    println!("  📈 Precision@5: {:.3}", precision);
    println!("  📈 Recall@5: {:.3}", recall);
    println!("  📈 F1 Score: {:.3}", f1_score);
    
    // 12. 演示向量检索
    println!("\n🔍 演示向量检索...");
    
    use algorithms::retriever::{InMemoryRetriever, VectorRetriever};
    
    let mut retriever = InMemoryRetriever::new(config.recommendation.embedding_dim);
    
    // 添加物品向量
    for item_feature in &item_features {
        retriever.add_vector(item_feature.item_id, item_feature.embedding.clone()).await?;
    }
    
    // 搜索相似物品
    let query_item = &item_features[0];
    let similar_items = retriever.search_similar(&query_item.embedding, 3).await?;
    
    println!("🎯 与物品 {} 最相似的物品:", query_item.item_id);
    for (i, (item_id, similarity)) in similar_items.iter().enumerate() {
        let item = item_features.iter().find(|f| f.item_id == *item_id).unwrap();
        println!("  {}. 物品: {} | 相似度: {:.3} | 类别: {}", 
                i + 1, item_id, similarity, item.category);
    }
    
    // 13. 演示不同的初始化方法
    println!("\n🎲 演示不同的初始化方法...");
    
    use algorithms::initializer::InitializationMethod;
    
    let methods = vec![
        ("Xavier Uniform", InitializationMethod::XavierUniform),
        ("He Normal", InitializationMethod::HeNormal),
        ("LeCun Uniform", InitializationMethod::LecunUniform),
    ];
    
    for (name, method) in methods {
        let weights = method.initialize(10);
        let mean: f32 = weights.iter().sum::<f32>() / weights.len() as f32;
        let variance: f32 = weights.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / weights.len() as f32;
        
        println!("  📊 {}: 均值={:.4}, 方差={:.4}", name, mean, variance);
    }
    
    // 14. 演示优化器
    println!("\n⚡ 演示不同的优化器...");
    
    use algorithms::optimizer::*;
    use nalgebra::DVector;
    
    let mut params = DVector::from_vec(vec![1.0, 2.0, 3.0]);
    let gradients = DVector::from_vec(vec![0.1, 0.2, 0.3]);
    
    let optimizers: Vec<(&str, Box<dyn Optimizer>)> = vec![
        ("SGD", Box::new(SGD::new(0.01))),
        ("Adam", Box::new(Adam::default())),
        ("AdaGrad", Box::new(AdaGrad::default())),
    ];
    
    for (name, mut optimizer) in optimizers {
        let mut test_params = params.clone();
        optimizer.update(&mut test_params, &gradients);
        println!("  🔧 {} 更新后参数: {:?}", name, test_params.as_slice());
    }
    
    println!("\n🎉 示例运行完成！");
    println!("💡 这个示例展示了 MilRustRec 推荐系统的核心功能:");
    println!("   - 用户画像和物品特征管理");
    println!("   - 协同过滤算法训练");
    println!("   - 实时推荐生成");
    println!("   - 向量相似度检索");
    println!("   - 推荐质量评估");
    println!("   - 多种优化算法");
    
    Ok(())
}
