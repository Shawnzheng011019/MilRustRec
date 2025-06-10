use milvuso::*;
use uuid::Uuid;
use chrono::Utc;
use std::collections::HashMap;

#[tokio::test]
async fn test_recommendation_flow() {
    // Initialize test configuration
    let config = Config::default();
    
    // Create test state (this would normally connect to real services)
    // For testing, we'll use mock implementations
    
    let user_id = Uuid::new_v4();
    let item_id = Uuid::new_v4();
    
    // Test user action creation
    let action = UserAction::new(user_id, item_id, ActionType::Click);
    assert_eq!(action.user_id, user_id);
    assert_eq!(action.item_id, item_id);
    assert!(matches!(action.action_type, ActionType::Click));
    
    // Test user profile creation
    let profile = UserProfile::new(user_id, 128);
    assert_eq!(profile.user_id, user_id);
    assert_eq!(profile.embedding.len(), 128);
    assert_eq!(profile.interaction_count, 0);
    
    // Test item feature creation
    let item_feature = ItemFeature::new(item_id, vec![0.1; 128], "electronics".to_string())
        .with_tags(vec!["smartphone".to_string(), "android".to_string()])
        .with_popularity(0.8);
    
    assert_eq!(item_feature.item_id, item_id);
    assert_eq!(item_feature.category, "electronics");
    assert_eq!(item_feature.popularity_score, 0.8);
    
    // Test recommendation request
    let request = RecommendationRequest {
        user_id,
        num_recommendations: 10,
        filter_categories: Some(vec!["electronics".to_string()]),
        exclude_items: None,
    };
    
    assert_eq!(request.user_id, user_id);
    assert_eq!(request.num_recommendations, 10);
}

#[tokio::test]
async fn test_algorithms() {
    use milvuso::algorithms::*;
    
    // Test collaborative filtering
    let mut cf = CollaborativeFiltering::new(64, 0.01, 0.001);
    
    let user_id = Uuid::new_v4();
    let item_id = Uuid::new_v4();
    
    // Initialize embeddings
    cf.initialize_user_embedding(user_id);
    cf.initialize_item_embedding(item_id);
    
    // Create training example
    let example = TrainingExample {
        user_id,
        item_id,
        label: 1.0,
        user_features: vec![0.1; 64],
        item_features: vec![0.2; 64],
        context_features: vec![0.0; 10],
        timestamp: Utc::now(),
    };
    
    // Test training
    let result = cf.train(&[example]).await;
    assert!(result.is_ok());
    
    // Test prediction
    let prediction = cf.predict(&vec![0.1; 64], &vec![0.2; 64]).await;
    assert!(prediction.is_ok());
    
    // Test embedding retrieval
    let user_embedding = cf.get_user_embedding(user_id).await;
    assert!(user_embedding.is_ok());
    assert_eq!(user_embedding.unwrap().len(), 64);
}

#[tokio::test]
async fn test_optimizers() {
    use milvuso::algorithms::optimizer::*;
    use nalgebra::DVector;
    
    // Test SGD optimizer
    let mut sgd = SGD::new(0.01);
    let mut params = DVector::from_vec(vec![1.0, 2.0, 3.0]);
    let gradients = DVector::from_vec(vec![0.1, 0.2, 0.3]);
    
    sgd.update(&mut params, &gradients);
    
    // Test Adam optimizer
    let mut adam = Adam::default();
    let mut params = DVector::from_vec(vec![1.0, 2.0, 3.0]);
    let gradients = DVector::from_vec(vec![0.1, 0.2, 0.3]);
    
    adam.update(&mut params, &gradients);
    
    // Test AdaGrad optimizer
    let mut adagrad = AdaGrad::default();
    let mut params = DVector::from_vec(vec![1.0, 2.0, 3.0]);
    let gradients = DVector::from_vec(vec![0.1, 0.2, 0.3]);
    
    adagrad.update(&mut params, &gradients);
}

#[tokio::test]
async fn test_retrievers() {
    use milvuso::algorithms::retriever::*;
    
    // Test in-memory retriever
    let mut retriever = InMemoryRetriever::new(64);
    
    let id1 = Uuid::new_v4();
    let id2 = Uuid::new_v4();
    let vector1 = vec![1.0; 64];
    let vector2 = vec![0.5; 64];
    
    // Add vectors
    retriever.add_vector(id1, vector1.clone()).await.unwrap();
    retriever.add_vector(id2, vector2.clone()).await.unwrap();
    
    // Search similar vectors
    let results = retriever.search_similar(&vector1, 2).await.unwrap();
    assert_eq!(results.len(), 2);
    assert_eq!(results[0].0, id1); // Most similar should be itself
    
    // Test HNSW retriever
    let mut hnsw = HNSWRetriever::new(64, 16, 200);
    
    hnsw.add_vector(id1, vector1.clone()).await.unwrap();
    hnsw.add_vector(id2, vector2.clone()).await.unwrap();
    
    let results = hnsw.search_similar(&vector1, 2).await.unwrap();
    assert!(!results.is_empty());
}

#[tokio::test]
async fn test_initializers() {
    use milvuso::algorithms::initializer::*;
    
    // Test Xavier uniform initialization
    let weights = xavier_uniform(100);
    assert_eq!(weights.len(), 100);
    
    // Check that values are within expected range
    let limit = (6.0 / 100.0_f32).sqrt();
    for &weight in &weights {
        assert!(weight >= -limit && weight <= limit);
    }
    
    // Test He normal initialization
    let weights = he_normal(100);
    assert_eq!(weights.len(), 100);
    
    // Test initialization method enum
    let method = InitializationMethod::XavierUniform;
    let weights = method.initialize(50);
    assert_eq!(weights.len(), 50);
    
    // Test embedding initializer
    let initializer = EmbeddingInitializer::new(InitializationMethod::XavierUniform, 64);
    let user_id = Uuid::new_v4();
    let embedding = initializer.initialize_user_embedding(user_id);
    assert_eq!(embedding.len(), 64);
    
    // Test reproducibility
    let embedding2 = initializer.initialize_user_embedding(user_id);
    assert_eq!(embedding, embedding2);
}

#[tokio::test]
async fn test_utils() {
    use milvuso::utils::*;
    
    // Test cosine similarity
    let a = vec![1.0, 0.0, 0.0];
    let b = vec![0.0, 1.0, 0.0];
    let similarity = cosine_similarity(&a, &b);
    assert!((similarity - 0.0).abs() < 1e-6);
    
    let a = vec![1.0, 1.0];
    let b = vec![1.0, 1.0];
    let similarity = cosine_similarity(&a, &b);
    assert!((similarity - 1.0).abs() < 1e-6);
    
    // Test euclidean distance
    let a = vec![0.0, 0.0];
    let b = vec![3.0, 4.0];
    let distance = euclidean_distance(&a, &b);
    assert!((distance - 5.0).abs() < 1e-6);
    
    // Test vector normalization
    let mut vector = vec![3.0, 4.0];
    normalize_vector(&mut vector);
    let norm: f32 = vector.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!((norm - 1.0).abs() < 1e-6);
    
    // Test top-k indices
    let scores = vec![0.1, 0.5, 0.3, 0.9, 0.2];
    let top_2 = top_k_indices(&scores, 2);
    assert_eq!(top_2, vec![3, 1]);
    
    // Test weighted average
    let vectors = vec![
        (vec![1.0, 0.0], 0.5),
        (vec![0.0, 1.0], 0.5),
    ];
    let avg = weighted_average(&vectors);
    assert_eq!(avg, vec![0.5, 0.5]);
}

#[tokio::test]
async fn test_metrics() {
    use milvuso::utils::metrics::*;
    
    let calculator = MetricsCalculator::new(5);
    
    // Test precision@k
    let recommended = vec![Uuid::new_v4(), Uuid::new_v4(), Uuid::new_v4()];
    let relevant = vec![recommended[0], recommended[2]]; // 2 out of 3 are relevant
    
    let precision = calculator.calculate_precision_at_k(&recommended, &relevant);
    assert!((precision - 2.0/3.0).abs() < 1e-6);
    
    // Test recall@k
    let recall = calculator.calculate_recall_at_k(&recommended, &relevant);
    assert!((recall - 1.0).abs() < 1e-6); // All relevant items were recommended
    
    // Test F1 score
    let f1 = calculator.calculate_f1_score(precision, recall);
    let expected_f1 = 2.0 * precision * recall / (precision + recall);
    assert!((f1 - expected_f1).abs() < 1e-6);
    
    // Test NDCG
    let mut relevant_scores = HashMap::new();
    relevant_scores.insert(recommended[0], 1.0);
    relevant_scores.insert(recommended[1], 0.0);
    relevant_scores.insert(recommended[2], 0.5);
    
    let ndcg = calculator.calculate_ndcg_at_k(&recommended, &relevant_scores);
    assert!(ndcg >= 0.0 && ndcg <= 1.0);
}

#[tokio::test]
async fn test_validation() {
    use milvuso::utils::validation::*;
    
    // Test valid user action
    let valid_action = UserAction {
        user_id: Uuid::new_v4(),
        item_id: Uuid::new_v4(),
        action_type: ActionType::Click,
        timestamp: Utc::now(),
        context: None,
    };
    assert!(validate_user_action(&valid_action).is_ok());
    
    // Test invalid user action (nil UUID)
    let invalid_action = UserAction {
        user_id: Uuid::nil(),
        item_id: Uuid::new_v4(),
        action_type: ActionType::Click,
        timestamp: Utc::now(),
        context: None,
    };
    assert!(validate_user_action(&invalid_action).is_err());
    
    // Test valid user profile
    let valid_profile = UserProfile {
        user_id: Uuid::new_v4(),
        embedding: vec![0.1, 0.2, 0.3],
        preferences: vec!["music".to_string()],
        last_updated: Utc::now(),
        interaction_count: 10,
    };
    assert!(validate_user_profile(&valid_profile).is_ok());
    
    // Test invalid user profile (NaN in embedding)
    let invalid_profile = UserProfile {
        user_id: Uuid::new_v4(),
        embedding: vec![f32::NAN, 0.2, 0.3],
        preferences: vec!["music".to_string()],
        last_updated: Utc::now(),
        interaction_count: 10,
    };
    assert!(validate_user_profile(&invalid_profile).is_err());
    
    // Test string sanitization
    let sanitized = sanitize_string("Hello, World! @#$%", 10);
    assert_eq!(sanitized, "Hello, Wor");
    
    // Test UUID validation
    let valid_uuid = validate_uuid_string("550e8400-e29b-41d4-a716-446655440000");
    assert!(valid_uuid.is_ok());
    
    let invalid_uuid = validate_uuid_string("invalid-uuid");
    assert!(invalid_uuid.is_err());
}
