use criterion::{black_box, criterion_group, criterion_main, Criterion};
use milvuso::*;
use uuid::Uuid;
use chrono::Utc;

fn benchmark_collaborative_filtering(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    
    c.bench_function("collaborative_filtering_train", |b| {
        b.to_async(&rt).iter(|| async {
            let mut cf = algorithms::CollaborativeFiltering::new(128, 0.01, 0.001);
            
            let user_id = Uuid::new_v4();
            let item_id = Uuid::new_v4();
            
            let example = TrainingExample {
                user_id,
                item_id,
                label: 1.0,
                user_features: vec![0.1; 128],
                item_features: vec![0.2; 128],
                context_features: vec![0.0; 10],
                timestamp: Utc::now(),
            };
            
            black_box(cf.train(&[example]).await.unwrap());
        });
    });
    
    c.bench_function("collaborative_filtering_predict", |b| {
        b.to_async(&rt).iter(|| async {
            let cf = algorithms::CollaborativeFiltering::new(128, 0.01, 0.001);
            let user_features = vec![0.1; 128];
            let item_features = vec![0.2; 128];
            
            black_box(cf.predict(&user_features, &item_features).await.unwrap());
        });
    });
}

fn benchmark_vector_retrieval(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    
    c.bench_function("in_memory_retriever_search", |b| {
        b.to_async(&rt).iter(|| async {
            let mut retriever = algorithms::retriever::InMemoryRetriever::new(128);
            
            // Add some vectors
            for i in 0..1000 {
                let id = Uuid::new_v4();
                let vector: Vec<f32> = (0..128).map(|j| (i + j) as f32 / 1000.0).collect();
                retriever.add_vector(id, vector).await.unwrap();
            }
            
            let query = vec![0.5; 128];
            black_box(retriever.search_similar(&query, 10).await.unwrap());
        });
    });
    
    c.bench_function("hnsw_retriever_search", |b| {
        b.to_async(&rt).iter(|| async {
            let mut retriever = algorithms::retriever::HNSWRetriever::new(128, 16, 200);
            
            // Add some vectors
            for i in 0..100 {
                let id = Uuid::new_v4();
                let vector: Vec<f32> = (0..128).map(|j| (i + j) as f32 / 100.0).collect();
                retriever.add_vector(id, vector).await.unwrap();
            }
            
            let query = vec![0.5; 128];
            black_box(retriever.search_similar(&query, 10).await.unwrap());
        });
    });
}

fn benchmark_optimizers(c: &mut Criterion) {
    use milvuso::algorithms::optimizer::*;
    use nalgebra::DVector;
    
    c.bench_function("sgd_optimizer", |b| {
        b.iter(|| {
            let mut optimizer = SGD::new(0.01);
            let mut params = DVector::from_vec(vec![1.0; 1000]);
            let gradients = DVector::from_vec(vec![0.01; 1000]);
            
            black_box(optimizer.update(&mut params, &gradients));
        });
    });
    
    c.bench_function("adam_optimizer", |b| {
        b.iter(|| {
            let mut optimizer = Adam::default();
            let mut params = DVector::from_vec(vec![1.0; 1000]);
            let gradients = DVector::from_vec(vec![0.01; 1000]);
            
            black_box(optimizer.update(&mut params, &gradients));
        });
    });
    
    c.bench_function("adagrad_optimizer", |b| {
        b.iter(|| {
            let mut optimizer = AdaGrad::default();
            let mut params = DVector::from_vec(vec![1.0; 1000]);
            let gradients = DVector::from_vec(vec![0.01; 1000]);
            
            black_box(optimizer.update(&mut params, &gradients));
        });
    });
}

fn benchmark_initializers(c: &mut Criterion) {
    use milvuso::algorithms::initializer::*;
    
    c.bench_function("xavier_uniform", |b| {
        b.iter(|| {
            black_box(xavier_uniform(1000));
        });
    });
    
    c.bench_function("he_normal", |b| {
        b.iter(|| {
            black_box(he_normal(1000));
        });
    });
    
    c.bench_function("orthogonal", |b| {
        b.iter(|| {
            black_box(orthogonal(100, 100));
        });
    });
}

fn benchmark_utils(c: &mut Criterion) {
    use milvuso::utils::*;
    
    let vec_a = vec![0.1; 1000];
    let vec_b = vec![0.2; 1000];
    
    c.bench_function("cosine_similarity", |b| {
        b.iter(|| {
            black_box(cosine_similarity(&vec_a, &vec_b));
        });
    });
    
    c.bench_function("euclidean_distance", |b| {
        b.iter(|| {
            black_box(euclidean_distance(&vec_a, &vec_b));
        });
    });
    
    c.bench_function("normalize_vector", |b| {
        b.iter(|| {
            let mut vec = vec_a.clone();
            black_box(normalize_vector(&mut vec));
        });
    });
    
    let scores = vec![0.1, 0.5, 0.3, 0.9, 0.2, 0.8, 0.4, 0.6, 0.7, 0.0];
    c.bench_function("top_k_indices", |b| {
        b.iter(|| {
            black_box(top_k_indices(&scores, 5));
        });
    });
}

fn benchmark_metrics(c: &mut Criterion) {
    use milvuso::utils::metrics::*;
    use std::collections::HashMap;
    
    let calculator = MetricsCalculator::new(10);
    let recommended: Vec<Uuid> = (0..10).map(|_| Uuid::new_v4()).collect();
    let relevant: Vec<Uuid> = recommended[0..5].to_vec();
    
    c.bench_function("precision_at_k", |b| {
        b.iter(|| {
            black_box(calculator.calculate_precision_at_k(&recommended, &relevant));
        });
    });
    
    c.bench_function("recall_at_k", |b| {
        b.iter(|| {
            black_box(calculator.calculate_recall_at_k(&recommended, &relevant));
        });
    });
    
    let mut relevant_scores = HashMap::new();
    for (i, &id) in recommended.iter().enumerate() {
        relevant_scores.insert(id, i as f64 / 10.0);
    }
    
    c.bench_function("ndcg_at_k", |b| {
        b.iter(|| {
            black_box(calculator.calculate_ndcg_at_k(&recommended, &relevant_scores));
        });
    });
    
    let mut item_features = HashMap::new();
    for &id in &recommended {
        item_features.insert(id, vec![0.1; 128]);
    }
    
    c.bench_function("diversity", |b| {
        b.iter(|| {
            black_box(calculator.calculate_diversity(&recommended, &item_features));
        });
    });
}

criterion_group!(
    benches,
    benchmark_collaborative_filtering,
    benchmark_vector_retrieval,
    benchmark_optimizers,
    benchmark_initializers,
    benchmark_utils,
    benchmark_metrics
);
criterion_main!(benches);
