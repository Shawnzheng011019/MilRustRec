use anyhow::Result;
use std::collections::HashMap;
use uuid::Uuid;

pub mod metrics;
pub mod validation;

pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return 0.0;
    }
    
    let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    
    if norm_a == 0.0 || norm_b == 0.0 {
        0.0
    } else {
        dot_product / (norm_a * norm_b)
    }
}

pub fn euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return f32::INFINITY;
    }
    
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f32>()
        .sqrt()
}

pub fn manhattan_distance(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return f32::INFINITY;
    }
    
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .sum()
}

pub fn normalize_vector(vector: &mut [f32]) {
    let norm: f32 = vector.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in vector.iter_mut() {
            *x /= norm;
        }
    }
}

pub fn normalize_vector_copy(vector: &[f32]) -> Vec<f32> {
    let mut normalized = vector.to_vec();
    normalize_vector(&mut normalized);
    normalized
}

pub fn weighted_average(vectors: &[(Vec<f32>, f32)]) -> Vec<f32> {
    if vectors.is_empty() {
        return Vec::new();
    }
    
    let dim = vectors[0].0.len();
    let mut result = vec![0.0; dim];
    let mut total_weight = 0.0;
    
    for (vector, weight) in vectors {
        if vector.len() != dim {
            continue;
        }
        
        for i in 0..dim {
            result[i] += vector[i] * weight;
        }
        total_weight += weight;
    }
    
    if total_weight > 0.0 {
        for x in result.iter_mut() {
            *x /= total_weight;
        }
    }
    
    result
}

pub fn top_k_indices(scores: &[f32], k: usize) -> Vec<usize> {
    let mut indexed_scores: Vec<(usize, f32)> = scores
        .iter()
        .enumerate()
        .map(|(i, &score)| (i, score))
        .collect();
    
    indexed_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    
    indexed_scores
        .into_iter()
        .take(k)
        .map(|(i, _)| i)
        .collect()
}

pub fn generate_user_id_from_string(user_str: &str) -> Uuid {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    
    let mut hasher = DefaultHasher::new();
    user_str.hash(&mut hasher);
    let hash = hasher.finish();
    
    // Convert hash to UUID
    let bytes = hash.to_be_bytes();
    let mut uuid_bytes = [0u8; 16];
    uuid_bytes[..8].copy_from_slice(&bytes);
    uuid_bytes[8..].copy_from_slice(&bytes);
    
    Uuid::from_bytes(uuid_bytes)
}

pub fn calculate_diversity_score(items: &[Uuid], item_categories: &HashMap<Uuid, String>) -> f32 {
    if items.len() <= 1 {
        return 0.0;
    }
    
    let mut categories = std::collections::HashSet::new();
    let mut total_items = 0;
    
    for item_id in items {
        if let Some(category) = item_categories.get(item_id) {
            categories.insert(category.clone());
            total_items += 1;
        }
    }
    
    if total_items == 0 {
        0.0
    } else {
        categories.len() as f32 / total_items as f32
    }
}

pub fn exponential_decay_weight(timestamp: chrono::DateTime<chrono::Utc>, decay_rate: f64) -> f32 {
    let now = chrono::Utc::now();
    let time_diff = now.signed_duration_since(timestamp).num_seconds() as f64;
    ((-decay_rate * time_diff / 3600.0).exp()) as f32 // decay per hour
}

pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

pub fn relu(x: f32) -> f32 {
    x.max(0.0)
}

pub fn softmax(scores: &[f32]) -> Vec<f32> {
    let max_score = scores.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let exp_scores: Vec<f32> = scores.iter().map(|&x| (x - max_score).exp()).collect();
    let sum_exp: f32 = exp_scores.iter().sum();
    
    if sum_exp > 0.0 {
        exp_scores.iter().map(|&x| x / sum_exp).collect()
    } else {
        vec![1.0 / scores.len() as f32; scores.len()]
    }
}

pub fn batch_process<T, F, R>(items: Vec<T>, batch_size: usize, mut processor: F) -> Vec<R>
where
    F: FnMut(&[T]) -> Vec<R>,
{
    let mut results = Vec::new();
    
    for chunk in items.chunks(batch_size) {
        let mut batch_results = processor(chunk);
        results.append(&mut batch_results);
    }
    
    results
}

pub async fn retry_with_backoff<F, Fut, T, E>(
    mut operation: F,
    max_retries: usize,
    initial_delay: std::time::Duration,
) -> Result<T, E>
where
    F: FnMut() -> Fut,
    Fut: std::future::Future<Output = Result<T, E>>,
    E: std::fmt::Debug,
{
    let mut delay = initial_delay;
    
    for attempt in 0..=max_retries {
        match operation().await {
            Ok(result) => return Ok(result),
            Err(e) => {
                if attempt == max_retries {
                    return Err(e);
                }
                
                tracing::warn!("Operation failed (attempt {}), retrying in {:?}: {:?}", 
                              attempt + 1, delay, e);
                tokio::time::sleep(delay).await;
                delay *= 2; // exponential backoff
            }
        }
    }
    
    unreachable!()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cosine_similarity() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        assert_eq!(cosine_similarity(&a, &b), 0.0);
        
        let a = vec![1.0, 1.0];
        let b = vec![1.0, 1.0];
        assert!((cosine_similarity(&a, &b) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_normalize_vector() {
        let mut v = vec![3.0, 4.0];
        normalize_vector(&mut v);
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_top_k_indices() {
        let scores = vec![0.1, 0.5, 0.3, 0.9, 0.2];
        let top_2 = top_k_indices(&scores, 2);
        assert_eq!(top_2, vec![3, 1]);
    }
}
