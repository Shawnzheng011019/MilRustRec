use crate::config::Config;
use crate::models::*;
use crate::services::{vector_db::VectorDbService, recommendation::RecommendationService};
use anyhow::Result;
use std::sync::Arc;
use tokio::sync::RwLock;
use uuid::Uuid;

use tracing::{info, error};
use dashmap::DashMap;
use std::collections::HashMap;

pub struct ServingService {
    vector_db: Arc<VectorDbService>,
    recommendation_service: Arc<RecommendationService>,
    config: Arc<Config>,
    model_parameters: Arc<RwLock<Option<ModelParameters>>>,
    serving_stats: Arc<DashMap<String, u64>>,
}

impl ServingService {
    pub async fn new(
        vector_db: Arc<VectorDbService>,
        recommendation_service: Arc<RecommendationService>,
        config: Arc<Config>,
    ) -> Result<Self> {
        Ok(Self {
            vector_db,
            recommendation_service,
            config,
            model_parameters: Arc::new(RwLock::new(None)),
            serving_stats: Arc::new(DashMap::new()),
        })
    }

    pub async fn serve_recommendations(&self, request: &RecommendationRequest) -> Result<RecommendationResponse> {
        self.increment_stat("total_requests").await;
        
        let start_time = std::time::Instant::now();
        
        let response = self.recommendation_service.get_recommendations(request).await?;
        
        let latency = start_time.elapsed().as_millis() as u64;
        self.update_latency_stat(latency).await;
        
        self.increment_stat("successful_requests").await;
        
        info!("Served recommendations for user {} in {}ms", request.user_id, latency);
        Ok(response)
    }

    pub async fn batch_serve_recommendations(&self, requests: &[RecommendationRequest]) -> Result<Vec<RecommendationResponse>> {
        self.increment_stat("batch_requests").await;
        
        let start_time = std::time::Instant::now();
        let mut responses = Vec::new();
        
        for request in requests {
            match self.recommendation_service.get_recommendations(request).await {
                Ok(response) => responses.push(response),
                Err(e) => {
                    error!("Failed to get recommendations for user {}: {}", request.user_id, e);
                    self.increment_stat("failed_requests").await;
                }
            }
        }
        
        let total_latency = start_time.elapsed().as_millis() as u64;
        self.update_latency_stat(total_latency).await;
        
        info!("Batch served {} recommendations in {}ms", responses.len(), total_latency);
        Ok(responses)
    }

    pub async fn get_similar_users(&self, user_id: Uuid, top_k: usize) -> Result<Vec<(Uuid, f32)>> {
        if let Some(user_profile) = self.vector_db.get_user_profile(user_id).await? {
            let similar_users = self.vector_db
                .search_similar_users(&user_profile.embedding, top_k + 1)
                .await?;
            
            // Filter out the user themselves
            let filtered_users: Vec<(Uuid, f32)> = similar_users
                .into_iter()
                .filter(|(id, _)| *id != user_id)
                .take(top_k)
                .collect();
            
            Ok(filtered_users)
        } else {
            Ok(Vec::new())
        }
    }

    pub async fn get_similar_items(&self, item_id: Uuid, top_k: usize) -> Result<Vec<(Uuid, f32)>> {
        if let Some(item_feature) = self.vector_db.get_item_feature(item_id).await? {
            let similar_items = self.vector_db
                .search_similar_items(&item_feature.embedding, top_k + 1)
                .await?;
            
            // Filter out the item itself
            let filtered_items: Vec<(Uuid, f32)> = similar_items
                .into_iter()
                .filter(|(id, _)| *id != item_id)
                .take(top_k)
                .collect();
            
            Ok(filtered_items)
        } else {
            Ok(Vec::new())
        }
    }

    pub async fn predict_user_item_score(&self, user_id: Uuid, item_id: Uuid) -> Result<f32> {
        let user_profile = self.vector_db.get_user_profile(user_id).await?;
        let item_feature = self.vector_db.get_item_feature(item_id).await?;
        
        match (user_profile, item_feature) {
            (Some(user), Some(item)) => {
                // Calculate cosine similarity as prediction score
                let score = crate::utils::cosine_similarity(&user.embedding, &item.embedding);
                Ok(score)
            }
            _ => Ok(0.0),
        }
    }

    pub async fn get_trending_items(&self, category: Option<String>, top_k: usize) -> Result<Vec<RecommendationItem>> {
        // This is a simplified implementation
        // In a real system, you would track item popularity and trends
        let mut trending_items = Vec::new();
        
        // For demonstration, we'll return some mock trending items
        for i in 0..top_k {
            trending_items.push(RecommendationItem {
                item_id: Uuid::new_v4(),
                score: 0.9 - (i as f32 * 0.1),
                reason: "Trending item".to_string(),
                category: category.clone().unwrap_or_else(|| "general".to_string()),
            });
        }
        
        Ok(trending_items)
    }

    pub async fn get_personalized_trending(&self, user_id: Uuid, top_k: usize) -> Result<Vec<RecommendationItem>> {
        // Get user profile
        let user_profile = self.vector_db.get_user_profile(user_id).await?;
        
        if let Some(profile) = user_profile {
            // Find items similar to user's preferences
            let similar_items = self.vector_db
                .search_similar_items(&profile.embedding, top_k * 2)
                .await?;
            
            let mut personalized_trending = Vec::new();
            
            for (item_id, score) in similar_items.into_iter().take(top_k) {
                if let Some(item_feature) = self.vector_db.get_item_feature(item_id).await? {
                    personalized_trending.push(RecommendationItem {
                        item_id,
                        score,
                        reason: format!("Personalized trending (score: {:.3})", score),
                        category: item_feature.category,
                    });
                }
            }
            
            Ok(personalized_trending)
        } else {
            // Fall back to general trending if no user profile
            self.get_trending_items(None, top_k).await
        }
    }

    pub async fn update_model_parameters(&self, parameters: ModelParameters) -> Result<()> {
        {
            let mut model_params = self.model_parameters.write().await;
            *model_params = Some(parameters);
        }
        
        self.increment_stat("model_updates").await;
        info!("Updated model parameters");
        Ok(())
    }

    pub async fn get_model_version(&self) -> Option<String> {
        let model_params = self.model_parameters.read().await;
        model_params.as_ref().map(|p| p.version.clone())
    }

    pub async fn health_check(&self) -> Result<HashMap<String, serde_json::Value>> {
        let mut health = HashMap::new();
        
        // Check vector database connection
        let vector_db_healthy = match self.vector_db.get_user_profile(Uuid::new_v4()).await {
            Ok(_) => true,
            Err(_) => false,
        };
        
        health.insert("vector_db".to_string(), serde_json::Value::Bool(vector_db_healthy));
        health.insert("model_loaded".to_string(), 
                     serde_json::Value::Bool(self.model_parameters.read().await.is_some()));
        health.insert("uptime".to_string(), 
                     serde_json::Value::String(format!("{:?}", std::time::SystemTime::now())));
        
        Ok(health)
    }

    pub async fn get_serving_stats(&self) -> HashMap<String, u64> {
        self.serving_stats.iter().map(|entry| (entry.key().clone(), *entry.value())).collect()
    }

    async fn increment_stat(&self, key: &str) {
        let mut counter = self.serving_stats.entry(key.to_string()).or_insert(0);
        *counter += 1;
    }

    async fn update_latency_stat(&self, latency_ms: u64) {
        // Simple moving average for latency
        let current_avg = self.serving_stats.get("avg_latency_ms").map(|v| *v).unwrap_or(0);
        let request_count = self.serving_stats.get("total_requests").map(|v| *v).unwrap_or(1);

        let new_avg = if request_count == 1 {
            latency_ms
        } else {
            (current_avg * (request_count - 1) + latency_ms) / request_count
        };

        self.serving_stats.insert("avg_latency_ms".to_string(), new_avg);

        // Track max latency
        let current_max = self.serving_stats.get("max_latency_ms").map(|v| *v).unwrap_or(0);
        if latency_ms > current_max {
            self.serving_stats.insert("max_latency_ms".to_string(), latency_ms);
        }
    }

    pub async fn get_user_recommendations_with_explanation(&self, user_id: Uuid, num_recommendations: usize) -> Result<Vec<(RecommendationItem, String)>> {
        let request = RecommendationRequest {
            user_id,
            num_recommendations,
            filter_categories: None,
            exclude_items: None,
        };
        
        let response = self.serve_recommendations(&request).await?;
        
        let mut explained_recommendations = Vec::new();
        
        for item in response.recommendations {
            let explanation = self.generate_explanation(&user_id, &item).await?;
            explained_recommendations.push((item, explanation));
        }
        
        Ok(explained_recommendations)
    }

    async fn generate_explanation(&self, user_id: &Uuid, item: &RecommendationItem) -> Result<String> {
        let user_profile = self.vector_db.get_user_profile(*user_id).await?;
        let item_feature = self.vector_db.get_item_feature(item.item_id).await?;
        
        match (user_profile, item_feature) {
            (Some(user), Some(item_feat)) => {
                let similarity = crate::utils::cosine_similarity(&user.embedding, &item_feat.embedding);
                
                let explanation = if similarity > 0.8 {
                    format!("Highly recommended based on your preferences in {} category", item_feat.category)
                } else if similarity > 0.6 {
                    format!("Recommended because you like similar {} items", item_feat.category)
                } else if item_feat.popularity_score > 0.8 {
                    format!("Popular {} item that might interest you", item_feat.category)
                } else {
                    format!("Recommended to help you discover new {} content", item_feat.category)
                };
                
                Ok(explanation)
            }
            _ => Ok("Recommended based on general popularity".to_string()),
        }
    }
}
