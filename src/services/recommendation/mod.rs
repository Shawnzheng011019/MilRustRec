use crate::config::Config;
use crate::models::*;
use crate::services::vector_db::VectorDbService;
use crate::algorithms::{CollaborativeFiltering, RecommendationAlgorithm};
use anyhow::Result;
use redis::AsyncCommands;
use std::sync::Arc;
use tokio::sync::RwLock;
use uuid::Uuid;
use chrono::{Utc, Timelike, Datelike};
use tracing::info;
use dashmap::DashMap;

pub struct RecommendationService {
    vector_db: Arc<VectorDbService>,
    redis_client: Arc<redis::Client>,
    algorithm: Arc<RwLock<CollaborativeFiltering>>,
    config: Arc<Config>,
    user_profiles_cache: Arc<DashMap<Uuid, UserProfile>>,
    item_features_cache: Arc<DashMap<Uuid, ItemFeature>>,
}

impl RecommendationService {
    pub async fn new(
        vector_db: Arc<VectorDbService>,
        redis_client: Arc<redis::Client>,
        config: Arc<Config>,
    ) -> Result<Self> {
        let algorithm = Arc::new(RwLock::new(
            CollaborativeFiltering::new(
                config.recommendation.embedding_dim,
                config.training.learning_rate,
                0.01, // regularization
            )
        ));

        Ok(Self {
            vector_db,
            redis_client,
            algorithm,
            config,
            user_profiles_cache: Arc::new(DashMap::new()),
            item_features_cache: Arc::new(DashMap::new()),
        })
    }

    pub async fn get_recommendations(&self, request: &RecommendationRequest) -> Result<RecommendationResponse> {
        let user_profile = self.get_or_create_user_profile(request.user_id).await?;
        
        // Get similar items based on user embedding
        let similar_items = self.vector_db
            .search_similar_items(&user_profile.embedding, request.num_recommendations * 2)
            .await?;

        let mut recommendations = Vec::new();
        
        for (item_id, similarity_score) in similar_items {
            // Skip excluded items
            if let Some(ref excluded) = request.exclude_items {
                if excluded.contains(&item_id) {
                    continue;
                }
            }

            // Get item feature
            if let Some(item_feature) = self.get_item_feature(item_id).await? {
                // Filter by category if specified
                if let Some(ref filter_categories) = request.filter_categories {
                    if !filter_categories.contains(&item_feature.category) {
                        continue;
                    }
                }

                // Calculate prediction score using the algorithm
                let prediction_score = self.algorithm
                    .read()
                    .await
                    .predict(&user_profile.embedding, &item_feature.embedding)
                    .await
                    .unwrap_or(0.0);

                // Combine similarity and prediction scores
                let final_score = (similarity_score + prediction_score) / 2.0;

                if final_score >= self.config.recommendation.similarity_threshold {
                    recommendations.push(RecommendationItem {
                        item_id,
                        score: final_score,
                        reason: format!("Similar to your preferences (score: {:.3})", final_score),
                        category: item_feature.category.clone(),
                    });
                }

                if recommendations.len() >= request.num_recommendations {
                    break;
                }
            }
        }

        // Sort by score descending
        recommendations.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));

        Ok(RecommendationResponse {
            user_id: request.user_id,
            recommendations,
            generated_at: Utc::now(),
        })
    }

    pub async fn process_user_action(&self, action: &UserAction) -> Result<()> {
        // Update user profile based on action
        let mut user_profile = self.get_or_create_user_profile(action.user_id).await?;
        
        // Get item feature
        if let Some(item_feature) = self.get_item_feature(action.item_id).await? {
            // Update user embedding based on interaction
            let weight = self.get_action_weight(&action.action_type);
            self.update_user_embedding(&mut user_profile, &item_feature, weight).await?;
            
            // Cache updated profile
            self.user_profiles_cache.insert(action.user_id, user_profile.clone());
            
            // Update in vector database
            self.vector_db.update_user_embedding(action.user_id, user_profile.embedding.clone()).await?;
            
            // Create training example
            let training_example = TrainingExample {
                user_id: action.user_id,
                item_id: action.item_id,
                label: weight,
                user_features: user_profile.embedding.clone(),
                item_features: item_feature.embedding.clone(),
                context_features: self.extract_context_features(action).await?,
                timestamp: action.timestamp,
            };

            // Train algorithm incrementally
            self.algorithm.write().await.train(&[training_example]).await?;
            
            info!("Processed user action: {:?} for user {}", action.action_type, action.user_id);
        }

        Ok(())
    }

    async fn get_or_create_user_profile(&self, user_id: Uuid) -> Result<UserProfile> {
        // Check cache first
        if let Some(profile) = self.user_profiles_cache.get(&user_id) {
            return Ok(profile.clone());
        }

        // Check Redis cache
        let mut redis_conn = self.redis_client.get_async_connection().await?;
        let cache_key = format!("user_profile:{}", user_id);
        
        if let Ok(cached_data) = redis_conn.get::<_, String>(&cache_key).await {
            if let Ok(profile) = serde_json::from_str::<UserProfile>(&cached_data) {
                self.user_profiles_cache.insert(user_id, profile.clone());
                return Ok(profile);
            }
        }

        // Check vector database
        if let Some(profile) = self.vector_db.get_user_profile(user_id).await? {
            // Cache in Redis and memory
            let profile_json = serde_json::to_string(&profile)?;
            let _: () = redis_conn.set_ex(&cache_key, profile_json, self.config.redis.ttl_seconds).await?;
            self.user_profiles_cache.insert(user_id, profile.clone());
            return Ok(profile);
        }

        // Create new profile
        let new_profile = UserProfile::new(user_id, self.config.recommendation.embedding_dim);
        
        // Save to vector database
        self.vector_db.insert_user_profile(&new_profile).await?;
        
        // Cache in Redis and memory
        let profile_json = serde_json::to_string(&new_profile)?;
        let _: () = redis_conn.set_ex(&cache_key, profile_json, self.config.redis.ttl_seconds).await?;
        self.user_profiles_cache.insert(user_id, new_profile.clone());

        info!("Created new user profile: {}", user_id);
        Ok(new_profile)
    }

    async fn get_item_feature(&self, item_id: Uuid) -> Result<Option<ItemFeature>> {
        // Check cache first
        if let Some(feature) = self.item_features_cache.get(&item_id) {
            return Ok(Some(feature.clone()));
        }

        // Check Redis cache
        let mut redis_conn = self.redis_client.get_async_connection().await?;
        let cache_key = format!("item_feature:{}", item_id);
        
        if let Ok(cached_data) = redis_conn.get::<_, String>(&cache_key).await {
            if let Ok(feature) = serde_json::from_str::<ItemFeature>(&cached_data) {
                self.item_features_cache.insert(item_id, feature.clone());
                return Ok(Some(feature));
            }
        }

        // Check vector database
        if let Some(feature) = self.vector_db.get_item_feature(item_id).await? {
            // Cache in Redis and memory
            let feature_json = serde_json::to_string(&feature)?;
            let _: () = redis_conn.set_ex(&cache_key, feature_json, self.config.redis.ttl_seconds).await?;
            self.item_features_cache.insert(item_id, feature.clone());
            return Ok(Some(feature));
        }

        Ok(None)
    }

    async fn update_user_embedding(&self, profile: &mut UserProfile, item_feature: &ItemFeature, weight: f32) -> Result<()> {
        // Simple weighted average update
        let learning_rate = 0.1;
        
        for i in 0..profile.embedding.len() {
            profile.embedding[i] = profile.embedding[i] * (1.0 - learning_rate) + 
                                  item_feature.embedding[i] * learning_rate * weight;
        }
        
        profile.increment_interactions();
        Ok(())
    }

    fn get_action_weight(&self, action_type: &ActionType) -> f32 {
        match action_type {
            ActionType::View => 0.1,
            ActionType::Click => 0.3,
            ActionType::Like => 0.7,
            ActionType::Share => 0.8,
            ActionType::Purchase => 1.0,
            ActionType::Convert => 1.0,
        }
    }

    async fn extract_context_features(&self, action: &UserAction) -> Result<Vec<f32>> {
        // Extract context features from action
        let mut features = vec![0.0; 10]; // Simple context features
        
        // Time-based features
        let hour = action.timestamp.hour() as f32 / 24.0;
        let day_of_week = action.timestamp.weekday().num_days_from_monday() as f32 / 7.0;
        
        features[0] = hour;
        features[1] = day_of_week;
        
        // Action type encoding
        features[2] = self.get_action_weight(&action.action_type);
        
        Ok(features)
    }

    pub async fn add_item_feature(&self, feature: ItemFeature) -> Result<()> {
        // Save to vector database
        self.vector_db.insert_item_feature(&feature).await?;
        
        // Cache in memory and Redis
        let mut redis_conn = self.redis_client.get_async_connection().await?;
        let cache_key = format!("item_feature:{}", feature.item_id);
        let feature_json = serde_json::to_string(&feature)?;
        let _: () = redis_conn.set_ex(&cache_key, feature_json, self.config.redis.ttl_seconds).await?;
        
        self.item_features_cache.insert(feature.item_id, feature);
        
        Ok(())
    }
}
