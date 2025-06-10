use serde::{Deserialize, Serialize};
use uuid::Uuid;
use chrono::{DateTime, Utc};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserAction {
    pub user_id: Uuid,
    pub item_id: Uuid,
    pub action_type: ActionType,
    pub timestamp: DateTime<Utc>,
    pub context: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ActionType {
    Click,
    Like,
    Share,
    Purchase,
    View,
    Convert,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserProfile {
    pub user_id: Uuid,
    pub embedding: Vec<f32>,
    pub preferences: Vec<String>,
    pub last_updated: DateTime<Utc>,
    pub interaction_count: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ItemFeature {
    pub item_id: Uuid,
    pub embedding: Vec<f32>,
    pub category: String,
    pub tags: Vec<String>,
    pub popularity_score: f32,
    pub created_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingExample {
    pub user_id: Uuid,
    pub item_id: Uuid,
    pub label: f32,
    pub user_features: Vec<f32>,
    pub item_features: Vec<f32>,
    pub context_features: Vec<f32>,
    pub timestamp: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecommendationRequest {
    pub user_id: Uuid,
    pub num_recommendations: usize,
    pub filter_categories: Option<Vec<String>>,
    pub exclude_items: Option<Vec<Uuid>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecommendationResponse {
    pub user_id: Uuid,
    pub recommendations: Vec<RecommendationItem>,
    pub generated_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecommendationItem {
    pub item_id: Uuid,
    pub score: f32,
    pub reason: String,
    pub category: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelParameters {
    pub version: String,
    pub user_embedding_weights: Vec<Vec<f32>>,
    pub item_embedding_weights: Vec<Vec<f32>>,
    pub bias_weights: Vec<f32>,
    pub updated_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureVector {
    pub id: Uuid,
    pub vector: Vec<f32>,
    pub metadata: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchTrainingData {
    pub batch_id: Uuid,
    pub examples: Vec<TrainingExample>,
    pub created_at: DateTime<Utc>,
}

impl UserAction {
    pub fn new(user_id: Uuid, item_id: Uuid, action_type: ActionType) -> Self {
        Self {
            user_id,
            item_id,
            action_type,
            timestamp: Utc::now(),
            context: None,
        }
    }
    
    pub fn with_context(mut self, context: serde_json::Value) -> Self {
        self.context = Some(context);
        self
    }
}

impl UserProfile {
    pub fn new(user_id: Uuid, embedding_dim: usize) -> Self {
        Self {
            user_id,
            embedding: vec![0.0; embedding_dim],
            preferences: Vec::new(),
            last_updated: Utc::now(),
            interaction_count: 0,
        }
    }
    
    pub fn update_embedding(&mut self, new_embedding: Vec<f32>) {
        self.embedding = new_embedding;
        self.last_updated = Utc::now();
    }
    
    pub fn increment_interactions(&mut self) {
        self.interaction_count += 1;
        self.last_updated = Utc::now();
    }
}

impl ItemFeature {
    pub fn new(item_id: Uuid, embedding: Vec<f32>, category: String) -> Self {
        Self {
            item_id,
            embedding,
            category,
            tags: Vec::new(),
            popularity_score: 0.0,
            created_at: Utc::now(),
        }
    }
    
    pub fn with_tags(mut self, tags: Vec<String>) -> Self {
        self.tags = tags;
        self
    }
    
    pub fn with_popularity(mut self, score: f32) -> Self {
        self.popularity_score = score;
        self
    }
}
