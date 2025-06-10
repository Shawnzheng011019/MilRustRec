use crate::config::Config;
use crate::models::*;
use crate::algorithms::retriever::{InMemoryRetriever, VectorRetriever};
use anyhow::Result;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::info;
use uuid::Uuid;

pub struct VectorDbService {
    user_retriever: Arc<RwLock<InMemoryRetriever>>,
    item_retriever: Arc<RwLock<InMemoryRetriever>>,
    user_profiles: Arc<RwLock<HashMap<Uuid, UserProfile>>>,
    item_features: Arc<RwLock<HashMap<Uuid, ItemFeature>>>,
    config: Arc<Config>,
}

impl VectorDbService {
    pub async fn new(config: &Config) -> Result<Self> {
        let user_retriever = Arc::new(RwLock::new(
            InMemoryRetriever::new(config.milvus.dimension)
        ));
        let item_retriever = Arc::new(RwLock::new(
            InMemoryRetriever::new(config.milvus.dimension)
        ));

        info!("Initialized in-memory vector database with dimension {}", config.milvus.dimension);

        Ok(Self {
            user_retriever,
            item_retriever,
            user_profiles: Arc::new(RwLock::new(HashMap::new())),
            item_features: Arc::new(RwLock::new(HashMap::new())),
            config: Arc::new(config.clone()),
        })
    }

    pub async fn insert_user_profile(&self, profile: &UserProfile) -> Result<()> {
        // Insert into user retriever
        {
            let mut retriever = self.user_retriever.write().await;
            retriever.add_vector(profile.user_id, profile.embedding.clone()).await?;
        }

        // Store profile metadata
        {
            let mut profiles = self.user_profiles.write().await;
            profiles.insert(profile.user_id, profile.clone());
        }

        info!("Inserted user profile: {}", profile.user_id);
        Ok(())
    }

    pub async fn insert_item_feature(&self, feature: &ItemFeature) -> Result<()> {
        // Insert into item retriever
        {
            let mut retriever = self.item_retriever.write().await;
            retriever.add_vector(feature.item_id, feature.embedding.clone()).await?;
        }

        // Store feature metadata
        {
            let mut features = self.item_features.write().await;
            features.insert(feature.item_id, feature.clone());
        }

        info!("Inserted item feature: {}", feature.item_id);
        Ok(())
    }

    pub async fn search_similar_users(&self, user_embedding: &[f32], top_k: usize) -> Result<Vec<(Uuid, f32)>> {
        let retriever = self.user_retriever.read().await;
        let results = retriever.search_similar(user_embedding, top_k).await?;
        Ok(results)
    }

    pub async fn search_similar_items(&self, item_embedding: &[f32], top_k: usize) -> Result<Vec<(Uuid, f32)>> {
        let retriever = self.item_retriever.read().await;
        let results = retriever.search_similar(item_embedding, top_k).await?;
        Ok(results)
    }

    pub async fn get_user_profile(&self, user_id: Uuid) -> Result<Option<UserProfile>> {
        let profiles = self.user_profiles.read().await;
        Ok(profiles.get(&user_id).cloned())
    }

    pub async fn get_item_feature(&self, item_id: Uuid) -> Result<Option<ItemFeature>> {
        let features = self.item_features.read().await;
        Ok(features.get(&item_id).cloned())
    }

    pub async fn update_user_embedding(&self, user_id: Uuid, new_embedding: Vec<f32>) -> Result<()> {
        // Update in retriever
        {
            let mut retriever = self.user_retriever.write().await;
            retriever.update_vector(user_id, new_embedding.clone()).await?;
        }

        // Update profile
        {
            let mut profiles = self.user_profiles.write().await;
            if let Some(profile) = profiles.get_mut(&user_id) {
                profile.update_embedding(new_embedding);
            }
        }

        Ok(())
    }

    pub async fn update_item_embedding(&self, item_id: Uuid, new_embedding: Vec<f32>) -> Result<()> {
        // Update in retriever
        {
            let mut retriever = self.item_retriever.write().await;
            retriever.update_vector(item_id, new_embedding.clone()).await?;
        }

        // Update feature
        {
            let mut features = self.item_features.write().await;
            if let Some(feature) = features.get_mut(&item_id) {
                feature.embedding = new_embedding;
            }
        }

        Ok(())
    }

    pub async fn batch_insert_profiles(&self, profiles: &[UserProfile]) -> Result<()> {
        for profile in profiles {
            self.insert_user_profile(profile).await?;
        }
        info!("Batch inserted {} user profiles", profiles.len());
        Ok(())
    }

    pub async fn batch_insert_features(&self, features: &[ItemFeature]) -> Result<()> {
        for feature in features {
            self.insert_item_feature(feature).await?;
        }
        info!("Batch inserted {} item features", features.len());
        Ok(())
    }
}
