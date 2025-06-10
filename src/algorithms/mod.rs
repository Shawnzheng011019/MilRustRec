pub mod optimizer;
pub mod retriever;
pub mod initializer;

use crate::models::*;
use anyhow::Result;
use nalgebra::DVector;
use std::collections::HashMap;

#[async_trait::async_trait]
pub trait RecommendationAlgorithm: Send + Sync {
    async fn train(&mut self, examples: &[TrainingExample]) -> Result<()>;
    async fn predict(&self, user_features: &[f32], item_features: &[f32]) -> Result<f32>;
    async fn get_user_embedding(&self, user_id: uuid::Uuid) -> Result<Vec<f32>>;
    async fn get_item_embedding(&self, item_id: uuid::Uuid) -> Result<Vec<f32>>;
    async fn update_parameters(&mut self, parameters: &ModelParameters) -> Result<()>;
}

#[derive(Debug, Clone)]
pub struct CollaborativeFiltering {
    pub user_embeddings: HashMap<uuid::Uuid, DVector<f32>>,
    pub item_embeddings: HashMap<uuid::Uuid, DVector<f32>>,
    pub embedding_dim: usize,
    pub learning_rate: f64,
    pub regularization: f64,
}

impl CollaborativeFiltering {
    pub fn new(embedding_dim: usize, learning_rate: f64, regularization: f64) -> Self {
        Self {
            user_embeddings: HashMap::new(),
            item_embeddings: HashMap::new(),
            embedding_dim,
            learning_rate,
            regularization,
        }
    }
    
    pub fn initialize_user_embedding(&mut self, user_id: uuid::Uuid) {
        if !self.user_embeddings.contains_key(&user_id) {
            let embedding = initializer::xavier_uniform(self.embedding_dim);
            self.user_embeddings.insert(user_id, DVector::from_vec(embedding));
        }
    }
    
    pub fn initialize_item_embedding(&mut self, item_id: uuid::Uuid) {
        if !self.item_embeddings.contains_key(&item_id) {
            let embedding = initializer::xavier_uniform(self.embedding_dim);
            self.item_embeddings.insert(item_id, DVector::from_vec(embedding));
        }
    }
    
    pub fn compute_loss(&self, examples: &[TrainingExample]) -> f64 {
        let mut total_loss = 0.0f64;
        let mut count = 0;

        for example in examples {
            if let (Some(user_emb), Some(item_emb)) = (
                self.user_embeddings.get(&example.user_id),
                self.item_embeddings.get(&example.item_id)
            ) {
                let prediction = user_emb.dot(item_emb);
                let error = example.label - prediction;
                total_loss += (error * error) as f64;
                count += 1;
            }
        }

        if count > 0 {
            total_loss / count as f64
        } else {
            0.0
        }
    }
    
    pub fn sgd_update(&mut self, example: &TrainingExample) -> Result<()> {
        self.initialize_user_embedding(example.user_id);
        self.initialize_item_embedding(example.item_id);
        
        let user_emb = self.user_embeddings.get(&example.user_id).unwrap().clone();
        let item_emb = self.item_embeddings.get(&example.item_id).unwrap().clone();
        
        let prediction = user_emb.dot(&item_emb);
        let error = example.label - prediction;
        
        let user_gradient = &item_emb * error - &user_emb * (self.regularization as f32);
        let item_gradient = &user_emb * error - &item_emb * (self.regularization as f32);
        
        let new_user_emb = &user_emb + &user_gradient * (self.learning_rate as f32);
        let new_item_emb = &item_emb + &item_gradient * (self.learning_rate as f32);
        
        self.user_embeddings.insert(example.user_id, new_user_emb);
        self.item_embeddings.insert(example.item_id, new_item_emb);
        
        Ok(())
    }
}

#[async_trait::async_trait]
impl RecommendationAlgorithm for CollaborativeFiltering {
    async fn train(&mut self, examples: &[TrainingExample]) -> Result<()> {
        for example in examples {
            self.sgd_update(example)?;
        }
        Ok(())
    }
    
    async fn predict(&self, user_features: &[f32], item_features: &[f32]) -> Result<f32> {
        let user_vec = DVector::from_vec(user_features.to_vec());
        let item_vec = DVector::from_vec(item_features.to_vec());
        Ok(user_vec.dot(&item_vec))
    }
    
    async fn get_user_embedding(&self, user_id: uuid::Uuid) -> Result<Vec<f32>> {
        if let Some(embedding) = self.user_embeddings.get(&user_id) {
            Ok(embedding.as_slice().to_vec())
        } else {
            Ok(initializer::xavier_uniform(self.embedding_dim))
        }
    }
    
    async fn get_item_embedding(&self, item_id: uuid::Uuid) -> Result<Vec<f32>> {
        if let Some(embedding) = self.item_embeddings.get(&item_id) {
            Ok(embedding.as_slice().to_vec())
        } else {
            Ok(initializer::xavier_uniform(self.embedding_dim))
        }
    }
    
    async fn update_parameters(&mut self, _parameters: &ModelParameters) -> Result<()> {
        // Update embeddings from model parameters
        // This would be implemented based on the specific parameter format
        Ok(())
    }
}
