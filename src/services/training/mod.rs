use crate::config::Config;
use crate::models::*;
use crate::services::{vector_db::VectorDbService, kafka::KafkaProducer};
use crate::algorithms::{CollaborativeFiltering, RecommendationAlgorithm};
use anyhow::Result;
use std::sync::Arc;
use tokio::sync::{RwLock, mpsc};
use uuid::Uuid;
use chrono::Utc;
use tracing::{info, error, warn};
use std::collections::HashMap;
use std::time::{Duration, Instant};

pub struct TrainingService {
    vector_db: Arc<VectorDbService>,
    kafka_producer: Arc<KafkaProducer>,
    algorithm: Arc<RwLock<CollaborativeFiltering>>,
    config: Arc<Config>,
    training_buffer: Arc<RwLock<Vec<TrainingExample>>>,
    last_model_save: Arc<RwLock<Instant>>,
}

impl TrainingService {
    pub async fn new(
        vector_db: Arc<VectorDbService>,
        kafka_producer: Arc<KafkaProducer>,
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
            kafka_producer,
            algorithm,
            config,
            training_buffer: Arc::new(RwLock::new(Vec::new())),
            last_model_save: Arc::new(RwLock::new(Instant::now())),
        })
    }

    pub async fn start_training_worker(&self) -> Result<()> {
        let (tx, rx) = mpsc::channel::<TrainingExample>(1000);
        
        // Start Kafka consumer for training examples
        let kafka_consumer = crate::services::kafka::KafkaConsumer::new(&self.config)?;
        let consumer_tx = tx.clone();
        
        tokio::spawn(async move {
            if let Err(e) = kafka_consumer.consume_training_examples(consumer_tx).await {
                error!("Training example consumer error: {}", e);
            }
        });

        // Start batch training worker
        let training_service = self.clone();
        tokio::spawn(async move {
            training_service.batch_training_worker(rx).await;
        });

        // Start model saving worker
        let saving_service = self.clone();
        tokio::spawn(async move {
            saving_service.model_saving_worker().await;
        });

        info!("Training workers started");
        Ok(())
    }

    async fn batch_training_worker(&self, mut rx: mpsc::Receiver<TrainingExample>) {
        let mut batch = Vec::new();
        let batch_timeout = Duration::from_secs(30);
        let mut last_batch_time = Instant::now();

        loop {
            tokio::select! {
                example = rx.recv() => {
                    match example {
                        Some(example) => {
                            batch.push(example);
                            
                            // Process batch if it's full or timeout reached
                            if batch.len() >= self.config.training.batch_size || 
                               last_batch_time.elapsed() > batch_timeout {
                                if let Err(e) = self.process_training_batch(&batch).await {
                                    error!("Failed to process training batch: {}", e);
                                }
                                batch.clear();
                                last_batch_time = Instant::now();
                            }
                        }
                        None => {
                            warn!("Training example channel closed");
                            break;
                        }
                    }
                }
                _ = tokio::time::sleep(batch_timeout) => {
                    if !batch.is_empty() {
                        if let Err(e) = self.process_training_batch(&batch).await {
                            error!("Failed to process training batch: {}", e);
                        }
                        batch.clear();
                        last_batch_time = Instant::now();
                    }
                }
            }
        }
    }

    async fn process_training_batch(&self, examples: &[TrainingExample]) -> Result<()> {
        if examples.is_empty() {
            return Ok(());
        }

        info!("Processing training batch of {} examples", examples.len());

        // Add negative sampling
        let augmented_examples = self.add_negative_samples(examples).await?;

        // Train the algorithm
        {
            let mut algorithm = self.algorithm.write().await;
            algorithm.train(&augmented_examples).await?;
        }

        // Update embeddings in vector database
        self.update_embeddings_from_training(&augmented_examples).await?;

        // Store training examples for batch processing
        {
            let mut buffer = self.training_buffer.write().await;
            buffer.extend_from_slice(&augmented_examples);
        }

        info!("Completed training batch processing");
        Ok(())
    }

    async fn add_negative_samples(&self, examples: &[TrainingExample]) -> Result<Vec<TrainingExample>> {
        let mut augmented = examples.to_vec();
        let negative_ratio = self.config.training.negative_sampling_ratio;
        
        for example in examples {
            if example.label > 0.5 { // Only add negatives for positive examples
                let num_negatives = (negative_ratio as usize).min(5);
                
                for _ in 0..num_negatives {
                    // Generate random negative item
                    let negative_item_id = Uuid::new_v4();
                    
                    // Create negative example
                    let negative_example = TrainingExample {
                        user_id: example.user_id,
                        item_id: negative_item_id,
                        label: 0.0,
                        user_features: example.user_features.clone(),
                        item_features: self.generate_random_item_features().await?,
                        context_features: example.context_features.clone(),
                        timestamp: example.timestamp,
                    };
                    
                    augmented.push(negative_example);
                }
            }
        }
        
        Ok(augmented)
    }

    async fn generate_random_item_features(&self) -> Result<Vec<f32>> {
        use crate::algorithms::initializer::xavier_uniform;
        Ok(xavier_uniform(self.config.recommendation.embedding_dim))
    }

    async fn update_embeddings_from_training(&self, examples: &[TrainingExample]) -> Result<()> {
        let mut user_updates = HashMap::new();
        let mut item_updates = HashMap::new();

        // Collect unique users and items
        for example in examples {
            user_updates.insert(example.user_id, &example.user_features);
            item_updates.insert(example.item_id, &example.item_features);
        }

        // Update user embeddings
        for (user_id, features) in user_updates {
            if let Err(e) = self.vector_db.update_user_embedding(user_id, features.clone()).await {
                warn!("Failed to update user embedding for {}: {}", user_id, e);
            }
        }

        // Update item embeddings
        for (item_id, features) in item_updates {
            if let Err(e) = self.vector_db.update_item_embedding(item_id, features.clone()).await {
                warn!("Failed to update item embedding for {}: {}", item_id, e);
            }
        }

        Ok(())
    }

    async fn model_saving_worker(&self) {
        let save_interval = Duration::from_secs(self.config.training.model_save_interval);
        
        loop {
            tokio::time::sleep(save_interval).await;
            
            if let Err(e) = self.save_model_parameters().await {
                error!("Failed to save model parameters: {}", e);
            }
        }
    }

    async fn save_model_parameters(&self) -> Result<()> {
        let algorithm = self.algorithm.read().await;
        
        // Extract model parameters
        let mut user_embeddings = Vec::new();
        let mut item_embeddings = Vec::new();
        
        for (_, embedding) in &algorithm.user_embeddings {
            user_embeddings.push(embedding.as_slice().to_vec());
        }
        
        for (_, embedding) in &algorithm.item_embeddings {
            item_embeddings.push(embedding.as_slice().to_vec());
        }

        let parameters = ModelParameters {
            version: format!("v{}", Utc::now().timestamp()),
            user_embedding_weights: user_embeddings,
            item_embedding_weights: item_embeddings,
            bias_weights: vec![0.0; self.config.recommendation.embedding_dim],
            updated_at: Utc::now(),
        };

        // Save to persistent storage (could be HDFS, S3, etc.)
        self.save_to_persistent_storage(&parameters).await?;
        
        // Update last save time
        {
            let mut last_save = self.last_model_save.write().await;
            *last_save = Instant::now();
        }

        info!("Model parameters saved successfully");
        Ok(())
    }

    async fn save_to_persistent_storage(&self, parameters: &ModelParameters) -> Result<()> {
        // In a real implementation, this would save to HDFS, S3, or another persistent store
        // For now, we'll just log the save operation
        info!("Saving model parameters version: {}", parameters.version);
        
        // Create batch training data
        let training_buffer = self.training_buffer.read().await;
        if !training_buffer.is_empty() {
            let batch_data = BatchTrainingData {
                batch_id: Uuid::new_v4(),
                examples: training_buffer.clone(),
                created_at: Utc::now(),
            };
            
            info!("Created batch training data with {} examples", batch_data.examples.len());
        }
        
        Ok(())
    }

    pub async fn load_model_parameters(&self, version: &str) -> Result<()> {
        // In a real implementation, this would load from persistent storage
        info!("Loading model parameters version: {}", version);
        
        // For now, just initialize with default parameters
        let _algorithm = self.algorithm.write().await;
        // algorithm.load_parameters(...);
        
        Ok(())
    }

    pub async fn get_training_stats(&self) -> Result<HashMap<String, serde_json::Value>> {
        let algorithm = self.algorithm.read().await;
        let buffer = self.training_buffer.read().await;
        let last_save = self.last_model_save.read().await;
        
        let mut stats = HashMap::new();
        stats.insert("user_embeddings_count".to_string(), 
                    serde_json::Value::Number(algorithm.user_embeddings.len().into()));
        stats.insert("item_embeddings_count".to_string(), 
                    serde_json::Value::Number(algorithm.item_embeddings.len().into()));
        stats.insert("training_buffer_size".to_string(), 
                    serde_json::Value::Number(buffer.len().into()));
        stats.insert("last_model_save".to_string(), 
                    serde_json::Value::String(format!("{:?}", *last_save)));
        
        Ok(stats)
    }
}

impl Clone for TrainingService {
    fn clone(&self) -> Self {
        Self {
            vector_db: self.vector_db.clone(),
            kafka_producer: self.kafka_producer.clone(),
            algorithm: self.algorithm.clone(),
            config: self.config.clone(),
            training_buffer: self.training_buffer.clone(),
            last_model_save: self.last_model_save.clone(),
        }
    }
}
