use crate::config::Config;
use crate::models::*;
use anyhow::Result;
use rdkafka::config::ClientConfig;
use rdkafka::producer::{FutureProducer, FutureRecord};
use rdkafka::consumer::{Consumer, StreamConsumer};
use rdkafka::Message;
use serde_json;
use std::time::Duration;
use tokio::sync::mpsc;
use tracing::{info, error, warn};

pub struct KafkaProducer {
    producer: FutureProducer,
    config: std::sync::Arc<Config>,
}

impl KafkaProducer {
    pub fn new(config: &Config) -> Result<Self> {
        let producer: FutureProducer = ClientConfig::new()
            .set("bootstrap.servers", &config.kafka.brokers)
            .set("message.timeout.ms", "5000")
            .set("queue.buffering.max.messages", "100000")
            .set("queue.buffering.max.kbytes", "1048576")
            .set("batch.num.messages", "1000")
            .create()?;

        Ok(Self {
            producer,
            config: std::sync::Arc::new(config.clone()),
        })
    }

    pub async fn send_user_action(&self, action: &UserAction) -> Result<()> {
        let payload = serde_json::to_string(action)?;
        let key = action.user_id.to_string();
        let record = FutureRecord::to(&self.config.kafka.log_topic)
            .payload(&payload)
            .key(&key);

        match self.producer.send(record, Duration::from_secs(5)).await {
            Ok(_) => {
                info!("User action sent to Kafka: {:?}", action.action_type);
                Ok(())
            }
            Err((e, _)) => {
                error!("Failed to send user action to Kafka: {}", e);
                Err(anyhow::anyhow!("Kafka send error: {}", e))
            }
        }
    }

    pub async fn send_feature_vector(&self, feature: &FeatureVector) -> Result<()> {
        let payload = serde_json::to_string(feature)?;
        let key = feature.id.to_string();
        let record = FutureRecord::to(&self.config.kafka.feature_topic)
            .payload(&payload)
            .key(&key);

        match self.producer.send(record, Duration::from_secs(5)).await {
            Ok(_) => {
                info!("Feature vector sent to Kafka: {}", feature.id);
                Ok(())
            }
            Err((e, _)) => {
                error!("Failed to send feature vector to Kafka: {}", e);
                Err(anyhow::anyhow!("Kafka send error: {}", e))
            }
        }
    }

    pub async fn send_training_example(&self, example: &TrainingExample) -> Result<()> {
        let payload = serde_json::to_string(example)?;
        let key = example.user_id.to_string();
        let record = FutureRecord::to(&self.config.kafka.training_topic)
            .payload(&payload)
            .key(&key);

        match self.producer.send(record, Duration::from_secs(5)).await {
            Ok(_) => {
                info!("Training example sent to Kafka: {} -> {}", example.user_id, example.item_id);
                Ok(())
            }
            Err((e, _)) => {
                error!("Failed to send training example to Kafka: {}", e);
                Err(anyhow::anyhow!("Kafka send error: {}", e))
            }
        }
    }
}

pub struct KafkaConsumer {
    consumer: StreamConsumer,
    config: std::sync::Arc<Config>,
}

impl KafkaConsumer {
    pub fn new(config: &Config) -> Result<Self> {
        let consumer: StreamConsumer = ClientConfig::new()
            .set("group.id", &config.kafka.group_id)
            .set("bootstrap.servers", &config.kafka.brokers)
            .set("enable.partition.eof", "false")
            .set("session.timeout.ms", "6000")
            .set("enable.auto.commit", "true")
            .set("auto.offset.reset", &config.kafka.auto_offset_reset)
            .create()?;

        Ok(Self {
            consumer,
            config: std::sync::Arc::new(config.clone()),
        })
    }

    pub async fn subscribe_user_actions(&self) -> Result<()> {
        self.consumer.subscribe(&[&self.config.kafka.log_topic])?;
        Ok(())
    }

    pub async fn subscribe_features(&self) -> Result<()> {
        self.consumer.subscribe(&[&self.config.kafka.feature_topic])?;
        Ok(())
    }

    pub async fn subscribe_training_examples(&self) -> Result<()> {
        self.consumer.subscribe(&[&self.config.kafka.training_topic])?;
        Ok(())
    }

    pub async fn consume_user_actions(&self, tx: mpsc::Sender<UserAction>) -> Result<()> {
        self.subscribe_user_actions().await?;
        
        loop {
            match self.consumer.recv().await {
                Ok(message) => {
                    if let Some(payload) = message.payload() {
                        match serde_json::from_slice::<UserAction>(payload) {
                            Ok(action) => {
                                if let Err(e) = tx.send(action).await {
                                    error!("Failed to send user action to channel: {}", e);
                                    break;
                                }
                            }
                            Err(e) => {
                                warn!("Failed to deserialize user action: {}", e);
                            }
                        }
                    }
                }
                Err(e) => {
                    error!("Kafka consumer error: {}", e);
                    tokio::time::sleep(Duration::from_secs(1)).await;
                }
            }
        }
        
        Ok(())
    }

    pub async fn consume_features(&self, tx: mpsc::Sender<FeatureVector>) -> Result<()> {
        self.subscribe_features().await?;
        
        loop {
            match self.consumer.recv().await {
                Ok(message) => {
                    if let Some(payload) = message.payload() {
                        match serde_json::from_slice::<FeatureVector>(payload) {
                            Ok(feature) => {
                                if let Err(e) = tx.send(feature).await {
                                    error!("Failed to send feature vector to channel: {}", e);
                                    break;
                                }
                            }
                            Err(e) => {
                                warn!("Failed to deserialize feature vector: {}", e);
                            }
                        }
                    }
                }
                Err(e) => {
                    error!("Kafka consumer error: {}", e);
                    tokio::time::sleep(Duration::from_secs(1)).await;
                }
            }
        }
        
        Ok(())
    }

    pub async fn consume_training_examples(&self, tx: mpsc::Sender<TrainingExample>) -> Result<()> {
        self.subscribe_training_examples().await?;
        
        loop {
            match self.consumer.recv().await {
                Ok(message) => {
                    if let Some(payload) = message.payload() {
                        match serde_json::from_slice::<TrainingExample>(payload) {
                            Ok(example) => {
                                if let Err(e) = tx.send(example).await {
                                    error!("Failed to send training example to channel: {}", e);
                                    break;
                                }
                            }
                            Err(e) => {
                                warn!("Failed to deserialize training example: {}", e);
                            }
                        }
                    }
                }
                Err(e) => {
                    error!("Kafka consumer error: {}", e);
                    tokio::time::sleep(Duration::from_secs(1)).await;
                }
            }
        }
        
        Ok(())
    }
}
