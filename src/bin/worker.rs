use milvuso::{init_tracing, AppState, Config};
use anyhow::Result;
use clap::Parser;
use tokio::sync::mpsc;
use tracing::{info, error};
use chrono::{Timelike, Datelike};

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(short, long, default_value = "config/default.toml")]
    config: String,
    
    #[arg(short, long, default_value = "info")]
    log_level: String,
    
    #[arg(short, long, default_value = "feature")]
    worker_type: String,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();
    
    // Initialize tracing with specified log level
    std::env::set_var("RUST_LOG", &args.log_level);
    init_tracing().await;

    info!("Starting Milvuso Worker: {}", args.worker_type);

    // Load configuration
    let config = if std::path::Path::new(&args.config).exists() {
        Config::from_file(&args.config)?
    } else {
        info!("Config file not found, using default configuration");
        Config::default()
    };

    // Initialize application state
    let state = AppState::new(config).await?;

    match args.worker_type.as_str() {
        "feature" => {
            start_feature_worker(state).await?;
        }
        "action" => {
            start_action_worker(state).await?;
        }
        "joiner" => {
            start_joiner_worker(state).await?;
        }
        _ => {
            error!("Unknown worker type: {}", args.worker_type);
            return Err(anyhow::anyhow!("Invalid worker type"));
        }
    }

    Ok(())
}

async fn start_feature_worker(state: AppState) -> Result<()> {
    info!("Starting Feature Generation Worker");
    
    let (tx, mut rx) = mpsc::channel::<milvuso::UserAction>(1000);
    
    // Start Kafka consumer for user actions
    let consumer = state.kafka_consumer.clone();
    tokio::spawn(async move {
        if let Err(e) = consumer.consume_user_actions(tx).await {
            error!("User action consumer error: {}", e);
        }
    });

    // Process user actions and generate features
    while let Some(action) = rx.recv().await {
        if let Err(e) = process_user_action_for_features(&state, &action).await {
            error!("Failed to process user action for features: {}", e);
        }
    }

    Ok(())
}

async fn start_action_worker(state: AppState) -> Result<()> {
    info!("Starting Action Processing Worker");
    
    let (tx, mut rx) = mpsc::channel::<milvuso::UserAction>(1000);
    
    // Start Kafka consumer for user actions
    let consumer = state.kafka_consumer.clone();
    tokio::spawn(async move {
        if let Err(e) = consumer.consume_user_actions(tx).await {
            error!("User action consumer error: {}", e);
        }
    });

    // Process user actions for real-time recommendations
    while let Some(action) = rx.recv().await {
        if let Err(e) = state.recommendation_service.process_user_action(&action).await {
            error!("Failed to process user action: {}", e);
        }
    }

    Ok(())
}

async fn start_joiner_worker(state: AppState) -> Result<()> {
    info!("Starting Joiner Worker (Flink Job simulation)");
    
    let (action_tx, mut action_rx) = mpsc::channel::<milvuso::UserAction>(1000);
    let (feature_tx, mut feature_rx) = mpsc::channel::<milvuso::FeatureVector>(1000);
    
    // Start Kafka consumers
    let action_consumer = state.kafka_consumer.clone();
    tokio::spawn(async move {
        if let Err(e) = action_consumer.consume_user_actions(action_tx).await {
            error!("User action consumer error: {}", e);
        }
    });

    let feature_consumer = state.kafka_consumer.clone();
    tokio::spawn(async move {
        if let Err(e) = feature_consumer.consume_features(feature_tx).await {
            error!("Feature consumer error: {}", e);
        }
    });

    // Join actions with features and create training examples
    let mut action_buffer = Vec::new();
    let mut feature_buffer = Vec::new();
    
    loop {
        tokio::select! {
            action = action_rx.recv() => {
                if let Some(action) = action {
                    action_buffer.push(action);
                    if action_buffer.len() >= 100 {
                        if let Err(e) = process_joined_data(&state, &action_buffer, &feature_buffer).await {
                            error!("Failed to process joined data: {}", e);
                        }
                        action_buffer.clear();
                    }
                }
            }
            feature = feature_rx.recv() => {
                if let Some(feature) = feature {
                    feature_buffer.push(feature);
                    if feature_buffer.len() >= 100 {
                        feature_buffer.clear(); // Keep buffer size manageable
                    }
                }
            }
            _ = tokio::time::sleep(tokio::time::Duration::from_secs(30)) => {
                if !action_buffer.is_empty() {
                    if let Err(e) = process_joined_data(&state, &action_buffer, &feature_buffer).await {
                        error!("Failed to process joined data: {}", e);
                    }
                    action_buffer.clear();
                }
            }
        }
    }
}

async fn process_user_action_for_features(state: &AppState, action: &milvuso::UserAction) -> Result<()> {
    // Generate feature vector from user action
    let feature_vector = milvuso::FeatureVector {
        id: action.user_id,
        vector: generate_feature_vector_from_action(action).await?,
        metadata: serde_json::json!({
            "action_type": action.action_type,
            "timestamp": action.timestamp,
            "item_id": action.item_id
        }),
    };

    // Send feature vector to Kafka
    state.kafka_producer.send_feature_vector(&feature_vector).await?;
    
    info!("Generated feature vector for user action: {:?}", action.action_type);
    Ok(())
}

async fn generate_feature_vector_from_action(action: &milvuso::UserAction) -> Result<Vec<f32>> {
    // Simple feature generation based on action
    let mut features = vec![0.0; 128];
    
    // Action type encoding
    match action.action_type {
        milvuso::ActionType::View => features[0] = 1.0,
        milvuso::ActionType::Click => features[1] = 1.0,
        milvuso::ActionType::Like => features[2] = 1.0,
        milvuso::ActionType::Share => features[3] = 1.0,
        milvuso::ActionType::Purchase => features[4] = 1.0,
        milvuso::ActionType::Convert => features[5] = 1.0,
    }
    
    // Time-based features
    let hour = action.timestamp.hour() as f32 / 24.0;
    let day_of_week = action.timestamp.weekday().num_days_from_monday() as f32 / 7.0;
    features[6] = hour;
    features[7] = day_of_week;
    
    // Add some random features for demonstration
    use rand::Rng;
    let mut rng = rand::thread_rng();
    for i in 8..128 {
        features[i] = rng.gen_range(-1.0..1.0);
    }
    
    Ok(features)
}

async fn process_joined_data(
    state: &AppState,
    actions: &[milvuso::UserAction],
    _features: &[milvuso::FeatureVector],
) -> Result<()> {
    for action in actions {
        // Create training example from joined data
        let user_profile = state.vector_db.get_user_profile(action.user_id).await?
            .unwrap_or_else(|| milvuso::UserProfile::new(action.user_id, 128));
        
        let item_feature = state.vector_db.get_item_feature(action.item_id).await?;
        
        if let Some(item_feature) = item_feature {
            let training_example = milvuso::TrainingExample {
                user_id: action.user_id,
                item_id: action.item_id,
                label: get_label_from_action(&action.action_type),
                user_features: user_profile.embedding,
                item_features: item_feature.embedding,
                context_features: generate_context_features(action).await?,
                timestamp: action.timestamp,
            };

            // Send training example to Kafka
            state.kafka_producer.send_training_example(&training_example).await?;
        }
    }

    info!("Processed {} joined actions", actions.len());
    Ok(())
}

fn get_label_from_action(action_type: &milvuso::ActionType) -> f32 {
    match action_type {
        milvuso::ActionType::View => 0.1,
        milvuso::ActionType::Click => 0.3,
        milvuso::ActionType::Like => 0.7,
        milvuso::ActionType::Share => 0.8,
        milvuso::ActionType::Purchase => 1.0,
        milvuso::ActionType::Convert => 1.0,
    }
}

async fn generate_context_features(action: &milvuso::UserAction) -> Result<Vec<f32>> {
    let mut features = vec![0.0; 10];
    
    // Time-based context
    let hour = action.timestamp.hour() as f32 / 24.0;
    let day_of_week = action.timestamp.weekday().num_days_from_monday() as f32 / 7.0;
    features[0] = hour;
    features[1] = day_of_week;
    
    // Action strength
    features[2] = get_label_from_action(&action.action_type);
    
    Ok(features)
}
