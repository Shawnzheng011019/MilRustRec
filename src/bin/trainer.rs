use milvuso::{init_tracing, AppState, Config};
use anyhow::Result;
use clap::Parser;
use tracing::info;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(short, long, default_value = "config/default.toml")]
    config: String,
    
    #[arg(short, long, default_value = "info")]
    log_level: String,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();
    
    // Initialize tracing with specified log level
    std::env::set_var("RUST_LOG", &args.log_level);
    init_tracing().await;

    info!("Starting MilRustRec Training Worker");

    // Load configuration
    let config = if std::path::Path::new(&args.config).exists() {
        Config::from_file(&args.config)?
    } else {
        info!("Config file not found, using default configuration");
        Config::default()
    };

    info!("Training worker configuration loaded: {:?}", config.training);

    // Initialize application state
    let state = AppState::new(config).await?;

    // Start training service
    state.training_service.start_training_worker().await?;

    info!("Training worker started successfully");

    // Keep the worker running
    loop {
        tokio::time::sleep(tokio::time::Duration::from_secs(60)).await;
        
        // Print training statistics
        match state.training_service.get_training_stats().await {
            Ok(stats) => {
                info!("Training stats: {:?}", stats);
            }
            Err(e) => {
                tracing::error!("Failed to get training stats: {}", e);
            }
        }
    }
}
