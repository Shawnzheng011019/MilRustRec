use milvuso::{init_tracing, AppState, Config};
use axum::{
    extract::{Path, Query, State},
    http::StatusCode,
    response::Json,
    routing::{get, post},
    Router,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tower::ServiceBuilder;
use tower_http::cors::CorsLayer;
use tower_http::trace::TraceLayer;
use tracing::info;
use uuid::Uuid;

#[derive(Debug, Deserialize)]
struct RecommendationQuery {
    num_recommendations: Option<usize>,
    filter_categories: Option<String>,
    exclude_items: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
struct ApiResponse<T> {
    success: bool,
    data: Option<T>,
    message: String,
}

impl<T> ApiResponse<T> {
    fn success(data: T) -> Self {
        Self {
            success: true,
            data: Some(data),
            message: "Success".to_string(),
        }
    }
    
    fn error(message: String) -> Self {
        Self {
            success: false,
            data: None,
            message,
        }
    }
}

async fn health_check() -> Json<ApiResponse<HashMap<String, String>>> {
    let mut status = HashMap::new();
    status.insert("status".to_string(), "healthy".to_string());
    status.insert("service".to_string(), "milvuso-recommendation".to_string());
    status.insert("version".to_string(), "0.1.0".to_string());
    
    Json(ApiResponse::success(status))
}

async fn get_recommendations(
    State(state): State<AppState>,
    Path(user_id): Path<Uuid>,
    Query(params): Query<RecommendationQuery>,
) -> Result<Json<ApiResponse<milvuso::RecommendationResponse>>, StatusCode> {
    let filter_categories = params.filter_categories
        .map(|s| s.split(',').map(|s| s.trim().to_string()).collect());
    
    let exclude_items = params.exclude_items
        .map(|s| s.split(',')
            .filter_map(|s| Uuid::parse_str(s.trim()).ok())
            .collect());

    let request = milvuso::RecommendationRequest {
        user_id,
        num_recommendations: params.num_recommendations.unwrap_or(10),
        filter_categories,
        exclude_items,
    };

    match state.recommendation_service.get_recommendations(&request).await {
        Ok(response) => Ok(Json(ApiResponse::success(response))),
        Err(e) => {
            tracing::error!("Failed to get recommendations: {}", e);
            Err(StatusCode::INTERNAL_SERVER_ERROR)
        }
    }
}

async fn record_user_action(
    State(state): State<AppState>,
    Json(action): Json<milvuso::UserAction>,
) -> Result<Json<ApiResponse<String>>, StatusCode> {
    // Send to Kafka
    if let Err(e) = state.kafka_producer.send_user_action(&action).await {
        tracing::error!("Failed to send user action to Kafka: {}", e);
        return Err(StatusCode::INTERNAL_SERVER_ERROR);
    }

    // Process immediately for real-time updates
    if let Err(e) = state.recommendation_service.process_user_action(&action).await {
        tracing::error!("Failed to process user action: {}", e);
        return Err(StatusCode::INTERNAL_SERVER_ERROR);
    }

    Ok(Json(ApiResponse::success("Action recorded successfully".to_string())))
}

async fn add_item(
    State(state): State<AppState>,
    Json(item_feature): Json<milvuso::ItemFeature>,
) -> Result<Json<ApiResponse<String>>, StatusCode> {
    match state.recommendation_service.add_item_feature(item_feature).await {
        Ok(_) => Ok(Json(ApiResponse::success("Item added successfully".to_string()))),
        Err(e) => {
            tracing::error!("Failed to add item: {}", e);
            Err(StatusCode::INTERNAL_SERVER_ERROR)
        }
    }
}

async fn get_user_profile(
    State(state): State<AppState>,
    Path(user_id): Path<Uuid>,
) -> Result<Json<ApiResponse<milvuso::UserProfile>>, StatusCode> {
    match state.vector_db.get_user_profile(user_id).await {
        Ok(Some(profile)) => Ok(Json(ApiResponse::success(profile))),
        Ok(None) => Err(StatusCode::NOT_FOUND),
        Err(e) => {
            tracing::error!("Failed to get user profile: {}", e);
            Err(StatusCode::INTERNAL_SERVER_ERROR)
        }
    }
}

async fn get_item_feature(
    State(state): State<AppState>,
    Path(item_id): Path<Uuid>,
) -> Result<Json<ApiResponse<milvuso::ItemFeature>>, StatusCode> {
    match state.vector_db.get_item_feature(item_id).await {
        Ok(Some(feature)) => Ok(Json(ApiResponse::success(feature))),
        Ok(None) => Err(StatusCode::NOT_FOUND),
        Err(e) => {
            tracing::error!("Failed to get item feature: {}", e);
            Err(StatusCode::INTERNAL_SERVER_ERROR)
        }
    }
}

fn create_router(state: AppState) -> Router {
    Router::new()
        .route("/health", get(health_check))
        .route("/recommendations/:user_id", get(get_recommendations))
        .route("/actions", post(record_user_action))
        .route("/items", post(add_item))
        .route("/users/:user_id", get(get_user_profile))
        .route("/items/:item_id", get(get_item_feature))
        .layer(
            ServiceBuilder::new()
                .layer(TraceLayer::new_for_http())
                .layer(CorsLayer::permissive())
        )
        .with_state(state)
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    init_tracing().await;

    let config = Config::default();
    info!("Starting MilRustRec Recommendation Server with config: {:?}", config.server);

    let state = AppState::new(config.clone()).await?;
    let app = create_router(state);

    let listener = tokio::net::TcpListener::bind(config.server.socket_addr()).await?;
    info!("Server listening on {}", config.server.socket_addr());

    axum::serve(listener, app).await?;

    Ok(())
}
