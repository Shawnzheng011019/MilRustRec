use crate::models::*;
use anyhow::{Result, anyhow};
use uuid::Uuid;

pub fn validate_user_action(action: &UserAction) -> Result<()> {
    if action.user_id.is_nil() {
        return Err(anyhow!("User ID cannot be nil"));
    }
    
    if action.item_id.is_nil() {
        return Err(anyhow!("Item ID cannot be nil"));
    }
    
    // Validate timestamp is not too far in the future
    let now = chrono::Utc::now();
    let max_future = now + chrono::Duration::hours(1);
    if action.timestamp > max_future {
        return Err(anyhow!("Timestamp cannot be more than 1 hour in the future"));
    }
    
    // Validate timestamp is not too old (e.g., more than 1 year)
    let min_past = now - chrono::Duration::days(365);
    if action.timestamp < min_past {
        return Err(anyhow!("Timestamp cannot be more than 1 year in the past"));
    }
    
    Ok(())
}

pub fn validate_user_profile(profile: &UserProfile) -> Result<()> {
    if profile.user_id.is_nil() {
        return Err(anyhow!("User ID cannot be nil"));
    }
    
    if profile.embedding.is_empty() {
        return Err(anyhow!("User embedding cannot be empty"));
    }
    
    // Check for valid embedding values
    for &value in &profile.embedding {
        if !value.is_finite() {
            return Err(anyhow!("User embedding contains invalid values (NaN or Infinity)"));
        }
    }
    
    // Check embedding dimension
    if profile.embedding.len() > 2048 {
        return Err(anyhow!("User embedding dimension too large (max 2048)"));
    }
    
    Ok(())
}

pub fn validate_item_feature(feature: &ItemFeature) -> Result<()> {
    if feature.item_id.is_nil() {
        return Err(anyhow!("Item ID cannot be nil"));
    }
    
    if feature.embedding.is_empty() {
        return Err(anyhow!("Item embedding cannot be empty"));
    }
    
    // Check for valid embedding values
    for &value in &feature.embedding {
        if !value.is_finite() {
            return Err(anyhow!("Item embedding contains invalid values (NaN or Infinity)"));
        }
    }
    
    // Check embedding dimension
    if feature.embedding.len() > 2048 {
        return Err(anyhow!("Item embedding dimension too large (max 2048)"));
    }
    
    if feature.category.is_empty() {
        return Err(anyhow!("Item category cannot be empty"));
    }
    
    if feature.category.len() > 100 {
        return Err(anyhow!("Item category too long (max 100 characters)"));
    }
    
    // Validate popularity score
    if feature.popularity_score < 0.0 || feature.popularity_score > 1.0 {
        return Err(anyhow!("Item popularity score must be between 0.0 and 1.0"));
    }
    
    Ok(())
}

pub fn validate_recommendation_request(request: &RecommendationRequest) -> Result<()> {
    if request.user_id.is_nil() {
        return Err(anyhow!("User ID cannot be nil"));
    }
    
    if request.num_recommendations == 0 {
        return Err(anyhow!("Number of recommendations must be greater than 0"));
    }
    
    if request.num_recommendations > 1000 {
        return Err(anyhow!("Number of recommendations too large (max 1000)"));
    }
    
    // Validate filter categories
    if let Some(ref categories) = request.filter_categories {
        if categories.is_empty() {
            return Err(anyhow!("Filter categories cannot be empty if specified"));
        }
        
        for category in categories {
            if category.is_empty() {
                return Err(anyhow!("Category name cannot be empty"));
            }
            if category.len() > 100 {
                return Err(anyhow!("Category name too long (max 100 characters)"));
            }
        }
    }
    
    // Validate exclude items
    if let Some(ref exclude_items) = request.exclude_items {
        if exclude_items.len() > 10000 {
            return Err(anyhow!("Too many items to exclude (max 10000)"));
        }
        
        for &item_id in exclude_items {
            if item_id.is_nil() {
                return Err(anyhow!("Excluded item ID cannot be nil"));
            }
        }
    }
    
    Ok(())
}

pub fn validate_training_example(example: &TrainingExample) -> Result<()> {
    if example.user_id.is_nil() {
        return Err(anyhow!("User ID cannot be nil"));
    }
    
    if example.item_id.is_nil() {
        return Err(anyhow!("Item ID cannot be nil"));
    }
    
    // Validate label
    if !example.label.is_finite() {
        return Err(anyhow!("Training label contains invalid values (NaN or Infinity)"));
    }
    
    if example.label < 0.0 || example.label > 1.0 {
        return Err(anyhow!("Training label must be between 0.0 and 1.0"));
    }
    
    // Validate user features
    if example.user_features.is_empty() {
        return Err(anyhow!("User features cannot be empty"));
    }
    
    for &value in &example.user_features {
        if !value.is_finite() {
            return Err(anyhow!("User features contain invalid values (NaN or Infinity)"));
        }
    }
    
    // Validate item features
    if example.item_features.is_empty() {
        return Err(anyhow!("Item features cannot be empty"));
    }
    
    for &value in &example.item_features {
        if !value.is_finite() {
            return Err(anyhow!("Item features contain invalid values (NaN or Infinity)"));
        }
    }
    
    // Validate context features
    for &value in &example.context_features {
        if !value.is_finite() {
            return Err(anyhow!("Context features contain invalid values (NaN or Infinity)"));
        }
    }
    
    // Check feature dimensions
    if example.user_features.len() > 2048 {
        return Err(anyhow!("User features dimension too large (max 2048)"));
    }
    
    if example.item_features.len() > 2048 {
        return Err(anyhow!("Item features dimension too large (max 2048)"));
    }
    
    if example.context_features.len() > 512 {
        return Err(anyhow!("Context features dimension too large (max 512)"));
    }
    
    Ok(())
}

pub fn validate_feature_vector(feature: &FeatureVector) -> Result<()> {
    if feature.id.is_nil() {
        return Err(anyhow!("Feature vector ID cannot be nil"));
    }
    
    if feature.vector.is_empty() {
        return Err(anyhow!("Feature vector cannot be empty"));
    }
    
    // Check for valid vector values
    for &value in &feature.vector {
        if !value.is_finite() {
            return Err(anyhow!("Feature vector contains invalid values (NaN or Infinity)"));
        }
    }
    
    // Check vector dimension
    if feature.vector.len() > 2048 {
        return Err(anyhow!("Feature vector dimension too large (max 2048)"));
    }
    
    Ok(())
}

pub fn validate_model_parameters(params: &ModelParameters) -> Result<()> {
    if params.version.is_empty() {
        return Err(anyhow!("Model version cannot be empty"));
    }
    
    if params.version.len() > 100 {
        return Err(anyhow!("Model version too long (max 100 characters)"));
    }
    
    // Validate user embedding weights
    for embedding in &params.user_embedding_weights {
        if embedding.is_empty() {
            return Err(anyhow!("User embedding weights cannot be empty"));
        }
        
        for &weight in embedding {
            if !weight.is_finite() {
                return Err(anyhow!("User embedding weights contain invalid values"));
            }
        }
    }
    
    // Validate item embedding weights
    for embedding in &params.item_embedding_weights {
        if embedding.is_empty() {
            return Err(anyhow!("Item embedding weights cannot be empty"));
        }
        
        for &weight in embedding {
            if !weight.is_finite() {
                return Err(anyhow!("Item embedding weights contain invalid values"));
            }
        }
    }
    
    // Validate bias weights
    for &bias in &params.bias_weights {
        if !bias.is_finite() {
            return Err(anyhow!("Bias weights contain invalid values"));
        }
    }
    
    Ok(())
}

pub fn sanitize_string(input: &str, max_length: usize) -> String {
    input
        .chars()
        .filter(|c| c.is_alphanumeric() || c.is_whitespace() || "-_.,!?".contains(*c))
        .take(max_length)
        .collect()
}

pub fn validate_uuid_string(uuid_str: &str) -> Result<Uuid> {
    Uuid::parse_str(uuid_str)
        .map_err(|_| anyhow!("Invalid UUID format: {}", uuid_str))
}

pub fn validate_embedding_dimension(embedding: &[f32], expected_dim: usize) -> Result<()> {
    if embedding.len() != expected_dim {
        return Err(anyhow!(
            "Embedding dimension mismatch: expected {}, got {}",
            expected_dim,
            embedding.len()
        ));
    }
    Ok(())
}

pub fn validate_batch_size(batch_size: usize, max_batch_size: usize) -> Result<()> {
    if batch_size == 0 {
        return Err(anyhow!("Batch size cannot be zero"));
    }
    
    if batch_size > max_batch_size {
        return Err(anyhow!(
            "Batch size too large: {} (max {})",
            batch_size,
            max_batch_size
        ));
    }
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;

    #[test]
    fn test_validate_user_action() {
        let valid_action = UserAction {
            user_id: Uuid::new_v4(),
            item_id: Uuid::new_v4(),
            action_type: ActionType::Click,
            timestamp: Utc::now(),
            context: None,
        };
        
        assert!(validate_user_action(&valid_action).is_ok());
        
        let invalid_action = UserAction {
            user_id: Uuid::nil(),
            item_id: Uuid::new_v4(),
            action_type: ActionType::Click,
            timestamp: Utc::now(),
            context: None,
        };
        
        assert!(validate_user_action(&invalid_action).is_err());
    }

    #[test]
    fn test_validate_user_profile() {
        let valid_profile = UserProfile {
            user_id: Uuid::new_v4(),
            embedding: vec![0.1, 0.2, 0.3],
            preferences: vec!["music".to_string()],
            last_updated: Utc::now(),
            interaction_count: 10,
        };
        
        assert!(validate_user_profile(&valid_profile).is_ok());
        
        let invalid_profile = UserProfile {
            user_id: Uuid::new_v4(),
            embedding: vec![f32::NAN, 0.2, 0.3],
            preferences: vec!["music".to_string()],
            last_updated: Utc::now(),
            interaction_count: 10,
        };
        
        assert!(validate_user_profile(&invalid_profile).is_err());
    }
}
