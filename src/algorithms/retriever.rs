use anyhow::Result;
use nalgebra::DVector;
use std::collections::HashMap;
use std::cmp::Ordering;

#[async_trait::async_trait]
pub trait VectorRetriever: Send + Sync {
    async fn search_similar(&self, query_vector: &[f32], top_k: usize) -> Result<Vec<(uuid::Uuid, f32)>>;
    async fn add_vector(&mut self, id: uuid::Uuid, vector: Vec<f32>) -> Result<()>;
    async fn remove_vector(&mut self, id: uuid::Uuid) -> Result<()>;
    async fn update_vector(&mut self, id: uuid::Uuid, vector: Vec<f32>) -> Result<()>;
}

#[derive(Debug, Clone)]
pub struct InMemoryRetriever {
    vectors: HashMap<uuid::Uuid, DVector<f32>>,
    dimension: usize,
}

impl InMemoryRetriever {
    pub fn new(dimension: usize) -> Self {
        Self {
            vectors: HashMap::new(),
            dimension,
        }
    }
    
    fn cosine_similarity(&self, a: &DVector<f32>, b: &DVector<f32>) -> f32 {
        let dot_product = a.dot(b);
        let norm_a = a.norm();
        let norm_b = b.norm();
        
        if norm_a == 0.0 || norm_b == 0.0 {
            0.0
        } else {
            dot_product / (norm_a * norm_b)
        }
    }
    
    fn euclidean_distance(&self, a: &DVector<f32>, b: &DVector<f32>) -> f32 {
        (a - b).norm()
    }
    
    fn manhattan_distance(&self, a: &DVector<f32>, b: &DVector<f32>) -> f32 {
        (a - b).iter().map(|x| x.abs()).sum()
    }
}

#[async_trait::async_trait]
impl VectorRetriever for InMemoryRetriever {
    async fn search_similar(&self, query_vector: &[f32], top_k: usize) -> Result<Vec<(uuid::Uuid, f32)>> {
        if query_vector.len() != self.dimension {
            return Err(anyhow::anyhow!("Query vector dimension mismatch"));
        }
        
        let query = DVector::from_vec(query_vector.to_vec());
        let mut similarities = Vec::new();
        
        for (id, vector) in &self.vectors {
            let similarity = self.cosine_similarity(&query, vector);
            similarities.push((*id, similarity));
        }
        
        // Sort by similarity in descending order
        similarities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
        
        // Return top k results
        similarities.truncate(top_k);
        Ok(similarities)
    }
    
    async fn add_vector(&mut self, id: uuid::Uuid, vector: Vec<f32>) -> Result<()> {
        if vector.len() != self.dimension {
            return Err(anyhow::anyhow!("Vector dimension mismatch"));
        }
        
        self.vectors.insert(id, DVector::from_vec(vector));
        Ok(())
    }
    
    async fn remove_vector(&mut self, id: uuid::Uuid) -> Result<()> {
        self.vectors.remove(&id);
        Ok(())
    }
    
    async fn update_vector(&mut self, id: uuid::Uuid, vector: Vec<f32>) -> Result<()> {
        if vector.len() != self.dimension {
            return Err(anyhow::anyhow!("Vector dimension mismatch"));
        }
        
        self.vectors.insert(id, DVector::from_vec(vector));
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct HNSWRetriever {
    // Hierarchical Navigable Small World implementation
    layers: Vec<HashMap<uuid::Uuid, Vec<uuid::Uuid>>>,
    vectors: HashMap<uuid::Uuid, DVector<f32>>,
    dimension: usize,
    max_connections: usize,
    ef_construction: usize,
    ml: f64,
}

impl HNSWRetriever {
    pub fn new(dimension: usize, max_connections: usize, ef_construction: usize) -> Self {
        Self {
            layers: vec![HashMap::new()],
            vectors: HashMap::new(),
            dimension,
            max_connections,
            ef_construction,
            ml: 1.0 / (2.0_f64).ln(),
        }
    }
    
    fn get_random_level(&self) -> usize {
        let mut level = 0;
        while rand::random::<f64>() < 0.5 && level < 16 {
            level += 1;
        }
        level
    }
    
    fn distance(&self, a: &DVector<f32>, b: &DVector<f32>) -> f32 {
        (a - b).norm_squared()
    }
    
    async fn search_layer(&self, query: &DVector<f32>, entry_points: Vec<uuid::Uuid>, 
                         num_closest: usize, layer: usize) -> Result<Vec<(uuid::Uuid, f32)>> {
        let mut visited = std::collections::HashSet::new();
        let mut candidates = std::collections::BinaryHeap::new();
        let mut w = std::collections::BinaryHeap::new();
        
        for ep in entry_points {
            if let Some(vector) = self.vectors.get(&ep) {
                let dist = self.distance(query, vector);
                candidates.push(std::cmp::Reverse((dist as i32, ep)));
                w.push((dist as i32, ep));
                visited.insert(ep);
            }
        }
        
        while let Some(std::cmp::Reverse((current_dist, current))) = candidates.pop() {
            if w.len() >= num_closest {
                if let Some((furthest_dist, _)) = w.peek() {
                    if current_dist > *furthest_dist {
                        break;
                    }
                }
            }
            
            if let Some(connections) = self.layers.get(layer).and_then(|l| l.get(&current)) {
                for &neighbor in connections {
                    if !visited.contains(&neighbor) {
                        visited.insert(neighbor);
                        
                        if let Some(neighbor_vector) = self.vectors.get(&neighbor) {
                            let dist = self.distance(query, neighbor_vector);
                            
                            if w.len() < num_closest {
                                candidates.push(std::cmp::Reverse((dist as i32, neighbor)));
                                w.push((dist as i32, neighbor));
                            } else if let Some((furthest_dist, _furthest_id)) = w.peek() {
                                if (dist as i32) < *furthest_dist {
                                    candidates.push(std::cmp::Reverse((dist as i32, neighbor)));
                                    w.pop();
                                    w.push((dist as i32, neighbor));
                                }
                            }
                        }
                    }
                }
            }
        }
        
        let mut result = Vec::new();
        while let Some((dist, id)) = w.pop() {
            result.push((id, dist as f32));
        }
        result.reverse();
        
        Ok(result)
    }
}

#[async_trait::async_trait]
impl VectorRetriever for HNSWRetriever {
    async fn search_similar(&self, query_vector: &[f32], top_k: usize) -> Result<Vec<(uuid::Uuid, f32)>> {
        if query_vector.len() != self.dimension {
            return Err(anyhow::anyhow!("Query vector dimension mismatch"));
        }
        
        let query = DVector::from_vec(query_vector.to_vec());
        
        // Start from the top layer and work down
        let mut entry_points = vec![];
        
        // Find entry point from top layer
        if let Some(top_layer) = self.layers.last() {
            if let Some(first_node) = top_layer.keys().next() {
                entry_points.push(*first_node);
            }
        }
        
        if entry_points.is_empty() {
            return Ok(Vec::new());
        }
        
        // Search through layers
        for layer in (1..self.layers.len()).rev() {
            let results = self.search_layer(&query, entry_points.clone(), 1, layer).await?;
            entry_points = results.into_iter().map(|(id, _)| id).collect();
        }
        
        // Search the bottom layer
        let results = self.search_layer(&query, entry_points, top_k, 0).await?;
        Ok(results)
    }
    
    async fn add_vector(&mut self, id: uuid::Uuid, vector: Vec<f32>) -> Result<()> {
        if vector.len() != self.dimension {
            return Err(anyhow::anyhow!("Vector dimension mismatch"));
        }
        
        let level = self.get_random_level();
        
        // Ensure we have enough layers
        while self.layers.len() <= level {
            self.layers.push(HashMap::new());
        }
        
        self.vectors.insert(id, DVector::from_vec(vector));
        
        // Add to all layers up to the determined level
        for l in 0..=level {
            self.layers[l].insert(id, Vec::new());
        }
        
        Ok(())
    }
    
    async fn remove_vector(&mut self, id: uuid::Uuid) -> Result<()> {
        self.vectors.remove(&id);
        
        for layer in &mut self.layers {
            layer.remove(&id);
            // Also remove from other nodes' connection lists
            for connections in layer.values_mut() {
                connections.retain(|&x| x != id);
            }
        }
        
        Ok(())
    }
    
    async fn update_vector(&mut self, id: uuid::Uuid, vector: Vec<f32>) -> Result<()> {
        if vector.len() != self.dimension {
            return Err(anyhow::anyhow!("Vector dimension mismatch"));
        }
        
        self.vectors.insert(id, DVector::from_vec(vector));
        Ok(())
    }
}
