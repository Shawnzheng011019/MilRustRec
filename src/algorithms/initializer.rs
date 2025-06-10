use rand::Rng;
use std::f32::consts::PI;

pub fn xavier_uniform(size: usize) -> Vec<f32> {
    let limit = (6.0 / size as f32).sqrt();
    let mut rng = rand::thread_rng();
    (0..size)
        .map(|_| rng.gen_range(-limit..limit))
        .collect()
}

pub fn xavier_normal(size: usize) -> Vec<f32> {
    let std_dev = (2.0 / size as f32).sqrt();
    let mut rng = rand::thread_rng();
    (0..size)
        .map(|_| {
            let u1: f32 = rng.gen();
            let u2: f32 = rng.gen();
            let z0 = (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos();
            z0 * std_dev
        })
        .collect()
}

pub fn he_uniform(size: usize) -> Vec<f32> {
    let limit = (6.0 / size as f32).sqrt();
    let mut rng = rand::thread_rng();
    (0..size)
        .map(|_| rng.gen_range(-limit..limit))
        .collect()
}

pub fn he_normal(size: usize) -> Vec<f32> {
    let std_dev = (2.0 / size as f32).sqrt();
    let mut rng = rand::thread_rng();
    (0..size)
        .map(|_| {
            let u1: f32 = rng.gen();
            let u2: f32 = rng.gen();
            let z0 = (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos();
            z0 * std_dev
        })
        .collect()
}

pub fn lecun_uniform(size: usize) -> Vec<f32> {
    let limit = (3.0 / size as f32).sqrt();
    let mut rng = rand::thread_rng();
    (0..size)
        .map(|_| rng.gen_range(-limit..limit))
        .collect()
}

pub fn lecun_normal(size: usize) -> Vec<f32> {
    let std_dev = (1.0 / size as f32).sqrt();
    let mut rng = rand::thread_rng();
    (0..size)
        .map(|_| {
            let u1: f32 = rng.gen();
            let u2: f32 = rng.gen();
            let z0 = (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos();
            z0 * std_dev
        })
        .collect()
}

pub fn uniform(size: usize, low: f32, high: f32) -> Vec<f32> {
    let mut rng = rand::thread_rng();
    (0..size)
        .map(|_| rng.gen_range(low..high))
        .collect()
}

pub fn normal(size: usize, mean: f32, std_dev: f32) -> Vec<f32> {
    let mut rng = rand::thread_rng();
    (0..size)
        .map(|_| {
            let u1: f32 = rng.gen();
            let u2: f32 = rng.gen();
            let z0 = (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos();
            z0 * std_dev + mean
        })
        .collect()
}

pub fn zeros(size: usize) -> Vec<f32> {
    vec![0.0; size]
}

pub fn ones(size: usize) -> Vec<f32> {
    vec![1.0; size]
}

pub fn constant(size: usize, value: f32) -> Vec<f32> {
    vec![value; size]
}

pub fn orthogonal(rows: usize, cols: usize) -> Vec<Vec<f32>> {
    let mut rng = rand::thread_rng();
    let mut matrix: Vec<Vec<f32>> = (0..rows)
        .map(|_| (0..cols).map(|_| rng.gen_range(-1.0..1.0)).collect())
        .collect();
    
    // Gram-Schmidt orthogonalization
    for i in 0..rows.min(cols) {
        // Normalize current vector
        let norm: f32 = matrix[i].iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 1e-8 {
            for j in 0..cols {
                matrix[i][j] /= norm;
            }
        }
        
        // Orthogonalize remaining vectors
        for k in (i + 1)..rows {
            let dot_product: f32 = (0..cols).map(|j| matrix[i][j] * matrix[k][j]).sum();
            for j in 0..cols {
                matrix[k][j] -= dot_product * matrix[i][j];
            }
        }
    }
    
    matrix
}

pub fn sparse_random(size: usize, sparsity: f32) -> Vec<f32> {
    let mut rng = rand::thread_rng();
    (0..size)
        .map(|_| {
            if rng.gen::<f32>() < sparsity {
                0.0
            } else {
                rng.gen_range(-1.0..1.0)
            }
        })
        .collect()
}

#[derive(Debug, Clone)]
pub enum InitializationMethod {
    XavierUniform,
    XavierNormal,
    HeUniform,
    HeNormal,
    LecunUniform,
    LecunNormal,
    Uniform { low: f32, high: f32 },
    Normal { mean: f32, std_dev: f32 },
    Zeros,
    Ones,
    Constant { value: f32 },
    SparseRandom { sparsity: f32 },
}

impl InitializationMethod {
    pub fn initialize(&self, size: usize) -> Vec<f32> {
        match self {
            InitializationMethod::XavierUniform => xavier_uniform(size),
            InitializationMethod::XavierNormal => xavier_normal(size),
            InitializationMethod::HeUniform => he_uniform(size),
            InitializationMethod::HeNormal => he_normal(size),
            InitializationMethod::LecunUniform => lecun_uniform(size),
            InitializationMethod::LecunNormal => lecun_normal(size),
            InitializationMethod::Uniform { low, high } => uniform(size, *low, *high),
            InitializationMethod::Normal { mean, std_dev } => normal(size, *mean, *std_dev),
            InitializationMethod::Zeros => zeros(size),
            InitializationMethod::Ones => ones(size),
            InitializationMethod::Constant { value } => constant(size, *value),
            InitializationMethod::SparseRandom { sparsity } => sparse_random(size, *sparsity),
        }
    }
    
    pub fn initialize_matrix(&self, rows: usize, cols: usize) -> Vec<Vec<f32>> {
        match self {
            _ => (0..rows).map(|_| self.initialize(cols)).collect(),
        }
    }
}

pub struct EmbeddingInitializer {
    method: InitializationMethod,
    dimension: usize,
}

impl EmbeddingInitializer {
    pub fn new(method: InitializationMethod, dimension: usize) -> Self {
        Self { method, dimension }
    }
    
    pub fn initialize_user_embedding(&self, user_id: uuid::Uuid) -> Vec<f32> {
        // Use user_id as seed for reproducible initialization
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        std::hash::Hash::hash(&user_id, &mut hasher);
        let seed = std::hash::Hasher::finish(&hasher);
        
        // Set seed for reproducible results
        use rand::SeedableRng;
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        
        match &self.method {
            InitializationMethod::XavierUniform => {
                let limit = (6.0 / self.dimension as f32).sqrt();
                (0..self.dimension)
                    .map(|_| rng.gen_range(-limit..limit))
                    .collect()
            }
            _ => self.method.initialize(self.dimension),
        }
    }
    
    pub fn initialize_item_embedding(&self, item_id: uuid::Uuid) -> Vec<f32> {
        // Use item_id as seed for reproducible initialization
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        std::hash::Hash::hash(&item_id, &mut hasher);
        let seed = std::hash::Hasher::finish(&hasher);
        
        // Set seed for reproducible results
        use rand::SeedableRng;
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        
        match &self.method {
            InitializationMethod::XavierUniform => {
                let limit = (6.0 / self.dimension as f32).sqrt();
                (0..self.dimension)
                    .map(|_| rng.gen_range(-limit..limit))
                    .collect()
            }
            _ => self.method.initialize(self.dimension),
        }
    }
}
