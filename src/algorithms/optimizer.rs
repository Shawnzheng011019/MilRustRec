use nalgebra::DVector;
use std::collections::HashMap;

pub trait Optimizer: Send + Sync {
    fn update(&mut self, params: &mut DVector<f32>, gradients: &DVector<f32>);
    fn reset(&mut self);
}

#[derive(Debug, Clone)]
pub struct SGD {
    learning_rate: f64,
}

impl SGD {
    pub fn new(learning_rate: f64) -> Self {
        Self { learning_rate }
    }
}

impl Optimizer for SGD {
    fn update(&mut self, params: &mut DVector<f32>, gradients: &DVector<f32>) {
        *params -= gradients * self.learning_rate as f32;
    }
    
    fn reset(&mut self) {
        // SGD doesn't maintain state
    }
}

#[derive(Debug, Clone)]
pub struct Adam {
    learning_rate: f64,
    beta1: f64,
    beta2: f64,
    epsilon: f64,
    t: usize,
    m: HashMap<String, DVector<f32>>,
    v: HashMap<String, DVector<f32>>,
}

impl Adam {
    pub fn new(learning_rate: f64, beta1: f64, beta2: f64, epsilon: f64) -> Self {
        Self {
            learning_rate,
            beta1,
            beta2,
            epsilon,
            t: 0,
            m: HashMap::new(),
            v: HashMap::new(),
        }
    }
    
    pub fn default() -> Self {
        Self::new(0.001, 0.9, 0.999, 1e-8)
    }
    
    pub fn update_with_key(&mut self, key: &str, params: &mut DVector<f32>, gradients: &DVector<f32>) {
        self.t += 1;
        
        let m = self.m.entry(key.to_string())
            .or_insert_with(|| DVector::zeros(params.len()));
        let v = self.v.entry(key.to_string())
            .or_insert_with(|| DVector::zeros(params.len()));
        
        // Update biased first moment estimate
        *m = m.scale(self.beta1 as f32) + gradients.scale(1.0 - self.beta1 as f32);
        
        // Update biased second raw moment estimate
        *v = v.scale(self.beta2 as f32) + gradients.component_mul(gradients).scale(1.0 - self.beta2 as f32);
        
        // Compute bias-corrected first moment estimate
        let m_hat = m.scale(1.0 / (1.0 - (self.beta1 as f32).powi(self.t as i32)));
        
        // Compute bias-corrected second raw moment estimate
        let v_hat = v.scale(1.0 / (1.0 - (self.beta2 as f32).powi(self.t as i32)));
        
        // Update parameters
        let denominator = v_hat.map(|x| (x + self.epsilon as f32).sqrt());
        let update = m_hat.component_div(&denominator).scale(self.learning_rate as f32);
        
        *params -= update;
    }
}

impl Optimizer for Adam {
    fn update(&mut self, params: &mut DVector<f32>, gradients: &DVector<f32>) {
        self.update_with_key("default", params, gradients);
    }
    
    fn reset(&mut self) {
        self.t = 0;
        self.m.clear();
        self.v.clear();
    }
}

#[derive(Debug, Clone)]
pub struct AdaGrad {
    learning_rate: f64,
    epsilon: f64,
    sum_squared_gradients: HashMap<String, DVector<f32>>,
}

impl AdaGrad {
    pub fn new(learning_rate: f64, epsilon: f64) -> Self {
        Self {
            learning_rate,
            epsilon,
            sum_squared_gradients: HashMap::new(),
        }
    }
    
    pub fn default() -> Self {
        Self::new(0.01, 1e-8)
    }
    
    pub fn update_with_key(&mut self, key: &str, params: &mut DVector<f32>, gradients: &DVector<f32>) {
        let sum_sq_grad = self.sum_squared_gradients.entry(key.to_string())
            .or_insert_with(|| DVector::zeros(params.len()));
        
        // Accumulate squared gradients
        *sum_sq_grad += gradients.component_mul(gradients);
        
        // Compute adaptive learning rate
        let adaptive_lr = sum_sq_grad.map(|x| (self.learning_rate as f32) / (x + (self.epsilon as f32)).sqrt());

        // Update parameters
        *params -= gradients.component_mul(&adaptive_lr);
    }
}

impl Optimizer for AdaGrad {
    fn update(&mut self, params: &mut DVector<f32>, gradients: &DVector<f32>) {
        self.update_with_key("default", params, gradients);
    }
    
    fn reset(&mut self) {
        self.sum_squared_gradients.clear();
    }
}

#[derive(Debug, Clone)]
pub struct RMSprop {
    learning_rate: f64,
    decay_rate: f64,
    epsilon: f64,
    cache: HashMap<String, DVector<f32>>,
}

impl RMSprop {
    pub fn new(learning_rate: f64, decay_rate: f64, epsilon: f64) -> Self {
        Self {
            learning_rate,
            decay_rate,
            epsilon,
            cache: HashMap::new(),
        }
    }
    
    pub fn default() -> Self {
        Self::new(0.001, 0.9, 1e-8)
    }
    
    pub fn update_with_key(&mut self, key: &str, params: &mut DVector<f32>, gradients: &DVector<f32>) {
        let cache = self.cache.entry(key.to_string())
            .or_insert_with(|| DVector::zeros(params.len()));
        
        // Update cache with exponential moving average of squared gradients
        *cache = cache.scale(self.decay_rate as f32) + 
                gradients.component_mul(gradients).scale(1.0 - self.decay_rate as f32);
        
        // Compute update
        let denominator = cache.map(|x| (x + self.epsilon as f32).sqrt());
        let update = gradients.component_div(&denominator).scale(self.learning_rate as f32);
        
        *params -= update;
    }
}

impl Optimizer for RMSprop {
    fn update(&mut self, params: &mut DVector<f32>, gradients: &DVector<f32>) {
        self.update_with_key("default", params, gradients);
    }
    
    fn reset(&mut self) {
        self.cache.clear();
    }
}
