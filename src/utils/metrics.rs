use std::collections::HashMap;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecommendationMetrics {
    pub precision_at_k: f64,
    pub recall_at_k: f64,
    pub f1_score: f64,
    pub ndcg_at_k: f64,
    pub map_score: f64,
    pub coverage: f64,
    pub diversity: f64,
    pub novelty: f64,
}

#[derive(Debug, Clone)]
pub struct MetricsCalculator {
    k: usize,
}

impl MetricsCalculator {
    pub fn new(k: usize) -> Self {
        Self { k }
    }

    pub fn calculate_precision_at_k(&self, recommended: &[Uuid], relevant: &[Uuid]) -> f64 {
        if recommended.is_empty() {
            return 0.0;
        }

        let relevant_set: std::collections::HashSet<_> = relevant.iter().collect();
        let relevant_recommended = recommended
            .iter()
            .take(self.k)
            .filter(|item| relevant_set.contains(item))
            .count();

        relevant_recommended as f64 / self.k.min(recommended.len()) as f64
    }

    pub fn calculate_recall_at_k(&self, recommended: &[Uuid], relevant: &[Uuid]) -> f64 {
        if relevant.is_empty() {
            return 0.0;
        }

        let relevant_set: std::collections::HashSet<_> = relevant.iter().collect();
        let relevant_recommended = recommended
            .iter()
            .take(self.k)
            .filter(|item| relevant_set.contains(item))
            .count();

        relevant_recommended as f64 / relevant.len() as f64
    }

    pub fn calculate_f1_score(&self, precision: f64, recall: f64) -> f64 {
        if precision + recall == 0.0 {
            0.0
        } else {
            2.0 * precision * recall / (precision + recall)
        }
    }

    pub fn calculate_ndcg_at_k(&self, recommended: &[Uuid], relevant_scores: &HashMap<Uuid, f64>) -> f64 {
        let dcg = self.calculate_dcg(recommended, relevant_scores);
        let idcg = self.calculate_ideal_dcg(relevant_scores);
        
        if idcg == 0.0 {
            0.0
        } else {
            dcg / idcg
        }
    }

    fn calculate_dcg(&self, recommended: &[Uuid], relevant_scores: &HashMap<Uuid, f64>) -> f64 {
        recommended
            .iter()
            .take(self.k)
            .enumerate()
            .map(|(i, item_id)| {
                let relevance = relevant_scores.get(item_id).unwrap_or(&0.0);
                let position = i + 1;
                relevance / (position as f64).log2()
            })
            .sum()
    }

    fn calculate_ideal_dcg(&self, relevant_scores: &HashMap<Uuid, f64>) -> f64 {
        let mut scores: Vec<f64> = relevant_scores.values().cloned().collect();
        scores.sort_by(|a, b| b.partial_cmp(a).unwrap());
        
        scores
            .iter()
            .take(self.k)
            .enumerate()
            .map(|(i, &score)| {
                let position = i + 1;
                score / (position as f64).log2()
            })
            .sum()
    }

    pub fn calculate_map(&self, all_recommended: &[Vec<Uuid>], all_relevant: &[Vec<Uuid>]) -> f64 {
        if all_recommended.len() != all_relevant.len() || all_recommended.is_empty() {
            return 0.0;
        }

        let total_ap: f64 = all_recommended
            .iter()
            .zip(all_relevant.iter())
            .map(|(recommended, relevant)| self.calculate_average_precision(recommended, relevant))
            .sum();

        total_ap / all_recommended.len() as f64
    }

    fn calculate_average_precision(&self, recommended: &[Uuid], relevant: &[Uuid]) -> f64 {
        if relevant.is_empty() {
            return 0.0;
        }

        let relevant_set: std::collections::HashSet<_> = relevant.iter().collect();
        let mut relevant_found = 0;
        let mut precision_sum = 0.0;

        for (i, item) in recommended.iter().take(self.k).enumerate() {
            if relevant_set.contains(item) {
                relevant_found += 1;
                precision_sum += relevant_found as f64 / (i + 1) as f64;
            }
        }

        if relevant_found == 0 {
            0.0
        } else {
            precision_sum / relevant.len() as f64
        }
    }

    pub fn calculate_coverage(&self, recommended_items: &[Uuid], all_items: &[Uuid]) -> f64 {
        if all_items.is_empty() {
            return 0.0;
        }

        let recommended_set: std::collections::HashSet<_> = recommended_items.iter().collect();
        let covered_items = all_items
            .iter()
            .filter(|item| recommended_set.contains(item))
            .count();

        covered_items as f64 / all_items.len() as f64
    }

    pub fn calculate_diversity(&self, recommended_items: &[Uuid], item_features: &HashMap<Uuid, Vec<f32>>) -> f64 {
        if recommended_items.len() < 2 {
            return 0.0;
        }

        let mut total_distance = 0.0;
        let mut pair_count = 0;

        for i in 0..recommended_items.len() {
            for j in (i + 1)..recommended_items.len() {
                if let (Some(features_i), Some(features_j)) = (
                    item_features.get(&recommended_items[i]),
                    item_features.get(&recommended_items[j])
                ) {
                    let distance = crate::utils::euclidean_distance(features_i, features_j);
                    total_distance += distance as f64;
                    pair_count += 1;
                }
            }
        }

        if pair_count == 0 {
            0.0
        } else {
            total_distance / pair_count as f64
        }
    }

    pub fn calculate_novelty(&self, recommended_items: &[Uuid], item_popularity: &HashMap<Uuid, f64>) -> f64 {
        if recommended_items.is_empty() {
            return 0.0;
        }

        let total_novelty: f64 = recommended_items
            .iter()
            .map(|item_id| {
                let popularity = item_popularity.get(item_id).unwrap_or(&0.0);
                if *popularity > 0.0 {
                    -popularity.log2()
                } else {
                    0.0
                }
            })
            .sum();

        total_novelty / recommended_items.len() as f64
    }

    pub fn calculate_all_metrics(
        &self,
        recommended: &[Uuid],
        relevant: &[Uuid],
        relevant_scores: &HashMap<Uuid, f64>,
        all_items: &[Uuid],
        item_features: &HashMap<Uuid, Vec<f32>>,
        item_popularity: &HashMap<Uuid, f64>,
    ) -> RecommendationMetrics {
        let precision = self.calculate_precision_at_k(recommended, relevant);
        let recall = self.calculate_recall_at_k(recommended, relevant);
        let f1_score = self.calculate_f1_score(precision, recall);
        let ndcg = self.calculate_ndcg_at_k(recommended, relevant_scores);
        let coverage = self.calculate_coverage(recommended, all_items);
        let diversity = self.calculate_diversity(recommended, item_features);
        let novelty = self.calculate_novelty(recommended, item_popularity);

        RecommendationMetrics {
            precision_at_k: precision,
            recall_at_k: recall,
            f1_score,
            ndcg_at_k: ndcg,
            map_score: 0.0, // Would need multiple queries to calculate MAP
            coverage,
            diversity,
            novelty,
        }
    }
}

#[derive(Debug, Clone)]
pub struct OnlineMetrics {
    pub click_through_rate: f64,
    pub conversion_rate: f64,
    pub engagement_rate: f64,
    pub session_length: f64,
    pub bounce_rate: f64,
}

pub struct OnlineMetricsCalculator {
    total_impressions: u64,
    total_clicks: u64,
    total_conversions: u64,
    total_engagements: u64,
    total_sessions: u64,
    total_bounces: u64,
    total_session_time: f64,
}

impl OnlineMetricsCalculator {
    pub fn new() -> Self {
        Self {
            total_impressions: 0,
            total_clicks: 0,
            total_conversions: 0,
            total_engagements: 0,
            total_sessions: 0,
            total_bounces: 0,
            total_session_time: 0.0,
        }
    }

    pub fn record_impression(&mut self) {
        self.total_impressions += 1;
    }

    pub fn record_click(&mut self) {
        self.total_clicks += 1;
    }

    pub fn record_conversion(&mut self) {
        self.total_conversions += 1;
    }

    pub fn record_engagement(&mut self) {
        self.total_engagements += 1;
    }

    pub fn record_session(&mut self, duration_seconds: f64, bounced: bool) {
        self.total_sessions += 1;
        self.total_session_time += duration_seconds;
        if bounced {
            self.total_bounces += 1;
        }
    }

    pub fn calculate_metrics(&self) -> OnlineMetrics {
        OnlineMetrics {
            click_through_rate: if self.total_impressions > 0 {
                self.total_clicks as f64 / self.total_impressions as f64
            } else {
                0.0
            },
            conversion_rate: if self.total_clicks > 0 {
                self.total_conversions as f64 / self.total_clicks as f64
            } else {
                0.0
            },
            engagement_rate: if self.total_impressions > 0 {
                self.total_engagements as f64 / self.total_impressions as f64
            } else {
                0.0
            },
            session_length: if self.total_sessions > 0 {
                self.total_session_time / self.total_sessions as f64
            } else {
                0.0
            },
            bounce_rate: if self.total_sessions > 0 {
                self.total_bounces as f64 / self.total_sessions as f64
            } else {
                0.0
            },
        }
    }

    pub fn reset(&mut self) {
        self.total_impressions = 0;
        self.total_clicks = 0;
        self.total_conversions = 0;
        self.total_engagements = 0;
        self.total_sessions = 0;
        self.total_bounces = 0;
        self.total_session_time = 0.0;
    }
}
