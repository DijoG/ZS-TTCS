// cpp/forman.cpp
#include "forman.h"
#include <iostream>
#include <algorithm>
#include <queue>
#include <stack>
#include <cmath>

namespace forman {

// ============================================================================
// Constructor
// ============================================================================
CellComplex::CellComplex(int width, int height, const double* data)
    : w_(width), h_(height) {
    
    // Copy data
    values_.resize(width * height);
    std::copy(data, data + width * height, values_.begin());
}

// ============================================================================
// Building the complex
// ============================================================================
void CellComplex::build() {
    build_vertices();
    build_edges();
    build_faces();
    compute_gradient();
    compute_persistence();
}

void CellComplex::build_vertices() {
    // Vertices are just pixels - no explicit storage needed beyond values_
    // But we need adjacency lists
    size_t n_vertices = values_.size();
    vertex_edges_.resize(n_vertices);
    vertex_faces_.resize(n_vertices);
}

void CellComplex::build_edges() {
    edge_vertices_.clear();
    edge_faces_.clear();
    
    // Horizontal edges (left-right)
    for (int y = 0; y < h_; ++y) {
        for (int x = 0; x < w_ - 1; ++x) {
            size_t e_id = edge_idx(x, y, x+1, y);
            size_t v1 = idx(x, y);
            size_t v2 = idx(x+1, y);
            
            if (edge_vertices_.size() <= e_id) {
                edge_vertices_.resize(e_id + 1);
                edge_faces_.resize(e_id + 1);
            }
            
            edge_vertices_[e_id] = {v1, v2};
            
            // Update vertex adjacency
            vertex_edges_[v1].push_back(e_id);
            vertex_edges_[v2].push_back(e_id);
        }
    }
    
    // Vertical edges (up-down)
    for (int y = 0; y < h_ - 1; ++y) {
        for (int x = 0; x < w_; ++x) {
            size_t e_id = edge_idx(x, y, x, y+1);
            size_t v1 = idx(x, y);
            size_t v2 = idx(x, y+1);
            
            if (edge_vertices_.size() <= e_id) {
                edge_vertices_.resize(e_id + 1);
                edge_faces_.resize(e_id + 1);
            }
            
            edge_vertices_[e_id] = {v1, v2};
            
            // Update vertex adjacency
            vertex_edges_[v1].push_back(e_id);
            vertex_edges_[v2].push_back(e_id);
        }
    }
}

void CellComplex::build_faces() {
    face_vertices_.clear();
    face_edges_.clear();
    
    // Each face is a 2x2 pixel square
    for (int y = 0; y < h_ - 1; ++y) {
        for (int x = 0; x < w_ - 1; ++x) {
            size_t f_id = face_idx(x, y);
            
            // Four vertices in counter-clockwise order
            size_t v0 = idx(x, y);        // top-left
            size_t v1 = idx(x+1, y);      // top-right
            size_t v2 = idx(x+1, y+1);    // bottom-right
            size_t v3 = idx(x, y+1);      // bottom-left
            
            // Four edges
            size_t e_top = edge_idx(x, y, x+1, y);
            size_t e_right = edge_idx(x+1, y, x+1, y+1);
            size_t e_bottom = edge_idx(x, y+1, x+1, y+1);
            size_t e_left = edge_idx(x, y, x, y+1);
            
            if (face_vertices_.size() <= f_id) {
                face_vertices_.resize(f_id + 1);
                face_edges_.resize(f_id + 1);
            }
            
            face_vertices_[f_id] = {v0, v1, v2, v3};
            face_edges_[f_id] = {e_top, e_right, e_bottom, e_left};
            
            // Update vertex adjacency
            vertex_faces_[v0].push_back(f_id);
            vertex_faces_[v1].push_back(f_id);
            vertex_faces_[v2].push_back(f_id);
            vertex_faces_[v3].push_back(f_id);
            
            // Update edge adjacency
            if (edge_faces_.size() <= e_top) edge_faces_.resize(e_top + 1);
            if (edge_faces_.size() <= e_right) edge_faces_.resize(e_right + 1);
            if (edge_faces_.size() <= e_bottom) edge_faces_.resize(e_bottom + 1);
            if (edge_faces_.size() <= e_left) edge_faces_.resize(e_left + 1);
            
            edge_faces_[e_top].push_back(f_id);
            edge_faces_[e_right].push_back(f_id);
            edge_faces_[e_bottom].push_back(f_id);
            edge_faces_[e_left].push_back(f_id);
        }
    }
}

// ============================================================================
// Edge and face index computation
// ============================================================================
size_t CellComplex::edge_idx(int x1, int y1, int x2, int y2) const {
    // Ensure consistent ordering
    if (x1 == x2) {
        // Vertical edge
        int y_min = std::min(y1, y2);
        return (x1 * (h_ - 1) + y_min) + (w_ * (h_ - 1));
    } else {
        // Horizontal edge
        int x_min = std::min(x1, x2);
        return y1 * (w_ - 1) + x_min;
    }
}

// ============================================================================
// Critical point detection
// ============================================================================
bool CellComplex::is_local_maximum(size_t idx) const {
    int x = idx % w_;
    int y = idx / w_;
    double val = values_[idx];
    
    // Check all 8 neighbors
    for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
            if (dx == 0 && dy == 0) continue;
            
            int nx = x + dx;
            int ny = y + dy;
            
            if (in_bounds(nx, ny)) {
                if (values_[ny * w_ + nx] > val) {
                    return false;
                }
            }
        }
    }
    return true;
}

std::vector<size_t> CellComplex::get_critical_points() {
    if (critical_points_.empty()) {
        // Find all local maxima
        for (size_t i = 0; i < values_.size(); ++i) {
            if (is_local_maximum(i)) {
                critical_points_.push_back(i);
            }
        }
    }
    return critical_points_;
}

// ============================================================================
// Gradient computation (simplified Forman)
// ============================================================================
void CellComplex::compute_gradient() {
    gradient_pairs_.clear();
    
    // Simple approach: each pixel flows to highest neighbor
    for (size_t i = 0; i < values_.size(); ++i) {
        int x = i % w_;
        int y = i / w_;
        double val = values_[i];
        
        // Find highest neighbor
        size_t best_neighbor = i;
        double best_val = val;
        
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) {
                if (dx == 0 && dy == 0) continue;
                
                int nx = x + dx;
                int ny = y + dy;
                
                if (in_bounds(nx, ny)) {
                    size_t nidx = ny * w_ + nx;
                    if (values_[nidx] > best_val) {
                        best_val = values_[nidx];
                        best_neighbor = nidx;
                    }
                }
            }
        }
        
        if (best_neighbor != i) {
            gradient_pairs_[i] = best_neighbor;
        }
    }
}

// ============================================================================
// Flow following and basin extraction
// ============================================================================
size_t CellComplex::follow_gradient(size_t start_idx) const {
    size_t current = start_idx;
    std::unordered_map<size_t, bool> visited;
    
    while (true) {
        if (visited[current]) break;  // Cycle detected
        
        auto it = gradient_pairs_.find(current);
        if (it == gradient_pairs_.end()) break;  // Critical point
        
        visited[current] = true;
        current = it->second;
    }
    
    return current;  // This is the critical point this pixel flows to
}

void CellComplex::flood_fill(size_t start_idx, std::vector<bool>& visited, 
                             std::vector<size_t>& basin) const {
    std::queue<size_t> q;
    q.push(start_idx);
    visited[start_idx] = true;
    
    while (!q.empty()) {
        size_t current = q.front();
        q.pop();
        basin.push_back(current);
        
        int x = current % w_;
        int y = current / w_;
        
        // Check all neighbors that flow to this point
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) {
                if (dx == 0 && dy == 0) continue;
                
                int nx = x + dx;
                int ny = y + dy;
                
                if (in_bounds(nx, ny)) {
                    size_t nidx = ny * w_ + nx;
                    if (!visited[nidx]) {
                        // Check if this neighbor flows to current
                        auto it = gradient_pairs_.find(nidx);
                        if (it != gradient_pairs_.end() && it->second == current) {
                            q.push(nidx);
                            visited[nidx] = true;
                        }
                    }
                }
            }
        }
    }
}

std::vector<size_t> CellComplex::get_basin(size_t critical_point) {
    std::vector<size_t> basin;
    std::vector<bool> visited(values_.size(), false);
    
    // Start flood fill from critical point
    flood_fill(critical_point, visited, basin);
    
    return basin;
}

std::unordered_map<size_t, std::vector<size_t>> CellComplex::get_all_basins() {
    std::unordered_map<size_t, std::vector<size_t>> basins;
    std::vector<bool> visited(values_.size(), false);
    
    // Get critical points first
    auto critical = get_critical_points();
    
    // For each critical point, get its basin
    for (size_t cp : critical) {
        if (!visited[cp]) {
            std::vector<size_t> basin;
            flood_fill(cp, visited, basin);
            basins[cp] = basin;
        }
    }
    
    return basins;
}

// ============================================================================
// Persistence filtering
// ============================================================================
void CellComplex::compute_persistence() {
    persistence_.clear();
    
    auto critical = get_critical_points();
    
    for (size_t cp : critical) {
        int x = cp % w_;
        int y = cp / w_;
        double val = values_[cp];
        
        // Find lowest saddle point connecting to another maximum
        double min_saddle = std::numeric_limits<double>::max();
        
        // Simple approach: look at boundary of basin
        auto basin = get_basin(cp);
        std::vector<bool> in_basin(values_.size(), false);
        for (size_t b : basin) in_basin[b] = true;
        
        // Find boundary pixels
        for (size_t b : basin) {
            int bx = b % w_;
            int by = b / w_;
            
            // Check neighbors outside basin
            for (int dy = -1; dy <= 1; ++dy) {
                for (int dx = -1; dx <= 1; ++dx) {
                    if (dx == 0 && dy == 0) continue;
                    
                    int nx = bx + dx;
                    int ny = by + dy;
                    
                    if (in_bounds(nx, ny)) {
                        size_t nidx = ny * w_ + nx;
                        if (!in_basin[nidx]) {
                            min_saddle = std::min(min_saddle, values_[nidx]);
                        }
                    }
                }
            }
        }
        
        if (min_saddle < std::numeric_limits<double>::max()) {
            persistence_[cp] = val - min_saddle;
        } else {
            persistence_[cp] = std::numeric_limits<double>::max();
        }
    }
}

void CellComplex::filter_by_persistence(double threshold) {
    if (persistence_.empty()) {
        compute_persistence();
    }
    
    std::vector<size_t> filtered;
    for (size_t cp : critical_points_) {
        auto it = persistence_.find(cp);
        if (it != persistence_.end() && it->second >= threshold) {
            filtered.push_back(cp);
        }
    }
    
    critical_points_ = filtered;
}

// ============================================================================
// Statistics and debugging
// ============================================================================
void CellComplex::print_statistics() const {
    std::cout << "=== CellComplex Statistics ===\n";
    std::cout << "Dimensions: " << w_ << " x " << h_ << "\n";
    std::cout << "Vertices: " << values_.size() << "\n";
    std::cout << "Edges: " << edge_vertices_.size() << "\n";
    std::cout << "Faces: " << face_vertices_.size() << "\n";
    std::cout << "Critical points: " << critical_points_.size() << "\n";
    std::cout << "Gradient pairs: " << gradient_pairs_.size() << "\n";
    std::cout << "============================\n";
}

} // namespace forman