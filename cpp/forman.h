// cpp/forman.h
#ifndef FORMAN_H
#define FORMAN_H

#include <vector>
#include <cstddef>
#include <unordered_map>
#include <algorithm>
#include <cmath>
#include <queue>
#include <stack>
#include <iostream>

namespace forman {

// Cell types in 2D complex
enum class CellDimension { VERTEX = 0, EDGE = 1, FACE = 2 };

// Simple 2D point
struct Point {
    int x, y;
    Point(int _x = 0, int _y = 0) : x(_x), y(_y) {}
};

class CellComplex {
public:
    /**
     * Constructor
     * @param width Image width (number of columns)
     * @param height Image height (number of rows)
     * @param data Flattened image data (row-major: row * width + col)
     */
    CellComplex(int width, int height, const double* data);
    ~CellComplex() = default;
    
    // Disable copying (too expensive)
    CellComplex(const CellComplex&) = delete;
    CellComplex& operator=(const CellComplex&) = delete;
    
    // Allow moving
    CellComplex(CellComplex&&) = default;
    CellComplex& operator=(CellComplex&&) = default;
    
    /**
     * Build the cell complex (vertices, edges, faces)
     * Must be called before any other operations
     */
    void build();
    
    /**
     * Get critical points (local maxima)
     * @return Vector of linear indices (row * width + col) of critical points
     */
    std::vector<size_t> get_critical_points();
    
    /**
     * Get basin of attraction for a critical point
     * @param critical_point Linear index of critical point
     * @return Vector of linear indices of all pixels in the basin
     */
    std::vector<size_t> get_basin(size_t critical_point);
    
    /**
     * Filter critical points by persistence
     * @param threshold Minimum persistence value to keep
     */
    void filter_by_persistence(double threshold);
    
    /**
     * Get all basins
     * @return Map from critical point to vector of basin pixels
     */
    std::unordered_map<size_t, std::vector<size_t>> get_all_basins();
    
    /**
     * Get gradient pairs (for debugging/visualization)
     * @return Map from cell index to paired cell index
     */
    std::unordered_map<size_t, size_t> get_gradient_pairs() const { return gradient_pairs_; }
    
    /**
     * Print statistics about the complex
     */
    void print_statistics() const;
    
    // Getters
    int width() const { return w_; }
    int height() const { return h_; }
    size_t num_vertices() const { return values_.size(); }
    double get_value(int x, int y) const { return values_[y * w_ + x]; }
    double get_value(size_t idx) const { return values_[idx]; }

private:
    int w_, h_;                          // Dimensions
    std::vector<double> values_;          // Original pixel values
    
    // Cell relationships
    std::vector<std::vector<size_t>> vertex_edges_;  // For each vertex, incident edges
    std::vector<std::vector<size_t>> vertex_faces_;  // For each vertex, incident faces
    std::vector<std::vector<size_t>> edge_vertices_; // For each edge, its two vertices
    std::vector<std::vector<size_t>> edge_faces_;    // For each edge, adjacent faces
    std::vector<std::vector<size_t>> face_vertices_; // For each face, its four vertices
    std::vector<std::vector<size_t>> face_edges_;    // For each face, its four edges
    
    // Gradient pairs (for Forman gradient)
    std::unordered_map<size_t, size_t> gradient_pairs_;  // cell -> paired cell
    
    // Critical points and their persistence
    std::vector<size_t> critical_points_;
    std::unordered_map<size_t, double> persistence_;
    
    // Helper methods
    size_t idx(int x, int y) const { return y * w_ + x; }
    bool in_bounds(int x, int y) const { return x >= 0 && x < w_ && y >= 0 && y < h_; }
    
    size_t edge_idx(int x1, int y1, int x2, int y2) const;
    size_t face_idx(int x, int y) const { return y * (w_-1) + x; }
    
    void build_vertices();
    void build_edges();
    void build_faces();
    
    bool is_local_maximum(size_t idx) const;
    void compute_gradient();
    void compute_persistence();
    
    // Gradient flow
    size_t follow_gradient(size_t start_idx) const;
    void flood_fill(size_t start_idx, std::vector<bool>& visited, std::vector<size_t>& basin) const;
};

} // namespace forman

#endif // FORMAN_H