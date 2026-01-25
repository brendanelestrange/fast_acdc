#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <map>
#include <algorithm>
#include <iomanip>
#include <chrono>
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <omp.h>

#include "lapjv.h"

using SparseMat = Eigen::SparseMatrix<double>;
using MatrixRowMaj = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using Triplet = Eigen::Triplet<double>;

// --- Timer Helpers ---
auto get_time() { return std::chrono::high_resolution_clock::now(); }
void print_elapsed(std::chrono::time_point<std::chrono::high_resolution_clock> start, std::string label) {
    auto end = get_time();
    std::chrono::duration<double> diff = end - start;
    std::cout << label << " took " << std::fixed << std::setprecision(4) << diff.count() << " s" << std::endl;
}
std::string clean_id(std::string s) {
    s.erase(remove_if(s.begin(), s.end(), [](unsigned char c) { 
        return std::isspace(c) || c == '\r' || c == '\n'; 
    }), s.end());
    return s;
}

// --- OPTIMIZED LAPJV Wrapper ---
Eigen::VectorXi solve_linear_assignment(Eigen::Ref<MatrixRowMaj> cost_matrix) {
    int n = cost_matrix.rows();
    std::vector<double*> rows(n);
    for(int i = 0; i < n; ++i) rows[i] = cost_matrix.row(i).data();

    std::vector<int> x(n), y(n);
    lapjv_internal(n, rows.data(), x.data(), y.data());
    return Eigen::Map<Eigen::VectorXi>(x.data(), n);
}

// --- Data Loader ---
struct DataLoader {
    std::map<std::string, int> male_map, female_map;
    std::vector<std::string> male_ids, female_ids;
    Eigen::VectorXi initial_perm; 

    void load_benchmark(const std::string& filename) {
        std::ifstream file(filename);
        if (!file.is_open()) { std::cerr << "Error opening " << filename << std::endl; exit(1); }
        std::string line, m_id, f_id;
        std::getline(file, line); 

        int idx = 0;
        while (std::getline(file, line)) {
            std::stringstream ss(line);
            std::getline(ss, m_id, ',');
            std::getline(ss, f_id, ',');
            m_id = clean_id(m_id);
            f_id = clean_id(f_id);

            if (male_map.find(m_id) == male_map.end()) {
                male_map[m_id] = idx;
                male_ids.push_back(m_id);
            }
            if (female_map.find(f_id) == female_map.end()) {
                female_map[f_id] = female_map.size(); 
                female_ids.push_back(f_id);
            }
            idx++;
        }
        
        initial_perm.resize(male_ids.size());
        file.clear(); file.seekg(0); std::getline(file, line);
        
        while (std::getline(file, line)) {
            std::stringstream ss(line);
            std::getline(ss, m_id, ',');
            std::getline(ss, f_id, ',');
            if (male_map.count(clean_id(m_id)) && female_map.count(clean_id(f_id)))
                initial_perm[male_map[clean_id(m_id)]] = female_map[clean_id(f_id)];
        }
    }

    SparseMat load_graph(const std::string& filename, const std::map<std::string, int>& map, int n) {
        std::ifstream file(filename);
        if (!file.is_open()) { std::cerr << "Error opening " << filename << std::endl; exit(1); }
        std::string line, u_str, v_str, w_str;
        std::vector<Triplet> coefficients;
        std::getline(file, line); 

        while (std::getline(file, line)) {
            std::stringstream ss(line);
            std::getline(ss, u_str, ',');
            std::getline(ss, v_str, ',');
            std::getline(ss, w_str, ',');
            u_str = clean_id(u_str);
            v_str = clean_id(v_str);

            if (map.count(u_str) && map.count(v_str)) {
                coefficients.push_back(Triplet(map.at(u_str), map.at(v_str), std::stod(w_str)));
            }
        }
        SparseMat mat(n, n);
        mat.setFromTriplets(coefficients.begin(), coefficients.end());
        return mat;
    }
};

// --- ACDC Solver ---
class ACDCSolver {
    int N;
    const SparseMat& A; 
    const SparseMat& B; 
    
public: 
    struct Component {
        double weight;
        Eigen::VectorXi perm;
    };
    std::vector<Component> P_components;

    ACDCSolver(const SparseMat& a, const SparseMat& b, const Eigen::VectorXi& p0) 
        : A(a), B(b), N(a.rows()) {
        reset_state(p0);
    }

    // NEW: Needed to reset the convex hull when switching from Discrete -> Relaxation
    void reset_state(const Eigen::VectorXi& p0) {
        P_components.clear();
        P_components.push_back({1.0, p0});
    }

    double calculate_score(const Eigen::VectorXi& pi) const {
        double score = 0;
        for (int k=0; k<A.outerSize(); ++k) {
            for (SparseMat::InnerIterator it(A, k); it; ++it) {
                int i = it.row();
                int j = it.col();
                double wB = B.coeff(pi[i], pi[j]); 
                if (wB > 0) score += std::min(it.value(), wB);
            }
        }
        return score;
    }

    void compute_gradient_internal(const SparseMat& B_T, const std::vector<Component>& components, MatrixRowMaj& G) {
        G.setZero(); 
        
        for (const auto& comp : components) {
            const Eigen::VectorXi& pi = comp.perm;
            double weight = comp.weight;
            
            #pragma omp parallel for schedule(dynamic)
            for (int k=0; k<A.outerSize(); ++k) { 
                for (SparseMat::InnerIterator itA(A, k); itA; ++itA) {
                    int i = itA.row();
                    int j = itA.col(); 
                    double val_A = itA.value();
                    int u = pi[i]; 

                    for (SparseMat::InnerIterator itB(B_T, u); itB; ++itB) {
                        G(j, itB.row()) += weight * std::min(val_A, itB.value());
                    }
                }
            }
            
            #pragma omp parallel for schedule(dynamic)
            for (int k=0; k<A.outerSize(); ++k) {
                for (SparseMat::InnerIterator itA(A, k); itA; ++itA) {
                    int j = itA.row(); 
                    int i = itA.col(); 
                    int u = pi[i]; 

                    for (SparseMat::InnerIterator itB(B, u); itB; ++itB) {
                        double contribution = weight * std::min(itA.value(), itB.value());
                        #pragma omp atomic
                        G(j, itB.row()) += contribution;
                    }
                }
            }
        }
    }

    Eigen::VectorXi get_best_permutation() {
        MatrixRowMaj P_dense = MatrixRowMaj::Zero(N, N);
        for(const auto& comp : P_components) {
            #pragma omp parallel for
            for(int i=0; i<N; ++i) P_dense(i, comp.perm[i]) += comp.weight;
        }

        #pragma omp parallel for collapse(2)
        for(int i=0; i<N; ++i) {
            for(int j=0; j<N; ++j) {
                P_dense(i, j) = -P_dense(i, j);
            }
        }
        return solve_linear_assignment(P_dense);
    }

    void frank_wolfe_step(MatrixRowMaj& G_buffer) {
        SparseMat B_T = B.transpose();
        
        // 1. Compute Gradient 
        compute_gradient_internal(B_T, P_components, G_buffer);

        // 2. Get Pi_t
        Eigen::VectorXi pi_t = get_best_permutation();

        // 3. Compute Omega 
        MatrixRowMaj Lambda(N, N);
        #pragma omp parallel for collapse(2)
        for(int i=0; i<N; ++i) {
            for(int j=0; j<N; ++j) {
                Lambda(i, j) = G_buffer(i, pi_t[j]);
            }
        }

        Eigen::VectorXd row_sums = Lambda.rowwise().sum();
        Eigen::VectorXd col_sums = Lambda.colwise().sum();
        Eigen::VectorXd diag_vals = Lambda.diagonal();

        MatrixRowMaj& Omega_ref = Lambda; // Alias

        #pragma omp parallel for collapse(2)
        for(int i=0; i<N; ++i) {
            for(int j=0; j<N; ++j) {
                double val = Omega_ref(i, j) + diag_vals(i) + diag_vals(j) - row_sums(i) - col_sums(j);
                Omega_ref(i, j) = -val; // In-place negation
            }
        }

        Eigen::VectorXi omega_perm = solve_linear_assignment(Omega_ref);

        Eigen::VectorXi q_perm(N);
        #pragma omp parallel for
        for(int i=0; i<N; ++i) q_perm[i] = pi_t[omega_perm[i]];

        // 4. Line Search
        MatrixRowMaj G_Q(N, N); 
        std::vector<Component> Q_comp = {{1.0, q_perm}};
        compute_gradient_internal(B_T, Q_comp, G_Q);
        
        double tr_QT_GP = 0; for(int i=0; i<N; ++i) tr_QT_GP += G_buffer(i, q_perm[i]);
        double tr_QT_GQ = 0; for(int i=0; i<N; ++i) tr_QT_GQ += G_Q(i, q_perm[i]);
        
        double tr_PT_GP = 0, tr_PT_GQ = 0;
        for(const auto& comp : P_components) {
             double sub_p = 0, sub_q = 0;
             for(int i=0; i<N; ++i) { sub_p += G_buffer(i, comp.perm[i]); sub_q += G_Q(i, comp.perm[i]); }
             tr_PT_GP += comp.weight * sub_p;
             tr_PT_GQ += comp.weight * sub_q;
        }

        double A_term = tr_QT_GP - tr_PT_GP; 
        double B_term = tr_QT_GQ - tr_PT_GQ; 
        double alpha = 0;

        if (std::abs(A_term - B_term) < 1e-9) alpha = (A_term > 0) ? 1.0 : 0.0;
        else alpha = std::max(0.0, std::min(1.0, A_term / (A_term - B_term)));

        // std::cout << "   - Alpha: " << alpha << std::endl; 
        if (alpha > 1e-4) {
            for(auto& comp : P_components) comp.weight *= (1.0 - alpha);
            P_components.push_back({alpha, q_perm});
        }
    }
    
    friend struct GreedyCoupler;
};

// --- Greedy Discrete Search ---
struct GreedyCoupler {
    ACDCSolver& solver;
    int N;
    MatrixRowMaj G_buffer;

    GreedyCoupler(ACDCSolver& s) : solver(s), N(s.N) {
        G_buffer.resize(N, N);
    }

    inline double H_term(double a, double B_ii, double B_jj, double B_ij, double B_ji) {
        return std::min(a, B_ii) + std::min(a, B_jj) - std::min(a, B_ij) - std::min(a, B_ji);
    }

    // UPDATED: Returns true if a swap occurred, false otherwise
    bool perform_pairwise_swap(Eigen::VectorXi& pi) {
        SparseMat B_T = solver.B.transpose();
        std::vector<ACDCSolver::Component> single_comp = {{1.0, pi}};
        
        solver.compute_gradient_internal(B_T, single_comp, G_buffer);
        
        struct SwapMove { int i; int j; double gain; };
        SwapMove best_move = {-1, -1, 0.0};
        std::vector<double> B_diag(N);
        for(int k=0; k<N; ++k) B_diag[k] = solver.B.coeff(pi[k], pi[k]);

        // std::cout << "   - [Discrete] Scanning pairs..." << std::flush;
        
        #pragma omp parallel
        {
            SwapMove local_best = {-1, -1, 0.0};
            
            #pragma omp for schedule(dynamic) nowait
            for (int i = 0; i < N; ++i) {
                for (int j = i + 1; j < N; ++j) {
                    double delta = G_buffer(i, pi[j]) + G_buffer(j, pi[i]) - G_buffer(i, pi[i]) - G_buffer(j, pi[j]);
                    
                    double A_ii = solver.A.coeff(i, i), A_jj = solver.A.coeff(j, j);
                    double A_ij = solver.A.coeff(i, j), A_ji = solver.A.coeff(j, i);
                    
                    double B_ii = B_diag[i], B_jj = B_diag[j];
                    double B_ij = solver.B.coeff(pi[i], pi[j]);
                    double B_ji = solver.B.coeff(pi[j], pi[i]);

                    delta += H_term(A_ii, B_ii, B_jj, B_ij, B_ji);
                    delta += H_term(A_jj, B_ii, B_jj, B_ij, B_ji);
                    delta -= H_term(A_ij, B_ii, B_jj, B_ij, B_ji);
                    delta -= H_term(A_ji, B_ii, B_jj, B_ij, B_ji);

                    if (delta > local_best.gain) local_best = {i, j, delta};
                }
            }
            #pragma omp critical
            { if (local_best.gain > best_move.gain) best_move = local_best; }
        }

        if (best_move.gain > 0) {
            std::cout << "SWAP " << best_move.i << "<->" << best_move.j << " Gain:" << best_move.gain << " ";
            std::swap(pi[best_move.i], pi[best_move.j]);
            return true;
        } else {
            return false;
        }
    }
};

int main() {
    std::string initial_path = "data/vnc_matching_submission_benchmark_5154247.csv";
    std::string male_path    = "data/male_connectome_graph.csv";
    std::string female_path  = "data/female_connectome_graph.csv";
    std::string output_path  = "final_submission.csv";

    std::cout << "--- Configuration ---" << std::endl;
    std::cout << "Initial Perm: " << initial_path << std::endl;
    std::cout << "Male Graph:   " << male_path << std::endl;
    std::cout << "Female Graph: " << female_path << std::endl;
    std::cout << "Output File:  " << output_path << std::endl;
    std::cout << "---------------------" << std::endl;

    auto t_total = get_time();
    DataLoader loader;
    
    loader.load_benchmark(initial_path);
    
    int N = loader.male_ids.size();
    SparseMat A = loader.load_graph(male_path, loader.male_map, N);
    SparseMat B = loader.load_graph(female_path, loader.female_map, N);
    
    ACDCSolver solver(A, B, loader.initial_perm);
    Eigen::VectorXi current_pi = loader.initial_perm;
    GreedyCoupler greedy(solver);

    MatrixRowMaj G_buffer(N, N);

    std::cout << "Starting Score: " << (long)solver.calculate_score(current_pi) << std::endl;
    std::cout << "\n=== Starting Interleaved Optimization ===" << std::endl;

    int epoch = 0;
    while(true) {
        epoch++;
        double start_epoch_score = solver.calculate_score(current_pi);
        std::cout << "\n--- Epoch " << epoch << " (Start Score: " << (long)start_epoch_score << ") ---" << std::endl;

        // 1. RELAXATION PHASE (10 Runs)
        // Reset solver history to focus exploration around the current best discrete permutation
        solver.reset_state(current_pi); 
        std::cout << "   [Relaxation] Running 10 steps..." << std::flush;
        auto t_relax = get_time();
        for(int t=0; t<10; ++t) {
            solver.frank_wolfe_step(G_buffer);
        }
        print_elapsed(t_relax, " Done.");
        
        // Project back to discrete space
        current_pi = solver.get_best_permutation();
        double post_relax_score = solver.calculate_score(current_pi);
        std::cout << "   [Relaxation] Result Score: " << (long)post_relax_score << std::endl;

        // 2. DISCRETE PHASE (10 Swaps)
        std::cout << "   [Discrete]   Running max 10 swaps..." << std::endl;
        int swaps_done = 0;
        for(int t=0; t<10; ++t) {
            std::cout << "     Run " << t+1 << ": ";
            bool swapped = greedy.perform_pairwise_swap(current_pi);
            if(swapped) {
                std::cout << "(Ok)" << std::endl;
                swaps_done++;
            } else {
                std::cout << "(No Move)" << std::endl;
                break; // Stop discrete phase if local max reached
            }
        }
        
        double end_epoch_score = solver.calculate_score(current_pi);
        std::cout << "   [End Epoch]  Score: " << (long)end_epoch_score << " (Delta: " << (long)(end_epoch_score - start_epoch_score) << ")" << std::endl;

        // 3. CONVERGENCE CHECK
        if (end_epoch_score <= start_epoch_score + 1e-6) {
            std::cout << "\n>>> CONVERGED <<<" << std::endl;
            break;
        }
    }

    std::cout << "Final Score: " << (long)solver.calculate_score(current_pi) << std::endl;
    
    std::ofstream out(output_path);
    out << "male_id,female_id" << std::endl;
    for(int i=0; i<N; ++i) {
        out << loader.male_ids[i] << "," << loader.female_ids[current_pi[i]] << std::endl;
    }
    
    print_elapsed(t_total, "Total Execution");
    return 0;
}