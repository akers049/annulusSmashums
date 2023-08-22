/*
 * NeoHookean_Newton_CompressedStrip.h
 *
 *  Created on: Aug 10, 2017
 *      Author: andrew
 */

#ifndef NEOHOOKEAN_NEWTON_COMPRESSEDSTRIP_H_
#define NEOHOOKEAN_NEWTON_COMPRESSEDSTRIP_H_

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/thread_management.h>
#include <deal.II/base/work_stream.h>
#include <deal.II/base/multithread_info.h>
#include <deal.II/distributed/grid_refinement.h>

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/tensor.h>

#include <deal.II/lac/constraint_matrix.h>

#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
//#include <deal.II/grid/tria_boundary_lib.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_in.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>

#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>

#include <deal.II/fe/mapping_q_eulerian.h>

#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/solver_minres.h>
#include <deal.II/lac/lapack_full_matrix.h>
#include <deal.II/grid/manifold_lib.h>


#include <deal.II/lac/trilinos_sparse_matrix.h>

#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>

#include "Constituitive.h"
#include "MMASolver.h"


#define MAXLINE 1024
#define DIM 3

namespace compressed_strip
{
  using namespace dealii;
  /****************************************************************
                       Class Declarations
  ****************************************************************/

  class timeHistory
  {
  public:
    timeHistory(){};
    timeHistory(unsigned int u_size_, unsigned int internal_size_);
    ~timeHistory(){};
    void init(unsigned int u_size_, unsigned int internal_size_)
    {
      u_size = u_size_;
      internal_size = internal_size_;
    }

    bool save_hisotry(std::vector<double> &eps_p_eff_,
                    std::vector<Tensor<2,DIM>> &eps_p_, Vector<double> &u_, Vector<double> &v_, Vector<double> &a_);

    void forward_problem_close();

    bool save_history_adjoint(unsigned int n, std::vector<double> &gamma_,
               std::vector<Tensor<2,DIM>> &mu_, Vector<double> &lambda_);
//    void get_instance(unsigned int n, std::vector<double> *eps_p_eff_,
//                     std::vector<Tensor<2,DIM>>*eps_p_, Vector<double> *u_);
    unsigned int get_size(){return size;};
    void clear_history();

    std::vector<std::vector<double>> eps_p_eff;
    std::vector< std::vector<Tensor<2,DIM>>> eps_p;
    std::vector<Vector<double>> u;
    std::vector<Vector<double>> v;
    std::vector<Vector<double>> a;

    std::vector<std::vector<double>> gamma;
    std::vector< std::vector<Tensor<2,DIM>>> mu;
    std::vector<Vector<double>> lambda;
    std::vector<Vector<double>> lambda_v;
    std::vector<Vector<double>> lambda_a;
  private:

    unsigned int size = 0;
    unsigned int u_size = 0;
    unsigned int internal_size = 0;

  };

  class Compute_eps_p : public DataPostprocessor<DIM>
  {
  public:

    Compute_eps_p(std::vector<double> *q_ptr, std::vector<double> *pressure_ptr)
    {
      q = q_ptr;
      pressure = pressure_ptr;
    }

    virtual void evaluate_vector_field(const DataPostprocessorInputs::Vector< DIM > &   input_data,
                                        std::vector< Vector< double > > &   computed_quantities) const;

    virtual std::vector<std::string> get_names() const;

    virtual UpdateFlags get_needed_update_flags() const;
    virtual std::vector<DataComponentInterpretation::DataComponentInterpretation> get_data_component_interpretation () const;



  private:

    std::vector<double> *q;
    std::vector<double> *pressure;

  };


  /****  ElasticProblem  *****
   * This is the primary class used, with all the dealii stuff
   */
  class ElasticProblem
  {
  public:
    ElasticProblem();
    ~ElasticProblem();

    void create_mesh();
    void setup_system ();

    void output_results(const unsigned int cycle) const;

    void read_input_file(char* filename);

    void save_current_state(unsigned int indx, bool firstTime);
    void load_state(unsigned int indx);

    void set_boundary_values();

    // get methods for important constants

    unsigned int get_n_dofs(){return dof_handler.n_dofs();};
    unsigned int get_number_active_cells(){return triangulation.n_active_cells();};

    void solve_forward_problem();

    double get_tol(){return tol;};
    unsigned int get_maxIter(){return maxIter;};

    double               system_energy = 0.0;


    char output_directory[MAXLINE];


  private:

    struct AssemblyScratchData
    {
      AssemblyScratchData (const FiniteElement<DIM> &fe, Quadrature<DIM> &quad, const unsigned int step_);
      AssemblyScratchData (const AssemblyScratchData &scratch_data);

      FEValues<DIM>     fe_values;
      unsigned int step;
    };

    struct RhsAssemblyCopyData
    {
      Vector<double>                       cell_rhs;
      std::vector<types::global_dof_index> local_dof_indices;
    };

    void parallel_assemble_rhs(unsigned int n = 0);


    void local_assemble_system_rhs (const typename DoFHandler<DIM>::active_cell_iterator &cell,
                                AssemblyScratchData                                  &scratch,
                                RhsAssemblyCopyData                                     &copy_data);


    void copy_local_to_global_rhs (const RhsAssemblyCopyData &copy_data);



    void right_hand_side (const std::vector<Point<DIM> > &points,
                          std::vector<Tensor<1, DIM> >   &values);

    void get_deformation_gradient(std::vector<Tensor<1,DIM> > &old_solution_gradient,
                                    Tensor<2,DIM> &F);
    void get_strain(std::vector<Tensor<1,DIM> > &old_solution_gradient,
                                    Tensor<2,DIM> &Eps);

    void setup_system_constraints();

    bool impactor_update(unsigned int k);

    void apply_boundaries_to_rhs(Vector<double> *rhs, std::vector<bool> *homogenous_dirichlet_dofs);

    void newton_iterate();
    void line_search_and_add_step_length(double last_residual, std::vector<bool> *homogenous_dirichlet_dofs);
    void update_internal_vars();

    void numerical_derivative(unsigned int n, double pert);

    void numerical_derivative_internal(double pert);

    void assemble_mass_vector();
    void assemble_mass_matrix();
    void assemble_system_matrix();
    void assemble_system_rhs();
    void indentor_contact_forces();


    void apply_boundaries_and_constraints_system_matrix(SparseMatrix<double> *mat,
        std::vector<bool> *homogenous_dofs);

    void solve();

    void getNextDataLine( FILE* const filePtr, char* nextLinePtr,
                            int const maxSize, int* const endOfFileFlag);

    void renumber_boundary_ids();


    Triangulation<DIM,DIM>   triangulation;
    DoFHandler<DIM>      dof_handler;
    Quadrature<DIM>      problemQuadrature;

    FESystem<DIM>        fe;

    ConstraintMatrix     constraints;

    std::vector<IndexSet>    owned_partitioning;
    std::vector<IndexSet>    relevant_partitioning;

    SparsityPattern      sparsity_pattern;
    SparseMatrix<double> system_matrix;
    SparseMatrix<double> mass_matrix;
    SparseDirectUMFPACK  *mass_direct = NULL;
    SparseMatrix<double> adjoint_system_matrix;

    Vector<double> mass_vector;

    Compressible_NeoHookean nh;

    std::vector<J2Isotropic> elmMats;

    Vector<double>       system_rhs;
    Vector<double>       system_rhs_just_loads;
    Vector<double>       present_solution;
    Vector<double>       velocity;
    Vector<double>       accel;


    Vector<double>       volumes;

    timeHistory tHist;


    std::vector<bool> load_dofs;
    Vector<double> load_factor;

    std::vector< std::vector<unsigned int> > neighborList;
    std::vector< std::vector<double > > neighborDists;


    std::vector<bool> homo_dofs;
    std::vector<unsigned int>  grid_dimensions;
    std::vector<double> domain_dimensions;

    std::vector<unsigned int> dead_design_vars;
    std::vector<bool> dead_design_vars_flags;
    std::vector< Vector<double> > values_history;

    std::vector<double> Epsp_eff;
    std::vector<double> pressure;
    std::vector< Tensor<2, DIM> > Epsp;
    Vector<double> ave_epsp_eff;
    Vector<double> ave_pressure;

    std::vector<double> sig_rr_r;
    std::vector<double> sig_tt_r;
    std::vector<double> sig_zz_r;
    std::vector<double> radii;
    std::vector<int> map_cell_to_rLine;

    double ave_sig_zz = 0.0;
    double TopVol = 0.0;
    std::vector<unsigned int> top_elms;

    Compute_eps_p *postprocess = NULL;

    double K = 0.0;


    unsigned int iter = 0;

    unsigned int maxIter = 100;
    double tol= 0.1;

    unsigned int N = 1;
    unsigned int qx = 2;
    unsigned int qy = 2;
    unsigned int qz = 2;

    double mu = 1.0;
    double lambda = 1.0;

    double V_tot_ratio = 0.0;
    double R_filter = 1.0;

    double load_val = 0.01;
    unsigned int load_steps = 10;



    double dT = 0.01;
    double T_final = 1.0;


    double density = 1.0;

    // plastic parameters
    double m_ = 1.0;
    double n_ = 5.0;
    double eps_p_0 = 1e-3;
    double eps_p_0_dot = 1.0;
    double yield = 1e-3;

    double M_impactor = 1.0;
    double u_impactor = 0.0;
    double v_impactor = -0.1;
    double a_impactor = 0.0;
    double z_impactor = 0.855e-3;
    double r_impactor = 0.35e-3;
    double penal_stiff = 1.0e7;


    bool inContact = true;
    bool firstFlag = true;
    bool fileLoadFlag = false;

  };
}

#endif /* NEOHOOKEAN_NEWTON_COMPRESSEDSTRIP_H_ */
