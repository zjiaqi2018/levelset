/* ---------------------------------------------------------------------
 *
 * Copyright (C) 2020 by the deal.II authors
 *
 * This file is part of the deal.II library.
 *
 * The deal.II library is free software; you can use it, redistribute
 * it, and/or modify it under the terms of the GNU Lesser General
 * Public License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 * The full text of the license can be found in the file LICENSE.md at
 * the top level directory of deal.II.
 *
 * ---------------------------------------------------------------------

 *
 * Author: Martin Kronbichler, 2020
 */

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/function.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/utilities.h>
#include <deal.II/base/vectorization.h>
#include <deal.II/base/time_stepping.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/la_parallel_vector.h>

#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/matrix_free.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>

#include <fstream>
#include <iomanip>
#include <iostream>

#include <deal.II/matrix_free/operators.h>

namespace LevelsetMatrixfree
{
  using namespace dealii;

  constexpr unsigned int dimension            = 2;
  constexpr unsigned int n_global_refinements = 6;
  constexpr unsigned int fe_degree            = 3;
  constexpr unsigned int vel_degree           = 2;
  constexpr unsigned int n_q_points_1d        = fe_degree + 2;

  using Number = double;

  constexpr double final_time  = 2.0;
  constexpr double output_tick = 0.05;

  const double courant_number = 0.05;
  enum LevelsetRungeKuttaScheme
  {
    stage_3_order_3, 
    stage_2_order_2, 
  };
  constexpr LevelsetRungeKuttaScheme lsrk_scheme = stage_2_order_2;

  enum EulerNumericalFlux
  {
    lax_friedrichs_modified,
    harten_lax_vanleer,
  };
  constexpr EulerNumericalFlux numerical_flux_type = lax_friedrichs_modified;





  class LevelsetTimeIntegrator
  {
  public:
    LevelsetTimeIntegrator(const LevelsetRungeKuttaScheme scheme)
    {
      switch (scheme)
        {
          case stage_3_order_3:
            {
              bi = {{1.0, 1.0/4.0, 2.0/3.0}};
              ai = {{0., 3./4., 1./3.}};

              break;
            }

          case stage_2_order_2:
            {
              bi = {{1., 1./2}};
              ai = {{0., 1./2.}};

              break;
            }

          default:
            AssertThrow(false, ExcNotImplemented());
        }
    }

    unsigned int n_stages() const
    {
      return bi.size();
    }

    template <typename VectorType, typename Operator>
    void perform_time_step(const Operator &levelset_operator,
                           const double old_time_step,
                           const double time_step,
                           const VectorType &velocity_old_solution,
                           const VectorType &velocity_old_old_solution,
                           std::vector<VectorType *> ui_and_velocity,
                           const VectorType &levelset_old_solution,
                           VectorType &levelset_solution) const
    {
      AssertDimension(ai.size(), bi.size());

      //compute time step

      for (unsigned int stage = 0; stage < bi.size(); ++stage)
      {
        ui_and_velocity[0]->swap(levelset_solution);

        if (stage == 1) //n+1
        {
          const double time_step_factor = time_step/old_time_step;
          ui_and_velocity[1]->equ(1. + time_step_factor, velocity_old_solution);
          ui_and_velocity[1]->add(-time_step_factor, velocity_old_old_solution);
        }
        else if (stage == 2) //n+1/2
        {
          ui_and_velocity[1]->sadd(0.5, 0.5, velocity_old_solution);
        }

        levelset_operator.perform_stage(time_step,
                                        bi[stage],
                                        ai[stage],
                                        levelset_old_solution,
                                        ui_and_velocity,
                                        levelset_solution);
      }
    }

  private:
    std::vector<double> bi;
    std::vector<double> ai;
  };


template <int dim> class LevelsetVelocity : public Function<dim> {
public:
  LevelsetVelocity()
      : Function<dim>(dim)
      {}
  virtual double value(const Point<dim> &point,
                       const unsigned int component = 0) const override;
};

template <int dim>
double LevelsetVelocity<dim>::value(const Point<dim> &point,
                                  const unsigned int component) const
{
  Assert(dim >= 2, ExcNotImplemented());
  const double pi = numbers::PI;
  const double x = point[0];
  const double y = point[1];

  if (dim == 2) {
    if(component == 0)
      return -std::sin(2. * pi * y) * std::sin(pi * x) * std::sin(pi * x); 
    else 
      return  std::sin(2. * pi * x) * std::sin(pi * y) * std::sin(pi * y); 
  }
  else /*(dim == 3)*/
   {
    const double z = point[2];
    if(component == 0)
      return  2. * std::sin(pi * x) * std::sin(pi * x) *
                 std::sin(2. * pi * y) * std::sin(2. * pi * z);
    else if (component == 1)
      return  -std::sin(2. * pi * x) * std::sin(pi * y) * std::sin(pi * y) *
                 std::sin(2. * pi * z);
    else                 
      return  -std::sin(2. * pi * x) * std::sin(2. * pi * y) *
                 std::sin(pi * z);
  }
}

template <int dim> class LevelsetInitialValues : public Function<dim> {
public:
  LevelsetInitialValues() : Function<dim>(1) {}

  virtual double value(const Point<dim> &p,
                       const unsigned int component = 0) const override;
};

template <int dim>
double LevelsetInitialValues<dim>::value(const Point<dim> &p,
                                         const unsigned int) const
{
  Assert(dim >= 2, ExcNotImplemented());
  Point<dim>   center;
  const double r = 0.15;
  if (dim == 2)
    {
      center(0) = 0.5;
      center(1) = 0.75;
    }
  else /*dim == 3*/
    {
      for (uint d = 0; d < dim; ++d)
        center(d) = 0.35;
    }
  return r - p.distance(center);
}

template <int dim, typename Number>
inline DEAL_II_ALWAYS_INLINE
    Tensor<1, dim, Number>
    levelset_flux(const Number &phi,
                  const Tensor<1, dim, Number> &velocity)
{
  Tensor<1, dim, Number> flux(velocity);
  flux *= phi;

  return flux;
}

template <int dim, typename Number>
inline DEAL_II_ALWAYS_INLINE //
  Number
  levelset_numerical_flux(const Number &                u_m,
                          const Number &                u_p,
                          const Tensor<1, dim, Number> &normal,
                          const Tensor<1, dim, Number> &velocity,
                          const Number &                lambda)
{
  return 0.5 * (u_m + u_p) * (velocity * normal) + 0.5 * lambda * (u_m - u_p);
}


  template <int dim, typename Number>
  VectorizedArray<Number>
  evaluate_velocity(const Function<dim> &                      function,
                    const Point<dim, VectorizedArray<Number>> &p_vectorized,
                    const unsigned int                         component)
  {
    VectorizedArray<Number> result;
    for (unsigned int v = 0; v < VectorizedArray<Number>::size(); ++v)
      {
        Point<dim> p;
        for (unsigned int d = 0; d < dim; ++d)
          p[d] = p_vectorized[d][v];
        result[v] = function.value(p, component);
      }
    return result;
  }


  template <int dim, typename Number, int n_components = dim >
  Tensor<1, n_components, VectorizedArray<Number>>
  evaluate_velocity(const Function<dim> &                      function,
                    const Point<dim, VectorizedArray<Number>> &p_vectorized)
  {
    AssertDimension(function.n_components, n_components);
    Tensor<1, n_components, VectorizedArray<Number>> result;
    for (unsigned int v = 0; v < VectorizedArray<Number>::size(); ++v)
      {
        Point<dim> p;
        for (unsigned int d = 0; d < dim; ++d)
          p[d] = p_vectorized[d][v];
        for (unsigned int d = 0; d < n_components; ++d)
          result[d][v] = function.value(p, d);
      }
    return result;
  }

template <int dim, int degree, int velocity_degree, int n_points_1d>
  class LevelsetOperator
  {
  public:
    static constexpr unsigned int n_quadrature_points_1d = n_points_1d;

    LevelsetOperator(TimerOutput &timer_output);

    void reinit(const Mapping<dim> &   mapping,
                const std::vector<const DoFHandler<dim> *> dof_handlers,
                const std::vector<const AffineConstraints<double> *> constraints,
                const std::vector<Quadrature<1>> quadratures);

    void initialize_vector(LinearAlgebra::distributed::Vector<Number> &vector,
                           const unsigned int no_dof = 0) const;

    void get_lambda(const double lambda);

    void update_global_lambda(const LinearAlgebra::distributed::Vector<Number> &velocity_solution);

    double compute_cell_convective_speed(const LinearAlgebra::distributed::Vector<Number> &velocity_solution);

    void apply_forward_euler(
      const double current_time,
      const double time_step,
      const std::vector<LinearAlgebra::distributed::Vector<Number> *> &src,
      LinearAlgebra::distributed::Vector<Number> &dst) const;

    void perform_stage(
      const double time_step,
      const double bi,
      const double ai,
      const LinearAlgebra::distributed::Vector<Number> levelset_old_solution,
      const std::vector<LinearAlgebra::distributed::Vector<Number> *> &src,
      LinearAlgebra::distributed::Vector<Number> &dst) const;      

    LinearAlgebra::distributed::Vector<Number> apply(
      const double current_time,
      const std::vector<LinearAlgebra::distributed::Vector<Number> *> &src,
      LinearAlgebra::distributed::Vector<Number> &dst) const;

    void perform_stage(const LinearAlgebra::distributed::Vector<double> &old_levelset_solution,
                              const LinearAlgebra::distributed::Vector<double> &velocity,
                              LinearAlgebra::distributed::Vector<double> &levelset_solution);

    void
    perform_stage(const Number cur_time,
                  const Number factor_solution,
                  const Number factor_ai,
                  const LinearAlgebra::distributed::Vector<Number> &current_ri,
                  LinearAlgebra::distributed::Vector<Number> &      vec_ki,
                  LinearAlgebra::distributed::Vector<Number> &      solution,
                  LinearAlgebra::distributed::Vector<Number> &next_ri) const;


    double compute_maximal_speed(
      const LinearAlgebra::distributed::Vector<Number> &solution) const;


  private:
    MatrixFree<dim, Number> data;

    TimerOutput &timer;

    std::map<types::boundary_id, std::unique_ptr<Function<dim>>>
      inflow_boundaries;
    std::map<types::boundary_id, std::unique_ptr<Function<dim>>>
                                   subsonic_outflow_boundaries;
    std::set<types::boundary_id>   wall_boundaries;
    std::unique_ptr<Function<dim>> body_force;

    VectorizedArray<Number> lambda;

    void local_apply_inverse_mass_matrix(
      const MatrixFree<dim, Number> &                   data,
      LinearAlgebra::distributed::Vector<Number> &      dst,
      const LinearAlgebra::distributed::Vector<double>  &src,
      const std::pair<unsigned int, unsigned int> &     cell_range) const;

    void local_apply_cell(
      const MatrixFree<dim, Number> &                                  data,
      LinearAlgebra::distributed::Vector<Number> &                     dst,
      const std::vector<LinearAlgebra::distributed::Vector<double> *> &src,
      const std::pair<unsigned int, unsigned int> &                    cell_range) const;

    void local_apply_face(
      const MatrixFree<dim, Number> &                   data,
      LinearAlgebra::distributed::Vector<Number> &      dst,
      const std::vector<LinearAlgebra::distributed::Vector<double> *> &src,
      const std::pair<unsigned int, unsigned int> &     cell_range) const;

    void local_apply_boundary_face(
      const MatrixFree<dim, Number> &                   data,
      LinearAlgebra::distributed::Vector<Number> &      dst,
      const std::vector<LinearAlgebra::distributed::Vector<double> *> &src,
      const std::pair<unsigned int, unsigned int> &     cell_range) const;
  };



  template <int dim, int degree, int velocity_degree, int n_points_1d>
  LevelsetOperator<dim, degree, velocity_degree, n_points_1d>::LevelsetOperator(TimerOutput &timer)
    : timer(timer)
  {}

  template <int dim, int degree, int velocity_degree, int n_points_1d>
  void LevelsetOperator<dim, degree, velocity_degree, n_points_1d>::
      get_lambda(const double lambda)
  {
    this->lambda = make_vectorized_array(lambda);
  }

  template <int dim, int degree, int velocity_degree, int n_points_1d>
  void LevelsetOperator<dim, degree, velocity_degree, n_points_1d>::
      update_global_lambda(const LinearAlgebra::distributed::Vector<Number> &velocity_solution)
  {
    const double max_speed = compute_maximal_speed(velocity_solution);
    this->lambda = make_vectorized_array(max_speed);
  }  

  template <int dim, int degree, int velocity_degree, int n_points_1d>
  double LevelsetOperator<dim, degree, velocity_degree, n_points_1d>::
      compute_cell_convective_speed(const LinearAlgebra::distributed::Vector<Number> &velocity_solution)
  {
    TimerOutput::Scope t(timer, "compute time step");
    Number             max_transport = 0; // v/h
    FEEvaluation<dim, velocity_degree, degree + 1, dim, Number> vel_phi(data, 1, 1);

    for (unsigned int cell = 0; cell < data.n_cell_batches(); ++cell)
      {
        // std::cout<< "cell batch: "<< cell <<std::endl;
        vel_phi.reinit(cell);
        
        vel_phi.gather_evaluate(velocity_solution, EvaluationFlags::values);
        VectorizedArray<Number> local_max = 0.;
        for (unsigned int q = 0; q < vel_phi.n_q_points; ++q)
          {
            const auto velocity = vel_phi.get_value(q);
            const auto inverse_jacobian = vel_phi.inverse_jacobian(q);
            const auto convective_speed = inverse_jacobian * velocity;
            VectorizedArray<Number> convective_limit = 0.;
            for (unsigned int d = 0; d < dim; ++d)
              convective_limit =
                  std::max(convective_limit, std::abs(convective_speed[d]));
            local_max = std::max(local_max, convective_limit);
          }

        for (unsigned int v = 0; v < data.n_active_entries_per_cell_batch(cell);
             ++v)
          {
            max_transport = std::max(max_transport, local_max[v]);
            // std::cout<< " v: "<<v<< " max speed: "<<max_speed;
          }
          // std::cout<<std::endl;
      }

    max_transport = Utilities::MPI::max(max_transport, MPI_COMM_WORLD);

    return max_transport;

  }

  template <int dim, int degree, int velocity_degree, int n_points_1d>
  double LevelsetOperator<dim, degree, velocity_degree, n_points_1d>::compute_maximal_speed(
    const LinearAlgebra::distributed::Vector<Number> &solution) const
  {
    TimerOutput::Scope t(timer, "compute max speed");
    Number             max_speed = 0;
    FEEvaluation<dim, velocity_degree, degree + 1, dim, Number> vel_phi(data, 1, 1);

    for (unsigned int cell = 0; cell < data.n_cell_batches(); ++cell)
      {
        // std::cout<< "cell batch: "<< cell <<std::endl;
        vel_phi.reinit(cell);
        vel_phi.gather_evaluate(solution, EvaluationFlags::values);
        VectorizedArray<Number> local_max = 0.;
        for (unsigned int q = 0; q < vel_phi.n_q_points; ++q)
          {
            const auto velocity = vel_phi.get_value(q);
            local_max =
              std::max(local_max,
                       velocity*velocity);
          }

        for (unsigned int v = 0; v < data.n_active_entries_per_cell_batch(cell);
             ++v)
          {
            max_speed = std::max(max_speed, local_max[v]);
            // std::cout<< " v: "<<v<< " max speed: "<<max_speed;
          }
          // std::cout<<std::endl;
      }

    max_speed = Utilities::MPI::max(max_speed, MPI_COMM_WORLD);

    return std::sqrt(max_speed);
  }

  template <int dim, int degree, int velocity_degree, int n_points_1d>
  void LevelsetOperator<dim, degree, velocity_degree, n_points_1d>::reinit(
    const Mapping<dim> &                                 mapping,
    const std::vector<const DoFHandler<dim> *>           dof_handlers,
    const std::vector<const AffineConstraints<double> *> constraints,
    const std::vector<Quadrature<1>>                     quadratures)
  {
    typename MatrixFree<dim, Number>::AdditionalData additional_data;
    additional_data.mapping_update_flags =
      (update_gradients | update_JxW_values | update_quadrature_points |
       update_values);
    additional_data.mapping_update_flags_inner_faces =
      (update_JxW_values | update_quadrature_points | update_normal_vectors |
       update_values);
    additional_data.mapping_update_flags_boundary_faces =
      (update_JxW_values | update_quadrature_points | update_normal_vectors |
       update_values);
    additional_data.tasks_parallel_scheme =
      MatrixFree<dim, Number>::AdditionalData::none;

    data.reinit(
      mapping, dof_handlers, constraints, quadratures, additional_data);    
  }

  template <int dim, int degree, int velocity_degree, int n_points_1d>
  void LevelsetOperator<dim, degree, velocity_degree, n_points_1d>::
  initialize_vector(
    LinearAlgebra::distributed::Vector<Number> &vector,
    const unsigned int no_dof) const
  {
    data.initialize_dof_vector(vector, no_dof);
  }

  template <int dim, int degree, int velocity_degree, int n_points_1d>
  void
  LevelsetOperator<dim, degree, velocity_degree, n_points_1d>::local_apply_cell(
    const MatrixFree<dim, Number> &                                  data,
    LinearAlgebra::distributed::Vector<Number> &                     dst,
    const std::vector<LinearAlgebra::distributed::Vector<double> *> &src,
    const std::pair<unsigned int, unsigned int> &cell_range) const
  {
// FEEvaluation< dim, fe_degree, n_q_points_1d, n_components_, Number, VectorizedArrayType >::FEEvaluation 	( 	
//    const MatrixFree< dim, Number, VectorizedArrayType > &  	matrix_free,
// 		const unsigned int  	dof_no = 0,
// 		const unsigned int  	quad_no = 0,
// 		const unsigned int  	first_selected_component = 0 
//       std::vector<LinearAlgebra::distributed::Vector<double> *>
//        src({&old_solution, &old_old_solution});
// 	) 
    FEEvaluation<dim, degree, n_points_1d, 1, Number> phi(data, 0);
    FEEvaluation<dim, velocity_degree, n_points_1d, dim, Number> vel_phi(data, 1);

    for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
      {
        phi.reinit(cell);
        phi.gather_evaluate(*src[0], EvaluationFlags::values);

        vel_phi.reinit(cell);
        vel_phi.gather_evaluate(*src[1], EvaluationFlags::values);

        for (unsigned int q = 0; q < phi.n_q_points; ++q)
          {
            const auto w_q = phi.get_value(q);
            const auto vel_q = vel_phi.get_value(q);
            phi.submit_gradient(levelset_flux(w_q, vel_q), q);
          }
        phi.integrate_scatter(EvaluationFlags::gradients,
                              dst);
      }    
  }


  template <int dim, int degree, int velocity_degree, int n_points_1d>
  void LevelsetOperator<dim, degree, velocity_degree, n_points_1d>::local_apply_face(
    const MatrixFree<dim, Number> &,
    LinearAlgebra::distributed::Vector<Number> &      dst,
    const std::vector<LinearAlgebra::distributed::Vector<Number>*> &src,
    const std::pair<unsigned int, unsigned int> &     face_range) const
  {
  //   FEFaceEvaluation 	( 	const MatrixFree< dim, Number, VectorizedArrayType > &  	matrix_free,
	// 	const bool  	is_interior_face = true,
	// 	const unsigned int  	dof_no = 0,
	// 	const unsigned int  	quad_no = 0,
	// 	const unsigned int  	first_selected_component = 0 
	// ) 		
    FEFaceEvaluation<dim, degree, n_points_1d, 1, Number>     phi_m(data, true, 0);
    FEFaceEvaluation<dim, degree, n_points_1d, 1, Number>     phi_p(data, false, 0);
    FEFaceEvaluation<dim, velocity_degree, n_points_1d, dim, Number> vel_phi(data, true, 1);                                                                

    for (unsigned int face = face_range.first; face < face_range.second; ++face)
      {
        phi_p.reinit(face);
        phi_p.gather_evaluate(*src[0], EvaluationFlags::values);

        phi_m.reinit(face);
        phi_m.gather_evaluate(*src[0], EvaluationFlags::values);

        vel_phi.reinit(face);
        vel_phi.gather_evaluate(*src[1], EvaluationFlags::values);
        //compute local lambda
        VectorizedArray<Number> lambda = 0.;
        for (unsigned int q = 0; q < phi_m.n_q_points; ++q)
        {
          const auto velq = vel_phi.get_value(q);
          lambda = std::max(lambda, velq.norm());
        }

        for (unsigned int q = 0; q < phi_m.n_q_points; ++q)
        {
          const auto numerical_flux =
              levelset_numerical_flux<dim>(phi_m.get_value(q),
                                           phi_p.get_value(q),
                                           phi_m.get_normal_vector(q),
                                           vel_phi.get_value(q),
                                           lambda);
          phi_m.submit_value(-numerical_flux, q);
          phi_p.submit_value(numerical_flux, q);
          }

        phi_p.integrate_scatter(EvaluationFlags::values, dst);
        phi_m.integrate_scatter(EvaluationFlags::values, dst);
      }
  }

  template <int dim, int degree, int velocity_degree, int n_points_1d>
  void LevelsetOperator<dim, degree, velocity_degree, n_points_1d>::
    local_apply_boundary_face(
      const MatrixFree<dim, Number> &,
      LinearAlgebra::distributed::Vector<Number> &                     dst,
      const std::vector<LinearAlgebra::distributed::Vector<Number> *> &src,
      const std::pair<unsigned int, unsigned int> &face_range) const
  {
    FEFaceEvaluation<dim, degree, n_points_1d, 1, Number> phi(data, true, 0);
    FEFaceEvaluation<dim, velocity_degree, n_points_1d, dim , Number> vel_phi(data, true, 1);

    for (unsigned int face = face_range.first; face < face_range.second; ++face)
      {
        phi.reinit(face);
        phi.gather_evaluate(*src[0], EvaluationFlags::values);

        vel_phi.reinit(face);
        vel_phi.gather_evaluate(*src[1], EvaluationFlags::values);

        //compute local lambda
        VectorizedArray<Number> lambda = 0.;
        for (unsigned int q = 0; q < phi.n_q_points; ++q)
        {
          const auto velq = vel_phi.get_value(q);
          lambda = std::max(lambda, velq.norm());
        }

        for (unsigned int q = 0; q < phi.n_q_points; ++q)
          {
            const auto w_m    = phi.get_value(q);
            const auto normal = phi.get_normal_vector(q);
            const auto vel_q  = vel_phi.get_value(q);
            auto flux = levelset_numerical_flux<dim>(w_m, w_m, normal, vel_q, lambda);
            phi.submit_value(-flux, q);
          }

        phi.integrate_scatter(EvaluationFlags::values, dst);
      }
  }

 template <int dim, int degree, int velocity_degree, int n_points_1d>
  void LevelsetOperator<dim, degree, velocity_degree, n_points_1d>::local_apply_inverse_mass_matrix(
    const MatrixFree<dim, Number> &,
    LinearAlgebra::distributed::Vector<Number> &      dst,
    const LinearAlgebra::distributed::Vector<Number> &src,
    const std::pair<unsigned int, unsigned int> &     cell_range) const
  {
    FEEvaluation<dim, degree, degree + 1, 1, Number> phi(data, 0, 1);
    // MatrixFreeOperators::CellwiseInverseMassMatrix
    // < dim, fe_degree, n_components, Number, VectorizedArrayType >
    // ::CellwiseInverseMassMatrix 	( 	const FEEvaluationBase< dim, n_components, 
    // Number, false, VectorizedArrayType > &  	fe_eval	) 	
    MatrixFreeOperators::CellwiseInverseMassMatrix<dim, degree, 1, Number>
      inverse(phi);

    for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
      {
        phi.reinit(cell);
        phi.read_dof_values(src);

        inverse.apply(phi.begin_dof_values(), phi.begin_dof_values());

        phi.set_dof_values(dst);
      }
  }


  template <int dim, int degree, int velocity_degree, int n_points_1d>
  void LevelsetOperator<dim, degree, velocity_degree, n_points_1d>::apply_forward_euler(
    const double /*current_time*/,
    const double time_step,
    const std::vector<LinearAlgebra::distributed::Vector<Number> *> &src,
    LinearAlgebra::distributed::Vector<Number> &                     dst) const
  {
    {
      TimerOutput::Scope t(timer, "apply - integrals");

      

      data.loop(&LevelsetOperator::local_apply_cell,
                &LevelsetOperator::local_apply_face,
                &LevelsetOperator::local_apply_boundary_face,
                this,
                dst,
                src,
                true,
                MatrixFree<dim, Number>::DataAccessOnFaces::values,
                MatrixFree<dim, Number>::DataAccessOnFaces::values);
    }

    {
      TimerOutput::Scope t(timer, "apply - inverse mass");

      data.cell_loop(&LevelsetOperator::local_apply_inverse_mass_matrix,
                     this,
                     dst,
                     dst,
                     std::function<void(const unsigned int, const unsigned int)>(),
                     [&](const unsigned int start_range, const unsigned int end_range) 
                     {
                       const Number ts = time_step;
                       DEAL_II_OPENMP_SIMD_PRAGMA
                       for (unsigned int i = start_range; i < end_range; ++i)
                       {
                         const Number old_sol_i = src[0]->local_element(i);
                         const Number sol_i = dst.local_element(i);
                         dst.local_element(i) = old_sol_i + ts * sol_i;
                       }
                     });
    }
  }

  template <int dim, int degree, int velocity_degree, int n_points_1d>
  void LevelsetOperator<dim, degree, velocity_degree, n_points_1d>::perform_stage(
      const double time_step,
      const double bi,
      const double ai,
      const LinearAlgebra::distributed::Vector<Number> levelset_old_solution,
      const std::vector<LinearAlgebra::distributed::Vector<Number> *> &src /*ui and velocity*/,
      LinearAlgebra::distributed::Vector<Number> &dst /*next ui*/) const
  {
    {
      TimerOutput::Scope t(timer, "apply - integrals");
      
      data.loop(&LevelsetOperator::local_apply_cell,
                &LevelsetOperator::local_apply_face,
                &LevelsetOperator::local_apply_boundary_face,
                this,
                dst,
                src,
                true,
                MatrixFree<dim, Number>::DataAccessOnFaces::values,
                MatrixFree<dim, Number>::DataAccessOnFaces::values);
    }
    
    {
      TimerOutput::Scope t(timer, "apply - inverse mass");

      data.cell_loop(&LevelsetOperator::local_apply_inverse_mass_matrix,
                     this,
                     dst,
                     dst,
                     std::function<void(const unsigned int, const unsigned int)>(),
                     [&](const unsigned int start_range, const unsigned int end_range) 
                     {
                       const Number ts = time_step;
                       DEAL_II_OPENMP_SIMD_PRAGMA
                       for (unsigned int i = start_range; i < end_range; ++i)
                       {
                         const Number u_i = dst.local_element(i);
                         const Number old_u_i = src[0]->local_element(i);
                         const Number old_sol_i = levelset_old_solution.local_element(i);
                         dst.local_element(i) = ai * old_sol_i + bi*(old_u_i + ts * u_i);
                       }
                     });
    }
  }


  template <int dim, int degree, int velocity_degree, int n_points_1d>
  LinearAlgebra::distributed::Vector<Number>
   LevelsetOperator<dim, degree, velocity_degree, n_points_1d>::apply(
    const double /*current_time*/,
    const std::vector<LinearAlgebra::distributed::Vector<Number> *> &src,
    LinearAlgebra::distributed::Vector<Number> &                     dst) const
  {
    {
      TimerOutput::Scope t(timer, "apply - integrals");

      

      data.loop(&LevelsetOperator::local_apply_cell,
                &LevelsetOperator::local_apply_face,
                &LevelsetOperator::local_apply_boundary_face,
                this,
                dst,
                src,
                true,
                MatrixFree<dim, Number>::DataAccessOnFaces::values,
                MatrixFree<dim, Number>::DataAccessOnFaces::values);
    }

    {
      TimerOutput::Scope t(timer, "apply - inverse mass");

      data.cell_loop(&LevelsetOperator::local_apply_inverse_mass_matrix,
                     this,
                     dst,
                     dst);
    }
    return dst;
  }

  template <int dim, int degree, int velocity_degree, int n_points_1d>
  void perform_stage(const std::vector<double> &time_steps,
                            const LinearAlgebra::distributed::Vector<double> &old_levelset_solution,
                            const LinearAlgebra::distributed::Vector<double> &velocity,
                            LinearAlgebra::distributed::Vector<double> &levelset_solution)
  {
    // setup TVB3 coefficient:
  	const double a_rk[] = {0.0, 3.0/4.0, 1.0/3.0};
  	const double b_rk[] = {1.0, 1.0/4.0, 2.0/3.0};


  }

    template <int dim>
    class LevelsetProblem
    {
    public:
      LevelsetProblem();

      void run();

    private:
      void make_grid_and_dofs();

      void output_results(const unsigned int result_number);

      LinearAlgebra::distributed::Vector<Number> levelset_solution;
      LinearAlgebra::distributed::Vector<Number> levelset_old_solution;

      LinearAlgebra::distributed::Vector<Number> velocity_solution, velocity_old_solution, velocity_old_old_solution;

      ConditionalOStream pcout;

#ifdef DEAL_II_WITH_P4EST
    parallel::distributed::Triangulation<dim> triangulation;
#else
    Triangulation<dim> triangulation;
#endif

    FE_DGQ<dim>          levelset_fe;
    FESystem<dim>        velocity_fe;
    MappingQGeneric<dim> mapping;
    DoFHandler<dim>      levelset_dof_handler;
    DoFHandler<dim>      velocity_dof_handler;

    TimerOutput timer;

    LevelsetOperator<dim, fe_degree, vel_degree, n_q_points_1d> levelset_operator;

    double time, time_step, old_time_step;

    IndexSet  velocity_locally_relevant_dofs;
    AffineConstraints<double> velocity_constraints;

  };


 template <int dim>
  LevelsetProblem<dim>::LevelsetProblem()
    : pcout(std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
#ifdef DEAL_II_WITH_P4EST
    , triangulation(MPI_COMM_WORLD)
#endif
    , levelset_fe(fe_degree)
    , velocity_fe(FE_Q<dim>(vel_degree), dim)
    , mapping(fe_degree)
    , levelset_dof_handler(triangulation)
    , velocity_dof_handler(triangulation)
    , timer(pcout, TimerOutput::never, TimerOutput::wall_times)
    , levelset_operator(timer)
    , time(0)
    , time_step(0)
  {}

    template <int dim>
  void LevelsetProblem<dim>::make_grid_and_dofs()
  {
    GridGenerator::hyper_cube(triangulation);
    triangulation.refine_global(n_global_refinements);

    levelset_dof_handler.distribute_dofs(levelset_fe);

    velocity_dof_handler.distribute_dofs(velocity_fe);
    DoFTools::extract_locally_relevant_dofs(velocity_dof_handler, velocity_locally_relevant_dofs);
    velocity_constraints.clear();
    velocity_constraints.reinit(velocity_locally_relevant_dofs);
    DoFTools::make_hanging_node_constraints(velocity_dof_handler, velocity_constraints);
    velocity_constraints.close();

    const std::vector<const DoFHandler<dim> *> dof_handlers = {&levelset_dof_handler, &velocity_dof_handler};
    const AffineConstraints<double>            dummy;
    const std::vector<const AffineConstraints<double> *> constraints = {&dummy, &velocity_constraints};
    const std::vector<Quadrature<1>> quadratures = {QGauss<1>(n_q_points_1d),
                                                    QGauss<1>(fe_degree + 1)};

    levelset_operator.reinit(mapping, dof_handlers, constraints, quadratures);
    
    levelset_operator.initialize_vector(levelset_solution, 0);
    levelset_old_solution.reinit(levelset_solution);

    levelset_operator.initialize_vector(velocity_solution, 1);
    velocity_old_solution.reinit(velocity_solution);
    velocity_old_old_solution.reinit(velocity_solution);

    std::locale s = pcout.get_stream().getloc();
    pcout.get_stream().imbue(std::locale(""));
    pcout << "Number of degrees of freedom: " << levelset_dof_handler.n_dofs()
          << " ( = " << 1 << " [vars] x "
          << triangulation.n_global_active_cells() << " [cells] x "
          << Utilities::pow(fe_degree + 1, dim) << " [dofs/cell/var] )"
          << std::endl;
    pcout.get_stream().imbue(s);
  }


template <int dim> void LevelsetProblem<dim>::output_results(const unsigned int result_number) 
{
  DataOut<dim> data_out;
  data_out.attach_dof_handler(levelset_dof_handler);
  data_out.add_data_vector(levelset_solution, "phi");
  Vector<float> subdomain(triangulation.n_active_cells());
  for (unsigned int i = 0; i < subdomain.size(); ++i)
    subdomain(i) = triangulation.locally_owned_subdomain();
  data_out.add_data_vector(subdomain, "subdomain");
  data_out.build_patches();

  data_out.write_vtu_with_pvtu_record("./", "levelset_solution", result_number,
                                      MPI_COMM_WORLD, 2, 8);
  
}

template <int dim>
void LevelsetProblem<dim>::run()
{
  {
    const unsigned int n_vect_number = VectorizedArray<Number>::size();
    const unsigned int n_vect_bits   = 8 * sizeof(Number) * n_vect_number;

    pcout << "Running with " << Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD)
          << " MPI processes" << std::endl;
    pcout << "Vectorization over " << n_vect_number << " "
          << (std::is_same<Number, double>::value ? "doubles" : "floats")
          << " = " << n_vect_bits << " bits ("
          << Utilities::System::get_current_vectorization_level() << ")"
          << std::endl;
  }
  make_grid_and_dofs();
  VectorTools::interpolate(levelset_dof_handler,
                           LevelsetInitialValues<dim>(),
                           levelset_solution);
  levelset_old_solution = levelset_solution;

  VectorTools::interpolate(velocity_dof_handler,
                           LevelsetVelocity<dim>(),
                           velocity_solution);
  velocity_old_old_solution = velocity_solution;
  velocity_old_solution = velocity_solution;

  std::vector<LinearAlgebra::distributed::Vector<double> *>
    levelset_src_vectors({&levelset_old_solution, &velocity_solution});

  output_results(0);

  const double max_speed =
    levelset_operator.compute_maximal_speed(velocity_solution);
  pcout << "max speed is" << max_speed << std::endl;

  double time_step    = 0.0008;
  double current_time = 0.;

  std::vector<LinearAlgebra::distributed::Vector<Number> *> old_sol_and_vel(
    {&levelset_old_solution, &velocity_solution});

  //setup TVB3 coefficient:
  // const double a_rk[] = {0.0, 3.0 / 4.0, 1.0 / 3.0};
  // const double b_rk[] = {1.0, 1.0 / 4.0, 2.0 / 3.0};
  // const unsigned int n_stages = 3;
  LinearAlgebra::distributed::Vector<Number> rk_register;
  rk_register.reinit(levelset_solution);

  std::vector<LinearAlgebra::distributed::Vector<Number> *> ui_and_vel(
      {&rk_register, &velocity_solution});

  levelset_operator.get_lambda(max_speed);
  double old_time_step = time_step;

  std::cout<<"hmin = "<<GridTools::minimal_cell_diameter(triangulation) /std::sqrt (1.*dim)
  <<" max_speed/h = "<<std::sqrt (1.*dim)/max_speed/GridTools::minimal_cell_diameter(triangulation)
  <<" max convective speed = "<<levelset_operator.compute_cell_convective_speed(velocity_solution)
  <<std::endl;

  const LevelsetTimeIntegrator time_integrator(lsrk_scheme);

  for (int step = 0; step < 1000; ++step)
    {
      //compute time step
      time_step = courant_number/levelset_operator.compute_cell_convective_speed(velocity_solution);
#if 0
      {
        //evolve one time step
        for (unsigned int stage = 0; stage < n_stages; ++stage)
        {
          rk_register.swap(levelset_solution);

          if (stage == 1) //n+1
          {
            const double factor = time_step / old_time_step;
            velocity_solution.equ(1. + factor, velocity_old_solution);
            velocity_solution.add(-factor, velocity_old_old_solution);
          }
          else if (stage == 2) //n+1/2
          {
            velocity_solution.sadd(0.5, 0.5, velocity_old_solution);
          }

          // levelset_operator.update_global_lambda(velocity_solution);
          levelset_operator.perform_stage(time_step,
                                          b_rk[stage],
                                          a_rk[stage],
                                          levelset_old_solution,
                                          ui_and_vel,
                                          levelset_solution);
        }
      }
#else
      time_integrator.perform_time_step(levelset_operator,
                                        old_time_step,
                                        time_step,
                                        velocity_old_solution,
                                        velocity_old_old_solution,
                                        ui_and_vel,
                                        levelset_old_solution,
                                        levelset_solution);
#endif
      levelset_old_solution = levelset_solution;
      old_time_step = time_step;

      pcout << " step " << step << " done! "
            << " time is " << current_time 
            << " time_step is " << time_step
            << " sol l2 norm: "<<levelset_solution.l2_norm()
            << std::endl;

      // pcout << " l2 norm " << levelset_solution.l2_norm()
      //       << " l1 norm" << levelset_solution.l1_norm()
      //       << std::endl;
      current_time += time_step;
      if ((step+1) % 100 == 0)
      {
        output_results(step);
      }
    }
  timer.print_wall_time_statistics(MPI_COMM_WORLD);
  pcout << std::endl;
}

  } // namespace Euler_DG

int main(int argc, char **argv)
{
  using namespace LevelsetMatrixfree;
  using namespace dealii;

  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

  try
    {
      deallog.depth_console(0);
      LevelsetProblem<dimension> ls_problem;
      ls_problem.run();
    }
  catch (std::exception &exc)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Exception on processing: " << std::endl
                << exc.what() << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;

      return 1;
    }
  catch (...)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Unknown exception!" << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }

  return 0;
}