/* ---------------------------------------------------------------------
 *
 * Copyright (C) 2008 - 2020 by the deal.II authors
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
 * Authors: Martin Kronbichler, Uppsala University,
 *          Wolfgang Bangerth, Texas A&M University,
 *          Timo Heister, University of Goettingen, 2008-2011
 */

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/function.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/utilities.h>
#include <deal.II/base/work_stream.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/block_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/lapack_full_matrix.h>
#include <deal.II/lac/solver_bicgstab.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/trilinos_block_sparse_matrix.h>
#include <deal.II/lac/trilinos_parallel_block_vector.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_solver.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/la_parallel_block_vector.h>
#include <deal.II/lac/generic_linear_algebra.h>

#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/matrix_free/operators.h>

#include <deal.II/grid/filtered_iterator.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>

#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_dgp.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/solution_transfer.h>
#include <deal.II/numerics/vector_tools.h>

#include <deal.II/fe/fe_interface_values.h>
#include <deal.II/meshworker/mesh_loop.h>

#include <fstream>
#include <iostream>
#include <limits>
#include <locale>
#include <string>

#include <deal.II/distributed/solution_transfer.h>

#include <deal.II/base/index_set.h>
#include <deal.II/distributed/grid_refinement.h>
#include <deal.II/distributed/tria.h>

namespace LevelsetMatrixbase
{

  using namespace dealii;

  constexpr unsigned int dimension = 2;
  constexpr unsigned int n_global_refinements = 6;
  constexpr unsigned int fe_degree = 3;
  constexpr unsigned int vel_degree = 2;
  constexpr unsigned int n_q_points_1d = fe_degree + 2;

  using Number = double;

  const double courant_number = 0.05;

  template <int dim, typename VectorType>
  void get_function_jump(const FEInterfaceValues<dim> &fe_iv,
                         const VectorType &solution, std::vector<double> &jump)
  {
    const unsigned n_q = fe_iv.n_quadrature_points;
    Assert(n_q == jump.size(), ExcInternalError());
    std::vector<double> face_values[2];
    jump.resize(n_q);
    for (unsigned i = 0; i < 2; ++i)
    {
      face_values[i].resize(n_q);
      fe_iv.get_fe_face_values(i).get_function_values(solution, face_values[i]);
    }
    for (unsigned int q = 0; q < n_q; ++q)
      jump[q] = face_values[0][q] - face_values[1][q];
  }

  template <int dim, typename VectorType>
  void get_function_average(const FEInterfaceValues<dim> &fe_iv,
                            const VectorType &solution,
                            std::vector<double> &average)
  {
    const unsigned n_q = fe_iv.n_quadrature_points;
    Assert(n_q == average.size(), ExcInternalError());
    std::vector<double> face_values[2];
    average.resize(n_q);
    for (unsigned i = 0; i < 2; ++i)
    {
      face_values[i].resize(n_q);
      fe_iv.get_fe_face_values(i).get_function_values(solution, face_values[i]);
    }
    for (unsigned int q = 0; q < n_q; ++q)
      average[q] = 0.5 * (face_values[0][q] + face_values[1][q]);
  }

  template <int dim, typename VectorType>
  void get_function_gradient_jump(const FEInterfaceValues<dim> &fe_iv,
                                  const VectorType &solution,
                                  std::vector<Tensor<1, dim>> &gradient_jump)
  {
    const unsigned n_q = fe_iv.n_quadrature_points;
    Assert(n_q == gradient_jump.size(), ExcInternalError());
    std::vector<Tensor<1, dim>> face_gradients[2];
    gradient_jump.resize(n_q);
    for (unsigned i = 0; i < 2; ++i)
    {
      face_gradients[i].resize(n_q);
      fe_iv.get_fe_face_values(i).get_function_gradients(solution,
                                                         face_gradients[i]);
    }
    for (unsigned int q = 0; q < n_q; ++q)
      gradient_jump[q] = face_gradients[0][q] - face_gradients[1][q];
  }

  namespace Assembly
  {
    namespace Scratch
    {
      template <int dim>
      struct LevelsetAssembly
      {
        LevelsetAssembly(
            const FiniteElement<dim> &fe, const QGauss<dim> &quadratureCell,
            const UpdateFlags updateFlagsCell, const QGauss<dim - 1> &quadratureFace,
            const UpdateFlags updateFlagsFace, 
            const FiniteElement<dim> &fe_ns, const UpdateFlags update_flag_ns)
            : fe_values(fe, quadratureCell, updateFlagsCell),
              fe_face_values(fe, quadratureFace, updateFlagsFace),
              fens_values(fe_ns, quadratureCell, update_flag_ns),
              fens_face_values(fe_ns, quadratureFace, update_flag_ns),
              fe_interface_values(fe, quadratureFace, updateFlagsFace) {}

        LevelsetAssembly(const LevelsetAssembly &scratch_data)
            : fe_values(scratch_data.fe_values.get_fe(),
                        scratch_data.fe_values.get_quadrature(),
                        scratch_data.fe_values.get_update_flags()),
              fe_face_values(scratch_data.fe_face_values.get_fe(),
                             scratch_data.fe_face_values.get_quadrature(),
                             scratch_data.fe_face_values.get_update_flags()),
              fens_values(scratch_data.fens_values.get_fe(),
                          scratch_data.fens_values.get_quadrature(),
                          scratch_data.fens_values.get_update_flags()),
              fens_face_values(scratch_data.fens_face_values.get_fe(),
                               scratch_data.fens_face_values.get_quadrature(),
                               scratch_data.fens_face_values.get_update_flags()),
              fe_interface_values(
                  scratch_data.fe_values.get_fe(),
                  scratch_data.fe_interface_values.get_quadrature(),
                  scratch_data.fe_interface_values.get_update_flags()) {}

        FEValues<dim> fe_values;
        FEFaceValues<dim> fe_face_values;
        FEValues<dim> fens_values;
        FEFaceValues<dim> fens_face_values;
        FEInterfaceValues<dim> fe_interface_values;
      };

    } // namespace Scratch

    namespace CopyData
    {
      template <int dim>
      struct LevelsetAssembly
      {
        Vector<double> cell_rhs;
        Vector<double> cell_sol;
        std::vector<types::global_dof_index> local_dof_indices;
        bool is_reinit;

        LevelsetAssembly(const FiniteElement<dim> &fe)
            : cell_rhs(fe.dofs_per_cell),
              cell_sol(fe.dofs_per_cell),
              local_dof_indices(fe.dofs_per_cell), is_reinit(false) {}
      };
    } // namespace CopyData
  }   // namespace Assembly

  enum LevelsetRungeKuttaScheme
  {
    stage_3_order_3,
    stage_2_order_2,
  };
  constexpr LevelsetRungeKuttaScheme lsrk_scheme = stage_3_order_3;

  class LevelsetTimeIntegrator
  {
  public:
    LevelsetTimeIntegrator(const LevelsetRungeKuttaScheme scheme)
    {
      switch (scheme)
      {
      case stage_3_order_3:
      {
        bi = {{1.0, 1.0 / 4.0, 2.0 / 3.0}};
        ai = {{0., 3. / 4., 1. / 3.}};

        break;
      }

      case stage_2_order_2:
      {
        bi = {{1., 1. / 2}};
        ai = {{0., 1. / 2.}};

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

      for (unsigned int stage = 0; stage < bi.size(); ++stage)
      {
        ui_and_velocity[0]->swap(levelset_solution);

        if (stage == 1) // n+1
        {
          const double time_step_factor = time_step / old_time_step;
          ui_and_velocity[1]->equ(1. + time_step_factor,
                                  velocity_old_solution);
          ui_and_velocity[1]->add(-time_step_factor,
                                  velocity_old_old_solution);
        }
        else if (stage == 2) // n+1/2
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

  template <int dim>
  class LevelsetVelocity : public Function<dim>
  {
  public:
    LevelsetVelocity()
        : Function<dim>(dim)
    {
    }
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

    if (dim == 2)
    {
      if (component == 0)
        return -std::sin(2. * pi * y) * std::sin(pi * x) * std::sin(pi * x);
      else
        return std::sin(2. * pi * x) * std::sin(pi * y) * std::sin(pi * y);
    }
    else /*(dim == 3)*/
    {
      const double z = point[2];
      if (component == 0)
        return 2. * std::sin(pi * x) * std::sin(pi * x) *
               std::sin(2. * pi * y) * std::sin(2. * pi * z);
      else if (component == 1)
        return -std::sin(2. * pi * x) * std::sin(pi * y) * std::sin(pi * y) *
               std::sin(2. * pi * z);
      else
        return -std::sin(2. * pi * x) * std::sin(2. * pi * y) *
               std::sin(pi * z);
    }
  }

  template <int dim>
  class LevelsetInitialValues : public Function<dim>
  {
  public:
    LevelsetInitialValues()
        : Function<dim>(1)
    {
    }

    virtual double value(const Point<dim> &p,
                         const unsigned int component = 0) const override;
  };

  template <int dim>
  double LevelsetInitialValues<dim>::value(const Point<dim> &p,
                                           const unsigned int) const
  {
    Assert(dim >= 2, ExcNotImplemented());
    Point<dim> center;
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
  inline DEAL_II_ALWAYS_INLINE Tensor<1, dim, Number>
  levelset_flux(const Number &phi, const Tensor<1, dim, Number> &velocity)
  {
    Tensor<1, dim, Number> flux(velocity);
    flux *= phi;

    return flux;
  }

  template <int dim, typename Number>
  inline DEAL_II_ALWAYS_INLINE //
      Number
      levelset_numerical_flux(const Number &u_m,
                              const Number &u_p,
                              const Tensor<1, dim, Number> &normal,
                              const Tensor<1, dim, Number> &velocity,
                              const Number &lambda)
  {
    return 0.5 * (u_m + u_p) * (velocity * normal) + 0.5 * lambda * (u_m - u_p);
  }

  template <int dim, typename Number>
  VectorizedArray<Number>
  evaluate_velocity(const Function<dim> &function,
                    const Point<dim, VectorizedArray<Number>> &p_vectorized,
                    const unsigned int component)
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

  template <int dim, typename Number, int n_components = dim>
  Tensor<1, n_components, VectorizedArray<Number>>
  evaluate_velocity(const Function<dim> &function,
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

  template <int dim>
  class LevelsetOperator
  {
  public:
    LevelsetOperator(TimerOutput &timer_output);

    void
    reinit(const Mapping<dim> &mapping,
           const std::vector<const DoFHandler<dim> *> &dof_handlers,
           const std::vector<const AffineConstraints<double> *> constraints,
           const std::vector<Quadrature<1>> quadratures);

  private:
    TimerOutput &timer;
  };

  template <int dim>
  LevelsetOperator<dim>::LevelsetOperator(
      TimerOutput &timer)
      : timer(timer)
  {
  }

  template <int dim>
  class LevelsetProblem
  {
  public:
    LevelsetProblem();

    void run_levelset();

  private:
    void make_grid_and_dofs();

    void output_results(const unsigned int result_number);

    void assemble_levelset(const double global_max_velocity,
                           const double time_step);

    double get_maximal_velocity() const;

    double compute_cell_convective_speed(
        const TrilinosWrappers::MPI::Vector &velocity_solution);

    TrilinosWrappers::MPI::Vector levelset_solution;
    TrilinosWrappers::MPI::Vector levelset_old_solution;

    TrilinosWrappers::MPI::Vector velocity_solution,
        velocity_old_solution, velocity_old_old_solution;

    ConditionalOStream pcout;

    parallel::distributed::Triangulation<dim> triangulation;

    FE_DGQ<dim> levelset_fe;
    FESystem<dim> velocity_fe;
    MappingQGeneric<dim> mapping;
    DoFHandler<dim> levelset_dof_handler;
    DoFHandler<dim> velocity_dof_handler;

    TimerOutput timer;

    IndexSet levelset_partitioning, levelset_relevant_partitioning;
    IndexSet velocity_partitioning, velocity_locally_relevant_dofs;
    AffineConstraints<double> velocity_constraints;

    LevelsetOperator<dim> levelset_operator;
    double time, time_step, old_time_step;
    double global_max_velocity;
  };

  template <int dim>
  LevelsetProblem<dim>::LevelsetProblem()
      : pcout(std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0), triangulation(MPI_COMM_WORLD), levelset_fe(fe_degree), velocity_fe(FE_Q<dim>(vel_degree), dim), mapping(fe_degree), levelset_dof_handler(triangulation), velocity_dof_handler(triangulation), timer(pcout, TimerOutput::never, TimerOutput::wall_times), levelset_operator(timer), time(0), time_step(0)
  {
  }

  template <int dim>
  void LevelsetProblem<dim>::make_grid_and_dofs()
  {
    {
      //make grid
      GridGenerator::hyper_cube(triangulation);
      triangulation.refine_global(n_global_refinements);
    }
    {
      //levelset dofs
      levelset_dof_handler.distribute_dofs(levelset_fe);
      levelset_partitioning = levelset_dof_handler.locally_owned_dofs();
      DoFTools::extract_locally_relevant_dofs(levelset_dof_handler,
                                              levelset_relevant_partitioning);
      levelset_solution.reinit(levelset_partitioning,
                               levelset_relevant_partitioning, MPI_COMM_WORLD,
                               true);
      levelset_old_solution.reinit(levelset_relevant_partitioning, MPI_COMM_WORLD);
    }
    {
      //velocity dofs and constraints
      velocity_dof_handler.distribute_dofs(velocity_fe);
      velocity_partitioning = velocity_dof_handler.locally_owned_dofs();
      DoFTools::extract_locally_relevant_dofs(velocity_dof_handler,
                                              velocity_locally_relevant_dofs);
      velocity_solution.reinit(velocity_locally_relevant_dofs, MPI_COMM_WORLD);
      velocity_old_solution.reinit(velocity_solution);
      velocity_old_old_solution.reinit(velocity_solution);

      velocity_constraints.clear();
      velocity_constraints.reinit(velocity_locally_relevant_dofs);
      DoFTools::make_hanging_node_constraints(velocity_dof_handler,
                                              velocity_constraints);
      velocity_constraints.close();
    }

    std::locale s = pcout.get_stream().getloc();
    pcout.get_stream().imbue(std::locale(""));
    pcout << "Number of degrees of freedom: " << levelset_dof_handler.n_dofs()
          << " ( = " << 1 << " [vars] x "
          << triangulation.n_global_active_cells() << " [cells] x "
          << Utilities::pow(fe_degree + 1, dim) << " [dofs/cell/var] )"
          << std::endl;
    pcout.get_stream().imbue(s);
  }

  template <int dim>
  void LevelsetProblem<dim>::output_results(const unsigned int result_number)
  {
    DataOut<dim> data_out;
    data_out.attach_dof_handler(levelset_dof_handler);
    data_out.add_data_vector(levelset_solution, "phi");
    Vector<float> subdomain(triangulation.n_active_cells());
    for (unsigned int i = 0; i < subdomain.size(); ++i)
      subdomain(i) = triangulation.locally_owned_subdomain();
    data_out.add_data_vector(subdomain, "subdomain");
    data_out.build_patches();

    data_out.write_vtu_with_pvtu_record(
        "./", "levelset_matrixbase", result_number, MPI_COMM_WORLD, 2, 8);
  }

  template <int dim>
  void LevelsetProblem<dim>::assemble_levelset(
      const double global_max_velocity, const double time_step)
  {
    typedef decltype(levelset_dof_handler.begin_active()) Iterator;
    const FEValuesExtractors::Vector velocities(0);
    const auto cell_worker =
        [&](const Iterator &cell,
            Assembly::Scratch::LevelsetAssembly<dim> &scratch_data,
            Assembly::CopyData::LevelsetAssembly<dim> &copy_data) {
          FEValues<dim> &levelset_fe_values = scratch_data.fe_values;
          levelset_fe_values.reinit(cell);
          const unsigned int dofs_per_cell = levelset_fe_values.dofs_per_cell;

          const unsigned int n_q_points = levelset_fe_values.n_quadrature_points;
          const std::vector<double> &JxW = levelset_fe_values.get_JxW_values();

          std::vector<double> old_ls_sol(n_q_points);
          levelset_fe_values.get_function_values(levelset_old_solution, old_ls_sol);

          std::vector<Tensor<1, dim>> velocity_values(n_q_points);
          typename DoFHandler<dim>::active_cell_iterator velocity_cell(
              &triangulation, cell->level(), cell->index(), &velocity_dof_handler);
          FEValues<dim> &velocity_fe_values = scratch_data.fens_values;
          velocity_fe_values.reinit(velocity_cell);
          velocity_fe_values[velocities].get_function_values(velocity_solution,
                                                             velocity_values);
          LAPACKFullMatrix<double> local_matrix(dofs_per_cell, dofs_per_cell);
          for (unsigned int point = 0; point < n_q_points; ++point)
          {
            for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
              for (unsigned int j = 0; j < dofs_per_cell; ++j)
                local_matrix(i, j) +=
                    levelset_fe_values.shape_value(i, point) *             // phi_i
                    levelset_fe_values.shape_value(j, point) * JxW[point]; // phi_j

              copy_data.cell_rhs(i) +=
                  old_ls_sol[point] * // old_phi
                  (velocity_values[point] *
                   levelset_fe_values.shape_grad(i, point)) * // vel grad phi_i
                  JxW[point];                                 // dx
            }
          }
#if 0
      std::cout << "cell: " << cell->active_cell_index() << " \n mat: " << std::endl;
      std::cout << "mesh loop dofs: " << dofs_per_cell << std::endl;
      {
        std::cout << "rhs: " << std::endl;
        for (unsigned int i = 0; i < dofs_per_cell; ++i)
          std::cout << " " << copy_data.cell_rhs(i) << std::endl;
        std::cout << std::endl;
      }
#endif

          cell->get_dof_indices(copy_data.local_dof_indices);
          local_matrix.invert();
          local_matrix.vmult(copy_data.cell_sol, copy_data.cell_rhs);
          copy_data.is_reinit = false;
        };

    auto boundary_worker =
        [&](const Iterator &cell, const unsigned int &face_no,
            Assembly::Scratch::LevelsetAssembly<dim> &scratch_data,
            Assembly::CopyData::LevelsetAssembly<dim> &copy_data) {
          if (!copy_data.is_reinit)
          {
            copy_data.cell_rhs = 0.;
            copy_data.cell_sol = 0.;
            copy_data.is_reinit = true;
          }
          FEFaceValues<dim> &levelset_fe_values = scratch_data.fe_face_values;
          levelset_fe_values.reinit(cell, face_no);

          const unsigned int n_q_points = levelset_fe_values.n_quadrature_points;
          const unsigned int dofs_per_cell = levelset_fe_values.dofs_per_cell;

          const std::vector<double> &JxW = levelset_fe_values.get_JxW_values();
          const std::vector<Tensor<1, dim>> &normals =
              levelset_fe_values.get_normal_vectors();

          std::vector<double> old_ls_sol(n_q_points);
          levelset_fe_values.get_function_values(levelset_old_solution, old_ls_sol);

          std::vector<Tensor<1, dim>> velocity_values(n_q_points);
          typename DoFHandler<dim>::active_cell_iterator velocity_cell(
              &triangulation, cell->level(), cell->index(), &velocity_dof_handler);
          FEFaceValues<dim> &velocity_fe_values = scratch_data.fens_face_values;
          velocity_fe_values.reinit(velocity_cell, face_no);
          velocity_fe_values[velocities].get_function_values(velocity_solution,
                                                             velocity_values);

          for (unsigned int point = 0; point < n_q_points; ++point)
          {
#if 0
        std::cout << " vel: " << velocity_values[point] << std::endl
                  << " old sol: " << old_ls_sol[point] << std::endl;
#endif
            const double normal_flux =
                velocity_values[point] * normals[point]; // vel . n
            for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
              copy_data.cell_rhs(i) -= // -
                  normal_flux * old_ls_sol[point] *
                  levelset_fe_values.shape_value(i, point) * // phi_i
                  JxW[point];                                // x
#if 0
        std::cout << " normal flux: " << normal_flux << std::endl
                  << " old sol:     " << old_ls_sol[point] << std::endl
                  << " shape val:   "<< levelset_fe_values.shape_value(i, point)<<std::endl
                  << " jxw :        " <<JxW[point]<<std::endl;
#endif
            }
          }
#if 0
      std::cout << "boundary face: " << face_no << std::endl;
      {
        std::cout << "rhs: " << std::endl;
        for (unsigned int i = 0; i < dofs_per_cell; ++i)
          std::cout << " " << copy_data.cell_rhs(i) << std::endl;
        std::cout << std::endl;
      }
#endif
        };

    auto face_worker = [&](const Iterator &cell, const unsigned int &f,
                           const unsigned int &sf, const Iterator &ncell,
                           const unsigned int &nf, const unsigned int &nsf,
                           Assembly::Scratch::LevelsetAssembly<dim> &scratch_data,
                           Assembly::CopyData::LevelsetAssembly<dim> &copy_data) {
      if (!copy_data.is_reinit)
      {
        copy_data.cell_rhs = 0.;
        copy_data.cell_sol = 0.;
        copy_data.is_reinit = true;
      }
      FEInterfaceValues<dim> &fe_interface_values =
          scratch_data.fe_interface_values;
      fe_interface_values.reinit(cell, f, sf, ncell, nf, nsf);

      const unsigned int n_q_points = fe_interface_values.n_quadrature_points;
      const std::vector<double> &JxW = fe_interface_values.get_JxW_values();
      const std::vector<Tensor<1, dim>> &normals =
          fe_interface_values.get_normal_vectors();

      std::vector<double> jump(n_q_points);
      get_function_jump(fe_interface_values, levelset_old_solution, jump);

      std::vector<double> average(n_q_points);
      get_function_average(fe_interface_values, levelset_old_solution, average);

      std::vector<Tensor<1, dim>> velocity_values(n_q_points);
      typename DoFHandler<dim>::active_cell_iterator velocity_cell(
          &triangulation, cell->level(), cell->index(), &velocity_dof_handler);
      FEFaceValues<dim> &velocity_fe_values = scratch_data.fens_face_values;
      velocity_fe_values.reinit(velocity_cell, f);
      velocity_fe_values[velocities].get_function_values(velocity_solution, velocity_values);

      const unsigned int dofs_per_cell =
          fe_interface_values.get_fe_face_values(0).dofs_per_cell;

      for (unsigned int point = 0; point < n_q_points; ++point)
      {
#if 0
        std::cout<<" vel: "<< velocity_values[point]<<std::endl
        <<" jump: "<< jump[point]<<std::endl
        <<" average: "<< average[point]
        <<std::endl;
#endif
        const double normal_flux = velocity_values[point] * normals[point]; //
        const double lf_flux =
            average[point] * normal_flux +           // {old_phi} (vel . n)
            0.5 * global_max_velocity * jump[point]; // + 0.5 lambda [old_phi]
        for (unsigned int i = 0; i < dofs_per_cell; ++i)
        {
          copy_data.cell_rhs(i) -=
              lf_flux *
              fe_interface_values.get_fe_face_values(0).shape_value(
                  i, point) * //    phi_i
              JxW[point];     //    dx
#if 0
        std::cout << " lf flux:   " << lf_flux << std::endl
                  << " tshape val: "<< fe_interface_values.shape_value(true, i, point)
                  << " fshape val: "<< fe_interface_values.shape_value(f, i, point)
                  << " 0shape val: "<<fe_interface_values.get_fe_face_values(0).shape_value(i,point)
                  <<std::endl
                  << " jxw :      " <<JxW[point]<<std::endl;
#endif
        }
      }
#if 0
      std::cout << "face: " << f <<std::endl;
      {
        std::cout << "rhs: " << std::endl;
        for (unsigned int i = 0; i < dofs_per_cell; ++i)
          std::cout << " " << copy_data.cell_rhs(i) << std::endl;
        std::cout << std::endl;
      }
#endif
    };

    auto copier =
        [&](const Assembly::CopyData::LevelsetAssembly<dim> &copy_data) {
          const uint dofs_per_cell = copy_data.cell_sol.size();
          for (unsigned int i = 0; i < dofs_per_cell; ++i)
          {
            levelset_solution[copy_data.local_dof_indices[i]] +=
                time_step * copy_data.cell_sol(i);
          }
        };

    QGauss<dim> quadrature_formula(levelset_fe.get_degree() + 2);
    QGauss<dim - 1> face_quadrature_formula(levelset_fe.get_degree() + 2);
    const UpdateFlags updateFlagsCell(update_values | update_gradients |
                                      update_quadrature_points |
                                      update_JxW_values);
    const UpdateFlags updateFlagsFace(update_values | update_normal_vectors |
                                      update_quadrature_points |
                                      update_JxW_values);
    
    const UpdateFlags updateFlagsNS(update_values);

    Assembly::CopyData::LevelsetAssembly<dim> copy_data(levelset_fe);
    Assembly::Scratch::LevelsetAssembly<dim> scratch_data(
        levelset_fe, quadrature_formula, updateFlagsCell, face_quadrature_formula,
        updateFlagsFace, velocity_fe, updateFlagsNS);

    MeshWorker::mesh_loop(
        levelset_dof_handler.begin_active(), levelset_dof_handler.end(),
        cell_worker, copier, scratch_data, copy_data,
        MeshWorker::assemble_own_cells | MeshWorker::assemble_boundary_faces |
            MeshWorker::assemble_ghost_faces_both |
            MeshWorker::assemble_own_interior_faces_both |
            MeshWorker::cells_after_faces,
        boundary_worker, face_worker);
    pcout << " done assembling levelset " << std::endl;
  }

  template <int dim>
  double LevelsetProblem<dim>::get_maximal_velocity() const
  {
    const QIterated<dim> quadrature_formula(QTrapez<1>(),
                                            levelset_fe.get_degree() + 2);
    const unsigned int n_q_points = quadrature_formula.size();

    FEValues<dim> fe_values(mapping, velocity_fe, quadrature_formula,
                            update_values);
    std::vector<Tensor<1, dim>> velocity_values(n_q_points);

    double max_local_velocity = 0;
    const FEValuesExtractors::Vector velocities(0);
    for (const auto &cell : velocity_dof_handler.active_cell_iterators())
      if (cell->is_locally_owned())
      {
        fe_values.reinit(cell);
        fe_values[velocities].get_function_values(velocity_solution,
                                                  velocity_values);

        for (unsigned int q = 0; q < n_q_points; ++q)
          max_local_velocity =
              std::max(max_local_velocity, velocity_values[q].norm());
      }

    return Utilities::MPI::max(max_local_velocity, MPI_COMM_WORLD);
  }

  template <int dim>
  double LevelsetProblem<dim>::
  compute_cell_convective_speed(const TrilinosWrappers::MPI::Vector &velocity_solution)
  {
    const QGauss<dim> quadrature_formula(fe_degree + 1);
    const unsigned int n_q_points = quadrature_formula.size();

    FEValues<dim> fe_values(mapping, velocity_fe, quadrature_formula,
                            update_values | update_inverse_jacobians);
    std::vector<Tensor<1, dim>> velocity_values(n_q_points);

    double max_transport = 0;
    const FEValuesExtractors::Vector velocities(0);
    for (const auto &cell : velocity_dof_handler.active_cell_iterators())
      if (cell->is_locally_owned())
      {
        fe_values.reinit(cell);
        fe_values[velocities].get_function_values(velocity_solution,
                                                  velocity_values);

        for (unsigned int q = 0; q < n_q_points; ++q)
        {
          const auto inverse_jacobian = Tensor<2,dim>(fe_values.inverse_jacobian(q));
          double convective_limit = 0;
          const auto convective_speed = inverse_jacobian * velocity_values[q];
          for(unsigned int d=0; d<dim; ++d)
              convective_limit = std::max(convective_limit, std::abs(convective_speed[d]));

          max_transport = std::max(max_transport, convective_limit);

        }
      }

    return Utilities::MPI::max(max_transport, MPI_COMM_WORLD);    
  }

  template <int dim>
  void LevelsetProblem<dim>::run_levelset()
  {
    std::cout << "run levelset" << std::endl;
    make_grid_and_dofs();
    {
      //levelset initial condition
      TrilinosWrappers::MPI::Vector tmp_vec(levelset_partitioning);
      VectorTools::interpolate(levelset_dof_handler,
                               LevelsetInitialValues<dim>(),
                               tmp_vec);
      levelset_solution = tmp_vec;
      levelset_old_solution = tmp_vec;
    }

    {
      TrilinosWrappers::MPI::Vector tmp_vec(velocity_partitioning);
      VectorTools::interpolate(velocity_dof_handler,
                               LevelsetVelocity<dim>(),
                               tmp_vec);
      velocity_solution = tmp_vec;
      velocity_old_old_solution = velocity_solution;
      velocity_old_solution = velocity_solution;
    }
    output_results(0);

    time_step = 0.0001;
    double current_time = 0.;
    global_max_velocity = get_maximal_velocity();
    pcout << " max vel = " << global_max_velocity << std::endl;

    for (int step = 0; step < 1000; ++step)
    {
      time_step = courant_number / compute_cell_convective_speed(velocity_solution);

      pcout << " step " << step << " done! "
            << " cell speed: " << compute_cell_convective_speed(velocity_solution)
            << " time is " << current_time << " time_step is "
            << time_step
            << " sol l2 norm: " << levelset_solution.l2_norm()
            << std::endl;
            return;

      assemble_levelset(global_max_velocity, time_step);
      levelset_old_solution = levelset_solution;
      current_time += time_step;
      if (step == 499)
        velocity_solution *= -1.;
      if ((step + 1) % 100 == 0)
        output_results(step);
    }
  }

} // namespace LevelsetMatrixbase

int main(int argc, char *argv[])
{
  try
  {
    using namespace LevelsetMatrixbase;
    using namespace dealii;

    Utilities::MPI::MPI_InitFinalize mpi_initialization(
        argc, argv, numbers::invalid_unsigned_int);

    LevelsetProblem<dimension> levelset_problem;
    levelset_problem.run_levelset();
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
