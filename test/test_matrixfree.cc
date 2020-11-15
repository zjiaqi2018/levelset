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
 * Author: Jiaqi Zhang, 2020
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


  template <int dim>
  class LevelsetVelocity : public Function<dim>
  {
  public:
    LevelsetVelocity()
      : Function<dim>(dim)
    {}
    virtual double value(const Point<dim> & point,
                         const unsigned int component = 0) const override;
  };

  template <int dim>
  double LevelsetVelocity<dim>::value(const Point<dim> & point,
                                      const unsigned int component) const
  {
    Assert(dim >= 2, ExcNotImplemented());
    const double pi = numbers::PI;
    const double x  = point[0];
    const double y  = point[1];

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

  template <int dim, int degree, int velocity_degree, int n_points_1d>
  class LevelsetOperator
  {
  public:
    LevelsetOperator(TimerOutput &timer_output);

    void
    reinit(const Mapping<dim> &                                 mapping,
           const std::vector<const DoFHandler<dim> *>          &dof_handlers,
           const std::vector<const AffineConstraints<double> *> constraints,
           const std::vector<Quadrature<1>>                     quadratures);

    void initialize_vector(LinearAlgebra::distributed::Vector<Number> &vector,
                           const unsigned int no_dof = 0) const;

    double compute_maximal_speed(
        const LinearAlgebra::distributed::Vector<Number> &solution) const;

  private:
    MatrixFree<dim, Number> data;

    TimerOutput &timer;
    
  };



  template <int dim, int degree, int velocity_degree, int n_points_1d>
  LevelsetOperator<dim, degree, velocity_degree, n_points_1d>::LevelsetOperator(
    TimerOutput &timer)
    : timer(timer)
  {}

  template <int dim, int degree, int velocity_degree, int n_points_1d>
  double LevelsetOperator<dim, degree, velocity_degree, n_points_1d>::
    compute_maximal_speed(
      const LinearAlgebra::distributed::Vector<Number> &solution) const
  {
    TimerOutput::Scope t(timer, "compute max speed");
    Number             max_speed = 0;
    FEEvaluation<dim, velocity_degree, degree + 1, dim, Number> vel_phi(data,
                                                                        1,
                                                                        1);

    for (unsigned int cell = 0; cell < data.n_cell_batches(); ++cell)
      {
        // std::cout<< "cell batch: "<< cell <<std::endl;
        vel_phi.reinit(cell);
        vel_phi.gather_evaluate(solution, EvaluationFlags::values);
        VectorizedArray<Number> local_max = 0.;
        for (unsigned int q = 0; q < vel_phi.n_q_points; ++q)
          {
            const auto velocity = vel_phi.get_value(q);
            local_max           = std::max(local_max, velocity * velocity);
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
    const std::vector<const DoFHandler<dim> *>          &dof_handlers,
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
    initialize_vector(LinearAlgebra::distributed::Vector<Number> &vector,
                      const unsigned int                          no_dof) const
  {
    data.initialize_dof_vector(vector, no_dof);
  }

  template <int dim>
  class LevelsetProblem
  {
  public:
    LevelsetProblem();

    void run();

  private:
    void make_grid_and_dofs();

    
    LinearAlgebra::distributed::Vector<Number> levelset_solution;
    LinearAlgebra::distributed::Vector<Number> levelset_old_solution;

    LinearAlgebra::distributed::Vector<Number> velocity_solution,
      velocity_old_solution, velocity_old_old_solution;

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

    LevelsetOperator<dim, fe_degree, vel_degree, n_q_points_1d>
      levelset_operator;

    double time, time_step, old_time_step;

    IndexSet                  velocity_locally_relevant_dofs;
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
    DoFTools::extract_locally_relevant_dofs(velocity_dof_handler,
                                            velocity_locally_relevant_dofs);
    velocity_constraints.clear();
    velocity_constraints.reinit(velocity_locally_relevant_dofs);
    DoFTools::make_hanging_node_constraints(velocity_dof_handler,
                                            velocity_constraints);
    velocity_constraints.close();

    const std::vector<const DoFHandler<dim> *> dof_handlers = {
      &levelset_dof_handler, &velocity_dof_handler};
    const AffineConstraints<double>                      dummy;
    const std::vector<const AffineConstraints<double> *> constraints = {
      &dummy, &velocity_constraints};
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

  template <int dim>
  void LevelsetProblem<dim>::run()
  {
    {
      const unsigned int n_vect_number = VectorizedArray<Number>::size();
      const unsigned int n_vect_bits   = 8 * sizeof(Number) * n_vect_number;

      pcout << "Running with "
            << Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD)
            << " MPI processes" << std::endl;
      pcout << "Vectorization over " << n_vect_number << " "
            << (std::is_same<Number, double>::value ? "doubles" : "floats")
            << " = " << n_vect_bits << " bits ("
            << Utilities::System::get_current_vectorization_level() << ")"
            << std::endl;
    }
    make_grid_and_dofs();

    VectorTools::interpolate(velocity_dof_handler,
                             LevelsetVelocity<dim>(),
                             velocity_solution);

    pcout << " max speed: "
              << levelset_operator.compute_maximal_speed(velocity_solution)
              << std::endl;
  }

} // namespace LevelsetMatrixfree

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
