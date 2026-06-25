"""
    NeuralFlow

Top-level module for the Neural ODE project.

Usage (after `julia --project=.`):
    using NeuralFlow
    example_random()
"""
module NeuralFlow

include("AbstractRepresentation.jl")
include("AbstractModels.jl")
include("LinearInterp.jl")
include("SplineRepresentation.jl")
include("ChebyshevRepresentation.jl")
include("ActivationFunctions.jl")
include("BlockField.jl")
include("ODEModel.jl")
include("ResBlockModel.jl")
include("Solvers.jl")
include("GradientUtils.jl")
include("Tasks.jl")
include("TrainingUtils.jl")
include("ModelUtils.jl")
include("ModelInit.jl")
include("FixedTrainingData.jl")
include("DiscreteNN.jl")
include("Derivatives.jl")
include("Results.jl")
include("Adjoint.jl")
include("BatchedProblems.jl")
include("ExperimentUtils.jl")
include("MNISTTask.jl")
include("FiniteDiff.jl")

const _LOAD_PLOTS = get(ENV, "NEURALFLOW_HEADLESS", "0") != "1"
if _LOAD_PLOTS
    include("Examples.jl")
end

using .AbstractRepresentation
using .AbstractModels
using .LinearInterp
using .SplineRepresentation
using .ChebyshevRepresentation
using .ActivationFunctions
using .BlockField
using .ODEModel
using .ResBlockModel
using .Solvers
using .GradientUtils
using .TrainingUtils
using .ModelUtils
using .ModelInit
using .Tasks
using .FixedTrainingData
using .DiscreteNN
using .Derivatives
using .Results
using .Adjoint
using .BatchedProblems
using .ExperimentUtils
using .MNISTData
using .FiniteDiff
if _LOAD_PLOTS
    using .Examples
end

export AbstractFunctionRepresentation, vals, basis_weights, nodes, reconstruct, random_like
export change_representation
export device_weights, device_basis_weights, BasisCache
export interpolate_samples
export Spline, RandomSpline, trapz
export ChebPoly, RandomChebPoly
export sigmoid, d_sigmoid, relu, d_relu, d_tanh, activation_derivative
export AbstractModel, AbstractStatefulModel
export NeuralFlowODE, f
export AbstractFlowModel, embed_input, readout, state_dim, field_eltype, n_control_points
export BlockSpec, build_block, cpu_field_ops, layer_widths, block_nparams
export ResBlockFlow, init_resblock
export block_fast_value, block_fast_value_and_gradient
export batched_field_ops, block_stack_group, block_unstack_group!
export block_fast_group_value, block_fast_group_value_and_gradient
export block_fast_group_errors, block_fast_group_predict, train_block_batched
export solve_euler, solve_tsit5, solve_picard
export compute_error
export init_model, init_output_layer!, feature_scale_winit!, INIT_SCHEMES
export Adam, AdamFlat, SGD
export CurveSink, with_curve_sink, take_curves, record_curve!
export run_parallel
export TeeLog, logln, loss_logger, _envint, euler_steps, cell_path, run_cell, seed_cell, report
export model_path, ckpt_path, save_models, load_models, resume_enabled
export combine_callbacks, periodic_save_models
export num_concurrent, paired_ttest, mean_ci, anova_rm
export target_baseline, normalized_eval, heldout_error
export ConvergenceTracker, tracker_callback, converged, plateaued, final_values, track_matrix
export train_run, tracked_train_run, DATA_OFFSET, EVAL_OFFSET
export check_gradients
export n_params, n_params_breakdown, copy_model
export param_vector, set_params!
export group_by_shape, stack_group, unstack_group!, batched_group_value_and_gradient, train_batched, batched_heldout_errors, batched_heldout_sample_errors
export fast_group_value_and_gradient, fast_group_value
export fastadaptive_group_value_and_gradient, fastadaptive_group_value
export picard_group_value_and_gradient, picard_group_value
export AbstractTask
export DeterminantTask, NestedDeterminantTask
export CofactorDeterminantTask, RandomMinorDeterminantTask
export generate, input_dim, output_dim, task_name, compute_target, has_target_fn
export near_training_perturb, decomposed_error
export DataPool, sample_from_pool, generate_sampler, build_data_pool
export DiscreteNetwork, forward, backprop, train_discrete
export fd_gradients, fd_gradient_flat, fd_loss
export experiment_dir, save_run, load_run, Checkpointer, checkpoint!, on_improve!
export D₁f, D₂f, solve_DPx
export forward_gradients, forward_gradients_tsit5
export adjoint_gradients, adjoint_value_and_gradients
export batched_value_and_gradient, batched_predict
export fast_value_and_gradient, fast_value
export train_adjoint, train_determinant_adjoint
export squared_error_loss, softmax_crossentropy_loss
export MNISTTask, load_mnist, onehot, accuracy
export example_random, example_train_determinant

end
