module ConstrainedOptim


using Optim
import Optim: optimize, Optimizer, @add_generic_fields()

export EqualityConstraint, BoxConstraint, BallConstraint, ProjectedGradientDescent, optimize
include("types.jl")
include("objective_helpers.jl")
include("augmented_lagrangian.jl")
include("optimize.jl")
end # module
