abstract Constraint
abstract ConvexConstraint <: Constraint
immutable BoxConstraint <: ConvexConstraint
    lower
    upper
end

immutable BallConstraint <: ConvexConstraint
    center # center
    radius # radius
    p # l^p norm
end

immutable EqualityConstraint <: Constraint
    c
    Jc!
    cJc!
end

function EqualityConstraint(c, Jc!)
    function cJc!(x::Array, storage::Array)
        Jc!(x, storage)
        return c(x)
    end
    return EqualityConstraint(c, Jc!, cJc!)
end

type AugmentedLagrangian{Tl<:Union{Real, Vector}}
   F::DifferentiableFunction
   C::EqualityConstraint
   lambda::Tl
   mu::Real
   Dc::Matrix
end

typealias AL AugmentedLagrangian
function AugmentedLagrangian(F::DifferentiableFunction,
                 C::EqualityConstraint,
                 x0::AbstractVector;
                 lambda=:auto, mu=10.0)
   # TODO: maybe not ideal to evaluate F, C here, maybe better to set
   # dimensions when C is first called??? But see #282
   C0 = C.c(x0)
   dim_x = length(x0)
   dim_c = length(C0)
   if lambda == :auto
      lambda = - mu * C0
   end
   return AugmentedLagrangian(F, C, lambda, mu, zeros(dim_c, dim_x))
end
