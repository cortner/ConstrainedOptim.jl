module AugmentedLagrangianMethod


using Optim

type AugmentedLagrangian
   F::DifferentiableFunction
   C::DifferentiableFunction
   lambda::Real
   mu::Real
   Dc::Matrix
end

typealias AL AugmentedLagrangian

dimx(al::AL) = size(Dc,1)  # dimension of state vector
dimc(al::AL) = size(Dc,2)  # number of constraints


function AugmentedLagrangian(F::DifferentiableFunction,
                 C::DifferentiableFunction,
                 x0::AbstractVector;
                 lambda=:auto, mu=10.0)
   # TODO: maybe not ideal to evaluate F, C here, maybe better to set
   # dimensions when C is first called??? But see #282
   C0 = C.f(x0)
   dimx = length(x0)
   dimc = length(C0)
   if lambda == :auto
      lambda = - mu * C0
   end
   return AugmentedLagrangian(F, C, lambda, mu, zeros(dimx, dimc))
end


function evaluate(x, al::AL)
   c = al.C.f(x)
   return al.F.f(x) - dot(c, al.lambda) + (0.5*al.mu) * sumabs2(c)
end

function gradient!(out, x, al::AL)
   eval_and_grad!(out, x, al)
   return out
end

function eval_and_grad!(out, x, al::AL)
   c = al.C.fg!(x, al.Dc)
   f = al.F.fg!(x, out)
   out[:] += - al.Dc * al.lambda + (1.0*al.mu) * al.Dc * c
   return f - dot(c, al.lambda) + (0.5*al.mu) * sumabs2(c)
end

gradient(x, al::AL) = gradient!(zeros(x), x, al)


#
# this is a trivial first write-up of NW, Algorithm 17.4
# with the new `update!` mechanism, I'd like to incorporate the
# Lagrange-multiplier and penalty updates into the minimisation
# procedure. But maybe this is not a good idea, to be tested.
#

function optimize( F::DifferentiableFunction,
                   C::DifferentiableFunction,
                   x0::AbstractVector;
                   iterations = 5000,
                   c_tol = 1e-6,
                   g_tol = 1e-6,
                   verbose = 1,
                   Optimizer = Optim.ConjugateGradient )

   # initialise
   al = AugmentedLagrangian(F, C, x0)
   eta = al.mu^(-0.1)
   # TODO: this needs to distinguish whether F, C are DifferentiableFunction
   # or TwiceDifferentiableFunction!!!!
   ALobj = DifferentiableFunction( x_ -> evaluate(x_, al),
                                   (x_,g_) -> gradient!(g_, x_, al),
                                   (x_,g_) -> eval_and_grad!(g_, x_, al) )


   x = copy(x0)
   iteration = 0
   if verbose >= 1
      @printf("  it  |     |c|_∞        |g_AL|_∞      μ         η  \n")
      @printf("------|----------------------------------------------\n")
   end
   while iteration < iterations

      # solve the sub-problem
      options = OptimizationOptions(g_tol=g_tol, iterations=iterations - iteration, store_trace = true)
      result = Optim.optimize(ALobj, x, Optimizer(), options)
      # TODO: unclear what to do if this step fails? it could still be ok after
      # we update the lagrange multipliers? For now just print a warning and continue.
      if !Optim.converged(result)
         warn("an inner AL iteration has not converged")
         # return minimizer(result)
      end

      # check for convergence, perform update for multipliers and tolerances
      iteration += Optim.iterations(result)
      x = Optim.minimizer(result)
      nrm_g = Optim.g_norm_trace(result)[end]

      c = C.f(x)    # TODO: this is a superfluous call to C (but #282 would fix this)

      if verbose >= 1
         @printf(" %4d |  %1.4e   %1.4e   %1.2e   %1.2e\n",
                  iteration, norm(x, Inf), nrm_g, al.mu, eta)
      end

      if norm(c, Inf) < c_tol
         return x, al
      elseif norm(c, Inf) <= eta
         al.lambda = al.lambda - al.mu * c
         eta /= al.mu^0.9  # tighten constraint tolerance
      else
         al.mu *= 100.0
         eta = al.mu^(-0.1)
      end
      if al.mu > 1e10
         warn("something horrible is happening")
         return x, al
      end

   end

   # iteration > iterations
   warn("too many iterations in `AugmentedLagrangianMethod.optimize` ")
   return x, al
end


end # module
