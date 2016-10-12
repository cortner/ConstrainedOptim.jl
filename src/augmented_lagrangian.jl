
#
# this is a trivial first write-up of NW, Algorithm 17.4
# with the new `update!` mechanism, I'd like to incorporate the
# Lagrange-multiplier and penalty updates into the minimisation
# procedure. But maybe this is not a good idea, to be tested.
#

function optimize( F::DifferentiableFunction,
                   C::EqualityConstraint,
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

      c = C.c(x)    # TODO: this is a superfluous call to C (but #282 would fix this)

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
   warn("too many iterations in `ConstrainedOptim.optimize` ")
   return x, al
end
