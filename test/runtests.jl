using AugmentedLagrangianMethod
ALM = AugmentedLagrangianMethod
using Optim
using Base.Test

f(x) = x[1]^2 + x[2]^2 + x[2]^4
df(x) = [2*x[1]; 2*x[2] + 4*x[2]^3]
c(x) = x[1]^2 + (x[2]-0.5)^2 - 2.0
dc(x) = [ 2 * x[1]; 2 * (x[2] - 0.5) ]

# the minimiser should be [0.0,0.0]
x0 = [1.0; 0.0]


F = DifferentiableFunction(f, (x,g) -> copy!(g, df(x)) )
C = DifferentiableFunction(c, (x,g) -> copy!(g, dc(x)) )


println("------------------------------------------------------------")
println("       Finite Difference Testing the AL ")
# finite-difference test of the augmented Lagrangian implementation
al = ALM.AugmentedLagrangian(F, C, x0)
al.lambda = 0.0
al.mu = 1.0
A = ALM.evaluate(x0, al)
dA = ALM.gradient(x0, al)
err = Float64[]
for p = 2:12
   h = 0.1^p
   dAh = zeros(dA)
   for n = 1:length(x0)
      x0[n] += h
      dAh[n] = (ALM.evaluate(x0, al) - A)  / h
      x0[n] -= h
   end
   push!(err, vecnorm(dA - dAh, Inf))
   println("p = ", p, "; err = ", err[end])
end
if minimum(err) < 1e-4 * err[1]
   println("looks like the FD test has passed...")
else
   warn("""the finite difference test for the augmented Lagrangien didn't
         pass; please check visually what happened and debug""")
end


println("------------------------------------------------------------")
println("      Try to optimise something simple")
x, al = AugmentedLagrangianMethod.optimize(F, C, x0)
println("Converged to ", x, "; λ = ", al.lambda)
println("First-order optimality: ")
al.mu = 0.0; g = ALM.gradient(x, al); C = al.C.f(x)
println("   ∇ₓL(x, λ) = ", g)
println("        c(x) = ", C)
println("  |∇L(x, λ)| = ", max(vecnorm(g), vecnorm(C)))
@test vecnorm(g) < 1e-6
@test vecnorm(C) < 1e-6
