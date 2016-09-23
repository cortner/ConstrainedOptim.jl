# AugmentedLagrangian.jl

[![Build Status](https://travis-ci.org/cortner/AugmentedLagrangian.jl.svg?branch=master)](https://travis-ci.org/cortner/AugmentedLagrangian.jl)

[![Coverage Status](https://coveralls.io/repos/cortner/AugmentedLagrangian.jl/badge.svg?branch=master&service=github)](https://coveralls.io/github/cortner/AugmentedLagrangian.jl?branch=master)

[![codecov.io](http://codecov.io/github/cortner/AugmentedLagrangian.jl/coverage.svg?branch=master)](http://codecov.io/github/cortner/AugmentedLagrangian.jl?branch=master)

This repository contains a straightforward implementation of
Nocedal & Wright, Sec. 17.4 "Augmented Lagrangian Methods", for
equality constraints only. This package is only intended as a starting
point for a more robust and more general implementation.

# How

Let us walk through the syntax (subject to changes without warning). Throughout
`x` is going to refer to the argument we are trying to vary to minimize our objective.
Say we want to minimize the sum of `x = [x1, x2]` subject to the constraint that `x` lies in
the circle with center at (0,0) and radius 2. We formulate our problem by specifying
our objective function `f` and its gradient `∇f`, and the constraint function `c`
and its jacobian `∇c`.
```julia
    f(x) = x[1]+x[2]
    g_f(x) = [1., 1.]
    c(x) = x[1]^2+x[2]^2-2
    J_c(x) = [2*x[1] 2*x[2]]

    # Initial value is arbitrary...
    initial_x = [-0.3, -0.5]
    solution_x = [-1.0, -1.0]
    F = DifferentiableFunction(f, (x,g) -> copy!(g, g_f(x)))
    C = DifferentiableFunction(c, (x,g) -> copy!(g, J_c(x)) )

    x, al = AugmentedLagrangianMethod.optimize(F, C, initial_x)
```
Notice how we specify the gradient as a (column) vector, but the Jacobian as a matrix
of dimension `length(c)` by `length(x)`. For consistancy, the constraint function should
also return a vector. If there is only one constraint, it doesn't matter, and the Jacobian
can be a vector as well. Keep the orientation in mind when adding multiple constraints.
Running the code, we see that indeed `x` is very close to `solution_x`.
```jlcon
julia> x
2-element Array{Float64,1}:
 -1.0
 -1.0

julia> solution_x
2-element Array{Float64,1}:
 -1.0
 -1.0
```

Say we have another problem. Now we want to minimize `x[1]+x[2]+x[3]^2` subject to
*two* constraints. The first is that `x[1]==1` and the second that `x[1]^2+x[2]^2==1`.
The (constrained) minimum is acheived at `x===[1.,0.,0.]`. Let us see if we can find it.
```julia
f(x) = x[1]+x[2]+x[3]^2
g_f(x) = [1.,
          1.,
          2x[3]]
c(x) = [x[1]-1,
        x[1]^2+x[2]^2-1]
J_c(x) = [1      0      0;
          2*x[1] 2*x[2] 0]

# Initial value is arbitrary...
initial_x = [0.3, 0.5, 0.1]
solution_x = [1.0, 0.0, 0.0]
F = DifferentiableFunction(f, (x,g) -> copy!(g, g_f(x)))
C = DifferentiableFunction(c, (x,g) -> copy!(g, J_c(x)))

x, al = AugmentedLagrangianMethod.optimize(F, C, initial_x)
```
The solution is seen to be quite close to the true solution, although
a tighter tolerance may be warranted in this case.
```jlcon
julia> x
3-element Array{Float64,1}:
  0.999999  
 -0.0012517
 -4.03099e-9
```
