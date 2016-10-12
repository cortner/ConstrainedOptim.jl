immutable ProjectedGradientDescent{T} <: Optimizer
    linesearch!::Function
    P::T
    precondprep!::Function
end

function trace!(tr, state, iteration, method::ProjectedGradientDescent, options)
    dt = Dict()
    if options.extended_trace
        dt["x"] = copy(state.x)
        dt["g(x)"] = copy(state.g)
        dt["Current step size"] = state.alpha
    end
    g_norm = vecnorm(state.g, Inf)
    Optim.update!(tr,
            iteration,
            state.f_x,
            g_norm,
            dt,
            options.store_trace,
            options.show_trace,
            options.show_every,
            options.callback)
end


project!(X, eq::ConvexConstraint) = nothing

function project!(X, eqc::BoxConstraint)
    @inbounds for (i, x) in enumerate(X)
        X[i] = max(min(x, eqc.upper[i]), eqc.lower[i])
    end
end

function project!(X, eqb::BallConstraint)
    @inbounds for (i, x) in enumerate(X)
        X[i] = x/max(eqb.radius, norm(x - eqb.center, eqb.p))
    end
end

ProjectedGradientDescent(; linesearch!::Function = Optim.backtracking_linesearch!,
                P = nothing, precondprep! = (P, x) -> nothing) =
                    ProjectedGradientDescent(linesearch!, P, precondprep!)


type ProjectedGradientDescentState{T}
    method_string::String
    n::Int64
    x::Array{T}
    f_x::T
    f_calls::Int64
    g_calls::Int64
    h_calls::Int64
    x_previous::Array{T}
    g::Array{T}
    f_x_previous::T
    s::Array{T}
    x_ls::Array{T}
    g_ls::Array{T}
    alpha::T
    mayterminate::Bool
    lsr::Optim.LineSearchResults
end

function initial_state{T}(method::ProjectedGradientDescent, options, d, eqc, initial_x::Array{T})
    g = similar(initial_x)
    f_x = d.fg!(initial_x, g)

    ProjectedGradientDescentState("Projected Gradient Descent",
                         length(initial_x),
                         copy(initial_x), # Maintain current state in state.x
                         f_x, # Store current f in state.f_x
                         1, # Track f calls in state.f_calls
                         1, # Track g calls in state.g_calls
                         0, # Track h calls in state.h_calls
                         copy(initial_x), # Maintain current state in state.x_previous
                         g, # Store current gradient in state.g
                         T(NaN), # Store previous f in state.f_x_previous
                         similar(initial_x), # Maintain current search direction in state.s
                         similar(initial_x), # Buffer of x for line search in state.x_ls
                         similar(initial_x), # Buffer of g for line search in state.g_ls
                         1., # Keep track of step size in state.alpha
                         false, # state.mayterminate
                         Optim.LineSearchResults(T))
end

function update_state!{T}(d, eqc, state::ProjectedGradientDescentState{T}, method::ProjectedGradientDescent)
    method.precondprep!(method.P, state.x)
    A_ldiv_B!(state.s, method.P, state.g)
    state.s = state.x - state.s
    project!(state.s, eqc)
    @simd for i in 1:state.n
        @inbounds state.s[i] = state.s[i]-state.x[i]
    end
    # Refresh the line search cache
    dphi0 = vecdot(state.g, state.s)
    Optim.clear!(state.lsr)
    Optim.push!(state.lsr, zero(T), state.f_x, dphi0)

    # Determine the distance of movement along the search line
    state.alpha, f_update, g_update =
    method.linesearch!(d, state.x, state.s, state.x_ls, state.g_ls,
                       state.lsr, state.alpha, state.mayterminate)
    state.f_calls, state.g_calls = state.f_calls + f_update, state.g_calls + g_update

    # Maintain a record of previous position
    copy!(state.x_previous, state.x)

    # Update current position # x = x + alpha * s
    LinAlg.axpy!(state.alpha, state.s, state.x)
    false # don't force break
end

function update_g!(d, state, method::ProjectedGradientDescent)
    # Update the function value and gradient
    state.f_x_previous, state.f_x = state.f_x, d.fg!(state.x, state.g)
    state.f_calls, state.g_calls = state.f_calls + 1, state.g_calls + 1
end


function optimize{T, M<:Optimizer}(d, initial_x::Array{T}, bc::ConvexConstraint, method::M, options::OptimizationOptions)
    t0 = time() # Initial time stamp used to control early stopping by options.time_limit

    if length(initial_x) == 1 && typeof(method) <: NelderMead
        error("Use optimize(f, scalar, scalar) for 1D problems")
    end


    state = initial_state(method, options, d, bc, initial_x)

    tr = OptimizationTrace{typeof(method)}()
    tracing = options.store_trace || options.show_trace || options.extended_trace || options.callback != nothing
    stopped, stopped_by_callback, stopped_by_time_limit = false, false, false

    x_converged, f_converged = false, false
    g_converged = if typeof(method) <: NelderMead
        nmobjective(state.f_simplex, state.m, state.n) < options.g_tol
    elseif  typeof(method) <: ParticleSwarm || typeof(method) <: SimulatedAnnealing
        g_converged = false
    else
        vecnorm(state.g, Inf) < options.g_tol
    end

    converged = g_converged
    iteration = 0

    options.show_trace && Optim.print_header(method)
    trace!(tr, state, iteration, method, options)

    while !converged && !stopped && iteration < options.iterations
        iteration += 1

        update_state!(d, bc, state, method) && break # it returns true if it's forced by something in update! to stop (eg dx_dg == 0.0 in BFGS)
        update_g!(d, state, method)
        x_converged, f_converged,
        g_converged, converged = Optim.assess_convergence(state, options)
        # We don't use the Hessian for anything if we have declared convergence,
        # so we might as well not make the (expensive) update if converged == true
        !converged && Optim.update_h!(d, state, method)

        # If tracing, update trace with trace!. If a callback is provided, it
        # should have boolean return value that controls the variable stopped_by_callback.
        # This allows for early stopping controlled by the callback.
        if tracing
            stopped_by_callback = trace!(tr, state, iteration, method, options)
        end

        # Check time_limit; if none is provided it is NaN and the comparison
        # will always return false.
        stopped_by_time_limit = time()-t0 > options.time_limit ? true : false

        # Combine the two, so see if the stopped flag should be changed to true
        # and stop the while loop
        stopped = stopped_by_callback || stopped_by_time_limit ? true : false
    end # while

    Optim.after_while!(d, state, method, options)

    return MultivariateOptimizationResults(state.method_string,
                                            initial_x,
                                            state.x,
                                            Float64(state.f_x),
                                            iteration,
                                            iteration == options.iterations,
                                            x_converged,
                                            options.x_tol,
                                            f_converged,
                                            options.f_tol,
                                            g_converged,
                                            options.g_tol,
                                            tr,
                                            state.f_calls,
                                            state.g_calls,
                                            state.h_calls)
end
