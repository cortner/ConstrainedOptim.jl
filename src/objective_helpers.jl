function evaluate(x, al::AL)
   c = al.C.c(x)
   return al.F.f(x) - dot(c, al.lambda) + (0.5*al.mu) * sumabs2(c)
end

function gradient!(out, x, al::AL)
   eval_and_grad!(out, x, al)
   return out
end

function eval_and_grad!(out, x, al::AL)
   c = al.C.cJc!(x, al.Dc)
   f = al.F.fg!(x, out)
   for i = 1:length(c)
       out[:] = out - al.lambda[i] * view(al.Dc, i, :) + (1.0*al.mu) * c[i] * view(al.Dc, i, :)
   end
   return f - dot(c, al.lambda) + (0.5*al.mu) * sumabs2(c)
end

gradient(x, al::AL) = gradient!(zeros(x), x, al)
