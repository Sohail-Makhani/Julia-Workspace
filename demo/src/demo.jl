using DifferentialEquations, BenchmarkTools, LinearAlgebra, SparseArrays, StatProfilerHTML
import Random

function odetest(U0, params, A0)
    A_op = DiffEqArrayOperator(A0, update_func=update_func!)
    prob = ODEProblem(A_op, U0, (0.0, 2.0), params, save_everystep=false)
    sol = solve(prob, MagnusGauss4(), dt=0.01)
end

function update_func!(A, u, p, t)
    A .= im * p[1] * cos(10t)
end

n = 150 # matrix size
Random.seed!(1)
X = SymTridiagonal(rand(n), rand(n-1))
U0 = Matrix{ComplexF64}(I, n, n)

# dense case
# params = [Matrix(X)]
# A0 = zeros(ComplexF64, n, n)
# # @btime odetest($U0, $params, $A0);
# @profilehtml odetest(U0, params, A0);

# sparse case
params = [sparse(X)]
A0 = spzeros(ComplexF64, n, n)
# @btime odetest($U0, $params, $A0);
@profilehtml odetest(U0, params, A0);