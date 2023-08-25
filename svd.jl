using LinearAlgebra
using Random
Random.seed!(39)

#random 4x4 matrix
dim = 4
M = rand(dim, dim)
U,S,V = svd(M) #the SVD in Julia returns actually the right singular vectors V but not V^\dagger
Vd = adjoint(V)

print("U is ")
display(U) #use print+display to monitor a var
print("S is ")
display(S)
print("Vd is ")
display(Vd)

#check the n-th singular vector in the self-consisitant equation
n = 2
uvec = U[:,n]
display(uvec) #all vectors are colomn vectors!
sv = S[n]
vvec = Vd[n,:]
println("s*v = ", sv * vvec)
println("u*M = ", uvec' * M) #left multiply -> transpose


#find the largest singular value of a matrix
function largest_singular_value(mat::Matrix, it_steps::Int=100, tol::Float64=1e-15)
    """
    :param mat: input matrix (assume to be real)
    :param it_time: max iteration steps
    :param tol: tolerance of error
    :return u: the dominant left singular vector
    :return s: the dominant singular value
    :return v: the dominant right singular vector
    """
    dim1, dim2 = size(mat)[1], size(mat)[2]
    
    u, v = rand(dim1), rand(dim2) #randomize
    u, v = u/norm(u), v/norm(v) #normalize
    s = 1 #initialize the singular value

    for i=1:it_steps
        #use u*M to update v and s
        v1 = u' * M
        s1 = norm(v1)
        v1 /= s1
        #use M*v to update u and s
        u1 = M * v1'
        s1 = norm(u1)
        u1 /= s1
        #convergence
        conv = norm(u.-u1)/dim1 + norm(v.-v1)/dim2
        u,s,v = u1, s1, v1
        if conv < tol
            break
        end
    end
    return u,s,v
end

#check whether the function is working
um,sm,vm = largest_singular_value(M)
vdm = adjoint(vm)
println("largest right singular vector by iteration", um)
println("largest right singular vector by svd", U[:,1])

println("largest left singular vector by iteration", vdm)
println("largest left singular vector by svd", Vd[1,:])

println("largest singular value by iteration", sm)
println("largest singular value by svd", S[1])